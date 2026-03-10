"""
Exp0.7b — Two-Stage Gate.

Stage 1: Binary "Is residual healthy?" check.
  If yes → collapse to residual-only (w_resid ≈ 1.0).
  If no  → Stage 2.

Stage 2: Utility-based soft weights with residual prior.
  U_i = median(gain_probe_i) - λ * FSR_probe_i - μ * instability_i
  Weights from U_i with residual minimum floor and var/hf ceiling.

Fixes vs Exp0.7:
  - Deterministic stratified probe (not random)
  - Two-stage prevents clean/blur collapse
  - Utility instead of raw correlation
  - Residual prior (innocent until proven guilty)
"""

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
import json

import sys
sys.path.insert(0, "/home/claude")
from exp04_combined_interest import (
    Exp04Config, quantile_normalize, aggregate_to_tiles,
    compute_hf, compute_residual_target_grid, compute_local_variance,
    compute_psnr, compute_false_split_rate, compute_miss_rates,
    select_tiles, compute_tile_gains_marginal, RunMetrics,
)
from exp05_break_oracle import (
    make_realistic_field, make_coarse_degraded,
    calibrate_delta_false_realistic, compute_delta_miss_realistic,
)
from exp06_adaptive import (
    degrade_clean, degrade_blur, degrade_noise, degrade_alias,
    degrade_jpeg, degrade_spatially_varying_noise,
)

ALL_DEGRADATIONS = {
    "clean": degrade_clean,
    "blur": degrade_blur,
    "noise": degrade_noise,
    "alias": degrade_alias,
    "jpeg": degrade_jpeg,
    "spatvar_noise": degrade_spatially_varying_noise,
}


# ─── Config ───────────────────────────────────────────────────────────

@dataclass
class TwoStageConfig:
    # Probe
    probe_fraction: float = 0.15   # 15% — larger than before
    jitter_sigma_frac: float = 0.005

    # Stage 1: residual health thresholds
    health_fsr_threshold: float = 0.20    # FSR on probe above this → unhealthy
    health_instab_threshold: float = 0.25 # fraction of probe tiles that flip → unhealthy

    # Stage 1: healthy mode
    healthy_resid_weight: float = 0.95
    healthy_other_weight: float = 0.025  # split equally among others

    # Stage 2: utility weights
    lambda_fsr: float = 2.0       # penalty for false split rate
    mu_instab: float = 1.0        # penalty for instability
    resid_min_weight: float = 0.20 # floor
    other_max_weight: float = 0.55 # ceiling per non-resid expert

    # Softmax
    temperature: float = 0.2


# ─── Expert signals ──────────────────────────────────────────────────

def compute_experts(coarse, coarse_interp, target, tile_size, clip_val=3.0):
    resid_full = compute_residual_target_grid(coarse_interp, target)
    resid_tile = aggregate_to_tiles(resid_full, tile_size)
    resid_norm = quantile_normalize(resid_tile, clip_val)

    hf = compute_hf(coarse)
    hf_norm = quantile_normalize(hf, clip_val)

    var = compute_local_variance(coarse)
    var_norm = quantile_normalize(var, clip_val)

    return {"resid": resid_norm, "hf": hf_norm, "var": var_norm}


# ─── Deterministic stratified probe ──────────────────────────────────

def select_probe_tiles(resid_tile: np.ndarray, probe_fraction: float) -> np.ndarray:
    """
    Deterministic stratified probe:
    - Split tiles into quartiles by residual energy
    - Sample equally from each quartile
    - Always include corners and center (boundary representatives)
    Returns boolean mask of probe tiles.
    """
    flat = resid_tile.flatten()
    n = len(flat)
    n_probe = max(4, int(probe_fraction * n))

    # Quartile stratification
    quartiles = np.percentile(flat, [25, 50, 75])
    bins = np.digitize(flat, quartiles)  # 0,1,2,3

    probe_idx = set()
    per_bin = max(1, n_probe // 4)

    for b in range(4):
        bin_indices = np.where(bins == b)[0]
        if len(bin_indices) > 0:
            # Evenly spaced within bin
            step = max(1, len(bin_indices) // per_bin)
            probe_idx.update(bin_indices[::step][:per_bin])

    # Always include corners and center
    ny, nx = resid_tile.shape
    corners = [0, nx-1, (ny-1)*nx, ny*nx-1, (ny//2)*nx + nx//2]
    for c in corners:
        if c < n:
            probe_idx.add(c)

    mask = np.zeros(n, dtype=bool)
    mask[list(probe_idx)] = True
    return mask.reshape(resid_tile.shape)


# ─── Probe diagnostics ───────────────────────────────────────────────

def run_probe_diagnostics(experts: dict, coarse_interp: np.ndarray,
                          target: np.ndarray, tile_size: int,
                          probe_mask: np.ndarray, delta_false: float) -> dict:
    """
    For each expert, compute on probe tiles:
    - median gain when expert's top tiles are selected
    - FSR (false split rate)
    """
    gains = compute_tile_gains_marginal(coarse_interp, target, tile_size)

    results = {}
    probe_gains = gains[probe_mask]
    n_probe = probe_mask.sum()

    for name, signal in experts.items():
        probe_scores = signal[probe_mask]

        # Select top 30% of probe tiles by this expert's score
        k = max(1, int(0.3 * n_probe))
        if len(probe_scores) > 0:
            threshold = np.partition(probe_scores.flatten(), -k)[-k]
            selected_in_probe = probe_scores >= threshold
            sel_gains = probe_gains[selected_in_probe]

            if len(sel_gains) > 0:
                median_gain = float(np.median(sel_gains))
                fsr = float(np.mean(sel_gains < delta_false))
            else:
                median_gain = 0.0
                fsr = 1.0
        else:
            median_gain = 0.0
            fsr = 1.0

        results[name] = {"median_gain": median_gain, "fsr": fsr}

    return results


def run_instability_probe(experts: dict, coarse, target, tile_size: int,
                          jitter_sigma_frac: float, seed: int,
                          clip_val: float = 3.0) -> dict:
    """Fraction of top-30% tiles that flip under input jitter."""
    from scipy.ndimage import zoom

    # Original top-30% masks
    original_tops = {}
    for name, signal in experts.items():
        flat = signal.flatten()
        k = max(1, int(0.3 * len(flat)))
        threshold = np.partition(flat, -k)[-k]
        original_tops[name] = signal >= threshold

    # Jittered
    drange = target.max() - target.min()
    sigma = jitter_sigma_frac * drange
    rng = np.random.RandomState(seed + 6666)
    target_j = target + rng.randn(*target.shape) * sigma

    factor = tile_size
    coarse_j = zoom(target_j, 1.0 / factor, order=1)
    coarse_interp_j = zoom(coarse_j, factor, order=1)[:target.shape[0], :target.shape[1]]

    experts_j = compute_experts(coarse_j, coarse_interp_j, target_j, tile_size, clip_val)

    instabilities = {}
    for name, signal_j in experts_j.items():
        flat_j = signal_j.flatten()
        k = max(1, int(0.3 * len(flat_j)))
        threshold_j = np.partition(flat_j, -k)[-k]
        top_j = signal_j >= threshold_j

        orig = original_tops[name]
        n_top = orig.sum()
        flipped = (orig != top_j).sum()
        instabilities[name] = float(flipped) / (2 * n_top + 1e-12)  # normalize by possible flips

    return instabilities


# ─── Two-stage gate ──────────────────────────────────────────────────

def two_stage_gate(probe_diag: dict, instabilities: dict,
                   cfg: TwoStageConfig) -> tuple[dict, str, str]:
    """
    Returns (weights, dominant_name, stage_used).

    Stage 1: check residual health.
    Stage 2: utility-based weights if residual unhealthy.
    """
    expert_names = list(probe_diag.keys())

    # Stage 1: Is residual healthy?
    resid_fsr = probe_diag["resid"]["fsr"]
    resid_instab = instabilities.get("resid", 0.0)

    residual_healthy = (
        resid_fsr <= cfg.health_fsr_threshold and
        resid_instab <= cfg.health_instab_threshold
    )

    if residual_healthy:
        # Collapse to residual
        weights = {}
        for name in expert_names:
            if name == "resid":
                weights[name] = cfg.healthy_resid_weight
            else:
                weights[name] = cfg.healthy_other_weight
        return weights, "resid", "stage1_healthy"

    # Stage 2: Utility-based weights
    utilities = {}
    for name in expert_names:
        d = probe_diag[name]
        inst = instabilities.get(name, 0.0)
        u = d["median_gain"] - cfg.lambda_fsr * d["fsr"] - cfg.mu_instab * inst
        utilities[name] = u

    # Softmax
    u_arr = np.array([utilities[n] for n in expert_names])
    u_arr = u_arr / (cfg.temperature + 1e-12)
    exp_u = np.exp(u_arr - u_arr.max())
    soft = exp_u / (exp_u.sum() + 1e-12)

    weights = {n: float(v) for n, v in zip(expert_names, soft)}

    # Apply floor/ceiling
    weights["resid"] = max(weights["resid"], cfg.resid_min_weight)
    for name in expert_names:
        if name != "resid":
            weights[name] = min(weights[name], cfg.other_max_weight)

    # Renormalize
    total = sum(weights.values())
    weights = {n: v / total for n, v in weights.items()}

    dominant = max(weights, key=weights.get)
    return weights, dominant, "stage2_utility"


# ─── Single run ──────────────────────────────────────────────────────

def run_single_twostage(target, seed, cfg, gate_cfg, delta_false, delta_miss, degrade_fn):
    from scipy.ndimage import zoom
    factor = cfg.tile_size

    coarse_clean = zoom(target, 1.0 / factor, order=1)
    coarse_interp_clean = zoom(coarse_clean, factor, order=1)[:target.shape[0], :target.shape[1]]
    coarse_deg = degrade_fn(coarse_clean, seed)
    coarse_interp_deg = zoom(coarse_deg, factor, order=1)[:target.shape[0], :target.shape[1]]

    # Experts from degraded coarse
    experts = compute_experts(coarse_deg, coarse_interp_deg, target, cfg.tile_size, cfg.norm_clip)

    # Deterministic probe
    probe_mask = select_probe_tiles(experts["resid"], gate_cfg.probe_fraction)

    # Probe diagnostics (gains against degraded — what system sees)
    probe_diag = run_probe_diagnostics(
        experts, coarse_interp_deg, target, cfg.tile_size, probe_mask, delta_false
    )

    # Instability
    instabilities = run_instability_probe(
        experts, coarse_deg, target, cfg.tile_size,
        gate_cfg.jitter_sigma_frac, seed, cfg.norm_clip
    )

    # Two-stage gate
    weights, dominant, stage = two_stage_gate(probe_diag, instabilities, gate_cfg)

    # Gated ρ
    rho = sum(weights[n] * experts[n] for n in experts)

    # Select tiles
    selected = select_tiles(rho, cfg.budget_fraction)

    # Gains (against clean coarse — ground truth)
    gains = compute_tile_gains_marginal(coarse_interp_clean, target, cfg.tile_size)

    # Reconstruct
    output = coarse_interp_clean.copy()
    ny, nx = selected.shape
    for iy in range(ny):
        for ix in range(nx):
            if selected[iy, ix]:
                sl = (slice(iy*cfg.tile_size, (iy+1)*cfg.tile_size),
                      slice(ix*cfg.tile_size, (ix+1)*cfg.tile_size))
                output[sl] = target[sl]

    mse = float(np.mean((output - target)**2))
    psnr = compute_psnr(mse)
    false_split = compute_false_split_rate(gains, selected, delta_false)
    miss_det, miss_bud = compute_miss_rates(rho, selected, gains, delta_miss)

    sel_gains = gains[selected]
    roi_median = float(np.median(sel_gains)) if len(sel_gains) > 0 else 0.0

    metrics = RunMetrics(
        variant=f"TwoStage({stage}/{dominant})",
        seed=seed, budget_fraction=cfg.budget_fraction,
        psnr=psnr, mse=mse,
        false_split_rate=false_split,
        miss_detector=miss_det, miss_budget=miss_bud,
        roi_median=roi_median, roi_q10=0.0,
    )

    diag = {
        "stage": stage, "dominant": dominant, "weights": weights,
        "probe_diag": {k: v for k, v in probe_diag.items()},
        "instabilities": instabilities,
        "resid_fsr_probe": probe_diag["resid"]["fsr"],
        "resid_instab": instabilities.get("resid", 0.0),
    }

    return metrics, diag


# ─── Baselines ────────────────────────────────────────────────────────

def run_baseline(variant_name, target, seed, cfg, delta_false, delta_miss, degrade_fn):
    from exp05_break_oracle import run_single_degraded
    from exp04_combined_interest import RhoResidualOnly, RhoFull
    variant = RhoResidualOnly() if variant_name == "Residual-only" else RhoFull()
    metrics, _, _ = run_single_degraded(variant, target, seed, cfg, delta_false, delta_miss, degrade_fn)
    return metrics


# ─── Main ─────────────────────────────────────────────────────────────

def run_exp07b():
    cfg = Exp04Config(
        field_size=256, tile_size=16,
        budget_fraction=0.3, n_seeds=10,
        results_dir="results/exp07b",
    )
    gate_cfg = TwoStageConfig()
    out_dir = Path(cfg.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for deg_name, deg_fn in ALL_DEGRADATIONS.items():
        print(f"\n{'#'*60}")
        print(f"# {deg_name}")
        print(f"{'#'*60}")

        delta_false = calibrate_delta_false_realistic(cfg, deg_fn)

        rm, cm, gm = [], [], []
        diags = []

        for seed in range(cfg.n_seeds):
            target = make_realistic_field(cfg.field_size, seed)
            dm = compute_delta_miss_realistic(target, cfg.tile_size)

            mr = run_baseline("Residual-only", target, seed, cfg, delta_false, dm, deg_fn)
            mc = run_baseline("HF+Residual+Var", target, seed, cfg, delta_false, dm, deg_fn)
            mg, diag = run_single_twostage(target, seed, cfg, gate_cfg, delta_false, dm, deg_fn)

            rm.append(mr); cm.append(mc); gm.append(mg)
            diags.append(diag)

            w = diag["weights"]
            print(f"  s{seed}: R={mr.psnr:.1f} C={mc.psnr:.1f} G={mg.psnr:.1f} "
                  f"[{diag['stage'][:6]}/{diag['dominant']}] "
                  f"w=[r:{w['resid']:.2f} h:{w['hf']:.2f} v:{w['var']:.2f}] "
                  f"rFSR={diag['resid_fsr_probe']:.2f} rInst={diag['resid_instab']:.2f}")

        rp = np.mean([m.psnr for m in rm]); cp = np.mean([m.psnr for m in cm])
        gp = np.mean([m.psnr for m in gm])
        rf = np.mean([m.false_split_rate for m in rm])
        cf = np.mean([m.false_split_rate for m in cm])
        gf = np.mean([m.false_split_rate for m in gm])
        best = max(rp, cp)

        n_s1 = sum(1 for d in diags if d["stage"].startswith("stage1"))
        avg_w = defaultdict(float)
        for d in diags:
            for k, v in d["weights"].items():
                avg_w[k] += v / len(diags)

        rows.append(dict(deg=deg_name, rp=rp, cp=cp, gp=gp, gap=gp-best,
                         rf=rf, cf=cf, gf=gf, s1=n_s1/10, w=dict(avg_w)))

        print(f"\n  [{deg_name}] R={rp:.2f} C={cp:.2f} G={gp:.2f} (Δ={gp-best:+.2f})")
        print(f"  [{deg_name}] FSR: R={rf:.3f} C={cf:.3f} G={gf:.3f}")
        print(f"  [{deg_name}] Stage1 rate: {n_s1}/{cfg.n_seeds}")
        print(f"  [{deg_name}] Avg w: {dict(avg_w)}")

    # Final
    print(f"\n{'='*105}")
    print(f"TWO-STAGE GATE SUMMARY")
    print(f"{'='*105}")
    print(f"{'Deg':<15} {'Resid':>7} {'Combo':>7} {'Gated':>7} {'Δbest':>7} "
          f"{'R_FSR':>7} {'C_FSR':>7} {'G_FSR':>7} "
          f"{'S1%':>5} {'w_r':>5} {'w_h':>5} {'w_v':>5}")
    print(f"{'-'*105}")

    for r in rows:
        w = r["w"]
        print(f"{r['deg']:<15} {r['rp']:>7.2f} {r['cp']:>7.2f} {r['gp']:>7.2f} "
              f"{r['gap']:>+7.2f} {r['rf']:>7.3f} {r['cf']:>7.3f} {r['gf']:>7.3f} "
              f"{r['s1']:>5.0%} {w.get('resid',0):>5.2f} {w.get('hf',0):>5.2f} {w.get('var',0):>5.2f}")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    run_exp07b()
