"""
Exp0.7 — Soft Gating: weighted expert mixture for ρ.

Instead of binary switch (Exp0.6), experts get continuous weights
based on probe-set diagnostics (correlation with gain, stability).
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
    compute_correlations, select_tiles, compute_tile_gains_marginal,
    holm_bonferroni, RunMetrics,
)
from exp05_break_oracle import (
    make_realistic_field, make_coarse_degraded,
    degrade_clean, degrade_blur, degrade_noise, degrade_alias,
    calibrate_delta_false_realistic, compute_delta_miss_realistic,
)
from exp06_adaptive import degrade_jpeg, degrade_spatially_varying_noise


# ─── Config ───────────────────────────────────────────────────────────

@dataclass
class GateConfig:
    # Probe
    probe_fraction: float = 0.08    # 8% of tiles for probe

    # Gate weights
    alpha: float = 0.7              # weight of correlation in gate
    beta: float = 0.3               # weight of stability in gate
    temperature: float = 0.3        # softmax temperature
    ema_lambda: float = 0.3         # EMA smoothing for weights
    hysteresis: float = 0.15        # min gap to switch dominant expert

    # Emergency
    emergency_corr_threshold: float = 0.3  # all experts below this → emergency

    # Stability probe
    jitter_sigma_frac: float = 0.005


ALL_DEGRADATIONS = {
    "clean": degrade_clean,
    "blur": degrade_blur,
    "noise": degrade_noise,
    "alias": degrade_alias,
    "jpeg": degrade_jpeg,
    "spatvar_noise": degrade_spatially_varying_noise,
}


# ─── Expert signals ──────────────────────────────────────────────────

def compute_expert_signals(coarse, coarse_interp, target, tile_size, clip_val=3.0):
    """Compute all expert signals at tile level, normalized."""
    # Residual (target-grid → tile)
    resid_full = compute_residual_target_grid(coarse_interp, target)
    resid_tile = aggregate_to_tiles(resid_full, tile_size)
    resid_norm = quantile_normalize(resid_tile, clip_val)

    # HF (coarse-grid)
    hf = compute_hf(coarse)
    hf_norm = quantile_normalize(hf, clip_val)

    # Variance (coarse-grid)
    var = compute_local_variance(coarse)
    var_norm = quantile_normalize(var, clip_val)

    return {
        "resid": resid_norm,
        "hf": hf_norm,
        "var": var_norm,
    }


# ─── Probe-based diagnostics ─────────────────────────────────────────

def run_probe(experts: dict, coarse_interp: np.ndarray, target: np.ndarray,
              tile_size: int, probe_fraction: float, seed: int) -> dict:
    """
    Select probe tiles, compute gain, return correlation per expert.

    Probe tiles are selected randomly (not by any expert — neutral probe).
    """
    gains = compute_tile_gains_marginal(coarse_interp, target, tile_size)
    gains_flat = gains.flatten()

    n_tiles = len(gains_flat)
    n_probe = max(1, int(probe_fraction * n_tiles))

    rng = np.random.RandomState(seed + 3333)
    probe_idx = rng.choice(n_tiles, size=n_probe, replace=False)

    probe_gains = gains_flat[probe_idx]

    correlations = {}
    for name, signal in experts.items():
        sig_flat = signal.flatten()
        probe_signal = sig_flat[probe_idx]
        if np.std(probe_signal) > 1e-12 and np.std(probe_gains) > 1e-12:
            correlations[name] = float(np.corrcoef(probe_signal, probe_gains)[0, 1])
        else:
            correlations[name] = 0.0

    return correlations


def run_stability_probe(experts: dict, coarse, target, tile_size: int,
                        jitter_sigma_frac: float, seed: int, clip_val: float = 3.0) -> dict:
    """Compute stability per expert: how much does tile ranking change under input jitter."""
    from scipy.ndimage import zoom

    # Original rankings
    original_ranks = {}
    for name, signal in experts.items():
        flat = signal.flatten()
        original_ranks[name] = np.argsort(np.argsort(-flat))  # rank (0 = highest)

    # Jittered input
    drange = target.max() - target.min()
    sigma = jitter_sigma_frac * drange
    rng = np.random.RandomState(seed + 4444)
    target_j = target + rng.randn(*target.shape) * sigma

    factor = tile_size
    coarse_j = zoom(target_j, 1.0 / factor, order=1)
    coarse_interp_j = zoom(coarse_j, factor, order=1)[:target.shape[0], :target.shape[1]]

    experts_j = compute_expert_signals(coarse_j, coarse_interp_j, target_j, tile_size, clip_val)

    stabilities = {}
    for name, signal_j in experts_j.items():
        flat_j = signal_j.flatten()
        ranks_j = np.argsort(np.argsort(-flat_j))
        # Rank correlation between original and jittered
        from scipy.stats import spearmanr
        if np.std(flat_j) > 1e-12:
            rc, _ = spearmanr(original_ranks[name], ranks_j)
            stabilities[name] = max(0.0, float(rc))
        else:
            stabilities[name] = 0.0

    return stabilities


# ─── Gate ─────────────────────────────────────────────────────────────

def compute_gate_weights(correlations: dict, stabilities: dict,
                         gate_cfg: GateConfig,
                         prev_weights: dict | None = None,
                         prev_dominant: str | None = None) -> tuple[dict, str, bool]:
    """
    Compute gated weights from diagnostics.

    Returns: (weights, dominant_expert_name, is_emergency)
    """
    expert_names = list(correlations.keys())

    # Check emergency: all experts bad
    all_corrs = [max(correlations[n], 0.0) for n in expert_names]
    is_emergency = all(c < gate_cfg.emergency_corr_threshold for c in all_corrs)

    if is_emergency:
        # Emergency: equal weights, minimal splitting expected
        weights = {n: 1.0 / len(expert_names) for n in expert_names}
        dominant = "EMERGENCY"
        return weights, dominant, True

    # Raw scores
    raw = {}
    for name in expert_names:
        c = max(correlations.get(name, 0.0), 0.0)
        s = max(stabilities.get(name, 0.0), 0.0)
        raw[name] = gate_cfg.alpha * c + gate_cfg.beta * s

    # Softmax
    values = np.array([raw[n] for n in expert_names])
    values = values / (gate_cfg.temperature + 1e-12)
    exp_values = np.exp(values - values.max())
    softmax_values = exp_values / (exp_values.sum() + 1e-12)

    weights = {n: float(v) for n, v in zip(expert_names, softmax_values)}

    # EMA smoothing
    if prev_weights is not None:
        for n in expert_names:
            if n in prev_weights:
                weights[n] = gate_cfg.ema_lambda * weights[n] + (1 - gate_cfg.ema_lambda) * prev_weights[n]
        # Re-normalize
        total = sum(weights.values())
        weights = {n: v / total for n, v in weights.items()}

    # Dominant expert with hysteresis
    current_dominant = max(weights, key=weights.get)
    if prev_dominant and prev_dominant in weights:
        if weights[current_dominant] - weights[prev_dominant] < gate_cfg.hysteresis:
            current_dominant = prev_dominant  # keep previous

    return weights, current_dominant, False


def compute_gated_rho(experts: dict, weights: dict) -> np.ndarray:
    """ρ = Σ w_i * E_i"""
    result = None
    for name, signal in experts.items():
        w = weights.get(name, 0.0)
        if result is None:
            result = w * signal
        else:
            result = result + w * signal
    return result


# ─── Single run with soft gating ─────────────────────────────────────

def run_single_gated(target: np.ndarray, seed: int, cfg: Exp04Config,
                     gate_cfg: GateConfig, delta_false: float, delta_miss: float,
                     degrade_fn) -> tuple[RunMetrics, dict]:
    """Run with soft-gated ρ."""
    from scipy.ndimage import zoom

    factor = cfg.tile_size
    coarse_clean = zoom(target, 1.0 / factor, order=1)
    coarse_interp_clean = zoom(coarse_clean, factor, order=1)[:target.shape[0], :target.shape[1]]

    coarse_deg = degrade_fn(coarse_clean, seed)
    coarse_interp_deg = zoom(coarse_deg, factor, order=1)[:target.shape[0], :target.shape[1]]

    # Expert signals (from degraded coarse)
    experts = compute_expert_signals(coarse_deg, coarse_interp_deg, target, cfg.tile_size, cfg.norm_clip)

    # Probe diagnostics (gains computed against DEGRADED coarse_interp — what system sees)
    # But we also need clean gains for evaluation
    correlations = run_probe(experts, coarse_interp_deg, target, cfg.tile_size,
                             gate_cfg.probe_fraction, seed)
    stabilities = run_stability_probe(experts, coarse_deg, target, cfg.tile_size,
                                      gate_cfg.jitter_sigma_frac, seed, cfg.norm_clip)

    # Gate
    weights, dominant, is_emergency = compute_gate_weights(
        correlations, stabilities, gate_cfg
    )

    # Gated ρ
    rho = compute_gated_rho(experts, weights)

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
    roi_q10 = float(np.percentile(sel_gains, 10)) if len(sel_gains) > 0 else 0.0

    # Oracle correlations (for logging — not used in gate)
    gains_flat = gains.flatten()
    oracle_corrs = {}
    for name, signal in experts.items():
        flat = signal.flatten()
        if np.std(flat) > 1e-12 and np.std(gains_flat) > 1e-12:
            oracle_corrs[f"oracle_corr_{name}"] = float(np.corrcoef(flat, gains_flat)[0, 1])

    # Gate entropy
    w_arr = np.array(list(weights.values()))
    w_arr = w_arr[w_arr > 1e-12]
    gate_entropy = float(-np.sum(w_arr * np.log(w_arr)))

    metrics = RunMetrics(
        variant=f"SoftGate(dom={dominant})",
        seed=seed, budget_fraction=cfg.budget_fraction,
        psnr=psnr, mse=mse,
        false_split_rate=false_split,
        miss_detector=miss_det, miss_budget=miss_bud,
        roi_median=roi_median, roi_q10=roi_q10,
        corr_components=oracle_corrs,
    )

    diagnosis = {
        "weights": weights,
        "dominant": dominant,
        "is_emergency": is_emergency,
        "probe_correlations": correlations,
        "stabilities": stabilities,
        "gate_entropy": gate_entropy,
        **oracle_corrs,
    }

    return metrics, diagnosis


# ─── Baselines (from exp05/06) ───────────────────────────────────────

def run_baseline(variant_name, target, seed, cfg, delta_false, delta_miss, degrade_fn):
    """Run a fixed-rho baseline."""
    from exp05_break_oracle import run_single_degraded
    if variant_name == "Residual-only":
        from exp04_combined_interest import RhoResidualOnly
        variant = RhoResidualOnly()
    elif variant_name == "HF+Residual+Var":
        from exp04_combined_interest import RhoFull
        variant = RhoFull()
    else:
        raise ValueError(f"Unknown variant: {variant_name}")

    metrics, _, _ = run_single_degraded(variant, target, seed, cfg, delta_false, delta_miss, degrade_fn)
    return metrics


# ─── Main ─────────────────────────────────────────────────────────────

def run_exp07():
    cfg = Exp04Config(
        field_size=256, tile_size=16,
        budget_fraction=0.3, n_seeds=10,
        results_dir="results/exp07",
    )
    gate_cfg = GateConfig()
    out_dir = Path(cfg.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for deg_name, deg_fn in ALL_DEGRADATIONS.items():
        print(f"\n{'#'*70}")
        print(f"# {deg_name}")
        print(f"{'#'*70}")

        delta_false = calibrate_delta_false_realistic(cfg, deg_fn)

        resid_metrics = []
        combo_metrics = []
        gated_metrics = []
        gated_diagnoses = []

        for seed in range(cfg.n_seeds):
            target = make_realistic_field(cfg.field_size, seed)
            delta_miss = compute_delta_miss_realistic(target, cfg.tile_size)

            # Baselines
            m_resid = run_baseline("Residual-only", target, seed, cfg, delta_false, delta_miss, deg_fn)
            m_combo = run_baseline("HF+Residual+Var", target, seed, cfg, delta_false, delta_miss, deg_fn)
            resid_metrics.append(m_resid)
            combo_metrics.append(m_combo)

            # Soft-gated
            m_gated, diag = run_single_gated(target, seed, cfg, gate_cfg, delta_false, delta_miss, deg_fn)
            gated_metrics.append(m_gated)
            gated_diagnoses.append(diag)

            w = diag["weights"]
            print(f"  seed={seed}: "
                  f"Resid={m_resid.psnr:.1f} Combo={m_combo.psnr:.1f} "
                  f"Gated={m_gated.psnr:.1f} [{diag['dominant']}] "
                  f"w=[r:{w['resid']:.2f} h:{w['hf']:.2f} v:{w['var']:.2f}] "
                  f"H={diag['gate_entropy']:.2f}")

        # Summary
        r_psnr = np.mean([m.psnr for m in resid_metrics])
        c_psnr = np.mean([m.psnr for m in combo_metrics])
        g_psnr = np.mean([m.psnr for m in gated_metrics])
        r_fsr = np.mean([m.false_split_rate for m in resid_metrics])
        c_fsr = np.mean([m.false_split_rate for m in combo_metrics])
        g_fsr = np.mean([m.false_split_rate for m in gated_metrics])

        best_fixed = max(r_psnr, c_psnr)
        gap = g_psnr - best_fixed

        n_emergency = sum(1 for d in gated_diagnoses if d["is_emergency"])
        avg_entropy = np.mean([d["gate_entropy"] for d in gated_diagnoses])

        # Average weights
        avg_w = defaultdict(float)
        for d in gated_diagnoses:
            for k, v in d["weights"].items():
                avg_w[k] += v / len(gated_diagnoses)

        summary_rows.append({
            "degradation": deg_name,
            "resid_psnr": r_psnr, "combo_psnr": c_psnr, "gated_psnr": g_psnr,
            "resid_fsr": r_fsr, "combo_fsr": c_fsr, "gated_fsr": g_fsr,
            "gap_vs_best": gap, "avg_entropy": avg_entropy,
            "emergency_rate": n_emergency / cfg.n_seeds,
            "avg_weights": dict(avg_w),
        })

        print(f"\n  [{deg_name}] Resid={r_psnr:.2f} Combo={c_psnr:.2f} "
              f"Gated={g_psnr:.2f} (Δ={gap:+.2f} vs best-fixed)")
        print(f"  [{deg_name}] FSR: Resid={r_fsr:.3f} Combo={c_fsr:.3f} Gated={g_fsr:.3f}")
        print(f"  [{deg_name}] Avg weights: {dict(avg_w)}")
        print(f"  [{deg_name}] Emergency: {n_emergency}/{cfg.n_seeds}, Entropy: {avg_entropy:.3f}")

    # Final table
    print(f"\n{'='*100}")
    print(f"SOFT GATING SUMMARY")
    print(f"{'='*100}")
    print(f"{'Deg':<15} {'Resid':>7} {'Combo':>7} {'Gated':>7} {'Δbest':>7} "
          f"{'R_FSR':>7} {'C_FSR':>7} {'G_FSR':>7} "
          f"{'w_r':>5} {'w_h':>5} {'w_v':>5} {'Emrg':>5} {'H':>5}")
    print(f"{'-'*100}")

    for row in summary_rows:
        w = row["avg_weights"]
        print(f"{row['degradation']:<15} "
              f"{row['resid_psnr']:>7.2f} {row['combo_psnr']:>7.2f} {row['gated_psnr']:>7.2f} "
              f"{row['gap_vs_best']:>+7.2f} "
              f"{row['resid_fsr']:>7.3f} {row['combo_fsr']:>7.3f} {row['gated_fsr']:>7.3f} "
              f"{w.get('resid',0):>5.2f} {w.get('hf',0):>5.2f} {w.get('var',0):>5.2f} "
              f"{row['emergency_rate']:>5.0%} {row['avg_entropy']:>5.2f}")

    print(f"\nΔbest > 0 = gated beats best fixed. Goal: Δbest ≈ 0 everywhere (never worse).")

    # Save
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2, default=str)
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    run_exp07()
