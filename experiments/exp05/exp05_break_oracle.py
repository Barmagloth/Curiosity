"""
Exp0.5 — Break the Oracle.

Goal: Find conditions where Residual-only ρ degrades and combined ρ becomes necessary.

Approach:
  - Generate realistic 2D fields with natural-image properties
    (textures, smooth gradients, edges, fine detail)
  - Apply 4 degradation modes to the "observed" coarse:
    1. Clean (baseline — same as Exp0.4)
    2. Blur (coarse is over-smoothed → residual underestimates edge importance)
    3. Noise (coarse has additive noise → residual sees noise as signal)
    4. Downsample-aliased (coarse loses fine structure → residual blind to it)
  - Run all 5 ρ variants × 4 degradation modes × 10 seeds
  - Same metrics, same statistical tests as Exp0.4

Key hypothesis: Residual oracle breaks when coarse representation is degraded,
because residual becomes a noisy/biased estimate of true importance.
"""

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
from typing import Literal
import json

# Import core machinery from Exp0.4
import sys
sys.path.insert(0, "/home/claude")
from exp04_combined_interest import (
    Exp04Config, quantile_normalize, aggregate_to_tiles, to_tile_level,
    compute_hf, compute_residual_target_grid, compute_local_variance,
    compute_psnr, compute_false_split_rate, compute_miss_rates,
    compute_stability_iou, compute_correlations, select_tiles,
    compute_tile_gains_marginal, holm_bonferroni,
    RhoVariant, RhoHFOnly, RhoResidualOnly, RhoHFResidual, RhoFull, RhoAntiControl,
    ALL_VARIANTS, SplitRecord, RunMetrics,
)


# ─── Realistic benchmark fields ──────────────────────────────────────

def make_realistic_field(size: int, seed: int) -> np.ndarray:
    """
    Generate a 2D field with natural-image-like properties:
    - Smooth gradients (sky-like)
    - Edges (object boundaries)
    - Textures (periodic + random fine detail)
    - Flat regions
    """
    rng = np.random.RandomState(seed)
    x = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(x, x)
    field = np.zeros((size, size), dtype=np.float64)

    # 1. Smooth gradient background (sky)
    angle = rng.uniform(0, 2 * np.pi)
    field += 0.3 * (np.cos(angle) * xx + np.sin(angle) * yy)

    # 2. Large smooth blobs (objects)
    for _ in range(3):
        cx, cy = rng.uniform(0.2, 0.8, 2)
        sx, sy = rng.uniform(0.05, 0.15, 2)
        amp = rng.uniform(0.2, 0.5)
        field += amp * np.exp(-((xx - cx)**2 / (2*sx**2) + (yy - cy)**2 / (2*sy**2)))

    # 3. Hard edges (rectangles and circles)
    for _ in range(2):
        cx, cy = rng.uniform(0.2, 0.8, 2)
        r = rng.uniform(0.05, 0.12)
        val = rng.uniform(0.2, 0.6)
        mask = ((xx - cx)**2 + (yy - cy)**2) < r**2
        field[mask] = val

    rect_x, rect_y = rng.uniform(0.1, 0.5, 2)
    rect_w, rect_h = rng.uniform(0.1, 0.3, 2)
    rect_val = rng.uniform(0.3, 0.7)
    mask_rect = (xx > rect_x) & (xx < rect_x + rect_w) & (yy > rect_y) & (yy < rect_y + rect_h)
    field[mask_rect] = rect_val

    # 4. Periodic texture in a region
    tx, ty = rng.uniform(0.3, 0.7, 2)
    tr = rng.uniform(0.08, 0.15)
    freq = rng.uniform(20, 60)
    texture_mask = ((xx - tx)**2 + (yy - ty)**2) < tr**2
    texture = 0.1 * np.sin(freq * xx) * np.cos(freq * 0.7 * yy)
    field[texture_mask] += texture[texture_mask]

    # 5. Fine detail (small gaussians — stars / speckles)
    for _ in range(15):
        cx, cy = rng.uniform(0.05, 0.95, 2)
        s = rng.uniform(0.002, 0.008)
        a = rng.uniform(0.05, 0.2)
        field += a * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2*s**2))

    # Normalize to [0, 1]
    field = (field - field.min()) / (field.max() - field.min() + 1e-10)

    return field


# ─── Degradation modes ───────────────────────────────────────────────

def degrade_clean(coarse_small: np.ndarray, seed: int) -> np.ndarray:
    """No degradation."""
    return coarse_small


def degrade_blur(coarse_small: np.ndarray, seed: int) -> np.ndarray:
    """Over-smooth the coarse: residual will underestimate edge importance."""
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(coarse_small, sigma=1.5)


def degrade_noise(coarse_small: np.ndarray, seed: int) -> np.ndarray:
    """Add noise to coarse: residual will see noise as signal."""
    rng = np.random.RandomState(seed + 7777)
    drange = coarse_small.max() - coarse_small.min()
    sigma = 0.15 * drange  # 15% noise — aggressive
    return coarse_small + rng.randn(*coarse_small.shape) * sigma


def degrade_alias(coarse_small: np.ndarray, seed: int) -> np.ndarray:
    """Downsample further and upsample back: lose fine structure in coarse."""
    from scipy.ndimage import zoom
    # Downsample 2x more, then back up — kills fine structure
    tiny = zoom(coarse_small, 0.5, order=0)  # nearest-neighbor down
    return zoom(tiny, 2.0, order=1)[:coarse_small.shape[0], :coarse_small.shape[1]]


DEGRADATIONS = {
    "clean": degrade_clean,
    "blur": degrade_blur,
    "noise": degrade_noise,
    "alias": degrade_alias,
}


# ─── Modified coarse pipeline ────────────────────────────────────────

def make_coarse_degraded(target: np.ndarray, tile_size: int, degrade_fn, seed: int):
    """
    Returns (coarse_small_degraded, coarse_interp_degraded, coarse_interp_clean).

    coarse_interp_degraded: what the system "sees" (used for ρ computation)
    coarse_interp_clean: clean coarse (used for gain ground truth)
    """
    from scipy.ndimage import zoom
    factor = tile_size

    # Clean coarse
    coarse_small_clean = zoom(target, 1.0 / factor, order=1)
    coarse_interp_clean = zoom(coarse_small_clean, factor, order=1)[:target.shape[0], :target.shape[1]]

    # Degraded coarse (what the system observes)
    coarse_small_deg = degrade_fn(coarse_small_clean, seed)
    coarse_interp_deg = zoom(coarse_small_deg, factor, order=1)[:target.shape[0], :target.shape[1]]

    return coarse_small_deg, coarse_interp_deg, coarse_interp_clean


# ─── Single run (adapted for degradation) ────────────────────────────

def run_single_degraded(variant: RhoVariant, target: np.ndarray, seed: int,
                        cfg: Exp04Config, delta_false: float, delta_miss: float,
                        degrade_fn) -> tuple[RunMetrics, list[SplitRecord], np.ndarray]:
    """Run with degraded coarse. Gain is computed against clean coarse (ground truth)."""

    coarse_deg, coarse_interp_deg, coarse_interp_clean = make_coarse_degraded(
        target, cfg.tile_size, degrade_fn, seed
    )

    # ρ uses degraded coarse (what the system sees)
    rho_result = variant.compute(
        coarse=coarse_deg, coarse_interp=coarse_interp_deg,
        target=target, seed=seed, clip_val=cfg.norm_clip,
    )
    tile_scores = rho_result["rho_total"]
    selected = select_tiles(tile_scores, cfg.budget_fraction)

    # Gain is computed against CLEAN coarse → ground truth importance
    gains = compute_tile_gains_marginal(coarse_interp_clean, target, cfg.tile_size)

    # Reconstruct (using clean coarse as base, selected tiles get target)
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
    miss_det, miss_bud = compute_miss_rates(tile_scores, selected, gains, delta_miss)

    sel_gains = gains[selected]
    roi_median = float(np.median(sel_gains)) if len(sel_gains) > 0 else 0.0
    roi_q10 = float(np.percentile(sel_gains, 10)) if len(sel_gains) > 0 else 0.0

    gains_flat = gains.flatten()
    tile_rho = {}
    for key, val in rho_result.items():
        if key != "rho_total":
            tile_rho[key] = to_tile_level(val, coarse_deg.shape, cfg.tile_size)
    corrs = compute_correlations(tile_rho, gains_flat)

    splits = []
    for iy in range(ny):
        for ix in range(nx):
            comp_vals = {k: float(v[iy, ix]) for k, v in tile_rho.items()}
            splits.append(SplitRecord(
                level=0, tile_iy=iy, tile_ix=ix,
                rho_total=float(tile_scores[iy, ix]),
                rho_components=comp_vals,
                gain_marginal=float(gains[iy, ix]),
                gain_type="marginal",
                selected=bool(selected[iy, ix]),
            ))

    metrics = RunMetrics(
        variant=variant.name, seed=seed,
        budget_fraction=cfg.budget_fraction,
        psnr=psnr, mse=mse,
        false_split_rate=false_split,
        miss_detector=miss_det, miss_budget=miss_bud,
        roi_median=roi_median, roi_q10=roi_q10,
        corr_components=corrs,
    )
    return metrics, splits, selected


# ─── Calibration ─────────────────────────────────────────────────────

def calibrate_delta_false_realistic(cfg: Exp04Config, degrade_fn) -> float:
    """δ_false from HF-only, seeds 0..2, with degradation."""
    variant = RhoHFOnly()
    all_gains = []
    for seed in range(3):
        target = make_realistic_field(cfg.field_size, seed)
        coarse_deg, coarse_interp_deg, coarse_interp_clean = make_coarse_degraded(
            target, cfg.tile_size, degrade_fn, seed
        )
        gains = compute_tile_gains_marginal(coarse_interp_clean, target, cfg.tile_size)
        rho = variant.compute(coarse_deg, coarse_interp_deg, target, seed, cfg.norm_clip)
        selected = select_tiles(rho["rho_total"], cfg.budget_fraction)
        sel_gains = gains[selected]
        all_gains.extend(sel_gains[sel_gains > 0].tolist())
    return cfg.delta_false_frac * np.median(all_gains) if all_gains else 1e-6


def compute_delta_miss_realistic(target: np.ndarray, tile_size: int) -> float:
    """δ_miss = q75(gain) in dense clean run."""
    from scipy.ndimage import zoom
    factor = tile_size
    coarse_small = zoom(target, 1.0 / factor, order=1)
    coarse_interp = zoom(coarse_small, factor, order=1)[:target.shape[0], :target.shape[1]]
    gains = compute_tile_gains_marginal(coarse_interp, target, tile_size)
    return float(np.percentile(gains, 75))


# ─── Statistical tests ───────────────────────────────────────────────

def run_stat_tests(all_metrics: list[RunMetrics], degradation: str):
    from scipy.stats import wilcoxon

    by_variant = defaultdict(list)
    for m in all_metrics:
        by_variant[m.variant].append(m)

    baseline_name = "Residual-only"
    if baseline_name not in by_variant:
        return

    baseline = sorted(by_variant[baseline_name], key=lambda m: m.seed)
    baseline_mse = np.array([m.mse for m in baseline])
    baseline_fsr = np.array([m.false_split_rate for m in baseline])

    print(f"\n  Statistical Tests [{degradation}] (vs {baseline_name}):")

    comparisons = [v for v in by_variant if v != baseline_name]
    all_pvalues = []
    test_labels = []

    for vname in comparisons:
        runs = sorted(by_variant[vname], key=lambda m: m.seed)
        if len(runs) != len(baseline):
            continue

        v_mse = np.array([m.mse for m in runs])
        v_fsr = np.array([m.false_split_rate for m in runs])

        try:
            _, p_mse = wilcoxon(baseline_mse, v_mse)
        except ValueError:
            p_mse = 1.0
        try:
            _, p_fsr = wilcoxon(baseline_fsr, v_fsr)
        except ValueError:
            p_fsr = 1.0

        all_pvalues.extend([p_mse, p_fsr])
        test_labels.extend([f"{vname}: MSE", f"{vname}: FSR"])

    significant = holm_bonferroni(all_pvalues)
    for label, p, sig in zip(test_labels, all_pvalues, significant):
        marker = "***" if sig else "   "
        print(f"    {marker} {label}: p={p:.4f}")


# ─── Main experiment ─────────────────────────────────────────────────

def run_exp05():
    cfg = Exp04Config(
        field_size=256,
        tile_size=16,
        budget_fraction=0.3,
        n_seeds=10,
        results_dir="results/exp05",
    )
    out_dir = Path(cfg.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}  # {degradation: {variant: [RunMetrics]}}

    for deg_name, deg_fn in DEGRADATIONS.items():
        print(f"\n{'#'*70}")
        print(f"# DEGRADATION: {deg_name}")
        print(f"{'#'*70}")

        delta_false = calibrate_delta_false_realistic(cfg, deg_fn)
        print(f"  δ_false = {delta_false:.6f}")

        deg_metrics = []

        for variant in ALL_VARIANTS:
            print(f"\n  --- {variant.name} ---")

            for seed in range(cfg.n_seeds):
                target = make_realistic_field(cfg.field_size, seed)
                delta_miss = compute_delta_miss_realistic(target, cfg.tile_size)

                metrics, splits, selected = run_single_degraded(
                    variant, target, seed, cfg, delta_false, delta_miss, deg_fn
                )
                deg_metrics.append(metrics)

                print(f"    seed={seed}: PSNR={metrics.psnr:.2f} "
                      f"FSR={metrics.false_split_rate:.3f} "
                      f"miss_d={metrics.miss_detector:.3f} "
                      f"ROI={metrics.roi_median:.4f} "
                      f"corr={list(metrics.corr_components.values())[:3]}")

        all_results[deg_name] = deg_metrics

        # Summary table per degradation
        print_deg_summary(deg_name, deg_metrics)
        run_stat_tests(deg_metrics, deg_name)

    # Save all results
    save_results(all_results, out_dir)

    # Final cross-degradation comparison
    print_cross_degradation(all_results)


def print_deg_summary(deg_name: str, metrics: list[RunMetrics]):
    by_variant = defaultdict(list)
    for m in metrics:
        by_variant[m.variant].append(m)

    print(f"\n  {'='*85}")
    print(f"  [{deg_name}] {'Variant':<28} {'PSNR':>7} {'MSE':>10} {'FSR':>7} "
          f"{'MissD':>7} {'MissB':>7} {'ROI_md':>8}")
    print(f"  {'-'*85}")

    for name, runs in by_variant.items():
        print(f"  [{deg_name}] {name:<28} "
              f"{np.mean([r.psnr for r in runs]):>7.2f} "
              f"{np.mean([r.mse for r in runs]):>10.6f} "
              f"{np.mean([r.false_split_rate for r in runs]):>7.3f} "
              f"{np.mean([r.miss_detector for r in runs]):>7.3f} "
              f"{np.mean([r.miss_budget for r in runs]):>7.3f} "
              f"{np.mean([r.roi_median for r in runs]):>8.5f}")

    # Correlations
    print(f"  Correlations [{deg_name}]:")
    for name, runs in by_variant.items():
        corr_keys = set()
        for r in runs:
            corr_keys.update(r.corr_components.keys())
        if corr_keys:
            parts = []
            for k in sorted(corr_keys):
                vs = [r.corr_components[k] for r in runs if k in r.corr_components]
                if vs:
                    parts.append(f"{k}={np.mean(vs):.3f}")
            print(f"    {name}: {', '.join(parts)}")


def print_cross_degradation(all_results: dict):
    """Show how each variant's advantage changes across degradation modes."""
    print(f"\n{'='*90}")
    print("CROSS-DEGRADATION COMPARISON: Residual-only dominance")
    print(f"{'='*90}")

    print(f"\n{'Degradation':<12} {'Resid PSNR':>11} {'Best-combo PSNR':>16} {'Δ':>7} "
          f"{'Resid FSR':>10} {'Best-combo FSR':>15} {'Δ':>7}")
    print(f"{'-'*80}")

    for deg_name, metrics in all_results.items():
        by_variant = defaultdict(list)
        for m in metrics:
            by_variant[m.variant].append(m)

        resid_psnr = np.mean([m.psnr for m in by_variant["Residual-only"]])
        resid_fsr = np.mean([m.false_split_rate for m in by_variant["Residual-only"]])

        combos = ["HF+Residual", "HF+Residual+Var"]
        best_combo_psnr = max(np.mean([m.psnr for m in by_variant[c]]) for c in combos if c in by_variant)
        best_combo_fsr = min(np.mean([m.false_split_rate for m in by_variant[c]]) for c in combos if c in by_variant)

        d_psnr = best_combo_psnr - resid_psnr
        d_fsr = best_combo_fsr - resid_fsr

        print(f"{deg_name:<12} {resid_psnr:>11.2f} {best_combo_psnr:>16.2f} {d_psnr:>+7.2f} "
              f"{resid_fsr:>10.3f} {best_combo_fsr:>15.3f} {d_fsr:>+7.3f}")

    print(f"\nΔ > 0 for PSNR = combo wins. Δ < 0 for FSR = combo wins.")
    print(f"Look for degradation modes where residual loses its dominance.")


def save_results(all_results: dict, out_dir: Path):
    for deg_name, metrics in all_results.items():
        run_dicts = []
        for m in metrics:
            d = {
                "degradation": deg_name,
                "variant": m.variant, "seed": m.seed,
                "budget": m.budget_fraction,
                "psnr": m.psnr, "mse": m.mse,
                "false_split_rate": m.false_split_rate,
                "miss_detector": m.miss_detector, "miss_budget": m.miss_budget,
                "roi_median": m.roi_median, "roi_q10": m.roi_q10,
                "gain_type": "marginal",
            }
            d.update(m.corr_components)
            run_dicts.append(d)

        with open(out_dir / f"metrics_{deg_name}.json", "w") as f:
            json.dump(run_dicts, f, indent=2)

    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    run_exp05()
