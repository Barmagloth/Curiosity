"""
Exp0.4 — Combined Interest.

Fixes applied:
  1. Residual always in target-grid (upsample → abs → tile-mean)
  2. Gain = marginal, explicitly named
  3. δ_false and δ_miss separated
  4. Anti-control = permute(norm(HF)), same normalization
  5. Paired Wilcoxon + Holm-Bonferroni on primary endpoints

Run: python exp04_combined_interest.py
"""

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from collections import defaultdict
import json

# ─── Config ───────────────────────────────────────────────────────────

@dataclass
class Exp04Config:
    field_size: int = 256
    tile_size: int = 16
    halo: int = 4

    budget_fraction: float = 0.3
    max_depth: int = 3
    n_seeds: int = 10

    input_jitter_sigma_frac: float = 0.005
    transform_jitter_shift: tuple = (0.3, 0.3)

    norm_clip: float = 3.0

    # δ_false: calibrated from HF-only seeds 0..2
    delta_false: float | None = None
    delta_false_frac: float = 0.05

    # δ_miss: q75 of gain in dense run, computed per seed
    # (not stored in config — computed on the fly)

    results_dir: str = "results/exp04"


# ─── Benchmark ────────────────────────────────────────────────────────

def make_benchmark_field(size: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    x = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, x)
    field = np.zeros((size, size), dtype=np.float64)

    field += 1.0 * np.exp(-((xx - 0.3)**2 + (yy - 0.2)**2) / 0.1)

    for _ in range(5):
        cx, cy = rng.uniform(-0.7, 0.7, 2)
        s = rng.uniform(0.005, 0.02)
        a = rng.uniform(0.3, 0.8)
        field += a * np.exp(-((xx - cx)**2 + (yy - cy)**2) / s)

    mask_circle = (xx**2 + yy**2) < 0.15
    field[mask_circle] += 0.5

    mask_rect = (np.abs(xx + 0.4) < 0.15) & (np.abs(yy + 0.3) < 0.2)
    field[mask_rect] += 0.4

    field += rng.randn(size, size) * 0.01
    return field


# ─── Coarse ───────────────────────────────────────────────────────────

def make_coarse(target: np.ndarray, tile_size: int):
    """Returns (coarse_small, coarse_interp). coarse_interp is in target-grid."""
    from scipy.ndimage import zoom
    factor = tile_size
    coarse_small = zoom(target, 1.0 / factor, order=1)
    coarse_interp = zoom(coarse_small, factor, order=1)[:target.shape[0], :target.shape[1]]
    return coarse_small, coarse_interp


# ─── Signals ──────────────────────────────────────────────────────────

def compute_hf(coarse: np.ndarray) -> np.ndarray:
    """HF energy on coarse (Laplacian)."""
    from scipy.ndimage import laplace
    return np.abs(laplace(coarse))


def compute_residual_target_grid(coarse_interp: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Residual in target-grid. Returns full-resolution abs error."""
    return np.abs(coarse_interp - target)


def compute_local_variance(coarse: np.ndarray, window: int = 3) -> np.ndarray:
    from scipy.ndimage import uniform_filter
    mean = uniform_filter(coarse, size=window)
    mean_sq = uniform_filter(coarse**2, size=window)
    return np.maximum(mean_sq - mean**2, 0.0)


# ─── Normalization ────────────────────────────────────────────────────

def quantile_normalize(signal: np.ndarray, clip_val: float = 3.0, eps: float = 1e-8) -> np.ndarray:
    q10, q50, q90 = np.percentile(signal, [10, 50, 90])
    z = (signal - q50) / (q90 - q10 + eps)
    return np.clip(z, -clip_val, clip_val)


# ─── Tile aggregation ────────────────────────────────────────────────

def aggregate_to_tiles(full_res: np.ndarray, tile_size: int) -> np.ndarray:
    """Mean-pool full-resolution signal to tile-level."""
    h, w = full_res.shape
    ny, nx = h // tile_size, w // tile_size
    scores = np.zeros((ny, nx))
    for iy in range(ny):
        for ix in range(nx):
            patch = full_res[iy*tile_size:(iy+1)*tile_size, ix*tile_size:(ix+1)*tile_size]
            scores[iy, ix] = patch.mean()
    return scores


def to_tile_level(signal: np.ndarray, coarse_shape: tuple, tile_size: int) -> np.ndarray:
    """Bring any signal to tile-level (coarse_shape)."""
    if signal.shape == coarse_shape:
        return signal
    return aggregate_to_tiles(signal, tile_size)


# ─── Rho Variants ────────────────────────────────────────────────────

@dataclass
class RhoVariant:
    name: str
    components: list[str]

    def compute(self, coarse, coarse_interp, target, seed, clip_val=3.0):
        raise NotImplementedError


class RhoHFOnly(RhoVariant):
    def __init__(self):
        super().__init__(name="HF-only", components=["HF"])

    def compute(self, coarse, coarse_interp, target, seed, clip_val=3.0):
        hf = compute_hf(coarse)
        hf_norm = quantile_normalize(hf, clip_val)
        return {"rho_total": hf_norm, "rho_HF": hf_norm}


class RhoResidualOnly(RhoVariant):
    def __init__(self):
        super().__init__(name="Residual-only", components=["Resid"])

    def compute(self, coarse, coarse_interp, target, seed, clip_val=3.0):
        tile_size = target.shape[0] // coarse.shape[0]
        # Residual in target-grid, then aggregate to tiles
        resid_full = compute_residual_target_grid(coarse_interp, target)
        resid_tile = aggregate_to_tiles(resid_full, tile_size)
        resid_norm = quantile_normalize(resid_tile, clip_val)
        return {"rho_total": resid_norm, "rho_resid": resid_norm}


class RhoHFResidual(RhoVariant):
    def __init__(self):
        super().__init__(name="HF+Residual", components=["HF", "Resid"])

    def compute(self, coarse, coarse_interp, target, seed, clip_val=3.0):
        tile_size = target.shape[0] // coarse.shape[0]
        hf_norm = quantile_normalize(compute_hf(coarse), clip_val)
        resid_full = compute_residual_target_grid(coarse_interp, target)
        resid_tile = aggregate_to_tiles(resid_full, tile_size)
        resid_norm = quantile_normalize(resid_tile, clip_val)
        rho = 0.5 * hf_norm + 0.5 * resid_norm
        return {"rho_total": rho, "rho_HF": hf_norm, "rho_resid": resid_norm}


class RhoFull(RhoVariant):
    def __init__(self):
        super().__init__(name="HF+Residual+Var", components=["HF", "Resid", "Var"])

    def compute(self, coarse, coarse_interp, target, seed, clip_val=3.0):
        tile_size = target.shape[0] // coarse.shape[0]
        hf_norm = quantile_normalize(compute_hf(coarse), clip_val)
        resid_full = compute_residual_target_grid(coarse_interp, target)
        resid_tile = aggregate_to_tiles(resid_full, tile_size)
        resid_norm = quantile_normalize(resid_tile, clip_val)
        var_norm = quantile_normalize(compute_local_variance(coarse), clip_val)
        rho = (1/3) * hf_norm + (1/3) * resid_norm + (1/3) * var_norm
        return {"rho_total": rho, "rho_HF": hf_norm, "rho_resid": resid_norm, "rho_var": var_norm}


class RhoAntiControl(RhoVariant):
    """Anti-control: HF + permuted(norm(HF)). Same distribution, destroyed structure."""
    def __init__(self):
        super().__init__(name="Anti-control(HF+permHF)", components=["HF", "PermHF"])

    def compute(self, coarse, coarse_interp, target, seed, clip_val=3.0):
        hf_norm = quantile_normalize(compute_hf(coarse), clip_val)
        # Permute: flatten, shuffle, reshape — same distribution, no structure
        rng = np.random.RandomState(seed + 9999)
        perm = hf_norm.flatten().copy()
        rng.shuffle(perm)
        perm_norm = perm.reshape(hf_norm.shape)
        rho = 0.5 * hf_norm + 0.5 * perm_norm
        return {"rho_total": rho, "rho_HF": hf_norm, "rho_permHF": perm_norm}


ALL_VARIANTS = [RhoHFOnly(), RhoResidualOnly(), RhoHFResidual(), RhoFull(), RhoAntiControl()]


# ─── Selection ────────────────────────────────────────────────────────

def select_tiles(scores: np.ndarray, budget_fraction: float) -> np.ndarray:
    flat = scores.flatten()
    k = max(1, int(budget_fraction * len(flat)))
    threshold = np.partition(flat, -k)[-k]
    return scores >= threshold


# ─── Gain (marginal) ─────────────────────────────────────────────────

def compute_tile_gains_marginal(coarse_interp: np.ndarray, target: np.ndarray,
                                tile_size: int) -> np.ndarray:
    """gain_marginal[tile] = MSE of that tile (coarse vs target).
    Since all selected tiles are applied simultaneously, marginal = isolated for this setup."""
    h, w = target.shape
    ny, nx = h // tile_size, w // tile_size
    gains = np.zeros((ny, nx), dtype=np.float64)
    for iy in range(ny):
        for ix in range(nx):
            sl = (slice(iy*tile_size, (iy+1)*tile_size),
                  slice(ix*tile_size, (ix+1)*tile_size))
            gains[iy, ix] = np.mean((coarse_interp[sl] - target[sl])**2)
    return gains


# ─── Dense run for δ_miss ────────────────────────────────────────────

def compute_delta_miss(target: np.ndarray, tile_size: int) -> float:
    """δ_miss = q75(gain_marginal) in dense run (all tiles)."""
    coarse_small, coarse_interp = make_coarse(target, tile_size)
    gains = compute_tile_gains_marginal(coarse_interp, target, tile_size)
    return float(np.percentile(gains, 75))


# ─── Metrics ─────────────────────────────────────────────────────────

@dataclass
class SplitRecord:
    level: int
    tile_iy: int
    tile_ix: int
    rho_total: float
    rho_components: dict
    gain_marginal: float
    gain_type: str  # always "marginal"
    selected: bool


@dataclass
class RunMetrics:
    variant: str
    seed: int
    budget_fraction: float
    psnr: float
    mse: float
    false_split_rate: float
    miss_detector: float
    miss_budget: float
    stability_iou_input: float | None = None
    stability_iou_transform: float | None = None
    delta_nodes_input: int | None = None
    delta_nodes_transform: int | None = None
    roi_median: float = 0.0
    roi_q10: float = 0.0
    corr_components: dict = field(default_factory=dict)


def compute_psnr(mse: float, max_val: float = 1.0) -> float:
    if mse < 1e-12:
        return 100.0
    return 10.0 * np.log10(max_val**2 / mse)


def compute_false_split_rate(gains: np.ndarray, selected: np.ndarray, delta_false: float) -> float:
    sel_gains = gains[selected]
    if len(sel_gains) == 0:
        return 0.0
    return float(np.mean(sel_gains < delta_false))


def compute_miss_rates(scores: np.ndarray, selected: np.ndarray,
                       gains: np.ndarray, delta_miss: float) -> tuple[float, float]:
    """
    miss_detector: GT tile never entered top by score
    miss_budget: GT tile was in top by score but cut by budget
    """
    gt_mask = gains > delta_miss
    if gt_mask.sum() == 0:
        return 0.0, 0.0

    median_score = np.median(scores)
    detected = scores >= median_score

    n_gt = gt_mask.sum()
    miss_detector = float((gt_mask & ~detected).sum()) / n_gt
    miss_budget = float((gt_mask & detected & ~selected).sum()) / n_gt
    return miss_detector, miss_budget


def compute_stability_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = (mask_a & mask_b).sum()
    union = (mask_a | mask_b).sum()
    return float(intersection) / float(union) if union > 0 else 1.0


def compute_correlations(rho_components: dict, gains_flat: np.ndarray) -> dict:
    corrs = {}
    for name, values in rho_components.items():
        flat = values.flatten()
        if len(flat) == len(gains_flat) and np.std(flat) > 1e-12:
            corrs[f"corr_{name}_gain"] = float(np.corrcoef(flat, gains_flat)[0, 1])
    return corrs


# ─── Core run ────────────────────────────────────────────────────────

def run_single(variant: RhoVariant, target: np.ndarray, seed: int,
               cfg: Exp04Config, delta_false: float, delta_miss: float
               ) -> tuple[RunMetrics, list[SplitRecord], np.ndarray]:
    """Returns (metrics, splits, selected_mask)."""
    coarse_small, coarse_interp = make_coarse(target, cfg.tile_size)

    rho_result = variant.compute(
        coarse=coarse_small, coarse_interp=coarse_interp,
        target=target, seed=seed, clip_val=cfg.norm_clip,
    )
    rho_total = rho_result["rho_total"]

    # All rho components should be at tile level (= coarse_small shape)
    tile_scores = rho_total
    selected = select_tiles(tile_scores, cfg.budget_fraction)

    # Gains (marginal)
    gains = compute_tile_gains_marginal(coarse_interp, target, cfg.tile_size)

    # Reconstruct
    output = coarse_interp.copy()
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

    # Correlations at tile level
    gains_flat = gains.flatten()
    tile_rho = {}
    for key, val in rho_result.items():
        if key != "rho_total":
            tile_rho[key] = to_tile_level(val, coarse_small.shape, cfg.tile_size)
    corrs = compute_correlations(tile_rho, gains_flat)

    # Split records
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


# ─── Stability ───────────────────────────────────────────────────────

def run_stability(variant: RhoVariant, target: np.ndarray, seed: int,
                  cfg: Exp04Config, base_mask: np.ndarray) -> dict:
    from scipy.ndimage import shift as ndshift

    results = {}

    # Input jitter
    drange = target.max() - target.min()
    sigma = cfg.input_jitter_sigma_frac * drange
    rng = np.random.RandomState(seed + 5000)
    target_j = target + rng.randn(*target.shape) * sigma

    cs_j, ci_j = make_coarse(target_j, cfg.tile_size)
    rho_j = variant.compute(cs_j, ci_j, target_j, seed, cfg.norm_clip)["rho_total"]
    mask_j = select_tiles(rho_j, cfg.budget_fraction)

    results["stability_iou_input"] = compute_stability_iou(base_mask, mask_j)
    results["delta_nodes_input"] = int(np.abs(base_mask.astype(int) - mask_j.astype(int)).sum())

    # Transform jitter
    target_s = ndshift(target, cfg.transform_jitter_shift, order=1)
    cs_s, ci_s = make_coarse(target_s, cfg.tile_size)
    rho_s = variant.compute(cs_s, ci_s, target_s, seed, cfg.norm_clip)["rho_total"]
    mask_s = select_tiles(rho_s, cfg.budget_fraction)

    results["stability_iou_transform"] = compute_stability_iou(base_mask, mask_s)
    results["delta_nodes_transform"] = int(np.abs(base_mask.astype(int) - mask_s.astype(int)).sum())

    return results


# ─── Statistical tests ───────────────────────────────────────────────

def holm_bonferroni(pvalues: list[float], alpha: float = 0.05) -> list[bool]:
    """Returns list of bools: True if significant after correction."""
    n = len(pvalues)
    indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
    significant = [False] * n
    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted_alpha = alpha / (n - rank)
        if p <= adjusted_alpha:
            significant[orig_idx] = True
        else:
            break  # all subsequent are non-significant
    return significant


def run_statistical_tests(all_metrics: list[RunMetrics]):
    """Paired Wilcoxon + Holm-Bonferroni on primary endpoints (MSE, false_split_rate)."""
    from scipy.stats import wilcoxon

    by_variant = defaultdict(list)
    for m in all_metrics:
        by_variant[m.variant].append(m)

    # Baseline = best single-signal variant (Residual-only based on previous results)
    baseline_name = "Residual-only"
    if baseline_name not in by_variant:
        print("WARNING: baseline not found, skipping stat tests")
        return

    baseline = sorted(by_variant[baseline_name], key=lambda m: m.seed)
    baseline_mse = np.array([m.mse for m in baseline])
    baseline_fsr = np.array([m.false_split_rate for m in baseline])

    comparisons = [v for v in by_variant if v != baseline_name]
    all_pvalues = []
    test_labels = []

    for vname in comparisons:
        runs = sorted(by_variant[vname], key=lambda m: m.seed)
        if len(runs) != len(baseline):
            continue

        v_mse = np.array([m.mse for m in runs])
        v_fsr = np.array([m.false_split_rate for m in runs])

        # MSE: test if variant is not worse (two-sided)
        try:
            _, p_mse = wilcoxon(baseline_mse, v_mse)
        except ValueError:
            p_mse = 1.0
        all_pvalues.append(p_mse)
        test_labels.append(f"{vname} vs {baseline_name}: MSE")

        # FSR: test if variant is different
        try:
            _, p_fsr = wilcoxon(baseline_fsr, v_fsr)
        except ValueError:
            p_fsr = 1.0
        all_pvalues.append(p_fsr)
        test_labels.append(f"{vname} vs {baseline_name}: FSR")

    # Holm-Bonferroni
    significant = holm_bonferroni(all_pvalues)

    print(f"\n{'='*80}")
    print("Statistical Tests (paired Wilcoxon + Holm-Bonferroni)")
    print(f"Baseline: {baseline_name}")
    print(f"{'-'*80}")
    for label, p, sig in zip(test_labels, all_pvalues, significant):
        marker = "***" if sig else "   "
        print(f"  {marker} {label}: p={p:.4f} {'(significant)' if sig else ''}")
    print(f"\nNote: secondary metrics reported WITHOUT multiple comparison correction.")


# ─── Runner ──────────────────────────────────────────────────────────

def calibrate_delta_false(cfg: Exp04Config) -> float:
    """Calibrate δ_false from HF-only, seeds 0..2."""
    variant = RhoHFOnly()
    all_gains = []
    for seed in range(3):
        target = make_benchmark_field(cfg.field_size, seed)
        coarse_small, coarse_interp = make_coarse(target, cfg.tile_size)
        gains = compute_tile_gains_marginal(coarse_interp, target, cfg.tile_size)
        rho = variant.compute(coarse_small, coarse_interp, target, seed, cfg.norm_clip)
        selected = select_tiles(rho["rho_total"], cfg.budget_fraction)
        sel_gains = gains[selected]
        all_gains.extend(sel_gains[sel_gains > 0].tolist())

    delta = cfg.delta_false_frac * np.median(all_gains) if all_gains else 1e-6
    print(f"Calibrated δ_false = {delta:.6f} (from {len(all_gains)} splits, HF-only seeds 0..2)")
    return delta


def run_experiment(cfg: Exp04Config):
    out_dir = Path(cfg.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Calibrate δ_false
    delta_false = calibrate_delta_false(cfg)

    all_run_metrics = []
    all_split_records = []

    for variant in ALL_VARIANTS:
        print(f"\n{'='*60}")
        print(f"Variant: {variant.name}")
        print(f"{'='*60}")

        for seed in range(cfg.n_seeds):
            target = make_benchmark_field(cfg.field_size, seed)

            # Step 2: δ_miss per seed (dense run)
            delta_miss = compute_delta_miss(target, cfg.tile_size)

            # Step 3: Main run
            metrics, splits, base_mask = run_single(
                variant, target, seed, cfg, delta_false, delta_miss
            )

            # Step 4: Stability
            stab = run_stability(variant, target, seed, cfg, base_mask)
            metrics.stability_iou_input = stab["stability_iou_input"]
            metrics.stability_iou_transform = stab["stability_iou_transform"]
            metrics.delta_nodes_input = stab["delta_nodes_input"]
            metrics.delta_nodes_transform = stab["delta_nodes_transform"]

            all_run_metrics.append(metrics)
            all_split_records.extend(splits)

            print(f"  seed={seed}: PSNR={metrics.psnr:.2f} "
                  f"FSR={metrics.false_split_rate:.3f} "
                  f"miss_d={metrics.miss_detector:.3f} "
                  f"miss_b={metrics.miss_budget:.3f} "
                  f"ROI={metrics.roi_median:.4f} "
                  f"stab={metrics.stability_iou_input:.3f}")

    # Save
    run_dicts = []
    for m in all_run_metrics:
        d = {
            "variant": m.variant, "seed": m.seed, "budget": m.budget_fraction,
            "psnr": m.psnr, "mse": m.mse,
            "false_split_rate": m.false_split_rate,
            "miss_detector": m.miss_detector, "miss_budget": m.miss_budget,
            "stability_iou_input": m.stability_iou_input,
            "stability_iou_transform": m.stability_iou_transform,
            "delta_nodes_input": m.delta_nodes_input,
            "delta_nodes_transform": m.delta_nodes_transform,
            "roi_median": m.roi_median, "roi_q10": m.roi_q10,
            "delta_false": delta_false,
            "gain_type": "marginal",
        }
        d.update(m.corr_components)
        run_dicts.append(d)

    with open(out_dir / "run_metrics.json", "w") as f:
        json.dump(run_dicts, f, indent=2)

    split_dicts = [
        {"level": s.level, "iy": s.tile_iy, "ix": s.tile_ix,
         "rho": s.rho_total, **s.rho_components,
         "gain_marginal": s.gain_marginal, "gain_type": s.gain_type, "sel": s.selected}
        for s in all_split_records
    ]
    with open(out_dir / "split_records.json", "w") as f:
        json.dump(split_dicts, f)

    print(f"\nResults saved to {out_dir}")
    print_summary(all_run_metrics)
    run_statistical_tests(all_run_metrics)


def print_summary(metrics: list[RunMetrics]):
    by_variant = defaultdict(list)
    for m in metrics:
        by_variant[m.variant].append(m)

    print(f"\n{'='*90}")
    print(f"{'Variant':<28} {'PSNR':>7} {'MSE':>10} {'FSR':>7} {'MissD':>7} "
          f"{'MissB':>7} {'ROI_md':>8} {'ROI_10':>8} {'StabI':>7} {'StabT':>7}")
    print(f"{'-'*90}")

    for name, runs in by_variant.items():
        vals = lambda attr: [getattr(r, attr) for r in runs]
        vals_nn = lambda attr: [v for v in vals(attr) if v is not None]

        print(f"{name:<28} "
              f"{np.mean(vals('psnr')):>7.2f} "
              f"{np.mean(vals('mse')):>10.6f} "
              f"{np.mean(vals('false_split_rate')):>7.3f} "
              f"{np.mean(vals('miss_detector')):>7.3f} "
              f"{np.mean(vals('miss_budget')):>7.3f} "
              f"{np.mean(vals('roi_median')):>8.5f} "
              f"{np.mean(vals('roi_q10')):>8.5f} "
              f"{np.mean(vals_nn('stability_iou_input')):>7.3f} "
              f"{np.mean(vals_nn('stability_iou_transform')):>7.3f}")

    print(f"\nCorrelations (ρ_i vs gain_marginal):")
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
            print(f"  {name}: {', '.join(parts)}")


if __name__ == "__main__":
    cfg = Exp04Config()
    run_experiment(cfg)
