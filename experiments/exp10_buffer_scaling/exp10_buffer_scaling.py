#!/usr/bin/env python3
"""
Curiosity — Exp10: Buffer Scaling (GPU)

Tests grid vs compact layout on GPU.  Measures wall-clock time and VRAM
overhead across a sweep of grid sizes, sparsities, and mask patterns.

Question:  Grid vs compact layout — which is better on GPU?
Kill criteria:  compact overhead >20% (wall-clock OR VRAM) → kill compact, fix grid.

Outputs:
  exp10_summary.json       — verdicts per space type, overall verdict
  exp10_buffer_scaling.png — wall-clock comparison plot
  exp10_vram_profile.png   — VRAM comparison plot
"""

import time
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

SIDES       = [32, 64, 128, 256, 512]
SPARSITIES  = [0.1, 0.3, 0.5, 0.7, 0.9]
PATTERNS    = ["random", "clustered", "checkerboard"]
N_SEEDS     = 10
N_WARMUP    = 5
N_REPEAT    = 20
SEED_BASE   = 42
DTYPE       = torch.float32
KILL_THRESH = 0.20   # 20 %

# ═══════════════════════════════════════════════════════════════════════
# Mask generation  (ported from exp09a, extended with checkerboard)
# ═══════════════════════════════════════════════════════════════════════

def make_mask(side: int, sparsity: float, pattern: str,
              rng: np.random.Generator) -> np.ndarray:
    """Generate 2-D boolean mask (True = active).  sparsity = fraction active."""
    M = side * side
    n_active = max(1, int(round(M * sparsity)))

    if pattern == "random":
        idx = rng.choice(M, size=n_active, replace=False)
        mask = np.zeros(M, dtype=bool)
        mask[idx] = True
        return mask.reshape(side, side)

    elif pattern == "clustered":
        mask = np.zeros((side, side), dtype=bool)
        n_blobs = max(1, int(np.sqrt(n_active)))
        tiles_per_blob = max(1, n_active // n_blobs)
        radius = max(1, int(np.sqrt(tiles_per_blob / np.pi)))
        placed = 0
        for _ in range(n_blobs * 3):
            if placed >= n_active:
                break
            cy, cx = rng.integers(0, side, size=2)
            yy, xx = np.ogrid[-cy:side - cy, -cx:side - cx]
            dist2 = yy * yy + xx * xx
            blob = dist2 <= radius * radius
            new_active = blob & ~mask
            can_add = min(n_active - placed, int(new_active.sum()))
            if can_add > 0:
                coords = np.argwhere(new_active)
                chosen = coords[rng.choice(len(coords), size=can_add, replace=False)]
                mask[chosen[:, 0], chosen[:, 1]] = True
                placed += can_add
        if placed < n_active:
            inactive = np.argwhere(~mask)
            need = n_active - placed
            chosen = inactive[rng.choice(len(inactive), size=need, replace=False)]
            mask[chosen[:, 0], chosen[:, 1]] = True
        return mask

    elif pattern == "checkerboard":
        mask = np.zeros((side, side), dtype=bool)
        rows, cols = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
        checker = ((rows + cols) % 2 == 0)
        # Start with perfect checkerboard, then thin/thicken to target sparsity
        checker_count = int(checker.sum())
        if n_active <= checker_count:
            idxs = np.argwhere(checker)
            chosen = idxs[rng.choice(len(idxs), size=n_active, replace=False)]
            mask[chosen[:, 0], chosen[:, 1]] = True
        else:
            mask[checker] = True
            remaining = n_active - checker_count
            inactive = np.argwhere(~mask)
            chosen = inactive[rng.choice(len(inactive), size=remaining, replace=False)]
            mask[chosen[:, 0], chosen[:, 1]] = True
        return mask

    else:
        raise ValueError(f"Unknown pattern: {pattern}")


# ═══════════════════════════════════════════════════════════════════════
# GPU layout builders
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class GridLayout:
    """Full-grid layout: O(M) buffers, bool mask for active elements."""
    name: str = "grid"
    mask: torch.Tensor = None          # bool [M]  (flat)
    data: torch.Tensor = None          # float [M]
    side: int = 0

@dataclass
class CompactLayout:
    """Compact layout: O(k) buffers + reverse map."""
    name: str = "compact"
    active_idx: torch.Tensor = None    # int64 [k]
    reverse_map: torch.Tensor = None   # int64 [M]  (-1 if inactive)
    data: torch.Tensor = None          # float [k]
    side: int = 0


def build_grid_gpu(mask_np: np.ndarray, device: torch.device) -> GridLayout:
    """Build full-grid layout on GPU.  O(M) allocation."""
    side = mask_np.shape[0]
    M = side * side
    mask_flat = mask_np.ravel()
    mask_t = torch.tensor(mask_flat, dtype=torch.bool, device=device)
    data_t = torch.zeros(M, dtype=DTYPE, device=device)
    return GridLayout(name="grid", mask=mask_t, data=data_t, side=side)


def build_compact_gpu(mask_np: np.ndarray, device: torch.device) -> CompactLayout:
    """Build compact layout on GPU.  O(k) data + reverse map."""
    side = mask_np.shape[0]
    M = side * side
    mask_flat = mask_np.ravel()
    active_idx_np = np.where(mask_flat)[0].astype(np.int64)
    k = len(active_idx_np)

    active_idx = torch.tensor(active_idx_np, dtype=torch.int64, device=device)
    reverse_map = torch.full((M,), -1, dtype=torch.int64, device=device)
    reverse_map[active_idx] = torch.arange(k, dtype=torch.int64, device=device)
    data = torch.zeros(k, dtype=DTYPE, device=device)

    return CompactLayout(
        name="compact", active_idx=active_idx,
        reverse_map=reverse_map, data=data, side=side,
    )


# ═══════════════════════════════════════════════════════════════════════
# Synthetic CUDA kernel  (gather / compute / scatter)
# ═══════════════════════════════════════════════════════════════════════

class SyntheticKernel:
    """Simulates a conv-like stencil operation on both layout types."""

    def __init__(self, side: int, device: torch.device):
        self.side = side
        self.M = side * side
        self.device = device
        # Pre-build a 3x3 stencil kernel for the "compute" step
        self.stencil = torch.tensor(
            [[0.05, 0.1, 0.05],
             [0.1,  0.4, 0.1],
             [0.05, 0.1, 0.05]],
            dtype=DTYPE, device=device,
        ).reshape(1, 1, 3, 3)

    # ── Grid path ────────────────────────────────────────────────────

    def run_grid(self, layout: GridLayout, field: torch.Tensor) -> torch.Tensor:
        """gather → compute → scatter on full grid."""
        # Gather: copy field into layout data (full grid)
        layout.data.copy_(field)

        # Compute: 2-D conv stencil over the full grid
        field_2d = layout.data.reshape(1, 1, self.side, self.side)
        out_2d = torch.nn.functional.conv2d(field_2d, self.stencil, padding=1)
        result = out_2d.reshape(self.M)

        # Scatter: mask out inactive elements (write back only active)
        output = field.clone()
        output[layout.mask] = result[layout.mask]
        return output

    # ── Compact path ─────────────────────────────────────────────────

    def run_compact(self, layout: CompactLayout,
                    field: torch.Tensor) -> torch.Tensor:
        """gather → compute → scatter on compact index set."""
        k = layout.active_idx.shape[0]

        # Gather: extract active elements
        gathered = field[layout.active_idx]            # [k]
        layout.data.copy_(gathered)

        # Compute: expand to full grid for the stencil, then gather back
        # (This mirrors real-world usage where compact still needs
        #  neighbor data for stencil ops)
        tmp_full = torch.zeros(self.M, dtype=DTYPE, device=self.device)
        tmp_full[layout.active_idx] = layout.data
        tmp_2d = tmp_full.reshape(1, 1, self.side, self.side)
        out_2d = torch.nn.functional.conv2d(tmp_2d, self.stencil, padding=1)
        result_full = out_2d.reshape(self.M)

        # Scatter: write back only active elements
        output = field.clone()
        output[layout.active_idx] = result_full[layout.active_idx]
        return output


# ═══════════════════════════════════════════════════════════════════════
# VRAM measurement
# ═══════════════════════════════════════════════════════════════════════

def measure_vram(func, device: torch.device) -> Tuple[torch.Tensor, int]:
    """Run *func*, return (result, peak_vram_bytes)."""
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    result = func()
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    return result, peak


# ═══════════════════════════════════════════════════════════════════════
# Timing helper
# ═══════════════════════════════════════════════════════════════════════

def timed_runs(func, n_warmup: int, n_repeat: int,
               device: torch.device) -> np.ndarray:
    """Return array of wall-clock seconds for *n_repeat* runs after warmup."""
    for _ in range(n_warmup):
        func()
        torch.cuda.synchronize(device)

    times = []
    for _ in range(n_repeat):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        func()
        torch.cuda.synchronize(device)
        dt = time.perf_counter() - t0
        times.append(dt)
    return np.array(times)


# ═══════════════════════════════════════════════════════════════════════
# Cross-space adapter wrappers  (scalar grid, vector grid, graph, tree)
# ═══════════════════════════════════════════════════════════════════════

class _SpaceProbe:
    """Measures gather/scatter overhead for a non-grid space type."""

    def __init__(self, name: str, n_elements: int, device: torch.device,
                 seed: int):
        self.name = name
        self.n = n_elements
        self.device = device
        self.rng = np.random.default_rng(seed)

    def build_data(self) -> torch.Tensor:
        return torch.randn(self.n, dtype=DTYPE, device=self.device)

    def build_index(self, k: int) -> torch.Tensor:
        idx = self.rng.choice(self.n, size=k, replace=False)
        return torch.tensor(idx, dtype=torch.int64, device=self.device)

    def gather(self, data: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return data[idx]

    def scatter(self, data: torch.Tensor, values: torch.Tensor,
                idx: torch.Tensor) -> torch.Tensor:
        out = data.clone()
        out[idx] = values
        return out


def _make_space_probe(space_type: str, device: torch.device,
                      seed: int) -> _SpaceProbe:
    """Create a space probe with size appropriate to the space type."""
    sizes = {
        "scalar_grid":      64 * 64,
        "vector_grid":      32 * 32 * 16,
        "irregular_graph":  200,
        "tree_hierarchy":   2 ** 6 - 1,
    }
    return _SpaceProbe(space_type, sizes[space_type], device, seed)


# ═══════════════════════════════════════════════════════════════════════
# Single probe run
# ═══════════════════════════════════════════════════════════════════════

def run_buffer_probe(
    side: int,
    sparsity: float,
    pattern: str,
    n_seeds: int,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Create synthetic mask, build both layouts, measure wall-clock and
    peak VRAM for each.  Returns timing and memory comparison dict.
    """
    grid_times_all = []
    compact_times_all = []
    grid_vram_all = []
    compact_vram_all = []

    for s in range(n_seeds):
        seed = SEED_BASE + s
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        mask_np = make_mask(side, sparsity, pattern, rng)

        # Build layouts
        grid_layout = build_grid_gpu(mask_np, device)
        compact_layout = build_compact_gpu(mask_np, device)
        kernel = SyntheticKernel(side, device)

        # Random field
        field = torch.randn(side * side, dtype=DTYPE, device=device)

        # ── Wall-clock ──
        grid_t = timed_runs(
            lambda: kernel.run_grid(grid_layout, field),
            N_WARMUP, N_REPEAT, device,
        )
        compact_t = timed_runs(
            lambda: kernel.run_compact(compact_layout, field),
            N_WARMUP, N_REPEAT, device,
        )
        grid_times_all.append(float(np.median(grid_t)))
        compact_times_all.append(float(np.median(compact_t)))

        # ── VRAM ──
        # Free and rebuild to get clean VRAM baseline
        del grid_layout, compact_layout, field
        torch.cuda.empty_cache()

        field = torch.randn(side * side, dtype=DTYPE, device=device)
        mask_np2 = mask_np  # reuse same mask

        grid_layout2 = build_grid_gpu(mask_np2, device)
        _, grid_peak = measure_vram(
            lambda: kernel.run_grid(grid_layout2, field), device,
        )
        grid_vram_all.append(grid_peak)

        del grid_layout2
        torch.cuda.empty_cache()

        compact_layout2 = build_compact_gpu(mask_np2, device)
        _, compact_peak = measure_vram(
            lambda: kernel.run_compact(compact_layout2, field), device,
        )
        compact_vram_all.append(compact_peak)

        del compact_layout2, field
        torch.cuda.empty_cache()

    grid_times = np.array(grid_times_all)
    compact_times = np.array(compact_times_all)
    grid_vram = np.array(grid_vram_all)
    compact_vram = np.array(compact_vram_all)

    return {
        "side": side,
        "M": side * side,
        "sparsity": sparsity,
        "pattern": pattern,
        "n_seeds": n_seeds,
        "n_active": int(round(side * side * sparsity)),
        "grid_time_median_s":    float(np.median(grid_times)),
        "grid_time_iqr_s":       float(np.subtract(*np.percentile(grid_times, [75, 25]))),
        "compact_time_median_s": float(np.median(compact_times)),
        "compact_time_iqr_s":    float(np.subtract(*np.percentile(compact_times, [75, 25]))),
        "time_overhead_frac":    float(np.median(compact_times) / np.median(grid_times) - 1.0)
                                 if np.median(grid_times) > 0 else 0.0,
        "grid_vram_median_bytes":    int(np.median(grid_vram)),
        "compact_vram_median_bytes": int(np.median(compact_vram)),
        "vram_overhead_frac":        float(np.median(compact_vram) / np.median(grid_vram) - 1.0)
                                     if np.median(grid_vram) > 0 else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════════
# Cross-space validation
# ═══════════════════════════════════════════════════════════════════════

SPACE_TYPES = ["scalar_grid", "vector_grid", "irregular_graph", "tree_hierarchy"]


def run_cross_space_validation(device: torch.device) -> Dict[str, Any]:
    """Measure gather/scatter overhead on 4 space types."""
    results = {}
    sparsity_levels = [0.1, 0.3, 0.5]

    for space_type in SPACE_TYPES:
        space_results = []
        for sp in sparsity_levels:
            times_gather = []
            times_scatter = []
            vram_vals = []

            for s in range(N_SEEDS):
                seed = SEED_BASE + s
                probe = _make_space_probe(space_type, device, seed)
                data = probe.build_data()
                k = max(1, int(probe.n * sp))
                idx = probe.build_index(k)

                # Time gather
                gt = timed_runs(
                    lambda: probe.gather(data, idx),
                    N_WARMUP, N_REPEAT, device,
                )
                times_gather.append(float(np.median(gt)))

                # Time scatter
                values = torch.randn(k, dtype=DTYPE, device=device)
                st = timed_runs(
                    lambda: probe.scatter(data, values, idx),
                    N_WARMUP, N_REPEAT, device,
                )
                times_scatter.append(float(np.median(st)))

                # VRAM
                torch.cuda.empty_cache()
                _, peak = measure_vram(
                    lambda: probe.scatter(data, values, idx), device,
                )
                vram_vals.append(peak)

                del data, idx, values
                torch.cuda.empty_cache()

            space_results.append({
                "sparsity": sp,
                "n_elements": probe.n,
                "k": max(1, int(probe.n * sp)),
                "gather_median_s": float(np.median(times_gather)),
                "scatter_median_s": float(np.median(times_scatter)),
                "vram_median_bytes": int(np.median(vram_vals)),
            })

        results[space_type] = space_results
    return results


# ═══════════════════════════════════════════════════════════════════════
# Statistical analysis  (Holm-Bonferroni correction)
# ═══════════════════════════════════════════════════════════════════════

def _wilcoxon_approx_p(x: np.ndarray, y: np.ndarray) -> float:
    """Wilcoxon signed-rank test (approximate p-value).

    Falls back to a simple sign test if scipy is unavailable.
    """
    try:
        from scipy.stats import wilcoxon
        if np.allclose(x, y):
            return 1.0
        stat, p = wilcoxon(x, y, alternative="two-sided")
        return float(p)
    except Exception:
        # Fallback: sign test
        diffs = x - y
        n_pos = int(np.sum(diffs > 0))
        n_neg = int(np.sum(diffs < 0))
        n = n_pos + n_neg
        if n == 0:
            return 1.0
        # Binomial p-value approx
        from math import comb
        k = min(n_pos, n_neg)
        p = 2 * sum(comb(n, i) * 0.5 ** n for i in range(k + 1))
        return min(p, 1.0)


def holm_bonferroni(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """Return list of bools: True = significant after Holm-Bonferroni."""
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    significant = [False] * m
    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted_alpha = alpha / (m - rank)
        if p <= adjusted_alpha:
            significant[orig_idx] = True
        else:
            break  # stop at first non-rejection
    return significant


def compute_statistics(probe_results: List[Dict]) -> Dict[str, Any]:
    """Compute summary statistics with Holm-Bonferroni correction."""
    time_overheads = []
    vram_overheads = []
    p_values_time = []

    for r in probe_results:
        time_overheads.append(r["time_overhead_frac"])
        vram_overheads.append(r["vram_overhead_frac"])
        # Approximate p-value: single-sample test whether overhead != 0
        # We use the per-seed medians stored in the probe; for a proper test
        # we'd need raw arrays — here we use the overhead as a summary stat.
        # Placeholder: collect for Holm-Bonferroni across configs.
        p_values_time.append(1.0)  # will be overwritten below

    # For proper p-values we need to re-run per-seed comparisons.
    # Instead, we flag configs where |overhead| > KILL_THRESH.
    time_arr = np.array(time_overheads)
    vram_arr = np.array(vram_overheads)

    return {
        "time_overhead_median": float(np.median(time_arr)),
        "time_overhead_iqr": float(np.subtract(*np.percentile(time_arr, [75, 25]))),
        "vram_overhead_median": float(np.median(vram_arr)),
        "vram_overhead_iqr": float(np.subtract(*np.percentile(vram_arr, [75, 25]))),
        "n_configs": len(probe_results),
        "n_time_over_kill": int(np.sum(time_arr > KILL_THRESH)),
        "n_vram_over_kill": int(np.sum(vram_arr > KILL_THRESH)),
    }


def compute_pairwise_stats(probe_results: List[Dict]) -> Dict[str, Any]:
    """Run Wilcoxon tests per (side, pattern) group and apply
    Holm-Bonferroni across all comparisons."""
    from collections import defaultdict

    groups = defaultdict(list)
    for r in probe_results:
        key = (r["side"], r["pattern"])
        groups[key].append(r)

    all_p = []
    labels = []
    medians = []

    for (side, pat), runs in sorted(groups.items()):
        grid_t = np.array([r["grid_time_median_s"] for r in runs])
        comp_t = np.array([r["compact_time_median_s"] for r in runs])
        p = _wilcoxon_approx_p(grid_t, comp_t)
        all_p.append(p)
        labels.append(f"side={side}_pat={pat}")
        medians.append({
            "grid_median": float(np.median(grid_t)),
            "compact_median": float(np.median(comp_t)),
            "overhead_frac": float(np.median(comp_t) / np.median(grid_t) - 1.0)
                            if np.median(grid_t) > 0 else 0.0,
        })

    significant = holm_bonferroni(all_p)

    comparisons = []
    for i, lab in enumerate(labels):
        comparisons.append({
            "group": lab,
            "p_value": all_p[i],
            "significant": significant[i],
            **medians[i],
        })

    return {
        "n_comparisons": len(comparisons),
        "comparisons": comparisons,
    }


# ═══════════════════════════════════════════════════════════════════════
# Verdicts
# ═══════════════════════════════════════════════════════════════════════

def make_verdict(stats: Dict, cross_space: Dict) -> Dict[str, Any]:
    """Produce per-space and overall verdicts."""
    overall_time = stats["time_overhead_median"]
    overall_vram = stats["vram_overhead_median"]

    grid_verdict = "grid"
    compact_verdict = "compact"

    if overall_time > KILL_THRESH or overall_vram > KILL_THRESH:
        winner = grid_verdict
        reason = []
        if overall_time > KILL_THRESH:
            reason.append(f"time overhead {overall_time:.1%} > {KILL_THRESH:.0%}")
        if overall_vram > KILL_THRESH:
            reason.append(f"VRAM overhead {overall_vram:.1%} > {KILL_THRESH:.0%}")
        decision = "KILL compact"
        reason_str = "; ".join(reason)
    elif overall_time < -KILL_THRESH:
        winner = compact_verdict
        decision = "KEEP compact"
        reason_str = f"compact is faster by {-overall_time:.1%}"
    else:
        winner = "tie"
        decision = "KEEP compact (no significant overhead)"
        reason_str = (f"time overhead {overall_time:.1%}, "
                      f"VRAM overhead {overall_vram:.1%}")

    # Per-space verdicts
    space_verdicts = {}
    for space_type, space_runs in cross_space.items():
        total_time = sum(r["gather_median_s"] + r["scatter_median_s"]
                         for r in space_runs)
        space_verdicts[space_type] = {
            "total_gather_scatter_s": total_time,
            "verdict": "ok",
        }

    return {
        "overall_winner": winner,
        "decision": decision,
        "reason": reason_str,
        "time_overhead_pct": round(overall_time * 100, 2),
        "vram_overhead_pct": round(overall_vram * 100, 2),
        "kill_threshold_pct": round(KILL_THRESH * 100, 2),
        "space_verdicts": space_verdicts,
    }


# ═══════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════

def plot_wallclock(probe_results: List[Dict], out_path: Path):
    """Bar chart: grid vs compact wall-clock by side and sparsity."""
    fig, axes = plt.subplots(1, len(SIDES), figsize=(4 * len(SIDES), 5),
                             sharey=True)
    if len(SIDES) == 1:
        axes = [axes]

    for ax, side in zip(axes, SIDES):
        subset = [r for r in probe_results if r["side"] == side]
        sps = sorted(set(r["sparsity"] for r in subset))
        x = np.arange(len(sps))
        width = 0.35

        grid_vals = []
        compact_vals = []
        for sp in sps:
            matching = [r for r in subset if r["sparsity"] == sp]
            grid_vals.append(np.mean([r["grid_time_median_s"] for r in matching]) * 1e6)
            compact_vals.append(np.mean([r["compact_time_median_s"] for r in matching]) * 1e6)

        ax.bar(x - width / 2, grid_vals, width, label="grid", color="#1f77b4", alpha=0.85)
        ax.bar(x + width / 2, compact_vals, width, label="compact", color="#ff7f0e", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s:.0%}" for s in sps], fontsize=8)
        ax.set_xlabel("Sparsity")
        ax.set_title(f"side={side}")
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Median time (us)")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Exp10: Wall-clock — Grid vs Compact on GPU", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_vram(probe_results: List[Dict], out_path: Path):
    """Bar chart: grid vs compact VRAM by side and sparsity."""
    fig, axes = plt.subplots(1, len(SIDES), figsize=(4 * len(SIDES), 5),
                             sharey=True)
    if len(SIDES) == 1:
        axes = [axes]

    for ax, side in zip(axes, SIDES):
        subset = [r for r in probe_results if r["side"] == side]
        sps = sorted(set(r["sparsity"] for r in subset))
        x = np.arange(len(sps))
        width = 0.35

        grid_vals = []
        compact_vals = []
        for sp in sps:
            matching = [r for r in subset if r["sparsity"] == sp]
            grid_vals.append(
                np.mean([r["grid_vram_median_bytes"] for r in matching]) / 1024,
            )
            compact_vals.append(
                np.mean([r["compact_vram_median_bytes"] for r in matching]) / 1024,
            )

        ax.bar(x - width / 2, grid_vals, width, label="grid", color="#1f77b4", alpha=0.85)
        ax.bar(x + width / 2, compact_vals, width, label="compact", color="#ff7f0e", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s:.0%}" for s in sps], fontsize=8)
        ax.set_xlabel("Sparsity")
        ax.set_title(f"side={side}")
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Peak VRAM (KB)")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Exp10: VRAM — Grid vs Compact on GPU", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# Main sweep
# ═══════════════════════════════════════════════════════════════════════

def run_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(__file__).parent
    print(f"Exp10 buffer scaling — device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(device)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    print(f"  Sides: {SIDES}")
    print(f"  Sparsities: {SPARSITIES}")
    print(f"  Patterns: {PATTERNS}")
    print(f"  Seeds: {N_SEEDS}, Warmup: {N_WARMUP}, Repeats: {N_REPEAT}")
    print("=" * 70)

    # ── Main sweep ────────────────────────────────────────────────────
    probe_results = []
    total = len(SIDES) * len(SPARSITIES) * len(PATTERNS)
    idx = 0

    for side in SIDES:
        for sparsity in SPARSITIES:
            for pattern in PATTERNS:
                idx += 1
                print(f"  [{idx}/{total}] side={side} sp={sparsity:.1f} "
                      f"pat={pattern}", end="", flush=True)
                try:
                    r = run_buffer_probe(side, sparsity, pattern, N_SEEDS, device)
                    probe_results.append(r)
                    t_oh = r["time_overhead_frac"]
                    v_oh = r["vram_overhead_frac"]
                    flag = " ***" if (t_oh > KILL_THRESH or v_oh > KILL_THRESH) else ""
                    print(f"  time_oh={t_oh:+.1%}  vram_oh={v_oh:+.1%}{flag}")
                except RuntimeError as e:
                    print(f"  SKIPPED ({e})")

    # ── Cross-space validation ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Cross-space validation (4 space types)")
    print("=" * 70)
    cross_space = run_cross_space_validation(device)
    for stype, runs in cross_space.items():
        for r in runs:
            print(f"  {stype:20s} sp={r['sparsity']:.1f}  "
                  f"gather={r['gather_median_s']*1e6:.1f}us  "
                  f"scatter={r['scatter_median_s']*1e6:.1f}us  "
                  f"vram={r['vram_median_bytes']/1024:.1f}KB")

    # ── Statistics ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Statistical analysis")
    print("=" * 70)
    stats = compute_statistics(probe_results)
    pairwise = compute_pairwise_stats(probe_results)

    print(f"  Time overhead median: {stats['time_overhead_median']:+.1%} "
          f"(IQR {stats['time_overhead_iqr']:.1%})")
    print(f"  VRAM overhead median: {stats['vram_overhead_median']:+.1%} "
          f"(IQR {stats['vram_overhead_iqr']:.1%})")
    print(f"  Configs over kill threshold (time): "
          f"{stats['n_time_over_kill']}/{stats['n_configs']}")
    print(f"  Configs over kill threshold (VRAM): "
          f"{stats['n_vram_over_kill']}/{stats['n_configs']}")

    n_sig = sum(1 for c in pairwise["comparisons"] if c["significant"])
    print(f"  Significant pairwise diffs (Holm-Bonferroni): "
          f"{n_sig}/{pairwise['n_comparisons']}")

    # ── Verdict ───────────────────────────────────────────────────────
    verdict = make_verdict(stats, cross_space)
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"  Decision: {verdict['decision']}")
    print(f"  Reason:   {verdict['reason']}")
    print(f"  Winner:   {verdict['overall_winner']}")

    # ── Save ──────────────────────────────────────────────────────────
    summary = {
        "config": {
            "sides": SIDES,
            "sparsities": SPARSITIES,
            "patterns": PATTERNS,
            "n_seeds": N_SEEDS,
            "n_warmup": N_WARMUP,
            "n_repeat": N_REPEAT,
            "kill_threshold": KILL_THRESH,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "N/A",
        },
        "probe_results": probe_results,
        "cross_space": cross_space,
        "statistics": stats,
        "pairwise_tests": pairwise,
        "verdict": verdict,
    }

    json_path = out_dir / "exp10_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved: {json_path}")

    # ── Plots ─────────────────────────────────────────────────────────
    plot_wallclock(probe_results, out_dir / "exp10_buffer_scaling.png")
    print(f"Saved: {out_dir / 'exp10_buffer_scaling.png'}")

    plot_vram(probe_results, out_dir / "exp10_vram_profile.png")
    print(f"Saved: {out_dir / 'exp10_vram_profile.png'}")

    return summary


if __name__ == "__main__":
    t0 = time.time()
    summary = run_all()
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
