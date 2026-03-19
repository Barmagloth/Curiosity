#!/usr/bin/env python3
"""
Curiosity -- Exp10e: Tile-Sparse Layouts (GPU)

Tests 3 tile-sparse layout candidates against grid baseline on GPU.
Each candidate uses tile-sparse, dense intra-tile storage with no
element-level reverse_map.  Lookup operates only on O(k) support set.

Candidates:
  A: Dense grid + bitset mask (improved baseline)
  B: Packed active tiles + sorted Morton keys + O(k) lookup
  C: Paged sparse tiles (macroblocks, 2-level addressing)

Kill criteria (per pattern class):
  overhead >20% vs grid in wall-clock OR VRAM --> killed for that class.

Outputs:
  results/exp10e_summary.json            -- all verdicts
  results/exp10e_time_comparison.png     -- wall-clock grouped by pattern
  results/exp10e_vram_comparison.png     -- VRAM grouped by pattern
  results/exp10e_halo_comparison.png     -- halo access grouped by pattern
  results/exp10e_memory_breakdown.png    -- stacked bar of memory components
"""

import time
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===================================================================
# Candidate imports
# ===================================================================

from candidate_a_bitset import benchmark_candidate_a as _raw_benchmark_A
from candidate_b_packed_tiles import benchmark_candidate_b as _raw_benchmark_B
from candidate_c_paged import benchmark_candidate_c as _raw_benchmark_C

# ===================================================================
# Configuration
# ===================================================================

SIDES       = [64, 128, 256]
SPARSITIES  = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70]
PATTERNS    = ["random", "clustered", "checkerboard"]
N_SEEDS     = 10
N_WARMUP    = 5
N_REPEAT    = 20
SEED_BASE   = 42
DTYPE       = torch.float32
KILL_THRESH = 0.20   # 20 %


# ===================================================================
# Adapters: normalize candidate outputs to flat format
#   {"wall_clock_ms": float, "peak_vram_bytes": int,
#    "halo_access_ms": float, "memory_breakdown": dict}
# ===================================================================

def _adapt_candidate_a(side, sparsity, pattern, n_seeds, device):
    """Adapter for Candidate A (bitset).

    Raw return has nested structure:
      aggregated.wall_clock_median_s  (seconds)
      aggregated.peak_vram_median_bytes
      aggregated.halo_time_median_s   (seconds)
      memory_breakdown: dict
    """
    raw = _raw_benchmark_A(side, sparsity, pattern, n_seeds, device)
    agg = raw["aggregated"]
    return {
        "wall_clock_ms": agg["wall_clock_median_s"] * 1e3,
        "peak_vram_bytes": int(agg["peak_vram_median_bytes"]),
        "halo_access_ms": agg["halo_time_median_s"] * 1e3,
        "memory_breakdown": raw["memory_breakdown"],
    }


def _adapt_candidate_b(side, sparsity, pattern, n_seeds, device):
    """Adapter for Candidate B (packed tiles).

    Raw return has nested structure:
      gcs_ms.median         (already ms)
      halo_ms.median        (already ms)
      peak_vram.median_bytes  (may be absent on CPU)
      memory: dict with tile_keys, tile_data, flat_idx, total, k, ratio_vs_full
    """
    raw = _raw_benchmark_B(side, sparsity, pattern, n_seeds, device)
    peak_vram = 0
    if "peak_vram" in raw:
        peak_vram = int(raw["peak_vram"]["median_bytes"])
    # Build memory_breakdown from the 'memory' sub-dict, excluding
    # non-byte fields so downstream code can iterate byte components.
    mem_raw = raw.get("memory", {})
    memory_breakdown = {
        "tile_keys": mem_raw.get("tile_keys_bytes", 0),
        "tile_data": mem_raw.get("tile_data_bytes", 0),
        "flat_idx": mem_raw.get("flat_idx_bytes", 0),
        "total": mem_raw.get("total_bytes", 0),
    }
    return {
        "wall_clock_ms": float(raw["gcs_ms"]["median"]),
        "peak_vram_bytes": peak_vram,
        "halo_access_ms": float(raw["halo_ms"]["median"]),
        "memory_breakdown": memory_breakdown,
    }


def _adapt_candidate_c(side, sparsity, pattern, n_seeds, device):
    """Adapter for Candidate C (paged).

    Raw return has per_page_size results keyed by page size (4, 8, 16).
    Each entry has gather_s, compute_s, scatter_s, halo_intra_s,
    halo_cross_s (all in seconds), peak_vram_bytes, memory_bytes (dict).

    Strategy: pick the page_size with lowest total wall-clock
    (gather + compute + scatter).
    """
    raw = _raw_benchmark_C(side, sparsity, pattern, n_seeds, device)
    per_ps = raw["per_page_size"]

    best_ps = None
    best_wall = float("inf")
    for ps, data in per_ps.items():
        wall_s = data["gather_s"] + data["compute_s"] + data["scatter_s"]
        if wall_s < best_wall:
            best_wall = wall_s
            best_ps = ps

    best = per_ps[best_ps]
    wall_clock_ms = (best["gather_s"] + best["compute_s"]
                     + best["scatter_s"]) * 1e3
    halo_access_ms = (best["halo_intra_s"] + best["halo_cross_s"]) * 1e3

    mem_raw = best.get("memory_bytes", {})
    memory_breakdown = {
        "page_data": mem_raw.get("page_data", 0),
        "page_keys": mem_raw.get("page_keys", 0),
        "page_masks": mem_raw.get("page_masks", 0),
        "page_lookup": mem_raw.get("page_lookup", 0),
        "total": mem_raw.get("total", 0),
    }

    return {
        "wall_clock_ms": float(wall_clock_ms),
        "peak_vram_bytes": int(best["peak_vram_bytes"]),
        "halo_access_ms": float(halo_access_ms),
        "memory_breakdown": memory_breakdown,
    }


CANDIDATES = {
    "A_bitset":       _adapt_candidate_a,
    "B_packed_tiles": _adapt_candidate_b,
    "C_paged":        _adapt_candidate_c,
}

# ===================================================================
# Mask generation (from exp10, extended)
# ===================================================================

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


# ===================================================================
# Grid baseline (inline)
# ===================================================================

def benchmark_grid_baseline(
    side: int,
    sparsity: float,
    pattern: str,
    n_seeds: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Benchmark the original full-grid layout as baseline.

    Full data tensor + bool mask (byte).
    Workflow: gather = data[mask], compute = conv2d on full grid,
              scatter = data[mask] = result.
    Returns dict with wall_clock_ms, peak_vram_bytes, halo_access_ms,
    memory_breakdown.
    """
    M = side * side
    stencil = torch.tensor(
        [[0.05, 0.1, 0.05],
         [0.1,  0.4, 0.1],
         [0.05, 0.1, 0.05]],
        dtype=DTYPE, device=device,
    ).reshape(1, 1, 3, 3)

    wall_times = []
    halo_times = []
    vram_peaks = []
    mem_breakdowns = []

    for s in range(n_seeds):
        seed = SEED_BASE + s
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        mask_np = make_mask(side, sparsity, pattern, rng)
        mask_flat = torch.tensor(mask_np.ravel(), dtype=torch.bool, device=device)
        data = torch.randn(M, dtype=DTYPE, device=device)

        # Memory breakdown
        data_bytes = data.nelement() * data.element_size()
        mask_bytes = mask_flat.nelement() * mask_flat.element_size()
        stencil_bytes = stencil.nelement() * stencil.element_size()
        mem_breakdowns.append({
            "data": data_bytes,
            "mask": mask_bytes,
            "stencil": stencil_bytes,
            "total": data_bytes + mask_bytes + stencil_bytes,
        })

        def _run():
            # Gather
            gathered = data[mask_flat]
            # Compute on full grid
            field_2d = data.reshape(1, 1, side, side)
            out_2d = torch.nn.functional.conv2d(field_2d, stencil, padding=1)
            result = out_2d.reshape(M)
            # Scatter
            output = data.clone()
            output[mask_flat] = result[mask_flat]
            return output

        # --- Halo access timing (separate: reading 1-ring neighbors) ---
        def _halo_access():
            field_2d = data.reshape(1, 1, side, side)
            # Simulate halo read: pad then crop is the halo access pattern
            padded = torch.nn.functional.pad(field_2d, (1, 1, 1, 1), mode="constant")
            # Read all 8 neighbors for each active cell
            _ = padded[:, :, 0:side, 0:side] + padded[:, :, 2:side+2, 2:side+2]
            torch.cuda.synchronize(device) if device.type == "cuda" else None
            return _

        # Warmup
        for _ in range(N_WARMUP):
            _run()
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        # Wall-clock: median of N_REPEAT
        times_ms = []
        for _ in range(N_REPEAT):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _run()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            dt = (time.perf_counter() - t0) * 1e3
            times_ms.append(dt)

        wall_times.append(float(np.median(times_ms)))

        # Halo access timing
        for _ in range(N_WARMUP):
            _halo_access()
        halo_ms = []
        for _ in range(N_REPEAT):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _halo_access()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            dt = (time.perf_counter() - t0) * 1e3
            halo_ms.append(dt)
        halo_times.append(float(np.median(halo_ms)))

        # VRAM measurement
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            _run()
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device)
        else:
            peak = 0
        vram_peaks.append(peak)

        # Cleanup
        del mask_flat, data
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Aggregate memory breakdown (use first seed -- layout-determined)
    avg_breakdown = {}
    for key in mem_breakdowns[0]:
        avg_breakdown[key] = int(np.mean([m[key] for m in mem_breakdowns]))

    return {
        "wall_clock_ms": float(np.median(wall_times)),
        "wall_clock_iqr_ms": float(
            np.subtract(*np.percentile(wall_times, [75, 25]))),
        "peak_vram_bytes": int(np.median(vram_peaks)),
        "halo_access_ms": float(np.median(halo_times)),
        "memory_breakdown": avg_breakdown,
    }


# ===================================================================
# Timing / VRAM helpers
# ===================================================================

def measure_vram(func, device: torch.device) -> Tuple[Any, int]:
    """Run *func*, return (result, peak_vram_bytes)."""
    if device.type != "cuda":
        return func(), 0
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    result = func()
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    return result, peak


# ===================================================================
# Statistical analysis
# ===================================================================

def _wilcoxon_p(x: np.ndarray, y: np.ndarray) -> float:
    """Wilcoxon signed-rank test (approximate p-value).
    Falls back to sign test if scipy unavailable."""
    try:
        from scipy.stats import wilcoxon
        if np.allclose(x, y):
            return 1.0
        stat, p = wilcoxon(x, y, alternative="two-sided")
        return float(p)
    except Exception:
        diffs = x - y
        n_pos = int(np.sum(diffs > 0))
        n_neg = int(np.sum(diffs < 0))
        n = n_pos + n_neg
        if n == 0:
            return 1.0
        from math import comb
        k = min(n_pos, n_neg)
        p = 2 * sum(comb(n, i) * 0.5 ** n for i in range(k + 1))
        return min(p, 1.0)


def holm_bonferroni(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """Return list of bools: True = significant after Holm-Bonferroni."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    significant = [False] * m
    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted_alpha = alpha / (m - rank)
        if p <= adjusted_alpha:
            significant[orig_idx] = True
        else:
            break
    return significant


def compute_pairwise_stats(
    baseline_results: Dict[str, Dict],
    candidate_results: Dict[str, Dict[str, Dict]],
    pattern: str,
) -> Dict[str, Any]:
    """Wilcoxon tests for each candidate vs baseline within a pattern group.

    Args:
        baseline_results: {config_key: result_dict} for grid baseline
        candidate_results: {candidate_name: {config_key: result_dict}}
        pattern: the pattern class to filter on

    Returns:
        Per-candidate p-values, significance, medians.
    """
    # Filter configs belonging to this pattern
    pattern_keys = [k for k in baseline_results if f"_pat={pattern}" in k]

    all_comparisons = []
    all_p = []
    labels = []

    for cand_name, cand_data in candidate_results.items():
        # Collect per-config baseline vs candidate wall-clock
        base_times = []
        cand_times = []
        for k in pattern_keys:
            if k in cand_data:
                base_times.append(baseline_results[k]["wall_clock_ms"])
                cand_times.append(cand_data[k]["wall_clock_ms"])

        if len(base_times) < 2:
            all_p.append(1.0)
            labels.append(cand_name)
            all_comparisons.append({
                "candidate": cand_name,
                "pattern": pattern,
                "n_configs": len(base_times),
                "p_value": 1.0,
                "significant": False,
                "baseline_median_ms": float(np.median(base_times)) if base_times else 0,
                "candidate_median_ms": float(np.median(cand_times)) if cand_times else 0,
            })
            continue

        base_arr = np.array(base_times)
        cand_arr = np.array(cand_times)
        p = _wilcoxon_p(base_arr, cand_arr)
        all_p.append(p)
        labels.append(cand_name)

        all_comparisons.append({
            "candidate": cand_name,
            "pattern": pattern,
            "n_configs": len(base_times),
            "p_value": p,
            "significant": False,  # will be updated after Holm
            "baseline_median_ms": float(np.median(base_arr)),
            "candidate_median_ms": float(np.median(cand_arr)),
            "overhead_pct": float(
                (np.median(cand_arr) / np.median(base_arr) - 1.0) * 100
            ) if np.median(base_arr) > 0 else 0.0,
        })

    # Holm-Bonferroni correction
    sig = holm_bonferroni(all_p)
    for i, comp in enumerate(all_comparisons):
        comp["significant"] = sig[i] if i < len(sig) else False

    return {
        "pattern": pattern,
        "n_comparisons": len(all_comparisons),
        "comparisons": all_comparisons,
    }


# ===================================================================
# Verdicts
# ===================================================================

def compute_verdicts(
    baseline_results: Dict[str, Dict],
    candidate_results: Dict[str, Dict[str, Dict]],
) -> Dict[str, Any]:
    """Compute per-candidate, per-pattern verdicts.

    Kill criterion: overhead >20% in VRAM or wall-clock for a pattern class.

    Returns:
        Structured verdict dict.
    """
    verdicts = {}

    for cand_name, cand_data in candidate_results.items():
        cand_verdicts = {}
        any_pass = False

        for pattern in PATTERNS:
            pattern_keys = [
                k for k in baseline_results
                if f"_pat={pattern}" in k
            ]

            if not pattern_keys:
                cand_verdicts[pattern] = "NO_DATA"
                continue

            time_overheads = []
            vram_overheads = []

            for k in pattern_keys:
                if k not in cand_data:
                    continue
                base = baseline_results[k]
                cand = cand_data[k]

                if base["wall_clock_ms"] > 0:
                    t_oh = (cand["wall_clock_ms"] / base["wall_clock_ms"] - 1.0)
                    time_overheads.append(t_oh)

                if base["peak_vram_bytes"] > 0:
                    v_oh = (cand["peak_vram_bytes"] / base["peak_vram_bytes"] - 1.0)
                    vram_overheads.append(v_oh)

            if not time_overheads:
                cand_verdicts[pattern] = "NO_DATA"
                continue

            med_time_oh = float(np.median(time_overheads))
            med_vram_oh = float(np.median(vram_overheads)) if vram_overheads else 0.0

            if med_time_oh > KILL_THRESH or med_vram_oh > KILL_THRESH:
                verdict = "FAIL"
                reason = []
                if med_time_oh > KILL_THRESH:
                    reason.append(f"time +{med_time_oh:.1%}")
                if med_vram_oh > KILL_THRESH:
                    reason.append(f"VRAM +{med_vram_oh:.1%}")
                reason_str = "; ".join(reason)
            elif med_time_oh < -0.05 and med_vram_oh < KILL_THRESH:
                verdict = "PASS"
                reason_str = f"time {med_time_oh:+.1%}, VRAM {med_vram_oh:+.1%}"
                any_pass = True
            else:
                verdict = "MARGINAL"
                reason_str = f"time {med_time_oh:+.1%}, VRAM {med_vram_oh:+.1%}"
                any_pass = True  # marginal still counts as alive

            cand_verdicts[pattern] = {
                "verdict": verdict,
                "reason": reason_str,
                "time_overhead_pct": round(med_time_oh * 100, 2),
                "vram_overhead_pct": round(med_vram_oh * 100, 2),
            }

            if verdict in ("PASS", "MARGINAL"):
                any_pass = True

        verdicts[cand_name] = {
            "per_pattern": cand_verdicts,
            "overall": "ALIVE" if any_pass else "KILLED",
        }

    return verdicts


def compute_hybrid_recommendation(
    verdicts: Dict[str, Any],
) -> Dict[str, str]:
    """Determine which candidate wins for each pattern class."""
    hybrid = {}
    for pattern in PATTERNS:
        best_cand = None
        best_time_oh = float("inf")

        for cand_name, cand_info in verdicts.items():
            pp = cand_info["per_pattern"].get(pattern)
            if pp is None or isinstance(pp, str):
                continue
            if pp["verdict"] in ("PASS", "MARGINAL"):
                if pp["time_overhead_pct"] < best_time_oh:
                    best_time_oh = pp["time_overhead_pct"]
                    best_cand = cand_name

        hybrid[pattern] = best_cand if best_cand else "grid_baseline"

    return hybrid


# ===================================================================
# Plotting
# ===================================================================

CAND_COLORS = {
    "grid_baseline": "#1f77b4",
    "A_bitset":      "#ff7f0e",
    "B_packed_tiles": "#2ca02c",
    "C_paged":       "#d62728",
}

ALL_LABELS = ["grid_baseline"] + list(CANDIDATES.keys())


def _grouped_bar(
    data_by_label: Dict[str, Dict[str, List[float]]],
    metric_key: str,
    ylabel: str,
    title: str,
    out_path: Path,
):
    """Grouped bar chart: x = pattern, bars = candidates.

    data_by_label[candidate_name][pattern] = list of values.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    n_patterns = len(PATTERNS)
    n_bars = len(ALL_LABELS)
    width = 0.8 / n_bars
    x = np.arange(n_patterns)

    for i, label in enumerate(ALL_LABELS):
        vals = []
        for pat in PATTERNS:
            raw = data_by_label.get(label, {}).get(pat, [])
            vals.append(float(np.median(raw)) if raw else 0.0)
        offset = (i - n_bars / 2 + 0.5) * width
        color = CAND_COLORS.get(label, "#888888")
        ax.bar(x + offset, vals, width, label=label, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(PATTERNS)
    ax.set_xlabel("Pattern")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_time_comparison(
    baseline_results: Dict[str, Dict],
    candidate_results: Dict[str, Dict[str, Dict]],
    out_path: Path,
):
    """Wall-clock grouped by pattern."""
    data = defaultdict(lambda: defaultdict(list))
    for k, v in baseline_results.items():
        pat = k.split("_pat=")[1]
        data["grid_baseline"][pat].append(v["wall_clock_ms"])
    for cand_name, cand_data in candidate_results.items():
        for k, v in cand_data.items():
            pat = k.split("_pat=")[1]
            data[cand_name][pat].append(v["wall_clock_ms"])

    _grouped_bar(data, "wall_clock_ms", "Median wall-clock (ms)",
                 "Exp10e: Wall-clock by Pattern", out_path)


def plot_vram_comparison(
    baseline_results: Dict[str, Dict],
    candidate_results: Dict[str, Dict[str, Dict]],
    out_path: Path,
):
    """VRAM grouped by pattern."""
    data = defaultdict(lambda: defaultdict(list))
    for k, v in baseline_results.items():
        pat = k.split("_pat=")[1]
        data["grid_baseline"][pat].append(v["peak_vram_bytes"] / 1024)
    for cand_name, cand_data in candidate_results.items():
        for k, v in cand_data.items():
            pat = k.split("_pat=")[1]
            data[cand_name][pat].append(v["peak_vram_bytes"] / 1024)

    _grouped_bar(data, "peak_vram_bytes", "Peak VRAM (KB)",
                 "Exp10e: VRAM by Pattern", out_path)


def plot_halo_comparison(
    baseline_results: Dict[str, Dict],
    candidate_results: Dict[str, Dict[str, Dict]],
    out_path: Path,
):
    """Halo access time grouped by pattern."""
    data = defaultdict(lambda: defaultdict(list))
    for k, v in baseline_results.items():
        pat = k.split("_pat=")[1]
        data["grid_baseline"][pat].append(v["halo_access_ms"])
    for cand_name, cand_data in candidate_results.items():
        for k, v in cand_data.items():
            pat = k.split("_pat=")[1]
            data[cand_name][pat].append(v["halo_access_ms"])

    _grouped_bar(data, "halo_access_ms", "Halo access (ms)",
                 "Exp10e: Halo Access by Pattern", out_path)


def plot_memory_breakdown(
    baseline_results: Dict[str, Dict],
    candidate_results: Dict[str, Dict[str, Dict]],
    out_path: Path,
):
    """Stacked bar chart of memory components, averaged across all configs."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Collect all memory component names across all results
    all_components = set()
    for v in baseline_results.values():
        all_components.update(v.get("memory_breakdown", {}).keys())
    for cand_data in candidate_results.values():
        for v in cand_data.values():
            all_components.update(v.get("memory_breakdown", {}).keys())
    all_components.discard("total")
    components = sorted(all_components)

    # For each label, average each component across all configs
    label_data = {}
    for label in ALL_LABELS:
        if label == "grid_baseline":
            source = baseline_results
        else:
            source = candidate_results.get(label, {})

        comp_avgs = {}
        for comp in components:
            vals = [
                v.get("memory_breakdown", {}).get(comp, 0)
                for v in source.values()
            ]
            comp_avgs[comp] = float(np.mean(vals)) / 1024 if vals else 0.0
        label_data[label] = comp_avgs

    # Stacked bars
    x = np.arange(len(ALL_LABELS))
    width = 0.5
    bottom = np.zeros(len(ALL_LABELS))
    comp_colors = plt.cm.Set2(np.linspace(0, 1, max(len(components), 1)))

    for ci, comp in enumerate(components):
        vals = [label_data[lab].get(comp, 0) for lab in ALL_LABELS]
        color = comp_colors[ci % len(comp_colors)]
        ax.bar(x, vals, width, bottom=bottom, label=comp, color=color, alpha=0.85)
        bottom += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(ALL_LABELS, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Memory (KB)")
    ax.set_title("Exp10e: Memory Breakdown by Candidate", fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ===================================================================
# Main sweep
# ===================================================================

def run_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Exp10e: Tile-Sparse Layouts")
    print("=" * 70)
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(device)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    print(f"  Device: {device}")
    print(f"  Sides: {SIDES}")
    print(f"  Sparsities: {SPARSITIES}")
    print(f"  Patterns: {PATTERNS}")
    print(f"  Seeds: {N_SEEDS}, Warmup: {N_WARMUP}, Repeats: {N_REPEAT}")
    print(f"  Kill threshold: {KILL_THRESH:.0%}")
    print(f"  Candidates: {list(CANDIDATES.keys())}")
    print("=" * 70)

    # ---- Collect results keyed by config string ----
    # Key format: "side={side}_sp={sparsity}_pat={pattern}"
    baseline_results: Dict[str, Dict] = {}
    candidate_results: Dict[str, Dict[str, Dict]] = {
        name: {} for name in CANDIDATES
    }

    total = len(SIDES) * len(SPARSITIES) * len(PATTERNS)
    idx = 0

    for side in SIDES:
        for sparsity in SPARSITIES:
            for pattern in PATTERNS:
                idx += 1
                config_key = f"side={side}_sp={sparsity}_pat={pattern}"
                print(f"\n  [{idx}/{total}] {config_key}")

                # ---- Grid baseline ----
                print(f"    grid_baseline ...", end="", flush=True)
                try:
                    base_r = benchmark_grid_baseline(
                        side, sparsity, pattern, N_SEEDS, device)
                    baseline_results[config_key] = base_r
                    print(f"  {base_r['wall_clock_ms']:.3f}ms  "
                          f"vram={base_r['peak_vram_bytes']/1024:.0f}KB  "
                          f"halo={base_r['halo_access_ms']:.3f}ms")
                except RuntimeError as e:
                    print(f"  SKIPPED ({e})")
                    continue

                # ---- Candidates ----
                for cand_name, cand_fn in CANDIDATES.items():
                    print(f"    {cand_name} ...", end="", flush=True)
                    try:
                        cand_r = cand_fn(
                            side, sparsity, pattern, N_SEEDS, device)
                        candidate_results[cand_name][config_key] = cand_r

                        # Compute overhead vs baseline
                        t_oh = 0.0
                        v_oh = 0.0
                        if config_key in baseline_results:
                            bw = baseline_results[config_key]["wall_clock_ms"]
                            bv = baseline_results[config_key]["peak_vram_bytes"]
                            if bw > 0:
                                t_oh = cand_r["wall_clock_ms"] / bw - 1.0
                            if bv > 0:
                                v_oh = cand_r["peak_vram_bytes"] / bv - 1.0

                        flag = " ***" if (
                            t_oh > KILL_THRESH or v_oh > KILL_THRESH
                        ) else ""
                        print(f"  {cand_r['wall_clock_ms']:.3f}ms  "
                              f"time_oh={t_oh:+.1%}  "
                              f"vram_oh={v_oh:+.1%}{flag}")
                    except RuntimeError as e:
                        print(f"  SKIPPED ({e})")

    # ================================================================
    # Statistical analysis per pattern
    # ================================================================
    print("\n" + "=" * 70)
    print("Statistical Analysis (per pattern, Wilcoxon + Holm-Bonferroni)")
    print("=" * 70)

    pairwise_by_pattern = {}
    for pattern in PATTERNS:
        pw = compute_pairwise_stats(
            baseline_results, candidate_results, pattern)
        pairwise_by_pattern[pattern] = pw

        print(f"\n  Pattern: {pattern}")
        for comp in pw["comparisons"]:
            sig_mark = "*" if comp["significant"] else " "
            print(f"    {comp['candidate']:18s}  "
                  f"base={comp['baseline_median_ms']:.3f}ms  "
                  f"cand={comp['candidate_median_ms']:.3f}ms  "
                  f"p={comp['p_value']:.4f} {sig_mark}  "
                  f"oh={comp.get('overhead_pct', 0):.1f}%")

    # ================================================================
    # Verdicts
    # ================================================================
    print("\n" + "=" * 70)
    print("VERDICTS")
    print("=" * 70)

    verdicts = compute_verdicts(baseline_results, candidate_results)
    hybrid = compute_hybrid_recommendation(verdicts)

    for cand_name, cand_info in verdicts.items():
        print(f"\n  {cand_name}: {cand_info['overall']}")
        for pat, pat_info in cand_info["per_pattern"].items():
            if isinstance(pat_info, str):
                print(f"    {pat:14s}: {pat_info}")
            else:
                print(f"    {pat:14s}: {pat_info['verdict']}  "
                      f"({pat_info['reason']})")

    print(f"\n  Hybrid recommendation:")
    for pat, winner in hybrid.items():
        print(f"    {pat:14s} -> {winner}")

    # ================================================================
    # Save JSON
    # ================================================================
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
            "gpu_name": (torch.cuda.get_device_name(device)
                         if device.type == "cuda" else "N/A"),
        },
        "baseline_results": baseline_results,
        "candidate_results": candidate_results,
        "pairwise_tests": pairwise_by_pattern,
        "verdicts": verdicts,
        "hybrid_recommendation": hybrid,
    }

    json_path = out_dir / "exp10e_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved: {json_path}")

    # ================================================================
    # Plots
    # ================================================================
    plot_time_comparison(
        baseline_results, candidate_results,
        out_dir / "exp10e_time_comparison.png")
    print(f"Saved: {out_dir / 'exp10e_time_comparison.png'}")

    plot_vram_comparison(
        baseline_results, candidate_results,
        out_dir / "exp10e_vram_comparison.png")
    print(f"Saved: {out_dir / 'exp10e_vram_comparison.png'}")

    plot_halo_comparison(
        baseline_results, candidate_results,
        out_dir / "exp10e_halo_comparison.png")
    print(f"Saved: {out_dir / 'exp10e_halo_comparison.png'}")

    plot_memory_breakdown(
        baseline_results, candidate_results,
        out_dir / "exp10e_memory_breakdown.png")
    print(f"Saved: {out_dir / 'exp10e_memory_breakdown.png'}")

    return summary


if __name__ == "__main__":
    t0 = time.time()
    summary = run_all()
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
