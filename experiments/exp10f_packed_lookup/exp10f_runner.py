#!/usr/bin/env python3
"""
Curiosity -- Exp10f: Packed Tile Storage with Direct tile_map Lookup

Benchmarks packed tile storage with O(1) tile_map[tile_id] -> slot lookup
against grid baseline AND A_bitset (exp10e winner).

Motivation: exp10e-B showed packed storage saves VRAM (-30%) but binary
search killed time (+1700%).  Replace binary search with O(1) tile_map.

Candidates:
  D_direct_onfly:    packed tiles + tile_map + on-the-fly halo lookup
  D_direct_prebuilt: packed tiles + tile_map + prebuilt neighbor_slots table

Comparison targets:
  grid_baseline: full grid + bool mask (current default)
  A_bitset:      exp10e winner (fast but +18% VRAM)

Kill criteria (per pattern class):
  overhead >20% vs grid in wall-clock OR VRAM --> FAIL for that class.

Competitive criteria:
  D must not be embarrassingly worse than A_bitset on time at moderate
  occupancy.  D should be better than A on VRAM in sparse regime.

Outputs:
  results/exp10f_summary.json         -- all verdicts, all data
  results/exp10f_time_comparison.png  -- grouped bar: grid vs A vs D_onfly vs D_prebuilt
  results/exp10f_vram_comparison.png  -- same for VRAM
  results/exp10f_build_cost.png       -- build time per config
  results/exp10f_report.md            -- text summary with verdicts
"""

import time
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===================================================================
# Candidate imports
# ===================================================================

# D candidates from this experiment (direct tile_map)
from packed_tile_layout import benchmark_packed_direct

# E candidates from this experiment (hash table)
from hash_tile_layout import benchmark_hash_lookup

# A_bitset from exp10e (reference)
sys.path.insert(0, str(Path(__file__).parent.parent / "exp10e_tile_sparse"))
from candidate_a_bitset import benchmark_candidate_a as _raw_benchmark_A

# ===================================================================
# Configuration
# ===================================================================

SIDES       = [64, 128, 256]  # NO 512!
SPARSITIES  = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70]
PATTERNS    = ["random", "clustered", "checkerboard"]
TILE_SIZES  = [8]  # start with 8, can extend later
N_SEEDS     = 10
N_WARMUP    = 5
N_REPEAT    = 20
SEED_BASE   = 42
DTYPE       = torch.float32
KILL_THRESH = 0.20   # 20%

# ===================================================================
# Mask generation (from exp10e, extended)
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
                chosen = coords[rng.choice(len(coords), size=can_add,
                                           replace=False)]
                mask[chosen[:, 0], chosen[:, 1]] = True
                placed += can_add
        if placed < n_active:
            inactive = np.argwhere(~mask)
            need = n_active - placed
            chosen = inactive[rng.choice(len(inactive), size=need,
                                         replace=False)]
            mask[chosen[:, 0], chosen[:, 1]] = True
        return mask

    elif pattern == "checkerboard":
        mask = np.zeros((side, side), dtype=bool)
        rows, cols = np.meshgrid(np.arange(side), np.arange(side),
                                 indexing="ij")
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
            chosen = inactive[rng.choice(len(inactive), size=remaining,
                                         replace=False)]
            mask[chosen[:, 0], chosen[:, 1]] = True
        return mask

    else:
        raise ValueError(f"Unknown pattern: {pattern}")


# ===================================================================
# Grid baseline (inline, same as exp10e runner)
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
    build_time_ms (always 0), memory_breakdown.
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
            gathered = data[mask_flat]
            field_2d = data.reshape(1, 1, side, side)
            out_2d = torch.nn.functional.conv2d(field_2d, stencil, padding=1)
            result = out_2d.reshape(M)
            output = data.clone()
            output[mask_flat] = result[mask_flat]
            return output

        # Halo access timing
        def _halo_access():
            field_2d = data.reshape(1, 1, side, side)
            padded = torch.nn.functional.pad(field_2d, (1, 1, 1, 1),
                                             mode="constant")
            _ = padded[:, :, 0:side, 0:side] + padded[:, :, 2:side+2, 2:side+2]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
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

        del mask_flat, data
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Aggregate memory breakdown (use first seed)
    avg_breakdown = {}
    for key in mem_breakdowns[0]:
        avg_breakdown[key] = int(np.mean([m[key] for m in mem_breakdowns]))

    n_total_tiles = (side // TILE_SIZES[0]) ** 2
    n_active_tiles = int(round(n_total_tiles * sparsity))

    return {
        "wall_clock_ms": float(np.median(wall_times)),
        "peak_vram_bytes": int(np.median(vram_peaks)),
        "halo_access_ms": float(np.median(halo_times)),
        "build_time_ms": 0.0,
        "memory_breakdown": avg_breakdown,
        "n_active_tiles": n_active_tiles,
        "n_total_tiles": n_total_tiles,
    }


# ===================================================================
# A_bitset adapter (same as exp10e runner)
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

    n_total_tiles = (side // TILE_SIZES[0]) ** 2
    n_active_tiles = int(round(n_total_tiles * sparsity))

    return {
        "wall_clock_ms": agg["wall_clock_median_s"] * 1e3,
        "peak_vram_bytes": int(agg["peak_vram_median_bytes"]),
        "halo_access_ms": agg["halo_time_median_s"] * 1e3,
        "build_time_ms": 0.0,
        "memory_breakdown": raw["memory_breakdown"],
        "n_active_tiles": n_active_tiles,
        "n_total_tiles": n_total_tiles,
    }


# ===================================================================
# D_direct adapters (packed tiles + tile_map)
# ===================================================================

def _adapt_d_direct_onfly(side, sparsity, pattern, n_seeds, device):
    """Adapter for D_direct_onfly: packed tiles + tile_map, halo on-the-fly."""
    raw = benchmark_packed_direct(
        side=side,
        sparsity=sparsity,
        pattern=pattern,
        n_seeds=n_seeds,
        device=device,
        tile_size=TILE_SIZES[0],
        use_prebuilt_neighbors=False,
    )
    return {
        "wall_clock_ms": float(raw["wall_clock_ms"]),
        "peak_vram_bytes": int(raw["peak_vram_bytes"]),
        "halo_access_ms": float(raw["halo_access_ms"]),
        "build_time_ms": float(raw["build_time_ms"]),
        "memory_breakdown": raw["memory_breakdown"],
        "n_active_tiles": int(raw["n_active_tiles"]),
        "n_total_tiles": int(raw["n_total_tiles"]),
    }


def _adapt_d_direct_prebuilt(side, sparsity, pattern, n_seeds, device):
    """Adapter for D_direct_prebuilt: packed tiles + tile_map + prebuilt
    neighbor_slots table."""
    raw = benchmark_packed_direct(
        side=side,
        sparsity=sparsity,
        pattern=pattern,
        n_seeds=n_seeds,
        device=device,
        tile_size=TILE_SIZES[0],
        use_prebuilt_neighbors=True,
    )
    return {
        "wall_clock_ms": float(raw["wall_clock_ms"]),
        "peak_vram_bytes": int(raw["peak_vram_bytes"]),
        "halo_access_ms": float(raw["halo_access_ms"]),
        "build_time_ms": float(raw["build_time_ms"]),
        "memory_breakdown": raw["memory_breakdown"],
        "n_active_tiles": int(raw["n_active_tiles"]),
        "n_total_tiles": int(raw["n_total_tiles"]),
    }


def _adapt_e_hash_onfly(side, sparsity, pattern, n_seeds, device):
    """Adapter for E_hash_onfly: packed tiles + hash table, halo on-the-fly."""
    raw = benchmark_hash_lookup(
        side=side,
        sparsity=sparsity,
        pattern=pattern,
        n_seeds=n_seeds,
        device=device,
        tile_size=TILE_SIZES[0],
        use_prebuilt_neighbors=False,
    )
    return {
        "wall_clock_ms": float(raw["wall_clock_ms"]),
        "peak_vram_bytes": int(raw["peak_vram_bytes"]),
        "halo_access_ms": float(raw["halo_access_ms"]),
        "build_time_ms": float(raw["build_time_ms"]),
        "memory_breakdown": raw["memory_breakdown"],
        "n_active_tiles": int(raw["n_active_tiles"]),
        "n_total_tiles": int(raw["n_total_tiles"]),
    }


def _adapt_e_hash_prebuilt(side, sparsity, pattern, n_seeds, device):
    """Adapter for E_hash_prebuilt: packed tiles + hash table + prebuilt neighbors."""
    raw = benchmark_hash_lookup(
        side=side,
        sparsity=sparsity,
        pattern=pattern,
        n_seeds=n_seeds,
        device=device,
        tile_size=TILE_SIZES[0],
        use_prebuilt_neighbors=True,
    )
    return {
        "wall_clock_ms": float(raw["wall_clock_ms"]),
        "peak_vram_bytes": int(raw["peak_vram_bytes"]),
        "halo_access_ms": float(raw["halo_access_ms"]),
        "build_time_ms": float(raw["build_time_ms"]),
        "memory_breakdown": raw["memory_breakdown"],
        "n_active_tiles": int(raw["n_active_tiles"]),
        "n_total_tiles": int(raw["n_total_tiles"]),
    }


CANDIDATES = {
    "A_bitset":          _adapt_candidate_a,
    "D_direct_onfly":    _adapt_d_direct_onfly,
    "D_direct_prebuilt": _adapt_d_direct_prebuilt,
    "E_hash_onfly":      _adapt_e_hash_onfly,
    "E_hash_prebuilt":   _adapt_e_hash_prebuilt,
}

ALL_LABELS = ["grid_baseline"] + list(CANDIDATES.keys())

CAND_COLORS = {
    "grid_baseline":     "#1f77b4",
    "A_bitset":          "#ff7f0e",
    "D_direct_onfly":    "#2ca02c",
    "D_direct_prebuilt": "#d62728",
    "E_hash_onfly":      "#9467bd",
    "E_hash_prebuilt":   "#8c564b",
}


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
    """Wilcoxon tests for each candidate vs baseline within a pattern group."""
    pattern_keys = [k for k in baseline_results if f"_pat={pattern}" in k]

    all_comparisons = []
    all_p = []

    for cand_name, cand_data in candidate_results.items():
        base_times = []
        cand_times = []
        for k in pattern_keys:
            if k in cand_data:
                base_times.append(baseline_results[k]["wall_clock_ms"])
                cand_times.append(cand_data[k]["wall_clock_ms"])

        if len(base_times) < 2:
            all_p.append(1.0)
            all_comparisons.append({
                "candidate": cand_name,
                "pattern": pattern,
                "n_configs": len(base_times),
                "p_value": 1.0,
                "significant": False,
                "baseline_median_ms": float(np.median(base_times))
                    if base_times else 0,
                "candidate_median_ms": float(np.median(cand_times))
                    if cand_times else 0,
            })
            continue

        base_arr = np.array(base_times)
        cand_arr = np.array(cand_times)
        p = _wilcoxon_p(base_arr, cand_arr)
        all_p.append(p)

        all_comparisons.append({
            "candidate": cand_name,
            "pattern": pattern,
            "n_configs": len(base_times),
            "p_value": p,
            "significant": False,
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
                    v_oh = (cand["peak_vram_bytes"] / base["peak_vram_bytes"]
                            - 1.0)
                    vram_overheads.append(v_oh)

            if not time_overheads:
                cand_verdicts[pattern] = "NO_DATA"
                continue

            med_time_oh = float(np.median(time_overheads))
            med_vram_oh = (float(np.median(vram_overheads))
                           if vram_overheads else 0.0)

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
                reason_str = (f"time {med_time_oh:+.1%}, "
                              f"VRAM {med_vram_oh:+.1%}")
                any_pass = True
            else:
                verdict = "MARGINAL"
                reason_str = (f"time {med_time_oh:+.1%}, "
                              f"VRAM {med_vram_oh:+.1%}")
                any_pass = True

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


def compute_competitive_analysis(
    candidate_results: Dict[str, Dict[str, Dict]],
) -> Dict[str, Any]:
    """Compare D candidates against A_bitset.

    D must not be embarrassingly worse than A on time at moderate occupancy.
    D should be better than A on VRAM in sparse regime.
    """
    a_data = candidate_results.get("A_bitset", {})
    analysis = {}

    for d_name in [n for n in CANDIDATES if n != "A_bitset"]:
        d_data = candidate_results.get(d_name, {})
        if not d_data or not a_data:
            analysis[d_name] = {"status": "NO_DATA"}
            continue

        per_pattern = {}
        for pattern in PATTERNS:
            pattern_keys = [k for k in a_data if f"_pat={pattern}" in k]

            time_diffs = []  # D - A (negative = D faster)
            vram_diffs = []  # D - A (negative = D smaller)
            sparse_vram_diffs = []  # only for sparsity <= 0.20

            for k in pattern_keys:
                if k not in d_data:
                    continue
                a_r = a_data[k]
                d_r = d_data[k]

                if a_r["wall_clock_ms"] > 0:
                    t_diff = (d_r["wall_clock_ms"] / a_r["wall_clock_ms"]
                              - 1.0)
                    time_diffs.append(t_diff)

                if a_r["peak_vram_bytes"] > 0:
                    v_diff = (d_r["peak_vram_bytes"] / a_r["peak_vram_bytes"]
                              - 1.0)
                    vram_diffs.append(v_diff)

                    # Sparse regime check
                    sp_str = k.split("_sp=")[1].split("_")[0]
                    sp = float(sp_str)
                    if sp <= 0.20:
                        sparse_vram_diffs.append(v_diff)

            if not time_diffs:
                per_pattern[pattern] = {"status": "NO_DATA"}
                continue

            med_time = float(np.median(time_diffs))
            med_vram = float(np.median(vram_diffs)) if vram_diffs else 0.0
            med_sparse_vram = (float(np.median(sparse_vram_diffs))
                               if sparse_vram_diffs else 0.0)

            # "Embarrassingly worse" = >50% slower than A
            embarrassing = med_time > 0.50
            sparse_vram_win = med_sparse_vram < -0.05

            per_pattern[pattern] = {
                "time_vs_A_pct": round(med_time * 100, 2),
                "vram_vs_A_pct": round(med_vram * 100, 2),
                "sparse_vram_vs_A_pct": round(med_sparse_vram * 100, 2),
                "embarrassingly_worse_on_time": embarrassing,
                "better_vram_in_sparse": sparse_vram_win,
            }

        analysis[d_name] = {"per_pattern": per_pattern}

    return analysis


# ===================================================================
# Plotting
# ===================================================================

def _grouped_bar(
    data_by_label: Dict[str, Dict[str, List[float]]],
    ylabel: str,
    title: str,
    out_path: Path,
):
    """Grouped bar chart: x = pattern, bars = candidates."""
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

    _grouped_bar(data, "Median wall-clock (ms)",
                 "Exp10f: Wall-clock by Pattern", out_path)


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

    _grouped_bar(data, "Peak VRAM (KB)",
                 "Exp10f: VRAM by Pattern", out_path)


def plot_build_cost(
    baseline_results: Dict[str, Dict],
    candidate_results: Dict[str, Dict[str, Dict]],
    out_path: Path,
):
    """Build time per config, grouped by pattern."""
    data = defaultdict(lambda: defaultdict(list))
    for k, v in baseline_results.items():
        pat = k.split("_pat=")[1]
        data["grid_baseline"][pat].append(v.get("build_time_ms", 0.0))
    for cand_name, cand_data in candidate_results.items():
        for k, v in cand_data.items():
            pat = k.split("_pat=")[1]
            data[cand_name][pat].append(v.get("build_time_ms", 0.0))

    _grouped_bar(data, "Build time (ms)",
                 "Exp10f: Build Cost by Pattern", out_path)


def generate_report(
    verdicts: Dict[str, Any],
    competitive: Dict[str, Any],
    pairwise_by_pattern: Dict[str, Any],
    baseline_results: Dict[str, Dict],
    candidate_results: Dict[str, Dict[str, Dict]],
    out_path: Path,
):
    """Generate a text report with verdicts."""
    lines = []
    lines.append("# Exp10f: Packed Tile Storage with Direct Lookup -- Report")
    lines.append("")
    lines.append("## Kill Criteria: >20% overhead vs grid in wall-clock OR VRAM")
    lines.append("")

    # Per-candidate verdicts
    for cand_name, cand_info in verdicts.items():
        lines.append(f"### {cand_name}: {cand_info['overall']}")
        for pat, pat_info in cand_info["per_pattern"].items():
            if isinstance(pat_info, str):
                lines.append(f"  - {pat}: {pat_info}")
            else:
                lines.append(f"  - {pat}: {pat_info['verdict']}  "
                             f"({pat_info['reason']})")
        lines.append("")

    # Competitive analysis vs A_bitset
    lines.append("## Competitive Analysis vs A_bitset")
    lines.append("")
    for d_name in [n for n in CANDIDATES if n != "A_bitset"]:
        d_analysis = competitive.get(d_name, {})
        lines.append(f"### {d_name}")
        if d_analysis.get("status") == "NO_DATA":
            lines.append("  No data available.")
            continue
        for pat, pat_info in d_analysis.get("per_pattern", {}).items():
            if pat_info.get("status") == "NO_DATA":
                lines.append(f"  - {pat}: no data")
                continue
            lines.append(
                f"  - {pat}: time vs A = {pat_info['time_vs_A_pct']:+.1f}%, "
                f"VRAM vs A = {pat_info['vram_vs_A_pct']:+.1f}%, "
                f"sparse VRAM vs A = {pat_info['sparse_vram_vs_A_pct']:+.1f}%"
            )
            if pat_info["embarrassingly_worse_on_time"]:
                lines.append(f"    WARNING: embarrassingly worse than A on time")
            if pat_info["better_vram_in_sparse"]:
                lines.append(f"    GOOD: better VRAM than A in sparse regime")
        lines.append("")

    # Build cost warnings
    lines.append("## Build Cost Analysis")
    lines.append("")
    for cand_name in [n for n in CANDIDATES if n != "A_bitset"]:
        cand_data = candidate_results.get(cand_name, {})
        warnings = []
        for k, v in cand_data.items():
            build_ms = v.get("build_time_ms", 0.0)
            compute_ms = v.get("wall_clock_ms", 1.0)
            if compute_ms > 0 and build_ms > 0.20 * compute_ms:
                warnings.append(
                    f"  - {k}: build={build_ms:.2f}ms "
                    f"({build_ms/compute_ms*100:.0f}% of compute)")
        if warnings:
            lines.append(f"### {cand_name} -- build overhead warnings:")
            lines.extend(warnings)
        else:
            lines.append(f"### {cand_name}: no build overhead warnings")
        lines.append("")

    # Statistical tests
    lines.append("## Statistical Tests (Wilcoxon + Holm-Bonferroni)")
    lines.append("")
    for pattern, pw in pairwise_by_pattern.items():
        lines.append(f"### {pattern}")
        for comp in pw["comparisons"]:
            sig_mark = "*" if comp["significant"] else ""
            lines.append(
                f"  - {comp['candidate']}: "
                f"base={comp['baseline_median_ms']:.3f}ms, "
                f"cand={comp['candidate_median_ms']:.3f}ms, "
                f"p={comp['p_value']:.4f}{sig_mark}, "
                f"oh={comp.get('overhead_pct', 0):.1f}%"
            )
        lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


# ===================================================================
# Main sweep
# ===================================================================

def run_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Exp10f: Packed Tile Storage with Direct Lookup")
    print("=" * 70)
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(device)}")
        print(f"  VRAM: "
              f"{torch.cuda.get_device_properties(device).total_memory / 1e9:.1f}"
              f" GB")
    print(f"  Device: {device}")
    print(f"  Sides: {SIDES}")
    print(f"  Sparsities: {SPARSITIES}")
    print(f"  Patterns: {PATTERNS}")
    print(f"  Tile sizes: {TILE_SIZES}")
    print(f"  Seeds: {N_SEEDS}, Warmup: {N_WARMUP}, Repeats: {N_REPEAT}")
    print(f"  Kill threshold: {KILL_THRESH:.0%}")
    print(f"  Candidates: {list(CANDIDATES.keys())}")
    print("=" * 70)

    # Collect results keyed by config string
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
                print(f"    grid_baseline     ...", end="", flush=True)
                try:
                    base_r = benchmark_grid_baseline(
                        side, sparsity, pattern, N_SEEDS, device)
                    baseline_results[config_key] = base_r
                    print(f"  {base_r['wall_clock_ms']:.2f}ms  "
                          f"vram={base_r['peak_vram_bytes']/1024:.0f}KB  "
                          f"halo={base_r['halo_access_ms']:.2f}ms  "
                          f"build={base_r['build_time_ms']:.2f}ms")
                except RuntimeError as e:
                    print(f"  SKIPPED ({e})")
                    continue

                # ---- Candidates ----
                for cand_name, cand_fn in CANDIDATES.items():
                    print(f"    {cand_name:20s} ...", end="", flush=True)
                    try:
                        cand_r = cand_fn(
                            side, sparsity, pattern, N_SEEDS, device)
                        candidate_results[cand_name][config_key] = cand_r

                        # Compute overhead vs baseline
                        t_oh = 0.0
                        v_oh = 0.0
                        bw = baseline_results[config_key]["wall_clock_ms"]
                        bv = baseline_results[config_key]["peak_vram_bytes"]
                        if bw > 0:
                            t_oh = cand_r["wall_clock_ms"] / bw - 1.0
                        if bv > 0:
                            v_oh = cand_r["peak_vram_bytes"] / bv - 1.0

                        flag = " ***" if (
                            t_oh > KILL_THRESH or v_oh > KILL_THRESH
                        ) else ""
                        build_str = ""
                        if cand_r.get("build_time_ms", 0) > 0:
                            build_str = (f"  build="
                                         f"{cand_r['build_time_ms']:.2f}ms")
                        print(f"  {cand_r['wall_clock_ms']:.2f}ms  "
                              f"time_oh={t_oh:+.1%}  "
                              f"vram_oh={v_oh:+.1%}"
                              f"{build_str}{flag}")
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
            print(f"    {comp['candidate']:20s}  "
                  f"base={comp['baseline_median_ms']:.3f}ms  "
                  f"cand={comp['candidate_median_ms']:.3f}ms  "
                  f"p={comp['p_value']:.4f} {sig_mark}  "
                  f"oh={comp.get('overhead_pct', 0):.1f}%")

    # ================================================================
    # Verdicts (vs grid)
    # ================================================================
    print("\n" + "=" * 70)
    print("VERDICTS (vs grid baseline, kill threshold 20%)")
    print("=" * 70)

    verdicts = compute_verdicts(baseline_results, candidate_results)

    for cand_name, cand_info in verdicts.items():
        print(f"\n  {cand_name}: {cand_info['overall']}")
        for pat, pat_info in cand_info["per_pattern"].items():
            if isinstance(pat_info, str):
                print(f"    {pat:14s}: {pat_info}")
            else:
                print(f"    {pat:14s}: {pat_info['verdict']}  "
                      f"({pat_info['reason']})")

    # ================================================================
    # Competitive analysis vs A_bitset
    # ================================================================
    print("\n" + "=" * 70)
    print("COMPETITIVE ANALYSIS (D vs A_bitset)")
    print("=" * 70)

    competitive = compute_competitive_analysis(candidate_results)

    for d_name in [n for n in CANDIDATES if n != "A_bitset"]:
        d_analysis = competitive.get(d_name, {})
        print(f"\n  {d_name}:")
        if d_analysis.get("status") == "NO_DATA":
            print("    No data.")
            continue
        for pat, pat_info in d_analysis.get("per_pattern", {}).items():
            if pat_info.get("status") == "NO_DATA":
                print(f"    {pat:14s}: no data")
                continue
            emb = " EMBARRASSING" if pat_info[
                "embarrassingly_worse_on_time"] else ""
            svw = " SPARSE_VRAM_WIN" if pat_info[
                "better_vram_in_sparse"] else ""
            print(f"    {pat:14s}: time_vs_A={pat_info['time_vs_A_pct']:+.1f}%"
                  f"  vram_vs_A={pat_info['vram_vs_A_pct']:+.1f}%"
                  f"  sparse_vram_vs_A="
                  f"{pat_info['sparse_vram_vs_A_pct']:+.1f}%"
                  f"{emb}{svw}")

    # ================================================================
    # Build cost warnings
    # ================================================================
    print("\n" + "=" * 70)
    print("BUILD COST (separate from compute)")
    print("=" * 70)

    for cand_name in [n for n in CANDIDATES if n != "A_bitset"]:
        cand_data = candidate_results.get(cand_name, {})
        print(f"\n  {cand_name}:")
        any_warning = False
        for k, v in sorted(cand_data.items()):
            build_ms = v.get("build_time_ms", 0.0)
            compute_ms = v.get("wall_clock_ms", 1.0)
            ratio = build_ms / compute_ms * 100 if compute_ms > 0 else 0
            warn = " << BUILD OVERHEAD WARNING" if ratio > 20 else ""
            if warn:
                any_warning = True
            print(f"    {k}: build={build_ms:.2f}ms  "
                  f"compute={compute_ms:.2f}ms  "
                  f"ratio={ratio:.0f}%{warn}")
        if not any_warning:
            print(f"    (no build overhead warnings)")

    # ================================================================
    # Save JSON
    # ================================================================
    summary = {
        "config": {
            "sides": SIDES,
            "sparsities": SPARSITIES,
            "patterns": PATTERNS,
            "tile_sizes": TILE_SIZES,
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
        "competitive_analysis": competitive,
    }

    json_path = out_dir / "exp10f_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved: {json_path}")

    # ================================================================
    # Plots
    # ================================================================
    plot_time_comparison(
        baseline_results, candidate_results,
        out_dir / "exp10f_time_comparison.png")
    print(f"Saved: {out_dir / 'exp10f_time_comparison.png'}")

    plot_vram_comparison(
        baseline_results, candidate_results,
        out_dir / "exp10f_vram_comparison.png")
    print(f"Saved: {out_dir / 'exp10f_vram_comparison.png'}")

    plot_build_cost(
        baseline_results, candidate_results,
        out_dir / "exp10f_build_cost.png")
    print(f"Saved: {out_dir / 'exp10f_build_cost.png'}")

    # ================================================================
    # Report
    # ================================================================
    report_path = out_dir / "exp10f_report.md"
    generate_report(
        verdicts, competitive, pairwise_by_pattern,
        baseline_results, candidate_results, report_path)
    print(f"Saved: {report_path}")

    return summary


if __name__ == "__main__":
    t0 = time.time()
    summary = run_all()
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
