#!/usr/bin/env python
"""exp10j — Per-level tree benchmark: D_direct vs A_bitset.

Tests D_direct (packed tiles + tile_map) against A_bitset at each tree level
independently. Finds break-even thresholds p*(N_l) per level.

Usage:
    # Run one branching factor:
    python exp10j_tree_perlevel.py --branching 2 --output results/chunk_br2.json

    # Merge all chunks:
    python exp10j_tree_perlevel.py --merge --output results/merged.json

    # Run all branching factors sequentially:
    python exp10j_tree_perlevel.py --all --output results/merged.json

Environment: Python 3.12.11, PyTorch 2.10.0+cu128
Venv: R:\\Projects\\Curiosity\\.venv-gpu
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import torch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BRANCHING_FACTORS = [2, 4, 8, 16, 32]
OCCUPANCIES = [0.01, 0.05, 0.1, 0.2, 0.4, 0.7]
PATTERNS = ["random", "clustered_subtree", "frontier"]
PAYLOADS_BYTES = [4, 16, 64, 256]  # 1, 4, 16, 64 floats (float32 = 4 bytes)
OPERATORS = ["stencil", "matmul"]

N_SEEDS = 10
N_WARMUP = 5
N_REPEATS = 20

# Minimum nodes at bottom level for the sweep to be meaningful
MIN_BOTTOM_NODES = 1024


def compute_max_depth(branching: int) -> int:
    """Compute max depth so bottom level has >= MIN_BOTTOM_NODES nodes."""
    depth = 1
    while branching ** depth < MIN_BOTTOM_NODES:
        depth += 1
    return depth


# Pre-computed depth table
DEPTH_TABLE: dict[int, int] = {br: compute_max_depth(br) for br in BRANCHING_FACTORS}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LevelConfig:
    branching: int
    depth: int  # total tree depth
    level: int  # 0 = root
    N_l: int  # nodes at this level = branching^level
    k_l: int  # active nodes
    p_l: float  # occupancy = k_l / N_l
    pattern: str
    feat_dim: int  # number of float32 features
    operator: str
    seed: int


@dataclass
class LevelResult:
    # Config echo
    branching: int
    depth: int
    level: int
    N_l: int
    k_l: int
    p_l: float
    pattern: str
    feat_dim: int
    operator: str
    seed: int
    # Memory (bytes)
    A_resident: int
    D_resident: int
    resident_ratio: float
    A_peak: int
    D_peak: int
    peak_ratio: float
    # Time (seconds, median over repeats)
    A_time: float
    D_time: float
    time_ratio: float
    # Build time (seconds)
    A_build: float
    D_build: float
    # Verdicts
    verdict_A: str  # "PASS" if D wins both mem and time, else "FAIL"
    verdict_B: str  # same for contour B (matmul uses this; stencil uses verdict_A)


# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("WARNING: CUDA not available, falling back to CPU", file=sys.stderr)
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Activation pattern generators
# ---------------------------------------------------------------------------

def generate_active_ids(
    N_l: int, k_l: int, pattern: str, seed: int, branching: int
) -> torch.Tensor:
    """Generate k_l active node IDs from [0, N_l) according to pattern.

    Returns int32 tensor of sorted unique IDs, length exactly k_l.
    """
    rng = torch.Generator().manual_seed(seed)

    if pattern == "random":
        perm = torch.randperm(N_l, generator=rng)
        ids = perm[:k_l].sort().values

    elif pattern == "clustered_subtree":
        # Pick a contiguous block of subtrees.  Each subtree at this level
        # has `branching` children in the next level, but at _this_ level
        # each node is one unit.  We pick a contiguous range of k_l nodes
        # starting from a random offset.
        max_start = max(0, N_l - k_l)
        start = torch.randint(0, max_start + 1, (1,), generator=rng).item()
        ids = torch.arange(start, start + k_l, dtype=torch.int64)

    elif pattern == "frontier":
        # Activate nodes near the "right edge" — simulating a wavefront.
        # Take the last k_l nodes (highest indices).  Add slight jitter.
        base = torch.arange(N_l - k_l, N_l, dtype=torch.int64)
        # Jitter: randomly swap up to 10% of positions with earlier nodes
        n_jitter = max(1, k_l // 10)
        jitter_targets = torch.randint(0, max(1, N_l - k_l), (n_jitter,), generator=rng)
        jitter_sources = torch.randint(0, k_l, (n_jitter,), generator=rng)
        base[jitter_sources] = jitter_targets
        ids = base.unique().sort().values
        # Ensure exactly k_l (pad or trim)
        if len(ids) < k_l:
            remaining = torch.tensor(
                [i for i in range(N_l) if i not in ids.tolist()], dtype=torch.int64
            )
            perm = torch.randperm(len(remaining), generator=rng)
            extra = remaining[perm[: k_l - len(ids)]]
            ids = torch.cat([ids, extra]).sort().values
        elif len(ids) > k_l:
            ids = ids[:k_l]
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return ids.to(torch.int32)


# ---------------------------------------------------------------------------
# Layout A: bitset (full-size allocation + mask)
# ---------------------------------------------------------------------------

def build_layout_A(
    N_l: int, k_l: int, active_ids: torch.Tensor, feat_dim: int, device: torch.device
) -> dict[str, torch.Tensor]:
    """Build A_bitset layout: full tensor + bitmask."""
    data = torch.zeros(N_l, feat_dim, dtype=torch.float32, device=device)
    mask_size = (N_l + 7) // 8
    mask = torch.zeros(mask_size, dtype=torch.uint8, device=device)

    # Set active entries
    ids_long = active_ids.long().to(device)
    data[ids_long] = torch.randn(k_l, feat_dim, device=device)

    # Set mask bits
    byte_idx = ids_long // 8
    bit_idx = (ids_long % 8).to(torch.uint8)
    mask.scatter_add_(0, byte_idx, (1 << bit_idx).to(torch.uint8))

    return {"data": data, "mask": mask}


# ---------------------------------------------------------------------------
# Layout D: packed (active-only allocation + tile_map)
# ---------------------------------------------------------------------------

def build_layout_D(
    N_l: int, k_l: int, active_ids: torch.Tensor, feat_dim: int, device: torch.device
) -> dict[str, torch.Tensor]:
    """Build D_direct layout: packed tensor + tile_map + active_ids."""
    packed_data = torch.randn(k_l, feat_dim, dtype=torch.float32, device=device)
    tile_map = torch.full((N_l,), -1, dtype=torch.int32, device=device)
    ids_dev = active_ids.to(device)
    tile_map[ids_dev.long()] = torch.arange(k_l, dtype=torch.int32, device=device)

    return {"packed_data": packed_data, "tile_map": tile_map, "active_ids": ids_dev}


# ---------------------------------------------------------------------------
# Operators — Contour A: stencil (parent-child gather)
# ---------------------------------------------------------------------------

def op_stencil_A(layout: dict[str, torch.Tensor], branching: int) -> torch.Tensor:
    """A_bitset stencil: gather parent features for each node.

    Parent of node i at level L is node i // branching at level L-1.
    Here we simulate within-level: treat node i's "parent" as node i // branching.
    Compute on full tensor, then mask.
    """
    data = layout["data"]
    N_l = data.shape[0]
    parent_idx = torch.arange(N_l, device=data.device) // branching
    parent_idx = parent_idx.clamp(max=N_l - 1)
    gathered = data[parent_idx]  # (N_l, feat_dim)
    result = data + gathered  # simple stencil: node + parent
    return result


def op_stencil_D(layout: dict[str, torch.Tensor], branching: int) -> torch.Tensor:
    """D_direct stencil: gather parent features for active nodes only.

    Uses tile_map for lookup.
    """
    packed = layout["packed_data"]
    tile_map = layout["tile_map"]
    active_ids = layout["active_ids"]
    k_l = packed.shape[0]

    # Parent IDs for active nodes
    parent_ids = (active_ids.long() // branching).clamp(max=tile_map.shape[0] - 1)
    parent_slots = tile_map[parent_ids]

    # For nodes whose parent is not active, use zero
    valid = parent_slots >= 0
    parent_feats = torch.zeros_like(packed)
    valid_mask = valid.unsqueeze(1).expand_as(packed)
    parent_feats[valid_mask] = packed[parent_slots[valid].long()].reshape(-1)

    result = packed + parent_feats
    return result


# ---------------------------------------------------------------------------
# Operators — Contour B: matmul (batched small matrix op)
# ---------------------------------------------------------------------------

_MATMUL_WEIGHT: dict[int, torch.Tensor] = {}


def _get_matmul_weight(feat_dim: int, device: torch.device) -> torch.Tensor:
    key = (feat_dim, str(device))
    if key not in _MATMUL_WEIGHT:
        _MATMUL_WEIGHT[key] = torch.randn(feat_dim, feat_dim, device=device) * 0.01
    return _MATMUL_WEIGHT[key]


def op_matmul_A(layout: dict[str, torch.Tensor]) -> torch.Tensor:
    """A_bitset matmul: (N_l, feat_dim) @ (feat_dim, feat_dim)."""
    data = layout["data"]
    W = _get_matmul_weight(data.shape[1], data.device)
    return data @ W


def op_matmul_D(layout: dict[str, torch.Tensor]) -> torch.Tensor:
    """D_direct matmul: (k_l, feat_dim) @ (feat_dim, feat_dim)."""
    packed = layout["packed_data"]
    W = _get_matmul_weight(packed.shape[1], packed.device)
    return packed @ W


# ---------------------------------------------------------------------------
# Memory measurement
# ---------------------------------------------------------------------------

def tensor_bytes(t: torch.Tensor) -> int:
    return t.nelement() * t.element_size()


def layout_resident_bytes(layout: dict[str, torch.Tensor]) -> int:
    return sum(tensor_bytes(v) for v in layout.values())


def measure_peak_bytes(
    fn, *args, device: torch.device
) -> tuple[float, int]:
    """Run fn(*args), return (wall_time_s, peak_bytes_delta).

    peak_bytes_delta is peak_memory - current_memory before the call.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated()

        start = time.perf_counter()
        result = fn(*args)
        torch.cuda.synchronize()
        end = time.perf_counter()

        peak = torch.cuda.max_memory_allocated()
        peak_delta = peak - mem_before
        del result
        return end - start, peak_delta
    else:
        start = time.perf_counter()
        result = fn(*args)
        end = time.perf_counter()
        del result
        return end - start, 0


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def benchmark_op(
    fn, *args, device: torch.device, n_warmup: int = N_WARMUP, n_repeats: int = N_REPEATS
) -> list[float]:
    """Run fn(*args) with warmup, return list of wall-clock times."""
    # Warmup
    for _ in range(n_warmup):
        result = fn(*args)
        if device.type == "cuda":
            torch.cuda.synchronize()
        del result

    times = []
    for _ in range(n_repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = fn(*args)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)
        del result

    return times


# ---------------------------------------------------------------------------
# Single level benchmark
# ---------------------------------------------------------------------------

def benchmark_level(cfg: LevelConfig, device: torch.device) -> LevelResult:
    """Run full benchmark for one level configuration."""
    # Generate active IDs
    active_ids = generate_active_ids(
        cfg.N_l, cfg.k_l, cfg.pattern, cfg.seed, cfg.branching
    )

    # --- Build layouts ---
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    layout_a = build_layout_A(cfg.N_l, cfg.k_l, active_ids, cfg.feat_dim, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    A_build = time.perf_counter() - t0

    A_resident = layout_resident_bytes(layout_a)

    t0 = time.perf_counter()
    layout_d = build_layout_D(cfg.N_l, cfg.k_l, active_ids, cfg.feat_dim, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    D_build = time.perf_counter() - t0

    D_resident = layout_resident_bytes(layout_d)

    # --- Select operator functions ---
    if cfg.operator == "stencil":
        fn_a = lambda: op_stencil_A(layout_a, cfg.branching)
        fn_d = lambda: op_stencil_D(layout_d, cfg.branching)
    elif cfg.operator == "matmul":
        fn_a = lambda: op_matmul_A(layout_a)
        fn_d = lambda: op_matmul_D(layout_d)
    else:
        raise ValueError(f"Unknown operator: {cfg.operator}")

    # --- Peak memory measurement ---
    _, A_peak_delta = measure_peak_bytes(fn_a, device=device)
    A_peak = A_resident + A_peak_delta

    _, D_peak_delta = measure_peak_bytes(fn_d, device=device)
    D_peak = D_resident + D_peak_delta

    # --- Timing ---
    times_a = benchmark_op(fn_a, device=device)
    times_d = benchmark_op(fn_d, device=device)

    A_time = sorted(times_a)[len(times_a) // 2]  # median
    D_time = sorted(times_d)[len(times_d) // 2]

    # --- Ratios ---
    resident_ratio = D_resident / max(A_resident, 1)
    peak_ratio = D_peak / max(A_peak, 1)
    time_ratio = D_time / max(A_time, 1e-12)

    # --- Verdicts ---
    # PASS = D wins both memory and time for this contour
    mem_pass = D_resident < A_resident
    time_pass = D_time <= A_time
    peak_pass = D_peak <= A_peak

    if cfg.operator == "stencil":
        verdict_A = "PASS" if (mem_pass and time_pass) else "FAIL"
        verdict_B = "N/A"
    elif cfg.operator == "matmul":
        verdict_A = "N/A"
        verdict_B = "PASS" if (mem_pass and time_pass and peak_pass) else "FAIL"
    else:
        verdict_A = verdict_B = "N/A"

    # Cleanup
    del layout_a, layout_d
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return LevelResult(
        branching=cfg.branching,
        depth=cfg.depth,
        level=cfg.level,
        N_l=cfg.N_l,
        k_l=cfg.k_l,
        p_l=cfg.p_l,
        pattern=cfg.pattern,
        feat_dim=cfg.feat_dim,
        operator=cfg.operator,
        seed=cfg.seed,
        A_resident=A_resident,
        D_resident=D_resident,
        resident_ratio=resident_ratio,
        A_peak=A_peak,
        D_peak=D_peak,
        peak_ratio=peak_ratio,
        A_time=A_time,
        D_time=D_time,
        time_ratio=time_ratio,
        A_build=A_build,
        D_build=D_build,
        verdict_A=verdict_A,
        verdict_B=verdict_B,
    )


# ---------------------------------------------------------------------------
# Sweep for one branching factor
# ---------------------------------------------------------------------------

def sweep_branching(branching: int, device: torch.device,
                    depth_min: int = 1, depth_max: int | None = None) -> list[dict[str, Any]]:
    """Run full sweep for one branching factor. Returns list of result dicts.
    depth_min/depth_max allow splitting large sweeps into chunks."""
    max_depth = DEPTH_TABLE[branching]
    if depth_max is not None:
        max_depth = min(max_depth, depth_max)
    results: list[dict[str, Any]] = []

    total_configs = 0
    for depth in range(depth_min, max_depth + 1):
        for level in range(depth + 1):
            N_l = branching ** level
            if N_l < 2:
                continue
            for p_l in OCCUPANCIES:
                k_l = max(1, int(N_l * p_l))
                if k_l >= N_l:
                    continue  # no point if fully occupied
                for payload_bytes in PAYLOADS_BYTES:
                    feat_dim = max(1, payload_bytes // 4)
                    for pattern in PATTERNS:
                        for operator in OPERATORS:
                            total_configs += N_SEEDS

    print(f"Branching={branching}: max_depth={max_depth}, total trials={total_configs}")

    trial = 0
    for depth in range(depth_min, max_depth + 1):
        for level in range(depth + 1):
            N_l = branching ** level
            if N_l < 2:
                continue
            for p_l in OCCUPANCIES:
                k_l = max(1, int(N_l * p_l))
                if k_l >= N_l:
                    continue
                for payload_bytes in PAYLOADS_BYTES:
                    feat_dim = max(1, payload_bytes // 4)
                    for pattern in PATTERNS:
                        for operator in OPERATORS:
                            for seed in range(N_SEEDS):
                                cfg = LevelConfig(
                                    branching=branching,
                                    depth=depth,
                                    level=level,
                                    N_l=N_l,
                                    k_l=k_l,
                                    p_l=round(k_l / N_l, 4),
                                    pattern=pattern,
                                    feat_dim=feat_dim,
                                    operator=operator,
                                    seed=seed,
                                )
                                try:
                                    result = benchmark_level(cfg, device)
                                    results.append(asdict(result))
                                except Exception as e:
                                    print(
                                        f"  ERROR: br={branching} L={level} "
                                        f"N_l={N_l} k_l={k_l} p={p_l} "
                                        f"pat={pattern} op={operator} "
                                        f"seed={seed}: {e}",
                                        file=sys.stderr,
                                    )
                                trial += 1
                                if trial % 100 == 0:
                                    print(
                                        f"  [{trial}/{total_configs}] "
                                        f"br={branching} L={level} N_l={N_l} "
                                        f"k_l={k_l} p={p_l:.2f}"
                                    )

    return results


# ---------------------------------------------------------------------------
# Break-even analysis
# ---------------------------------------------------------------------------

def analyze_breakeven(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute break-even thresholds from merged results.

    For each (N_l, operator) pair, find the occupancy threshold p* where
    D starts winning.

    Returns analysis dict with:
    - mem_breakeven: {N_l -> p*_mem}
    - time_breakeven: {N_l -> p*_time}
    - policy: {N_l -> {"use_D_below": p*, "recommendation": str}}
    """
    from collections import defaultdict

    # Group by (N_l, operator), aggregate across seeds
    groups: dict[tuple[int, str], dict[float, dict]] = defaultdict(
        lambda: defaultdict(lambda: {"mem_wins": 0, "time_wins": 0, "total": 0})
    )

    for r in results:
        key = (r["N_l"], r["operator"])
        p = r["p_l"]
        g = groups[key][p]
        g["total"] += 1
        if r["D_resident"] < r["A_resident"]:
            g["mem_wins"] += 1
        if r["D_time"] <= r["A_time"]:
            g["time_wins"] += 1

    mem_breakeven: dict[str, dict[str, float | None]] = {}
    time_breakeven: dict[str, dict[str, float | None]] = {}
    policy: dict[str, dict[str, Any]] = {}

    for (N_l, operator), p_groups in sorted(groups.items()):
        key_str = f"N={N_l}_op={operator}"

        # Find threshold: highest p where D wins >50% of trials
        sorted_ps = sorted(p_groups.keys())
        p_star_mem = None
        p_star_time = None

        for p in sorted_ps:
            g = p_groups[p]
            if g["total"] == 0:
                continue
            mem_rate = g["mem_wins"] / g["total"]
            time_rate = g["time_wins"] / g["total"]
            if mem_rate > 0.5:
                p_star_mem = p
            if time_rate > 0.5:
                p_star_time = p

        mem_breakeven[key_str] = {"p_star": p_star_mem}
        time_breakeven[key_str] = {"p_star": p_star_time}

        # Policy
        if p_star_mem is not None and p_star_time is not None:
            p_combined = min(p_star_mem, p_star_time)
            policy[key_str] = {
                "use_D_below": p_combined,
                "recommendation": f"Use D when p_l < {p_combined:.3f}",
            }
        elif p_star_mem is not None:
            policy[key_str] = {
                "use_D_below": None,
                "recommendation": f"D saves memory below p={p_star_mem:.3f} but not time",
            }
        elif p_star_time is not None:
            policy[key_str] = {
                "use_D_below": None,
                "recommendation": f"D saves time below p={p_star_time:.3f} but not memory",
            }
        else:
            policy[key_str] = {
                "use_D_below": None,
                "recommendation": "D never wins — use A_bitset",
            }

    return {
        "mem_breakeven": mem_breakeven,
        "time_breakeven": time_breakeven,
        "policy": policy,
    }


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def compute_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute high-level summary statistics."""
    if not results:
        return {"total": 0, "pass_A": 0, "pass_B": 0}

    total = len(results)
    pass_A = sum(1 for r in results if r.get("verdict_A") == "PASS")
    pass_B = sum(1 for r in results if r.get("verdict_B") == "PASS")
    fail_A = sum(1 for r in results if r.get("verdict_A") == "FAIL")
    fail_B = sum(1 for r in results if r.get("verdict_B") == "FAIL")

    # Resident ratio stats
    ratios = [r["resident_ratio"] for r in results]
    time_ratios = [r["time_ratio"] for r in results]

    return {
        "total_trials": total,
        "contour_A": {"PASS": pass_A, "FAIL": fail_A},
        "contour_B": {"PASS": pass_B, "FAIL": fail_B},
        "resident_ratio": {
            "min": round(min(ratios), 4),
            "max": round(max(ratios), 4),
            "median": round(sorted(ratios)[len(ratios) // 2], 4),
        },
        "time_ratio": {
            "min": round(min(time_ratios), 4),
            "max": round(max(time_ratios), 4),
            "median": round(sorted(time_ratios)[len(time_ratios) // 2], 4),
        },
    }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_results(
    results: list[dict[str, Any]],
    output_path: str,
    *,
    analysis: dict[str, Any] | None = None,
    summary: dict[str, Any] | None = None,
) -> None:
    """Save results to JSON."""
    out = {
        "experiment": "exp10j_tree_perlevel",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": str(get_device()),
        "torch_version": torch.__version__,
        "config": {
            "branching_factors": BRANCHING_FACTORS,
            "occupancies": OCCUPANCIES,
            "patterns": PATTERNS,
            "payloads_bytes": PAYLOADS_BYTES,
            "operators": OPERATORS,
            "n_seeds": N_SEEDS,
            "n_warmup": N_WARMUP,
            "n_repeats": N_REPEATS,
        },
        "n_results": len(results),
        "results": results,
    }
    if summary is not None:
        out["summary"] = summary
    if analysis is not None:
        out["breakeven_analysis"] = analysis

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {len(results)} results to {output_path}")


def load_results(path: str) -> list[dict[str, Any]]:
    """Load results from a JSON chunk file."""
    with open(path) as f:
        data = json.load(f)
    return data.get("results", [])


def merge_chunks(output_path: str) -> None:
    """Merge all chunk_*.json files in the results directory into one file."""
    results_dir = Path(output_path).parent
    if not results_dir.exists():
        print(f"Results directory {results_dir} does not exist.", file=sys.stderr)
        sys.exit(1)

    all_results: list[dict[str, Any]] = []
    chunk_files = sorted(results_dir.glob("chunk_*.json"))

    if not chunk_files:
        print(f"No chunk files found in {results_dir}", file=sys.stderr)
        sys.exit(1)

    for chunk_file in chunk_files:
        print(f"Loading {chunk_file.name}...")
        chunk_results = load_results(str(chunk_file))
        all_results.extend(chunk_results)
        print(f"  -> {len(chunk_results)} results")

    print(f"\nTotal: {len(all_results)} results from {len(chunk_files)} chunks")

    summary = compute_summary(all_results)
    analysis = analyze_breakeven(all_results)

    save_results(all_results, output_path, analysis=analysis, summary=summary)

    # Print summary
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print("\n=== BREAK-EVEN POLICY ===")
    for key, pol in sorted(analysis["policy"].items()):
        print(f"  {key}: {pol['recommendation']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="exp10j: Per-level tree benchmark (D_direct vs A_bitset)"
    )
    parser.add_argument(
        "--branching",
        type=int,
        choices=BRANCHING_FACTORS,
        help="Run sweep for this branching factor only",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all branching factors sequentially",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge all chunk files into one output",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/merged.json",
        help="Output file path (default: results/merged.json)",
    )
    parser.add_argument(
        "--min-depth", type=int, default=1,
        help="Min tree depth for sweep (default: 1)",
    )
    parser.add_argument(
        "--max-depth", type=int, default=None,
        help="Max tree depth for sweep (default: use DEPTH_TABLE)",
    )
    args = parser.parse_args()

    if args.merge:
        merge_chunks(args.output)
        return

    if not args.branching and not args.all:
        parser.error("Specify --branching <N>, --all, or --merge")

    device = get_device()
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"PyTorch: {torch.__version__}")
    print()

    if args.all:
        all_results: list[dict[str, Any]] = []
        for br in BRANCHING_FACTORS:
            print(f"\n{'='*60}")
            print(f"  BRANCHING = {br}")
            print(f"{'='*60}\n")
            chunk_results = sweep_branching(br, device)
            all_results.extend(chunk_results)

            # Save incremental chunk
            chunk_path = str(
                Path(args.output).parent / f"chunk_br{br}.json"
            )
            summary = compute_summary(chunk_results)
            save_results(chunk_results, chunk_path, summary=summary)

        # Save merged
        summary = compute_summary(all_results)
        analysis = analyze_breakeven(all_results)
        save_results(all_results, args.output, analysis=analysis, summary=summary)

        print("\n=== FINAL SUMMARY ===")
        print(json.dumps(summary, indent=2))
        print("\n=== BREAK-EVEN POLICY ===")
        for key, pol in sorted(analysis["policy"].items()):
            print(f"  {key}: {pol['recommendation']}")

    elif args.branching:
        br = args.branching
        results = sweep_branching(br, device,
                                  depth_min=args.min_depth,
                                  depth_max=args.max_depth)
        summary = compute_summary(results)
        analysis = analyze_breakeven(results)

        # Default output path for single branching
        output = args.output
        if output == "results/merged.json":
            output = f"results/chunk_br{br}.json"

        save_results(results, output, analysis=analysis, summary=summary)

        print(f"\n=== SUMMARY (branching={br}) ===")
        print(json.dumps(summary, indent=2))
        print("\n=== BREAK-EVEN POLICY ===")
        for key, pol in sorted(analysis["policy"].items()):
            print(f"  {key}: {pol['recommendation']}")


if __name__ == "__main__":
    main()
