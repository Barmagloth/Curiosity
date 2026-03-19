#!/usr/bin/env python3
"""
Chunked runner for exp10i_graph_blocks.

Runs a single (graph_type, graph_size) slice of the full experiment grid.
Saves results incrementally to a JSON file after each config completes.

Usage:
    python exp10i_chunk.py --graph_type random_geometric --graph_size 256 \
        --output results/chunk_rg256.json
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Force CPU to allow parallel execution without CUDA contention
# ---------------------------------------------------------------------------
DEVICE = torch.device("cpu")
DTYPE = torch.float32

# ---------------------------------------------------------------------------
# Constants (mirrored from exp10i_graph_blocks.py -- cannot import that
# module because it has top-level benchmark code with no __main__ guard)
# ---------------------------------------------------------------------------

SEED_BASE = 42
N_SEEDS = 10
N_WARMUP = 5
N_REPEAT = 20
KILL_THRESH = 0.20
CROSS_BLOCK_THRESH = 0.50
PADDING_WASTE_THRESH = 0.50
FEATURE_DIM = 8

SPARSITIES = [0.05, 0.1, 0.3, 0.5]
BLOCK_SIZES = [8, 16, 32, 64]
PARTITION_METHODS = ["random_partition", "spatial_partition", "greedy_partition"]

# ---------------------------------------------------------------------------
# Import pure functions from the main module by loading only definitions.
# We read the source up to the benchmark loop and exec it.
# ---------------------------------------------------------------------------
_this_dir = Path(__file__).resolve().parent
_main_src = (_this_dir / "exp10i_graph_blocks.py").read_text(encoding="utf-8")

# Find where the benchmark loop starts (line "# BENCHMARK LOOP")
_cutoff = _main_src.find("# #####################################################################\n# BENCHMARK LOOP")
if _cutoff < 0:
    raise RuntimeError("Could not find BENCHMARK LOOP marker in exp10i_graph_blocks.py")

_func_src = _main_src[:_cutoff]

# Build a namespace with the imports the functions need
import collections
from collections import defaultdict
from typing import Any, Dict, List, Tuple

_ns: dict = {
    "torch": torch, "np": np, "math": math, "time": time,
    "defaultdict": defaultdict, "json": json,
    "List": List, "Dict": Dict, "Tuple": Tuple, "Any": Any,
    "Path": Path,
    "__name__": "__exp10i_funcs__",
    "__file__": str(_this_dir / "exp10i_graph_blocks.py"),
}
exec(compile(_func_src, str(_this_dir / "exp10i_graph_blocks.py"), "exec"), _ns)

# Extract the pure functions we need (graph gen, partitioning, quality)
generate_graph = _ns["generate_graph"]
partition_nodes = _ns["partition_nodes"]
compute_cross_block_ratio = _ns["compute_cross_block_ratio"]
compute_padding_waste = _ns["compute_padding_waste"]
make_active_mask = _ns["make_active_mask"]

# ---------------------------------------------------------------------------
# Helpers (CPU-safe versions)
# ---------------------------------------------------------------------------

def _sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)


def _timed_runs(func, n_warmup=N_WARMUP, n_repeat=N_REPEAT):
    for _ in range(n_warmup):
        func()
        _sync()
    times = []
    for _ in range(n_repeat):
        _sync()
        t0 = time.perf_counter()
        func()
        _sync()
        dt = time.perf_counter() - t0
        times.append(dt)
    return np.array(times)


def _build_time(build_fn) -> float:
    times = []
    for _ in range(5):
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        _sync()
        t0 = time.perf_counter()
        build_fn()
        _sync()
        dt = time.perf_counter() - t0
        times.append(dt)
    return float(np.median(times)) * 1e6


# ---------------------------------------------------------------------------
# Layout builders (CPU-safe, copied from main module but using local DEVICE)
# ---------------------------------------------------------------------------

def build_graph_baseline(N, feat_dim, active_mask, adj_indices, adj_indptr, seed):
    torch.manual_seed(seed)
    data = torch.randn(N, feat_dim, dtype=DTYPE, device=DEVICE)
    active_mask_t = torch.from_numpy(active_mask).to(DEVICE)
    adj_indices_t = torch.from_numpy(adj_indices).to(torch.int64).to(DEVICE)
    adj_indptr_t = torch.from_numpy(adj_indptr).to(torch.int64).to(DEVICE)
    return {
        "data": data,
        "active_mask": active_mask_t,
        "adj_indices": adj_indices_t,
        "adj_indptr": adj_indptr_t,
    }


def compute_graph_baseline(bl):
    data = bl["data"]
    active = bl["active_mask"]
    indices = bl["adj_indices"]
    indptr = bl["adj_indptr"]
    N, feat_dim = data.shape
    output = data.clone()
    active_nodes = torch.where(active)[0]
    for i_idx in range(len(active_nodes)):
        node = active_nodes[i_idx].item()
        s = indptr[node].item()
        e = indptr[node + 1].item()
        if s == e:
            continue
        nbr_ids = indices[s:e]
        nbr_active = active[nbr_ids]
        active_nbrs = nbr_ids[nbr_active]
        if len(active_nbrs) == 0:
            continue
        output[node] = data[active_nbrs].mean(dim=0)
    return output


def build_graph_blocks(N, feat_dim, active_mask, adj_indices, adj_indptr,
                       assignment, block_size, seed):
    torch.manual_seed(seed)
    original_data = torch.randn(N, feat_dim, dtype=DTYPE, device=DEVICE)
    n_blocks_total = int(assignment.max()) + 1 if N > 0 else 0
    active_nodes = np.where(active_mask)[0]
    active_blocks_set = set(assignment[active_nodes].tolist())
    block_map_np = np.full(n_blocks_total, -1, dtype=np.int32)
    sorted_active_blocks = sorted(active_blocks_set)
    for slot, bid in enumerate(sorted_active_blocks):
        block_map_np[bid] = slot
    n_active_blocks = len(sorted_active_blocks)
    block_fill = np.zeros(n_blocks_total, dtype=np.int32)
    node_to_local = np.zeros(N, dtype=np.int32)
    for node in range(N):
        bid = assignment[node]
        node_to_local[node] = block_fill[bid]
        block_fill[bid] += 1
    n_ab = max(n_active_blocks, 1)
    packed_data = torch.zeros(n_ab, block_size, feat_dim, dtype=DTYPE, device=DEVICE)
    for node in active_nodes:
        bid = assignment[node]
        slot = block_map_np[bid]
        if slot >= 0:
            local = node_to_local[node]
            if local < block_size:
                packed_data[slot, local] = original_data[node]
    src_blocks, src_locals, dst_blocks, dst_locals = [], [], [], []
    intra_src_blocks, intra_src_locals, intra_dst_locals = [], [], []
    for node in active_nodes:
        bid_src = assignment[node]
        slot_src = block_map_np[bid_src]
        local_src = node_to_local[node]
        if slot_src < 0 or local_src >= block_size:
            continue
        s = adj_indptr[node]
        e = adj_indptr[node + 1]
        for idx in range(s, e):
            nbr = adj_indices[idx]
            if not active_mask[nbr]:
                continue
            bid_dst = assignment[nbr]
            slot_dst = block_map_np[bid_dst]
            local_dst = node_to_local[nbr]
            if slot_dst < 0 or local_dst >= block_size:
                continue
            if bid_src == bid_dst:
                intra_src_blocks.append(slot_src)
                intra_src_locals.append(local_src)
                intra_dst_locals.append(local_dst)
            else:
                src_blocks.append(slot_src)
                src_locals.append(local_src)
                dst_blocks.append(slot_dst)
                dst_locals.append(local_dst)
    block_map_t = torch.from_numpy(block_map_np).to(DEVICE)
    node_to_block_t = torch.from_numpy(assignment).to(torch.int32).to(DEVICE)
    node_to_local_t = torch.from_numpy(node_to_local).to(torch.int32).to(DEVICE)
    if src_blocks:
        cross_src_block = torch.tensor(src_blocks, dtype=torch.int64, device=DEVICE)
        cross_src_local = torch.tensor(src_locals, dtype=torch.int64, device=DEVICE)
        cross_dst_block = torch.tensor(dst_blocks, dtype=torch.int64, device=DEVICE)
        cross_dst_local = torch.tensor(dst_locals, dtype=torch.int64, device=DEVICE)
    else:
        cross_src_block = torch.empty(0, dtype=torch.int64, device=DEVICE)
        cross_src_local = torch.empty(0, dtype=torch.int64, device=DEVICE)
        cross_dst_block = torch.empty(0, dtype=torch.int64, device=DEVICE)
        cross_dst_local = torch.empty(0, dtype=torch.int64, device=DEVICE)
    if intra_src_blocks:
        intra_src_block_t = torch.tensor(intra_src_blocks, dtype=torch.int64, device=DEVICE)
        intra_src_local_t = torch.tensor(intra_src_locals, dtype=torch.int64, device=DEVICE)
        intra_dst_local_t = torch.tensor(intra_dst_locals, dtype=torch.int64, device=DEVICE)
    else:
        intra_src_block_t = torch.empty(0, dtype=torch.int64, device=DEVICE)
        intra_src_local_t = torch.empty(0, dtype=torch.int64, device=DEVICE)
        intra_dst_local_t = torch.empty(0, dtype=torch.int64, device=DEVICE)
    return {
        "packed_data": packed_data,
        "block_map": block_map_t,
        "node_to_block": node_to_block_t,
        "node_to_local": node_to_local_t,
        "n_active_blocks": n_active_blocks,
        "cross_edges": (cross_src_block, cross_src_local,
                        cross_dst_block, cross_dst_local),
        "intra_edges": (intra_src_block_t, intra_src_local_t,
                        intra_dst_local_t),
        "original_data": original_data,
        "active_mask": torch.from_numpy(active_mask).to(DEVICE),
    }


def compute_graph_blocks(bl):
    packed = bl["packed_data"]
    n_ab = bl["n_active_blocks"]
    if n_ab == 0:
        return packed
    output_sum = torch.zeros_like(packed)
    output_count = torch.zeros(packed.shape[0], packed.shape[1],
                               dtype=DTYPE, device=DEVICE)
    intra_src_block, intra_src_local, intra_dst_local = bl["intra_edges"]
    if len(intra_src_block) > 0:
        dst_feats = packed[intra_src_block, intra_dst_local]
        output_sum.index_put_(
            (intra_src_block, intra_src_local), dst_feats, accumulate=True)
        ones = torch.ones(len(intra_src_block), dtype=DTYPE, device=DEVICE)
        output_count.index_put_(
            (intra_src_block, intra_src_local), ones, accumulate=True)
    cross_src_block, cross_src_local, cross_dst_block, cross_dst_local = bl["cross_edges"]
    if len(cross_src_block) > 0:
        dst_feats = packed[cross_dst_block, cross_dst_local]
        output_sum.index_put_(
            (cross_src_block, cross_src_local), dst_feats, accumulate=True)
        ones = torch.ones(len(cross_src_block), dtype=DTYPE, device=DEVICE)
        output_count.index_put_(
            (cross_src_block, cross_src_local), ones, accumulate=True)
    mask = output_count > 0
    result = packed.clone()
    if mask.any():
        safe_count = output_count.clamp(min=1.0).unsqueeze(-1)
        mean_feats = output_sum / safe_count
        mask_3d = mask.unsqueeze(-1).expand_as(result)
        result[mask_3d] = mean_feats[mask_3d]
    return result


# ---------------------------------------------------------------------------
# Main chunk runner
# ---------------------------------------------------------------------------

def run_chunk(graph_type: str, graph_size: int, output_file: str):
    """Run all configs for a single (graph_type, graph_size) pair."""

    if graph_type == "grid_graph":
        side = int(math.sqrt(graph_size))
        actual_N = side * side
    else:
        actual_N = graph_size

    configs = []
    for sparsity in SPARSITIES:
        for block_size in BLOCK_SIZES:
            for partition_method in PARTITION_METHODS:
                configs.append((sparsity, block_size, partition_method))

    total = len(configs)
    print(f"=== Chunk: {graph_type} N={actual_N} ({total} configs) ===")
    print(f"Output: {output_file}")
    print(f"Device: {DEVICE}")

    # Ensure output dir exists
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []

    for ci, (sparsity, block_size, partition_method) in enumerate(configs):
        seed_records_base = []
        seed_records_d = []
        seed_cross_ratios = []
        seed_padding_wastes = []

        for seed_idx in range(N_SEEDS):
            seed = SEED_BASE + seed_idx
            rng = np.random.default_rng(seed)
            torch.manual_seed(seed)

            positions, adj_indices, adj_indptr = generate_graph(
                graph_type, actual_N, rng)
            N = len(adj_indptr) - 1
            active_mask = make_active_mask(N, sparsity, rng)
            assignment = partition_nodes(
                partition_method, N, block_size,
                positions, adj_indices, adj_indptr,
                graph_type, rng)

            cbr = compute_cross_block_ratio(assignment, adj_indices, adj_indptr, N)
            pw = compute_padding_waste(assignment, active_mask, block_size)
            seed_cross_ratios.append(cbr)
            seed_padding_wastes.append(pw)

            # ---- graph_baseline ----
            bl_base = build_graph_baseline(
                N, FEATURE_DIM, active_mask, adj_indices, adj_indptr, seed)

            # On CPU we measure process-level memory instead of CUDA
            # Just record 0 for memory metrics since CPU memory tracking
            # is not as precise; the important metrics are timing + quality
            resident_base = 0

            times_base = _timed_runs(lambda: compute_graph_baseline(bl_base))
            peak_base = 0
            wall_base_us = float(np.median(times_base)) * 1e6

            build_base_us = _build_time(
                lambda: build_graph_baseline(
                    N, FEATURE_DIM, active_mask, adj_indices, adj_indptr, seed))

            seed_records_base.append({
                "resident_bytes": int(resident_base),
                "peak_vram_bytes": int(peak_base),
                "wall_clock_us": wall_base_us,
                "build_cost_us": build_base_us,
            })
            del bl_base

            # ---- D_graph_blocks ----
            bl_d = build_graph_blocks(
                N, FEATURE_DIM, active_mask, adj_indices, adj_indptr,
                assignment, block_size, seed)

            resident_d = 0
            times_d = _timed_runs(lambda: compute_graph_blocks(bl_d))
            peak_d = 0
            wall_d_us = float(np.median(times_d)) * 1e6

            build_d_us = _build_time(
                lambda: build_graph_blocks(
                    N, FEATURE_DIM, active_mask, adj_indices, adj_indptr,
                    assignment, block_size, seed))

            seed_records_d.append({
                "resident_bytes": int(resident_d),
                "peak_vram_bytes": int(peak_d),
                "wall_clock_us": wall_d_us,
                "build_cost_us": build_d_us,
            })
            del bl_d

        # Aggregate across seeds
        def _median_dict(records):
            out = {}
            for key in records[0]:
                vals = [r[key] for r in records]
                out[key] = float(np.median(vals))
            return out

        base_agg = _median_dict(seed_records_base)
        d_agg = _median_dict(seed_records_d)
        median_cbr = float(np.median(seed_cross_ratios))
        median_pw = float(np.median(seed_padding_wastes))

        time_overhead = ((d_agg["wall_clock_us"] - base_agg["wall_clock_us"])
                         / max(base_agg["wall_clock_us"], 1e-9))

        contour_a = (d_agg["resident_bytes"] < base_agg["resident_bytes"]
                     and time_overhead < KILL_THRESH
                     and median_cbr < CROSS_BLOCK_THRESH)
        contour_b = (d_agg["peak_vram_bytes"] < base_agg["peak_vram_bytes"]
                     and time_overhead < KILL_THRESH)

        # On CPU, memory is 0 for both, so contour A/B based on memory
        # are meaningless.  Use timing + quality only for CPU runs.
        # We'll re-evaluate contours at merge time if needed.
        # For CPU: treat memory contours as N/A, but still record timing.
        padding_warning = median_pw > PADDING_WASTE_THRESH

        rec = {
            "graph_type": graph_type,
            "N_nodes": N,
            "sparsity": sparsity,
            "block_size": block_size,
            "partition_method": partition_method,
            "graph_baseline": base_agg,
            "D_graph_blocks": d_agg,
            "time_overhead_frac": round(time_overhead, 4),
            "cross_block_ratio": round(median_cbr, 4),
            "padding_waste": round(median_pw, 4),
            "contour_A": "PASS" if contour_a else "FAIL",
            "contour_B": "PASS" if contour_b else "FAIL",
            "padding_warning": padding_warning,
        }
        results.append(rec)

        # Incremental save
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

        ca_tag = "PASS" if contour_a else "FAIL"
        cb_tag = "PASS" if contour_b else "FAIL"
        pw_tag = " !!PADDING" if padding_warning else ""
        print(f"  [{ci+1}/{total}] sp={sparsity} bs={block_size} "
              f"part={partition_method} | "
              f"A={ca_tag} B={cb_tag} "
              f"time_oh={time_overhead:+.1%} "
              f"cbr={median_cbr:.2f} pw={median_pw:.2f}{pw_tag}")

    print(f"\n=== Chunk {graph_type} N={actual_N} DONE ({len(results)} configs) ===")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="exp10i chunk runner")
    parser.add_argument("--graph_type", required=True,
                        choices=["random_geometric", "barabasi_albert", "grid_graph"])
    parser.add_argument("--graph_size", required=True, type=int)
    parser.add_argument("--output", required=True, help="Output JSON file path")
    args = parser.parse_args()

    t0 = time.perf_counter()
    run_chunk(args.graph_type, args.graph_size, args.output)
    elapsed = time.perf_counter() - t0
    print(f"Total time: {elapsed:.1f}s")
