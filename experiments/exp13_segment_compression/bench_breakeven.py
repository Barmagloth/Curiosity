#!/usr/bin/env python3
"""
Microbenchmark: empirically measure C_refine, C_track, C_init to compute
N_critical -- the minimum number of degree-2 nodes where segment compression
becomes profitable.

Breakeven formula:
    N_critical = C_init / (C_refine * (1 - 1/L_avg) - C_track)

Where L_avg = 3 (conservative average segment length).
"""

import json
import os
import time
import numpy as np

from segment_compress import (
    hamming12,
    component_diff,
    SignatureStabilityChecker,
    RefinementTree,
    SegmentTree,
)

ITERATIONS = 10_000
L_AVG = 3


# ─────────────────────────────────────────────────────────────────────
# Helpers: build a complete binary tree of given depth
# ─────────────────────────────────────────────────────────────────────

def make_binary_tree(depth: int):
    """Return (parent, children) dicts for a complete binary tree of given depth.
    Nodes numbered 1..2^depth - 1 (root=1)."""
    parent = {}
    children = {}
    n_nodes = (1 << depth) - 1  # 2^depth - 1

    for i in range(1, n_nodes + 1):
        children[i] = []
    for i in range(1, n_nodes + 1):
        p = i // 2
        if p >= 1 and i != 1:
            parent[i] = p
            children[p].append(i)
        elif i == 1:
            parent[i] = None

    return parent, children


# ─────────────────────────────────────────────────────────────────────
# SC-Enforce simulation helpers (R_fn / Up_fn for grid-like delta)
# ─────────────────────────────────────────────────────────────────────

def R_fn(delta: np.ndarray) -> np.ndarray:
    """Gaussian restrict: average 2x2 blocks (coarsen by factor 2)."""
    h, w = delta.shape
    return delta.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))


def Up_fn(coarse: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Bilinear prolong: repeat elements 2x in each dimension."""
    return coarse.repeat(2, axis=0).repeat(2, axis=1)


# ─────────────────────────────────────────────────────────────────────
# Benchmark: C_refine
# ─────────────────────────────────────────────────────────────────────

def bench_c_refine(n: int) -> np.ndarray:
    """Simulate pipeline cost for one degree-2 transit node.

    Per iteration:
      1. Two-stage gate check (FSR + instability from probe data, utility-weighted)
      2. rho computation (np.linalg.norm on ~64 elements)
      3. SC-Enforce: R_fn(delta) + Up_fn(R_delta, shape) + two norms
    """
    rng = np.random.default_rng(42)

    # Pre-allocate realistic data
    delta = rng.standard_normal((8, 8))
    probe_fsr = rng.random(4)
    probe_instab = rng.random(4)
    weights = rng.random(4)
    weights /= weights.sum()
    rho_vec = rng.standard_normal(64)

    times = np.empty(n, dtype=np.int64)

    for i in range(n):
        t0 = time.perf_counter_ns()

        # 1) Two-stage gate: compute FSR + instability, then utility-weighted soft weights
        fsr_score = np.dot(probe_fsr, weights)
        instab_score = np.dot(probe_instab, weights)
        gate_pass = (fsr_score > 0.3) and (instab_score < 0.7)

        # 2) rho computation
        rho = np.linalg.norm(rho_vec)

        # 3) SC-Enforce: d_parent_lf_frac
        R_delta = R_fn(delta)
        Up_R_delta = Up_fn(R_delta, delta.shape)
        norm_diff = np.linalg.norm(delta - Up_R_delta)
        norm_orig = np.linalg.norm(delta)

        t1 = time.perf_counter_ns()
        times[i] = t1 - t0

    return times


# ─────────────────────────────────────────────────────────────────────
# Benchmark: C_track
# ─────────────────────────────────────────────────────────────────────

def bench_c_track(n: int) -> np.ndarray:
    """Cost of one node's compression bookkeeping per step.

    Per iteration:
      - hamming12(sig_a, sig_b)
      - component_diff(sig_a, sig_b)
      - SignatureStabilityChecker.record(node_id, signature, step)
      - One dict lookup in _node_to_seg
    """
    rng = np.random.default_rng(99)
    checker = SignatureStabilityChecker(
        hamming_threshold=4, component_threshold=4, stability_window=3
    )
    # Pre-populate checker with a baseline for node 0
    checker.record(0, 0xABC, 0)

    # Fake _node_to_seg dict similar to SegmentTree internals
    node_to_seg = {i: i % 10 for i in range(256)}

    # Pre-generate random signatures
    sigs_a = rng.integers(0, 0xFFF, size=n, endpoint=True)
    sigs_b = rng.integers(0, 0xFFF, size=n, endpoint=True)

    times = np.empty(n, dtype=np.int64)

    for i in range(n):
        sa = int(sigs_a[i])
        sb = int(sigs_b[i])
        t0 = time.perf_counter_ns()

        # 1) hamming12
        hd = hamming12(sa, sb)

        # 2) component_diff
        cd = component_diff(sa, sb)

        # 3) stability checker record
        checker.record(0, sa, i + 1)

        # 4) dict lookup
        seg_idx = node_to_seg.get(0)

        t1 = time.perf_counter_ns()
        times[i] = t1 - t0

    return times


# ─────────────────────────────────────────────────────────────────────
# Benchmark: C_init
# ─────────────────────────────────────────────────────────────────────

def bench_c_init(n: int) -> np.ndarray:
    """Fixed cost of instantiating SegmentTree + SignatureStabilityChecker.

    Per iteration:
      - Build RefinementTree from binary tree of depth 8 (255 nodes)
      - Instantiate SegmentTree (which calls find_degree2_chains internally)
    """
    parent, children = make_binary_tree(depth=8)

    times = np.empty(n, dtype=np.int64)

    for i in range(n):
        t0 = time.perf_counter_ns()

        tree = RefinementTree(parent, children)
        st = SegmentTree(tree, max_length=8, stability_window=3)

        t1 = time.perf_counter_ns()
        times[i] = t1 - t0

    return times


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Segment Compression Breakeven Microbenchmark")
    print(f"Iterations per measurement: {ITERATIONS:,}")
    print(f"Timer: time.perf_counter_ns")
    print()

    # Warmup
    print("Warming up...")
    bench_c_refine(100)
    bench_c_track(100)
    bench_c_init(100)

    # Measure
    print("Measuring C_refine...")
    t_refine = bench_c_refine(ITERATIONS)
    print("Measuring C_track...")
    t_track = bench_c_track(ITERATIONS)
    print("Measuring C_init...")
    t_init = bench_c_init(ITERATIONS)

    c_refine = float(np.median(t_refine))
    c_track = float(np.median(t_track))
    c_init = float(np.median(t_init))

    # Compute N_critical
    denominator = c_refine * (1.0 - 1.0 / L_AVG) - c_track

    print()
    print("=" * 60)
    print(f"  BREAKEVEN ANALYSIS  (L_avg = {L_AVG})")
    print("=" * 60)
    print(f"  C_refine  (median) : {c_refine:>12,.0f} ns")
    print(f"  C_track   (median) : {c_track:>12,.0f} ns")
    print(f"  C_init    (median) : {c_init:>12,.0f} ns")
    print("-" * 60)
    print(f"  C_refine * (1 - 1/L_avg)  : {c_refine * (1 - 1/L_AVG):>12,.1f} ns")
    print(f"  denominator               : {denominator:>12,.1f} ns")
    print("-" * 60)

    if denominator <= 0:
        n_critical = float("inf")
        print("  N_critical : COMPRESSION NEVER PROFITABLE")
        print("  (per-node tracking overhead exceeds refine savings)")
    else:
        n_critical = c_init / denominator
        print(f"  N_critical : {n_critical:>12,.1f} degree-2 nodes")
        print(f"             ~ {int(np.ceil(n_critical)):>6d} (ceil)")

    print("=" * 60)

    # Percentiles for context
    print()
    print("Percentiles (ns):")
    print(f"  {'':18s} {'p25':>10s} {'p50':>10s} {'p75':>10s} {'p99':>10s}")
    for name, data in [("C_refine", t_refine), ("C_track", t_track), ("C_init", t_init)]:
        p25, p50, p75, p99 = np.percentile(data, [25, 50, 75, 99])
        print(f"  {name:18s} {p25:>10,.0f} {p50:>10,.0f} {p75:>10,.0f} {p99:>10,.0f}")

    # Save results
    results = {
        "iterations": ITERATIONS,
        "L_avg": L_AVG,
        "C_refine_median_ns": c_refine,
        "C_track_median_ns": c_track,
        "C_init_median_ns": c_init,
        "denominator_ns": denominator,
        "N_critical": n_critical if denominator > 0 else None,
        "N_critical_ceil": int(np.ceil(n_critical)) if denominator > 0 else None,
        "compression_profitable": denominator > 0,
        "percentiles": {
            "C_refine": {f"p{p}": float(np.percentile(t_refine, p)) for p in [25, 50, 75, 99]},
            "C_track": {f"p{p}": float(np.percentile(t_track, p)) for p in [25, 50, 75, 99]},
            "C_init": {f"p{p}": float(np.percentile(t_init, p)) for p in [25, 50, 75, 99]},
        },
    }

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "bench_breakeven.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
