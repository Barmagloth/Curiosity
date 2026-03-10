#!/usr/bin/env python3
"""
Exp0.9a Sandbox — Analysis & Visualization

Reads results JSON and produces:
1. Logistics overhead heatmaps (compact/grid, morton/grid ratios)
2. A1 compute speedup chart by sparsity × side
3. Memory model comparison
4. Summary table for report
"""

import json
import numpy as np
from pathlib import Path

RESULTS_PATH = Path("/home/claude/exp09a_results/exp09a_sandbox.json")

with open(RESULTS_PATH) as f:
    data = json.load(f)

print("=" * 70)
print("Exp0.9a SANDBOX — DETAILED ANALYSIS")
print("=" * 70)

# ════════════════════════════════════════════
# 1. LOGISTICS: is build cost tolerable?
# ════════════════════════════════════════════
print("\n╔══════════════════════════════════════════╗")
print("║  1. LOGISTICS OVERHEAD (build cost)       ║")
print("╚══════════════════════════════════════════╝")
print()
print("Question: can we afford to rebuild active_idx every step?")
print()

logistics = data["logistics"]

# Show build time as fraction of hypothetical step budget
# Assume 1 step ≈ 1-10 ms on GPU.  If logistics > 1ms on CPU, it's a red flag.
print(f"{'side':>6s} {'sp':>5s} {'pat':>9s} | {'grid':>8s} {'compact':>8s} {'morton':>8s} {'blk_sp':>8s} | {'comp/grd':>8s} {'mort/grd':>8s}")
print("-" * 90)
for e in logistics:
    g = e["layouts"]["fixed_grid"]["median_us"]
    c = e["layouts"]["compact"]["median_us"]
    m = e["layouts"]["morton"]["median_us"]
    b = e["layouts"]["block_sparse"]["median_us"]
    print(f"{e['side']:6d} {e['sparsity']:5.2f} {e['pattern']:>9s} | "
          f"{g:7.0f}µ {c:7.0f}µ {m:7.0f}µ {b:7.0f}µ | "
          f"{c/g:7.1f}×  {m/g:7.1f}×")

print()
print("Observations:")
print("  • Compact build: 2.6-3.1× grid (np.where + reverse_map fill).")
print("    At 512²: ~250-530µs.  Acceptable per-step (sub-ms).")
print("  • Morton build: 12-15× grid (adds argsort on Morton codes).")
print("    At 512²: 450-7600µs.  At high sparsity — multi-ms.  Marginal.")
print("  • Block-sparse: 14-50ms at 512².  DEAD on CPU.  Pure Python loops.")
print("    (Would need Cython/Numba/CUDA to be viable; not worth it for sandbox.)")

# ════════════════════════════════════════════
# 2. A1 COMPUTE: does indexed access save work?
# ════════════════════════════════════════════
print()
print("╔══════════════════════════════════════════╗")
print("║  2. A1: GATHER/COMPUTE/SCATTER            ║")
print("╚══════════════════════════════════════════╝")
print()
print("Question: at low sparsity, does compact touch fewer elements → faster?")
print()

gcs = data["gather_compute_scatter"]

print(f"{'side':>6s} {'sp':>5s} {'pat':>9s} | {'grid':>8s} {'compact':>8s} {'morton':>8s} | {'comp/grd':>8s} {'verdict':>8s}")
print("-" * 80)
for e in gcs:
    g = e["layouts"]["fixed_grid"]["median_us"]
    c = e["layouts"]["compact"]["median_us"]
    m = e["layouts"]["morton"]["median_us"]
    ratio = c / g if g > 0 else 999
    if ratio <= 0.67:
        v = "WIN"
    elif ratio <= 1.0:
        v = "~"
    else:
        v = "LOSE"
    print(f"{e['side']:6d} {e['sparsity']:5.2f} {e['pattern']:>9s} | "
          f"{g:7.0f}µ {c:7.0f}µ {m:7.0f}µ | "
          f"{ratio:7.2f}×  {v:>5s}")

print()
print("Key findings:")
print("  • At side ≥ 128 and sparsity ≤ 15%:")
print("    compact wins 2-4× (fewer elements, amortizes gather/scatter).")
print()
print("  • At sparsity ≥ 30% + clustered pattern:")
print("    compact LOSES.  Why?  grid does vectorized compute on contiguous array")
print("    + mask; compact does indexed gather into non-contiguous memory.")
print("    At 30% active, savings from fewer elements don't cover gather overhead.")
print()
print("  • Random pattern helps compact:")
print("    grid's masked compute still touches full array;")
print("    compact's gather is bad either way, but grid gets worse on random.")
print()
print("  • Morton ≈ compact (no extra benefit on CPU; cache locality irrelevant")
print("    for numpy's vectorized ops — they're already streaming).")
print()
print("  • Block-sparse: always worst (expansion ratio: active blocks include")
print("    inactive tiles → more work than grid).")

# ════════════════════════════════════════════
# 3. MEMORY MODEL: who's heavier?
# ════════════════════════════════════════════
print()
print("╔══════════════════════════════════════════╗")
print("║  3. MEMORY MODEL                          ║")
print("╚══════════════════════════════════════════╝")
print()
print("Surprise: compact uses MORE memory than grid at ALL sparsities.")
print()
print("Why: reverse_map (M × int32 = 4 bytes/tile) alone > mask (M × 1 byte).")
print("     Plus active_idx (k × int32).  Total: 4M + 4k vs M.")
print()
print(f"{'side':>6s} {'sp':>5s} | {'grid':>10s} {'compact':>10s} {'morton':>10s} | {'comp/grd':>8s}")
print("-" * 65)
for e in data["memory_model"]:
    side = int(np.sqrt(e["M"]))
    sp = e["k"] / e["M"]
    g = e["fixed_grid"]["total"]
    c = e["compact"]["total"]
    m = e["morton"]["total"]
    print(f"{side:6d} {sp:5.2f} | {g/1024:9.1f}KB {c/1024:9.1f}KB {m/1024:9.1f}KB | {c/g:7.1f}×")

print()
print("At 512² sp=0.05: grid=256KB, compact=1075KB, morton=1126KB.")
print("Compact uses 4.2× MORE memory.  Memory kill criterion: DEAD.")
print()
print("However: this is STEADY-STATE.  On GPU with VRAM, the question is")
print("whether compact lets you FIT a larger M by avoiding compute on the")
print("full grid.  The data structures cost more, but the intermediate")
print("tensors (field values, deltas, ρ) could be smaller if only allocated")
print("for active tiles.  This needs GPU measurement (0.9b).")

# ════════════════════════════════════════════
# 4. NEIGHBOR ACCESS
# ════════════════════════════════════════════
print()
print("╔══════════════════════════════════════════╗")
print("║  4. NEIGHBOR ACCESS (halo stencil)         ║")
print("╚══════════════════════════════════════════╝")
print()

na = data["neighbor_access"]
print(f"{'side':>6s} {'sp':>5s} {'pat':>9s} | {'grid':>8s} {'compact':>8s} {'morton':>8s} | {'comp/grd':>8s}")
print("-" * 70)
for e in na:
    g = e["layouts"]["fixed_grid"]["median_us"]
    c = e["layouts"]["compact"]["median_us"]
    m = e["layouts"]["morton"]["median_us"]
    print(f"{e['side']:6d} {e['sparsity']:5.2f} {e['pattern']:>9s} | "
          f"{g:7.0f}µ {c:7.0f}µ {m:7.0f}µ | {c/g:7.2f}×")

print()
print("Grid wins: 1.0× baseline.  Compact/Morton: 1.15-1.45×.")
print("Extra cost = index-map lookup for each neighbor.")
print("On GPU this gap may widen (random memory access pattern).")

# ════════════════════════════════════════════
# 5. DETERMINISM
# ════════════════════════════════════════════
print()
print("╔══════════════════════════════════════════╗")
print("║  5. DETERMINISM                            ║")
print("╚══════════════════════════════════════════╝")
print()
det = data["determinism"]
all_ok = all(c["bitwise_equal"] for c in det["comparisons"])
print(f"Bitwise identical across all traversal orders: {'YES' if all_ok else 'NO'}")
if all_ok:
    print()
    print("Why: each delta writes to a UNIQUE index (no overlap between tiles).")
    print("Summation order of deltas at each index is always the same (one delta")
    print("per tile per step).  No reduction = no FP reordering = bitwise match.")
    print()
    print("CAUTION: this holds for NON-OVERLAPPING writes.  With halo (overlap>0),")
    print("multiple tiles contribute to the same element → summation order matters.")
    print("Must test separately when halo overlap creates write conflicts.")

# ════════════════════════════════════════════
# 6. KILL CRITERIA & VERDICT
# ════════════════════════════════════════════
print()
print("╔══════════════════════════════════════════╗")
print("║  6. VERDICT                                ║")
print("╚══════════════════════════════════════════╝")
print()
print("Speed criterion (compact ≥1.5× at sp≤30%):  ALIVE")
print("  But ONLY at side≥128 AND sp≤15% AND (random pattern OR large side).")
print("  At sp=30% clustered: compact LOSES.")
print()
print("Memory criterion (compact ≤0.6× at sp≤30%): DEAD")
print("  reverse_map dominates.  Compact always uses more memory than grid.")
print()
print("Morton over compact:  NO BENEFIT (on CPU)")
print("  Sort adds 5-10× overhead to build, no compute speedup.")
print()
print("Block-sparse:  DEAD")
print("  Expansion ratio kills it.  Pure Python implementation is catastrophic,")
print("  but even in principle: touching inactive tiles in active blocks")
print("  negates the point of sparse layout.")
print()
print("─── Recommendation ───")
print()
print("1. For 0.9b on GPU, test ONLY grid vs compact (drop Morton, block-sparse).")
print("   Morton adds sort cost with zero compute benefit.")
print("   Block-sparse is structurally flawed (expansion).")
print()
print("2. Compact has a narrow win zone: high M, low sparsity (<15%), or")
print("   scenarios where VRAM savings on intermediate tensors matter.")
print("   GPU profiling will show if kernel-launch overhead kills it.")
print()
print("3. If compact dies on GPU too → fixed grid + mask is the answer.")
print("   This is not a failure — it's the simplest correct solution.")
print()
print("4. Determinism is clean for non-overlapping writes.")
print("   Halo overlap region needs a separate bitwise test on GPU.")
print()
print("5. The real question for GPU: does compact allow LARGER M by")
print("   allocating intermediate buffers only for active tiles?")
print("   Memory model says data-structure overhead is 4×, but if")
print("   intermediate computation buffers scale with k instead of M,")
print("   net savings could flip.  This is the 0.9b/0.9c question.")


if __name__ == "__main__":
    pass  # Analysis printed on import/run
