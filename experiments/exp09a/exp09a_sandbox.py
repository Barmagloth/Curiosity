#!/usr/bin/env python3
"""
Curiosity — Phase 3, Exp0.9a (sandbox / CPU-only)

Three blocks:
  Block 1 — A2 "logistics": compaction, Morton sort, scatter.
             Measures CPU wall-clock, scaling behaviour, constants.
  Block 2 — Memory model: steady-state bytes for each layout × sparsity × M.
  Block 3 — Determinism: bitwise checks that traversal order
             (natural vs Morton vs random) doesn't change additive results
             beyond FP summation noise.
  Block 4 — A1 "index imitation": gather/compute/scatter on CPU
             with prebaked active_idx.  Gives relative cost comparison
             across layouts (not absolute GPU truth).

Layouts:
  1. fixed_grid   — bool mask, arithmetic indexing
  2. compact      — dense active_idx array + reverse index-map
  3. morton       — compact list sorted by Morton/Z-order code
  4. block_sparse — blocks of BxB tiles, block-level mask + compact block list

All units relative.  Grid is 2D: M = grid_side².
"""

import time
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

GRID_SIDES = [64, 128, 256, 512]          # M = side²
SPARSITIES = [0.05, 0.15, 0.30, 0.50, 0.80, 0.95]  # fraction of ACTIVE tiles
PATTERNS   = ["clustered", "random"]
BLOCK_SIZE = 8                              # for block_sparse layout
N_REPEAT   = 20                             # timing repeats
SEED       = 42
DTYPE      = np.float32
VEC_DIM    = 1                              # scalar field for sandbox

# ─────────────────────────────────────────────
# Morton encoding (2D)
# ─────────────────────────────────────────────

def _part1by1(n: np.ndarray) -> np.ndarray:
    """Spread bits of n for 2D Morton interleave (works up to 16-bit coords)."""
    n = n.astype(np.uint32)
    n = (n | (n << 8)) & np.uint32(0x00FF00FF)
    n = (n | (n << 4)) & np.uint32(0x0F0F0F0F)
    n = (n | (n << 2)) & np.uint32(0x33333333)
    n = (n | (n << 1)) & np.uint32(0x55555555)
    return n

def morton_encode_2d(row: np.ndarray, col: np.ndarray) -> np.ndarray:
    """Return Morton (Z-order) code for 2D coordinates."""
    return _part1by1(col) | (_part1by1(row) << np.uint32(1))

# ─────────────────────────────────────────────
# Mask generation
# ─────────────────────────────────────────────

def make_mask(side: int, sparsity: float, pattern: str, rng: np.random.Generator) -> np.ndarray:
    """
    Generate 2D boolean mask (True = active).
    sparsity = fraction of active tiles.
    """
    M = side * side
    n_active = max(1, int(round(M * sparsity)))

    if pattern == "random":
        idx = rng.choice(M, size=n_active, replace=False)
        mask = np.zeros(M, dtype=bool)
        mask[idx] = True
        return mask.reshape(side, side)

    elif pattern == "clustered":
        # Place ~sqrt(n_active) gaussian blobs
        mask = np.zeros((side, side), dtype=bool)
        n_blobs = max(1, int(np.sqrt(n_active)))
        tiles_per_blob = max(1, n_active // n_blobs)
        radius = max(1, int(np.sqrt(tiles_per_blob / np.pi)))

        placed = 0
        for _ in range(n_blobs * 3):  # extra attempts
            if placed >= n_active:
                break
            cy, cx = rng.integers(0, side, size=2)
            yy, xx = np.ogrid[-cy:side - cy, -cx:side - cx]
            dist2 = yy * yy + xx * xx
            blob = dist2 <= radius * radius
            new_active = blob & ~mask
            can_add = min(n_active - placed, new_active.sum())
            if can_add > 0:
                coords = np.argwhere(new_active)
                chosen = coords[rng.choice(len(coords), size=can_add, replace=False)]
                mask[chosen[:, 0], chosen[:, 1]] = True
                placed += can_add

        # top-up if blobs didn't reach target
        if placed < n_active:
            inactive = np.argwhere(~mask)
            need = n_active - placed
            chosen = inactive[rng.choice(len(inactive), size=need, replace=False)]
            mask[chosen[:, 0], chosen[:, 1]] = True

        return mask
    else:
        raise ValueError(f"Unknown pattern: {pattern}")


# ─────────────────────────────────────────────
# Layout implementations
# ─────────────────────────────────────────────

@dataclass
class LayoutResult:
    """Holds layout structures + timing of construction."""
    name: str
    active_idx: np.ndarray           # flat indices of active tiles
    reverse_map: Optional[np.ndarray] # flat grid → position in active_idx (-1 if inactive)
    build_time_s: float = 0.0
    extra_bytes: int = 0             # bytes beyond the mask itself

def build_fixed_grid(mask_flat: np.ndarray) -> LayoutResult:
    """Baseline: just the bool mask. No explicit active_idx needed."""
    t0 = time.perf_counter()
    active_idx = np.where(mask_flat)[0].astype(np.int32)
    dt = time.perf_counter() - t0
    return LayoutResult(
        name="fixed_grid",
        active_idx=active_idx,
        reverse_map=None,  # grid uses arithmetic
        build_time_s=dt,
        extra_bytes=mask_flat.nbytes,  # just the mask
    )

def build_compact(mask_flat: np.ndarray) -> LayoutResult:
    """Compact list + reverse index-map."""
    t0 = time.perf_counter()
    active_idx = np.where(mask_flat)[0].astype(np.int32)
    reverse_map = np.full(len(mask_flat), -1, dtype=np.int32)
    reverse_map[active_idx] = np.arange(len(active_idx), dtype=np.int32)
    dt = time.perf_counter() - t0
    return LayoutResult(
        name="compact",
        active_idx=active_idx,
        reverse_map=reverse_map,
        build_time_s=dt,
        extra_bytes=active_idx.nbytes + reverse_map.nbytes,
    )

def build_morton(mask_flat: np.ndarray, side: int) -> LayoutResult:
    """Compact list sorted by Morton code."""
    t0 = time.perf_counter()
    active_idx = np.where(mask_flat)[0].astype(np.int32)
    rows = active_idx // side
    cols = active_idx % side
    codes = morton_encode_2d(rows, cols)
    sort_order = np.argsort(codes)
    active_idx_sorted = active_idx[sort_order]
    reverse_map = np.full(len(mask_flat), -1, dtype=np.int32)
    reverse_map[active_idx_sorted] = np.arange(len(active_idx_sorted), dtype=np.int32)
    dt = time.perf_counter() - t0
    return LayoutResult(
        name="morton",
        active_idx=active_idx_sorted,
        reverse_map=reverse_map,
        build_time_s=dt,
        extra_bytes=active_idx_sorted.nbytes + reverse_map.nbytes + codes.nbytes,
    )

def build_block_sparse(mask_2d: np.ndarray, block_size: int) -> LayoutResult:
    """Block-sparse: BxB blocks, block-level mask + compact block list."""
    t0 = time.perf_counter()
    side = mask_2d.shape[0]
    n_blocks_side = (side + block_size - 1) // block_size
    # Block is active if ANY tile in it is active
    block_mask = np.zeros((n_blocks_side, n_blocks_side), dtype=bool)
    for by in range(n_blocks_side):
        for bx in range(n_blocks_side):
            y0, y1 = by * block_size, min((by + 1) * block_size, side)
            x0, x1 = bx * block_size, min((bx + 1) * block_size, side)
            if mask_2d[y0:y1, x0:x1].any():
                block_mask[by, bx] = True

    block_mask_flat = block_mask.ravel()
    active_blocks = np.where(block_mask_flat)[0].astype(np.int32)

    # Expand to tile-level active_idx (all tiles in active blocks)
    tile_indices = []
    for bidx in active_blocks:
        by = bidx // n_blocks_side
        bx = bidx % n_blocks_side
        y0, y1 = by * block_size, min((by + 1) * block_size, side)
        x0, x1 = bx * block_size, min((bx + 1) * block_size, side)
        for y in range(y0, y1):
            for x in range(x0, x1):
                tile_indices.append(y * side + x)
    active_idx = np.array(tile_indices, dtype=np.int32)
    # Remove duplicates and sort
    active_idx = np.unique(active_idx)

    reverse_map = np.full(side * side, -1, dtype=np.int32)
    reverse_map[active_idx] = np.arange(len(active_idx), dtype=np.int32)

    dt = time.perf_counter() - t0
    return LayoutResult(
        name="block_sparse",
        active_idx=active_idx,
        reverse_map=reverse_map,
        build_time_s=dt,
        extra_bytes=(active_idx.nbytes + reverse_map.nbytes
                     + block_mask_flat.nbytes + active_blocks.nbytes),
    )


# ─────────────────────────────────────────────
# Block 1: A2 Logistics benchmark
# ─────────────────────────────────────────────

def bench_logistics(side: int, sparsity: float, pattern: str, rng: np.random.Generator):
    """Time layout construction N_REPEAT times, return median + stats."""
    mask_2d = make_mask(side, sparsity, pattern, rng)
    mask_flat = mask_2d.ravel()
    n_active = mask_flat.sum()

    results = {}
    builders = {
        "fixed_grid": lambda: build_fixed_grid(mask_flat),
        "compact":    lambda: build_compact(mask_flat),
        "morton":     lambda: build_morton(mask_flat, side),
        "block_sparse": lambda: build_block_sparse(mask_2d, BLOCK_SIZE),
    }

    for name, builder in builders.items():
        times = []
        for _ in range(N_REPEAT):
            lr = builder()
            times.append(lr.build_time_s)
        times = np.array(times)
        results[name] = {
            "median_us": float(np.median(times) * 1e6),
            "p95_us":    float(np.percentile(times, 95) * 1e6),
            "extra_bytes": int(builder().extra_bytes),
            "n_active_tiles": int(n_active),
        }
        # block_sparse expands: log actual tile count
        if name == "block_sparse":
            lr = builder()
            results[name]["n_tiles_in_layout"] = len(lr.active_idx)
            results[name]["expansion_ratio"] = len(lr.active_idx) / max(1, n_active)

    return {
        "side": side,
        "M": side * side,
        "sparsity": sparsity,
        "pattern": pattern,
        "n_active": int(n_active),
        "layouts": results,
    }


# ─────────────────────────────────────────────
# Block 2: Memory model
# ─────────────────────────────────────────────

def memory_model(side: int, sparsity: float):
    """
    Analytical memory estimate (bytes) for each layout.
    No randomness — pure formula.
    """
    M = side * side
    k = max(1, int(round(M * sparsity)))
    n_blocks_side = (side + BLOCK_SIZE - 1) // BLOCK_SIZE
    n_blocks = n_blocks_side * n_blocks_side
    # Worst case: each active tile in its own block → k blocks
    # Best case (clustered): k / BLOCK_SIZE² blocks
    k_blocks_best = max(1, k // (BLOCK_SIZE * BLOCK_SIZE))
    k_blocks_worst = min(n_blocks, k)
    tiles_in_blocks_best = k_blocks_best * BLOCK_SIZE * BLOCK_SIZE
    tiles_in_blocks_worst = k_blocks_worst * BLOCK_SIZE * BLOCK_SIZE

    return {
        "M": M, "k": k,
        "fixed_grid": {
            "mask": M * 1,             # bool mask
            "total": M * 1,
        },
        "compact": {
            "active_idx": k * 4,       # int32
            "reverse_map": M * 4,      # int32
            "total": k * 4 + M * 4,
        },
        "morton": {
            "active_idx": k * 4,
            "reverse_map": M * 4,
            "morton_codes": k * 4,     # uint32
            "total": k * 4 + M * 4 + k * 4,
        },
        "block_sparse_best": {
            "block_mask": n_blocks * 1,
            "active_blocks": k_blocks_best * 4,
            "active_idx": tiles_in_blocks_best * 4,
            "reverse_map": M * 4,
            "total": n_blocks + k_blocks_best * 4 + tiles_in_blocks_best * 4 + M * 4,
            "n_blocks_active": k_blocks_best,
            "n_tiles_in_layout": tiles_in_blocks_best,
        },
        "block_sparse_worst": {
            "block_mask": n_blocks * 1,
            "active_blocks": k_blocks_worst * 4,
            "active_idx": tiles_in_blocks_worst * 4,
            "reverse_map": M * 4,
            "total": n_blocks + k_blocks_worst * 4 + tiles_in_blocks_worst * 4 + M * 4,
            "n_blocks_active": k_blocks_worst,
            "n_tiles_in_layout": tiles_in_blocks_worst,
        },
    }


# ─────────────────────────────────────────────
# Block 3: Determinism check
# ─────────────────────────────────────────────

def check_determinism(side: int = 128, sparsity: float = 0.3, n_trials: int = 5):
    """
    Additive refinement in different traversal orders.
    Checks that final result is bitwise identical (or documents FP diff).

    Simulates: output[i] = coarse[i] + sum_of_deltas[i]
    where deltas are applied in different orders.
    """
    rng = np.random.default_rng(SEED + 7)
    M = side * side
    mask_2d = make_mask(side, sparsity, "clustered", rng)
    mask_flat = mask_2d.ravel()
    active_idx = np.where(mask_flat)[0]
    k = len(active_idx)

    # Simulate coarse + N deltas
    coarse = rng.standard_normal(M).astype(DTYPE)
    n_deltas = 5
    # Each delta is sparse: only touches active tiles
    deltas = [rng.standard_normal(k).astype(DTYPE) * 0.1 for _ in range(n_deltas)]

    results = {}

    # --- Order 1: natural (active_idx as-is) ---
    out_natural = coarse.copy()
    for d in deltas:
        out_natural[active_idx] += d
    results["natural"] = out_natural.copy()

    # --- Order 2: Morton-sorted active_idx ---
    rows = active_idx // side
    cols = active_idx % side
    codes = morton_encode_2d(rows, cols)
    morton_order = np.argsort(codes)
    active_morton = active_idx[morton_order]
    out_morton = coarse.copy()
    for d in deltas:
        d_reordered = d[morton_order]
        out_morton[active_morton] += d_reordered
    results["morton"] = out_morton.copy()

    # --- Order 3: random permutation ---
    for trial in range(n_trials):
        perm = rng.permutation(k)
        active_perm = active_idx[perm]
        out_perm = coarse.copy()
        for d in deltas:
            d_reordered = d[perm]
            out_perm[active_perm] += d_reordered
        results[f"random_{trial}"] = out_perm.copy()

    # Compare all against natural
    report = {"side": side, "k": k, "n_deltas": n_deltas, "comparisons": []}
    ref = results["natural"]
    for name, out in results.items():
        if name == "natural":
            continue
        bitwise_equal = np.array_equal(ref, out)
        max_abs_diff = float(np.max(np.abs(ref - out)))
        n_diff_bits = int(np.sum(ref != out))
        report["comparisons"].append({
            "name": name,
            "bitwise_equal": bitwise_equal,
            "max_abs_diff": max_abs_diff,
            "n_different_elements": n_diff_bits,
        })

    return report


# ─────────────────────────────────────────────
# Block 4: A1 "index imitation" — gather/compute/scatter
# ─────────────────────────────────────────────

def bench_gather_compute_scatter(side: int, sparsity: float, pattern: str,
                                  rng: np.random.Generator):
    """
    Simulates one refinement step on CPU:
      1. gather: read values at active tiles
      2. compute: apply a mock "refinement" (e.g. local Laplacian → delta)
      3. scatter: write deltas back

    For fixed_grid: compute over full grid, masked.
    For compact/morton/block_sparse: gather → compute on dense subarray → scatter.
    """
    mask_2d = make_mask(side, sparsity, pattern, rng)
    mask_flat = mask_2d.ravel()
    M = side * side

    # Build field
    field = rng.standard_normal(M).astype(DTYPE)
    field_2d = field.reshape(side, side)

    # Build layouts
    layouts = {
        "fixed_grid":   build_fixed_grid(mask_flat),
        "compact":      build_compact(mask_flat),
        "morton":       build_morton(mask_flat, side),
        "block_sparse": build_block_sparse(mask_2d, BLOCK_SIZE),
    }

    def _mock_compute_dense(values_1d: np.ndarray) -> np.ndarray:
        """Mock refinement: just a scaled tanh (nonlinear, non-trivial)."""
        return np.tanh(values_1d) * 0.1

    results = {}

    # --- fixed_grid: compute on full grid, mask out ---
    def run_fixed_grid():
        vals = field.copy()
        delta = _mock_compute_dense(vals)
        delta[~mask_flat] = 0.0
        out = vals + delta
        return out

    # --- compact/morton/block_sparse: gather → compute → scatter ---
    def run_indexed(layout: LayoutResult):
        out = field.copy()
        gathered = field[layout.active_idx]          # gather
        delta = _mock_compute_dense(gathered)         # compute (dense on active)
        out[layout.active_idx] += delta               # scatter
        return out

    # Time each
    for name, layout in layouts.items():
        runner = run_fixed_grid if name == "fixed_grid" else (lambda l=layout: run_indexed(l))
        times = []
        for _ in range(N_REPEAT):
            t0 = time.perf_counter()
            _ = runner()
            dt = time.perf_counter() - t0
            times.append(dt)
        times_arr = np.array(times)
        results[name] = {
            "median_us": float(np.median(times_arr) * 1e6),
            "p95_us":    float(np.percentile(times_arr, 95) * 1e6),
            "n_elements_touched": int(len(layout.active_idx)),
        }

    return {
        "side": side,
        "M": M,
        "sparsity": sparsity,
        "pattern": pattern,
        "n_active": int(mask_flat.sum()),
        "layouts": results,
    }


# ─────────────────────────────────────────────
# Block 5: Neighbor access cost (halo stencil)
# ─────────────────────────────────────────────

def bench_neighbor_access(side: int, sparsity: float, pattern: str,
                           rng: np.random.Generator, halo_w: int = 3):
    """
    Measure cost of fetching halo neighbors for each active tile.

    For fixed_grid: arithmetic indexing (row ± w, col ± w), clamped.
    For compact/morton: index-map lookup.
    For block_sparse: block-level + intra-block arithmetic.
    """
    mask_2d = make_mask(side, sparsity, pattern, rng)
    mask_flat = mask_2d.ravel()
    M = side * side
    field = rng.standard_normal(M).astype(DTYPE)

    layouts = {
        "fixed_grid":   build_fixed_grid(mask_flat),
        "compact":      build_compact(mask_flat),
        "morton":       build_morton(mask_flat, side),
        "block_sparse": build_block_sparse(mask_2d, BLOCK_SIZE),
    }

    def _get_halo_grid(active_idx, w):
        """Grid-arithmetic neighbor fetch: for each active tile, read w-ring."""
        total = 0.0
        rows = active_idx // side
        cols = active_idx % side
        for dy in range(-w, w + 1):
            for dx in range(-w, w + 1):
                nr = np.clip(rows + dy, 0, side - 1)
                nc = np.clip(cols + dx, 0, side - 1)
                nidx = nr * side + nc
                total += field[nidx].sum()
        return total

    def _get_halo_indexmap(active_idx, reverse_map, w):
        """Index-map neighbor fetch: same stencil but through reverse_map."""
        total = 0.0
        rows = active_idx // side
        cols = active_idx % side
        for dy in range(-w, w + 1):
            for dx in range(-w, w + 1):
                nr = np.clip(rows + dy, 0, side - 1)
                nc = np.clip(cols + dx, 0, side - 1)
                nidx = nr * side + nc
                # Check if neighbor is active (extra lookup)
                pos = reverse_map[nidx]
                # Read from field regardless (halo reads frozen values too)
                total += field[nidx].sum()
        return total

    results = {}
    for name, layout in layouts.items():
        if name == "fixed_grid":
            func = lambda: _get_halo_grid(layout.active_idx, halo_w)
        else:
            func = lambda l=layout: _get_halo_indexmap(l.active_idx, l.reverse_map, halo_w)

        times = []
        for _ in range(N_REPEAT):
            t0 = time.perf_counter()
            _ = func()
            dt = time.perf_counter() - t0
            times.append(dt)
        times_arr = np.array(times)
        results[name] = {
            "median_us": float(np.median(times_arr) * 1e6),
            "p95_us": float(np.percentile(times_arr, 95) * 1e6),
        }

    return {
        "side": side, "sparsity": sparsity, "pattern": pattern,
        "halo_w": halo_w, "layouts": results,
    }


# ─────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────

def run_all():
    rng_master = np.random.default_rng(SEED)
    output_dir = Path("/home/claude/exp09a_results")
    output_dir.mkdir(exist_ok=True)

    all_results = {
        "config": {
            "grid_sides": GRID_SIDES,
            "sparsities": SPARSITIES,
            "patterns": PATTERNS,
            "block_size": BLOCK_SIZE,
            "n_repeat": N_REPEAT,
            "seed": SEED,
            "dtype": str(DTYPE),
        },
        "logistics": [],
        "memory_model": [],
        "determinism": None,
        "gather_compute_scatter": [],
        "neighbor_access": [],
    }

    # ── Block 1: Logistics ──
    print("=" * 60)
    print("BLOCK 1: Logistics (A2)")
    print("=" * 60)
    for side in GRID_SIDES:
        for sp in SPARSITIES:
            for pat in PATTERNS:
                rng = np.random.default_rng(rng_master.integers(0, 2**31))
                res = bench_logistics(side, sp, pat, rng)
                all_results["logistics"].append(res)
                n = res["n_active"]
                grid_t = res["layouts"]["fixed_grid"]["median_us"]
                comp_t = res["layouts"]["compact"]["median_us"]
                mort_t = res["layouts"]["morton"]["median_us"]
                blk_t  = res["layouts"]["block_sparse"]["median_us"]
                print(f"  {side:4d}² sp={sp:.2f} {pat:9s} | "
                      f"k={n:6d} | grid={grid_t:8.1f} comp={comp_t:8.1f} "
                      f"mort={mort_t:8.1f} blk={blk_t:8.1f} µs")

    # ── Block 2: Memory model ──
    print("\n" + "=" * 60)
    print("BLOCK 2: Memory model (analytical)")
    print("=" * 60)
    for side in GRID_SIDES:
        for sp in SPARSITIES:
            mm = memory_model(side, sp)
            all_results["memory_model"].append(mm)
            M = mm["M"]
            print(f"  {side:4d}² sp={sp:.2f} | "
                  f"grid={mm['fixed_grid']['total']/1024:7.1f}KB "
                  f"comp={mm['compact']['total']/1024:7.1f}KB "
                  f"mort={mm['morton']['total']/1024:7.1f}KB "
                  f"blk_best={mm['block_sparse_best']['total']/1024:7.1f}KB "
                  f"blk_worst={mm['block_sparse_worst']['total']/1024:7.1f}KB")

    # ── Block 3: Determinism ──
    print("\n" + "=" * 60)
    print("BLOCK 3: Determinism check")
    print("=" * 60)
    det = check_determinism(side=128, sparsity=0.3)
    all_results["determinism"] = det
    for c in det["comparisons"]:
        status = "BITWISE OK" if c["bitwise_equal"] else f"DIFF: max={c['max_abs_diff']:.2e} n={c['n_different_elements']}"
        print(f"  {c['name']:12s}: {status}")

    # ── Block 4: Gather/compute/scatter (A1 imitation) ──
    print("\n" + "=" * 60)
    print("BLOCK 4: Gather/compute/scatter (A1 index imitation)")
    print("=" * 60)
    # Run on subset to keep sandbox fast
    a1_sides = [64, 128, 256]
    a1_sparsities = [0.05, 0.15, 0.30, 0.50]
    for side in a1_sides:
        for sp in a1_sparsities:
            for pat in PATTERNS:
                rng = np.random.default_rng(rng_master.integers(0, 2**31))
                res = bench_gather_compute_scatter(side, sp, pat, rng)
                all_results["gather_compute_scatter"].append(res)
                grid_t = res["layouts"]["fixed_grid"]["median_us"]
                comp_t = res["layouts"]["compact"]["median_us"]
                mort_t = res["layouts"]["morton"]["median_us"]
                blk_t  = res["layouts"]["block_sparse"]["median_us"]
                print(f"  {side:4d}² sp={sp:.2f} {pat:9s} | "
                      f"grid={grid_t:8.1f} comp={comp_t:8.1f} "
                      f"mort={mort_t:8.1f} blk={blk_t:8.1f} µs")

    # ── Block 5: Neighbor access ──
    print("\n" + "=" * 60)
    print("BLOCK 5: Neighbor access (halo stencil)")
    print("=" * 60)
    na_sides = [64, 128]
    na_sparsities = [0.05, 0.15, 0.30]
    for side in na_sides:
        for sp in na_sparsities:
            for pat in PATTERNS:
                rng = np.random.default_rng(rng_master.integers(0, 2**31))
                res = bench_neighbor_access(side, sp, pat, rng, halo_w=3)
                all_results["neighbor_access"].append(res)
                grid_t = res["layouts"]["fixed_grid"]["median_us"]
                comp_t = res["layouts"]["compact"]["median_us"]
                mort_t = res["layouts"]["morton"]["median_us"]
                blk_t  = res["layouts"]["block_sparse"]["median_us"]
                print(f"  {side:4d}² sp={sp:.2f} {pat:9s} w=3 | "
                      f"grid={grid_t:8.1f} comp={comp_t:8.1f} "
                      f"mort={mort_t:8.1f} blk={blk_t:8.1f} µs")

    # ── Save ──
    json_path = output_dir / "exp09a_sandbox.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")

    return all_results


# ─────────────────────────────────────────────
# Analysis & summary
# ─────────────────────────────────────────────

def print_summary(results: dict):
    """Print condensed summary with kill-criterion evaluation."""
    print("\n" + "=" * 60)
    print("SUMMARY & KILL CRITERIA EVALUATION")
    print("=" * 60)

    # --- Logistics: is compaction cost acceptable? ---
    print("\n── Logistics (A2): compaction overhead ──")
    logistics = results["logistics"]
    # Group by side, show worst-case ratio (morton / grid)
    for side in GRID_SIDES:
        entries = [e for e in logistics if e["side"] == side]
        ratios_compact = []
        ratios_morton = []
        for e in entries:
            g = e["layouts"]["fixed_grid"]["median_us"]
            c = e["layouts"]["compact"]["median_us"]
            m = e["layouts"]["morton"]["median_us"]
            if g > 0:
                ratios_compact.append(c / g)
                ratios_morton.append(m / g)
        if ratios_compact:
            print(f"  {side:4d}²: compact/grid = {np.median(ratios_compact):.2f}× "
                  f"(p95={np.percentile(ratios_compact, 95):.2f}×)  "
                  f"morton/grid = {np.median(ratios_morton):.2f}× "
                  f"(p95={np.percentile(ratios_morton, 95):.2f}×)")

    # --- Memory: where does compact win? ---
    print("\n── Memory model: compact vs grid savings ──")
    for side in GRID_SIDES:
        entries = [e for e in results["memory_model"] if e["M"] == side * side]
        for e in entries:
            sp = e["k"] / e["M"]
            grid_b = e["fixed_grid"]["total"]
            comp_b = e["compact"]["total"]
            ratio = comp_b / grid_b if grid_b > 0 else float("inf")
            if sp <= 0.30:  # show only sparse regime
                print(f"  {side:4d}² sp={sp:.2f}: compact/grid memory = {ratio:.2f}×")

    # --- Gather/compute/scatter: does indexed win at low sparsity? ---
    print("\n── A1 imitation: gather/compute/scatter ──")
    gcs = results["gather_compute_scatter"]
    for side in [64, 128, 256]:
        entries = [e for e in gcs if e["side"] == side]
        for e in entries:
            sp = e["sparsity"]
            g = e["layouts"]["fixed_grid"]["median_us"]
            c = e["layouts"]["compact"]["median_us"]
            if g > 0:
                ratio = c / g
                marker = "✓" if ratio < 0.67 else ("~" if ratio < 1.0 else "✗")
                if sp <= 0.30:
                    print(f"  {side:4d}² sp={sp:.2f} {e['pattern']:9s}: "
                          f"compact/grid = {ratio:.2f}× {marker}")

    # --- Determinism ---
    print("\n── Determinism ──")
    det = results["determinism"]
    all_ok = all(c["bitwise_equal"] for c in det["comparisons"])
    if all_ok:
        print("  All traversal orders produce BITWISE IDENTICAL results.")
    else:
        max_diff = max(c["max_abs_diff"] for c in det["comparisons"])
        print(f"  FP summation order matters. Max diff: {max_diff:.2e}")
        print("  → Must fix accumulation order or accept near-bitwise + document threshold.")

    # --- Kill criterion ---
    print("\n── Kill criteria ──")
    # Check: at sparsity <= 0.30, does compact give >= 1.5× speedup in A1?
    speed_alive = False
    for e in gcs:
        if e["sparsity"] <= 0.30:
            g = e["layouts"]["fixed_grid"]["median_us"]
            c = e["layouts"]["compact"]["median_us"]
            if g > 0 and (c / g) <= (1.0 / 1.5):
                speed_alive = True
                break
    # Check: at sparsity >= 0.70 (i.e. coverage <= 30%), does compact save >= 40% memory?
    mem_alive = False
    for e in results["memory_model"]:
        sp = e["k"] / e["M"]
        if sp <= 0.30:
            grid_b = e["fixed_grid"]["total"]
            comp_b = e["compact"]["total"]
            if grid_b > 0 and (comp_b / grid_b) <= 0.60:
                mem_alive = True
                break

    print(f"  Speed criterion (compact ≥1.5× at sparsity≤30%): "
          f"{'ALIVE' if speed_alive else 'DEAD'}")
    print(f"  Memory criterion (compact ≤0.6× at sparsity≤30%): "
          f"{'ALIVE' if mem_alive else 'DEAD'}")

    if speed_alive or mem_alive:
        print("  → Compact layout ALIVE. Proceed to 0.9b on GPU.")
    else:
        print("  → Compact layout DEAD on sandbox. GPU may resurrect (memory locality).")
        print("  → Still run 0.9a-A1 on GPU before final verdict.")


if __name__ == "__main__":
    results = run_all()
    print_summary(results)
