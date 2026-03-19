#!/usr/bin/env python3
"""
Curiosity -- Exp10h: Cross-Space Validation of D_direct

Motivation:
  Exp10g proved D_direct (packed tiles + tile_map) works on scalar 2D grids
  with -54% to -82% time savings and -36% to -86% peak VRAM savings.
  This experiment validates D_direct on two additional space types:

  Section 1 -- Vector Grid:
    Grid where each cell holds a C-dimensional vector (C>1).
    D_direct tiles[k, Ht, Wt, C] with C>1.
    Compute kernel: 3x3 depthwise conv (F.conv2d with groups=C).

  Section 2 -- Tree Hierarchy:
    Multi-level tree with per-level tile_map.
    Each level: tile_map[L][node_id] -> slot, packed_data[L][slot, feat_dim].
    Compute kernel: parent-child aggregation across levels.

Candidates:
  Vector Grid:
    grid_baseline: full tensor [H, W, C] + bool mask
    D_direct:      packed tiles[k, Ht, Wt, C] + tile_map[n_tiles] int32

  Tree Hierarchy:
    tree_baseline: full arrays per level + bool mask per level
    D_tree:        per-level packed_data + per-level tile_map

Dual kill criteria:
  Contour A (architectural): D resident < grid resident AND time overhead < 20%
  Contour B (operational):   D peak_vram < grid peak_vram AND time overhead < 20%

Outputs (in results/ subdirectory):
  exp10h_summary.json        -- all data + verdicts
  exp10h_vector_grid.png     -- time + VRAM comparison for vector grid
  exp10h_tree.png            -- time + VRAM comparison by depth for tree
  exp10h_report.md           -- human-readable report
"""

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =====================================================================
# Configuration
# =====================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
SEED_BASE = 42
N_SEEDS = 10
N_WARMUP = 5
N_REPEAT = 20
KILL_THRESH = 0.20  # 20% overhead tolerance

RESULTS_DIR = Path(__file__).parent / "results"

# -- Vector Grid sweep --
VG_SIDES = [64, 128, 256]
VG_CHANNELS = [3, 8, 16]
VG_SPARSITIES = [0.05, 0.1, 0.3, 0.5]
VG_PATTERNS = ["random", "clustered"]
VG_TILE_SIZE = 8

# -- Tree Hierarchy sweep --
TREE_MAX_DEPTHS = [3, 4, 5]
TREE_BRANCHING = [2, 4, 8]
TREE_FEAT_DIMS = [4, 8, 16]
TREE_SPARSITIES = [0.1, 0.3, 0.5, 0.7]

# =====================================================================
# Helpers
# =====================================================================

def _sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)


def _timed_runs(func, n_warmup: int = N_WARMUP,
                n_repeat: int = N_REPEAT) -> np.ndarray:
    """Return array of wall-clock seconds for n_repeat runs after warmup."""
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


def _measure_memory(build_fn, compute_fn) -> Dict[str, int]:
    """Build layout, measure resident; run compute, measure peak.

    Returns dict with resident_bytes, peak_vram_bytes.
    """
    # Clean slate
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(DEVICE)

    build_fn()
    _sync()

    if DEVICE.type == "cuda":
        resident = torch.cuda.memory_allocated(DEVICE)
        torch.cuda.reset_peak_memory_stats(DEVICE)
        compute_fn()
        _sync()
        peak = torch.cuda.max_memory_allocated(DEVICE)
    else:
        resident = 0
        compute_fn()
        peak = 0

    return {"resident_bytes": int(resident), "peak_vram_bytes": int(peak)}


def _build_time(build_fn) -> float:
    """Measure build cost in microseconds (median of 5 runs)."""
    times = []
    for _ in range(5):
        torch.cuda.empty_cache()
        _sync()
        t0 = time.perf_counter()
        build_fn()
        _sync()
        dt = time.perf_counter() - t0
        times.append(dt)
    return float(np.median(times)) * 1e6


# =====================================================================
# Mask generation (from exp10g, reused)
# =====================================================================

def make_mask(side: int, sparsity: float, pattern: str,
              rng: np.random.Generator) -> np.ndarray:
    """Generate 2D boolean mask (True = active).  sparsity = fraction active."""
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

    else:
        raise ValueError(f"Unknown pattern: {pattern}")


# #####################################################################
# SECTION 1: VECTOR GRID
# #####################################################################

print("=" * 72)
print("SECTION 1: Vector Grid (tiles[k, Ht, Wt, C] with C > 1)")
print("=" * 72)

# =====================================================================
# Vector Grid: Layout builders
# =====================================================================

def build_vector_grid_baseline(side: int, C: int, mask_2d: np.ndarray,
                               rng: np.random.Generator):
    """Full tensor [H, W, C] + bool mask. Returns (data, mask_t)."""
    data = torch.randn(side, side, C, dtype=DTYPE, device=DEVICE)
    mask_t = torch.from_numpy(mask_2d).to(DEVICE)
    return data, mask_t


def build_vector_d_direct(side: int, C: int, tile_size: int,
                          mask_2d: np.ndarray, rng: np.random.Generator):
    """Packed tiles[k, Ht, Wt, C] + tile_map[n_tiles] int32.

    Returns (tiles, tile_map, n_tiles_grid, tile_size, active_tile_ids).
    """
    Ht = tile_size
    Wt = tile_size
    n_ty = side // Ht
    n_tx = side // Wt
    n_tiles_grid = n_ty * n_tx

    # Determine which tiles are active (any True cell in tile footprint)
    tile_active = np.zeros(n_tiles_grid, dtype=bool)
    for ty in range(n_ty):
        for tx in range(n_tx):
            patch = mask_2d[ty * Ht:(ty + 1) * Ht, tx * Wt:(tx + 1) * Wt]
            if patch.any():
                tile_active[ty * n_tx + tx] = True

    active_ids = np.where(tile_active)[0].astype(np.int32)
    k = len(active_ids)

    # Build tile_map
    tile_map = torch.full((n_tiles_grid,), -1, dtype=torch.int32, device=DEVICE)
    if k > 0:
        tile_map[torch.from_numpy(active_ids).to(DEVICE)] = torch.arange(
            k, dtype=torch.int32, device=DEVICE
        )

    # Pack tiles
    tiles = torch.zeros(max(k, 1), Ht, Wt, C, dtype=DTYPE, device=DEVICE)
    full_data = torch.randn(side, side, C, dtype=DTYPE, device=DEVICE)
    for slot, tid in enumerate(active_ids):
        ty = tid // n_tx
        tx = tid % n_tx
        tiles[slot] = full_data[ty * Ht:(ty + 1) * Ht, tx * Wt:(tx + 1) * Wt]

    return tiles, tile_map, n_tiles_grid, active_ids, full_data


# =====================================================================
# Vector Grid: Compute kernels (3x3 depthwise conv)
# =====================================================================

def compute_grid_depthwise_conv(data: torch.Tensor, mask_t: torch.Tensor,
                                C: int):
    """3x3 depthwise conv on full grid [H, W, C].

    Uses F.conv2d with groups=C.
    """
    H, W, _ = data.shape
    # F.conv2d expects [N, C, H, W]
    x = data.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
    weight = torch.ones(C, 1, 3, 3, dtype=DTYPE, device=DEVICE) / 9.0
    out = F.conv2d(x, weight, padding=1, groups=C)  # [1, C, H, W]
    out = out.squeeze(0).permute(1, 2, 0)  # [H, W, C]
    # Apply mask
    result = data.clone()
    result[mask_t] = out[mask_t]
    return result


def compute_d_depthwise_conv(tiles: torch.Tensor, C: int):
    """3x3 depthwise conv on packed tiles[k, Ht, Wt, C].

    Each tile is convolved independently (boundary = zero pad).
    """
    k, Ht, Wt, _ = tiles.shape
    # Reshape to [k, C, Ht, Wt] for F.conv2d
    x = tiles.permute(0, 3, 1, 2)  # [k, C, Ht, Wt]
    weight = torch.ones(C, 1, 3, 3, dtype=DTYPE, device=DEVICE) / 9.0
    out = F.conv2d(x, weight, padding=1, groups=C)  # [k, C, Ht, Wt]
    return out.permute(0, 2, 3, 1)  # [k, Ht, Wt, C]


# =====================================================================
# Vector Grid: Benchmark loop
# =====================================================================

vg_results: List[Dict[str, Any]] = []

total_vg = (len(VG_SIDES) * len(VG_CHANNELS) * len(VG_SPARSITIES)
            * len(VG_PATTERNS) * N_SEEDS)
vg_count = 0

for side in VG_SIDES:
    for C in VG_CHANNELS:
        for sparsity in VG_SPARSITIES:
            for pattern in VG_PATTERNS:
                seed_records_grid = []
                seed_records_d = []

                for seed_idx in range(N_SEEDS):
                    seed = SEED_BASE + seed_idx
                    rng = np.random.default_rng(seed)
                    torch.manual_seed(seed)

                    mask_2d = make_mask(side, sparsity, pattern, rng)

                    # ---- Grid Baseline ----
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(DEVICE)

                    data_g, mask_t = build_vector_grid_baseline(
                        side, C, mask_2d, rng)
                    _sync()

                    resident_grid = (torch.cuda.memory_allocated(DEVICE)
                                     if DEVICE.type == "cuda" else 0)

                    torch.cuda.reset_peak_memory_stats(DEVICE)
                    times_grid = _timed_runs(
                        lambda: compute_grid_depthwise_conv(data_g, mask_t, C))
                    peak_grid = (torch.cuda.max_memory_allocated(DEVICE)
                                 if DEVICE.type == "cuda" else 0)

                    wall_grid_us = float(np.median(times_grid)) * 1e6

                    # Build cost for grid (trivial)
                    build_grid_us = _build_time(
                        lambda: build_vector_grid_baseline(
                            side, C, mask_2d, rng))

                    seed_records_grid.append({
                        "resident_bytes": int(resident_grid),
                        "peak_vram_bytes": int(peak_grid),
                        "wall_clock_us": wall_grid_us,
                        "build_cost_us": build_grid_us,
                    })

                    del data_g, mask_t
                    torch.cuda.empty_cache()

                    # ---- D_direct ----
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(DEVICE)

                    tiles, tile_map, n_tiles_grid, active_ids, _ = \
                        build_vector_d_direct(side, C, VG_TILE_SIZE, mask_2d, rng)
                    _sync()

                    resident_d = (torch.cuda.memory_allocated(DEVICE)
                                  if DEVICE.type == "cuda" else 0)

                    torch.cuda.reset_peak_memory_stats(DEVICE)
                    times_d = _timed_runs(
                        lambda: compute_d_depthwise_conv(tiles, C))
                    peak_d = (torch.cuda.max_memory_allocated(DEVICE)
                              if DEVICE.type == "cuda" else 0)

                    wall_d_us = float(np.median(times_d)) * 1e6

                    build_d_us = _build_time(
                        lambda: build_vector_d_direct(
                            side, C, VG_TILE_SIZE, mask_2d, rng))

                    seed_records_d.append({
                        "resident_bytes": int(resident_d),
                        "peak_vram_bytes": int(peak_d),
                        "wall_clock_us": wall_d_us,
                        "build_cost_us": build_d_us,
                    })

                    del tiles, tile_map
                    torch.cuda.empty_cache()

                    vg_count += 1

                # Aggregate across seeds (median)
                def _median_dict(records):
                    out = {}
                    for key in records[0]:
                        vals = [r[key] for r in records]
                        out[key] = float(np.median(vals))
                    return out

                grid_agg = _median_dict(seed_records_grid)
                d_agg = _median_dict(seed_records_d)

                # Kill criteria
                time_overhead = ((d_agg["wall_clock_us"] - grid_agg["wall_clock_us"])
                                 / max(grid_agg["wall_clock_us"], 1e-9))

                contour_a = (d_agg["resident_bytes"] < grid_agg["resident_bytes"]
                             and time_overhead < KILL_THRESH)
                contour_b = (d_agg["peak_vram_bytes"] < grid_agg["peak_vram_bytes"]
                             and time_overhead < KILL_THRESH)

                rec = {
                    "space": "vector_grid",
                    "side": side,
                    "channels": C,
                    "sparsity": sparsity,
                    "pattern": pattern,
                    "grid_baseline": grid_agg,
                    "D_direct": d_agg,
                    "time_overhead_frac": round(time_overhead, 4),
                    "contour_A": "PASS" if contour_a else "FAIL",
                    "contour_B": "PASS" if contour_b else "FAIL",
                }
                vg_results.append(rec)

                ca_tag = "PASS" if contour_a else "FAIL"
                cb_tag = "PASS" if contour_b else "FAIL"
                print(f"  [{vg_count}/{total_vg}] side={side} C={C} "
                      f"sp={sparsity} pat={pattern} | "
                      f"A={ca_tag} B={cb_tag} "
                      f"time_oh={time_overhead:+.1%} "
                      f"res={d_agg['resident_bytes']}/{grid_agg['resident_bytes']} "
                      f"peak={d_agg['peak_vram_bytes']}/{grid_agg['peak_vram_bytes']}")

print(f"\nVector Grid: {len(vg_results)} configs done.")
vg_pass_a = sum(1 for r in vg_results if r["contour_A"] == "PASS")
vg_pass_b = sum(1 for r in vg_results if r["contour_B"] == "PASS")
print(f"  Contour A: {vg_pass_a}/{len(vg_results)} PASS")
print(f"  Contour B: {vg_pass_b}/{len(vg_results)} PASS")


# #####################################################################
# SECTION 2: TREE HIERARCHY
# #####################################################################

print("\n" + "=" * 72)
print("SECTION 2: Tree Hierarchy (per-level packed tiles + tile_map)")
print("=" * 72)

# =====================================================================
# Tree: Synthetic tree generation
# =====================================================================

def build_tree_levels(max_depth: int, branching: int, feat_dim: int,
                      sparsity: float, rng: np.random.Generator,
                      seed: int):
    """Build synthetic tree structure.

    Returns list of dicts per level:
      [{
        "n_nodes": int,
        "active_mask": np.ndarray bool [n_nodes],
        "features": torch.Tensor [n_nodes, feat_dim]  (full, for baseline),
        "parent_ids": np.ndarray int32 [n_nodes] (-1 for root level),
      }, ...]
    """
    torch.manual_seed(seed)
    levels = []

    for L in range(max_depth):
        if L == 0:
            n_nodes = 1  # root
            parent_ids = np.array([-1], dtype=np.int32)
        else:
            n_parent = levels[L - 1]["n_nodes"]
            n_nodes = n_parent * branching
            # parent_ids[child_i] = child_i // branching
            parent_ids = np.arange(n_nodes, dtype=np.int32) // branching

        # Activate a fraction of nodes (root always active)
        if L == 0:
            active_mask = np.ones(n_nodes, dtype=bool)
        else:
            n_active = max(1, int(round(n_nodes * sparsity)))
            idx = rng.choice(n_nodes, size=n_active, replace=False)
            active_mask = np.zeros(n_nodes, dtype=bool)
            active_mask[idx] = True

        features = torch.randn(n_nodes, feat_dim, dtype=DTYPE, device=DEVICE)

        levels.append({
            "n_nodes": n_nodes,
            "active_mask": active_mask,
            "features": features,
            "parent_ids": parent_ids,
        })

    return levels


# =====================================================================
# Tree: Layout builders
# =====================================================================

def build_tree_baseline(levels: List[Dict]):
    """Baseline: keep full arrays per level + bool mask tensors.

    Returns list of (features_t, mask_t, parent_ids_t) per level.
    """
    result = []
    for lev in levels:
        features_t = lev["features"]  # already on device
        mask_t = torch.from_numpy(lev["active_mask"]).to(DEVICE)
        parent_ids_t = torch.from_numpy(lev["parent_ids"]).to(DEVICE)
        result.append((features_t, mask_t, parent_ids_t))
    return result


def build_tree_d(levels: List[Dict]):
    """D_tree: per-level packed_data + per-level tile_map.

    For each level L:
      tile_map[L] = int32 tensor [n_nodes], -1 for inactive
      packed_data[L] = float tensor [k_L, feat_dim]
      parent_ids_packed[L] = int32 tensor [k_L] (parent node IDs)

    Returns list of (packed_data, tile_map, parent_ids_packed, n_nodes) per level.
    """
    result = []
    for lev in levels:
        n_nodes = lev["n_nodes"]
        active = lev["active_mask"]
        active_ids = np.where(active)[0].astype(np.int32)
        k = len(active_ids)

        tile_map = torch.full((n_nodes,), -1, dtype=torch.int32, device=DEVICE)
        if k > 0:
            tile_map[torch.from_numpy(active_ids).to(DEVICE)] = torch.arange(
                k, dtype=torch.int32, device=DEVICE
            )

        packed_data = lev["features"][active_ids] if k > 0 else \
            torch.zeros(1, lev["features"].shape[1], dtype=DTYPE, device=DEVICE)

        parent_ids_packed = torch.from_numpy(
            lev["parent_ids"][active_ids]).to(DEVICE) if k > 0 else \
            torch.zeros(1, dtype=torch.int32, device=DEVICE)

        result.append((packed_data, tile_map, parent_ids_packed, n_nodes))
    return result


# =====================================================================
# Tree: Compute kernels (parent-child aggregation)
# =====================================================================

def compute_tree_baseline(baseline_levels: List[Tuple]):
    """For each active node at level L>0, read parent feature from L-1,
    add local feature, write back.

    baseline_levels: list of (features_t, mask_t, parent_ids_t)
    """
    for L in range(1, len(baseline_levels)):
        feat_L, mask_L, parent_ids_L = baseline_levels[L]
        feat_parent, _, _ = baseline_levels[L - 1]

        # Indices of active nodes at level L
        active_idx = torch.where(mask_L)[0]
        if len(active_idx) == 0:
            continue

        # Parent lookup: read parent features
        parent_node_ids = parent_ids_L[active_idx]  # [n_active]
        parent_feats = feat_parent[parent_node_ids]  # [n_active, feat_dim]

        # Aggregate: active node feature += parent feature
        feat_L[active_idx] = feat_L[active_idx] + parent_feats


def compute_tree_d(d_levels: List[Tuple]):
    """For each active node at level L>0, resolve parent slot via tile_map[L-1],
    read parent packed feature, add local packed feature.

    d_levels: list of (packed_data, tile_map, parent_ids_packed, n_nodes)
    """
    for L in range(1, len(d_levels)):
        packed_L, tile_map_L, parent_ids_L, _ = d_levels[L]
        packed_parent, tile_map_parent, _, _ = d_levels[L - 1]

        k_L = packed_L.shape[0]
        if k_L == 0:
            continue

        # parent_ids_L[slot] = parent node ID for slot-th active node at level L
        # tile_map_parent[parent_node_id] -> parent slot (-1 if inactive)
        parent_slots = tile_map_parent[parent_ids_L.long()]  # [k_L] int32

        # Handle inactive parents (-1 sentinel) -> use zero features
        valid = parent_slots >= 0
        parent_slots_safe = torch.where(
            valid, parent_slots.long(),
            torch.zeros_like(parent_slots, dtype=torch.int64)
        )

        parent_feats = packed_parent[parent_slots_safe]  # [k_L, feat_dim]
        # Zero out contributions from inactive parents
        parent_feats = parent_feats * valid.unsqueeze(1).float()

        packed_L[:] = packed_L + parent_feats


# =====================================================================
# Tree: Benchmark loop
# =====================================================================

tree_results: List[Dict[str, Any]] = []

total_tree = (len(TREE_MAX_DEPTHS) * len(TREE_BRANCHING) * len(TREE_FEAT_DIMS)
              * len(TREE_SPARSITIES) * N_SEEDS)
tree_count = 0

for max_depth in TREE_MAX_DEPTHS:
    for branching in TREE_BRANCHING:
        for feat_dim in TREE_FEAT_DIMS:
            for sparsity in TREE_SPARSITIES:
                seed_records_base = []
                seed_records_d = []

                for seed_idx in range(N_SEEDS):
                    seed = SEED_BASE + seed_idx
                    rng = np.random.default_rng(seed)
                    torch.manual_seed(seed)

                    levels = build_tree_levels(
                        max_depth, branching, feat_dim, sparsity, rng, seed)

                    # ---- Tree Baseline ----
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(DEVICE)

                    bl = build_tree_baseline(levels)
                    _sync()

                    resident_base = (torch.cuda.memory_allocated(DEVICE)
                                     if DEVICE.type == "cuda" else 0)

                    torch.cuda.reset_peak_memory_stats(DEVICE)
                    times_base = _timed_runs(
                        lambda: compute_tree_baseline(bl))
                    peak_base = (torch.cuda.max_memory_allocated(DEVICE)
                                 if DEVICE.type == "cuda" else 0)

                    wall_base_us = float(np.median(times_base)) * 1e6

                    # Per-level resident
                    per_level_res_base = []
                    for lev in levels:
                        nbytes = (lev["features"].nelement() * lev["features"].element_size()
                                  + lev["active_mask"].nbytes
                                  + lev["parent_ids"].nbytes)
                        per_level_res_base.append(int(nbytes))

                    build_base_us = _build_time(
                        lambda: build_tree_baseline(levels))

                    seed_records_base.append({
                        "resident_bytes": int(resident_base),
                        "peak_vram_bytes": int(peak_base),
                        "wall_clock_us": wall_base_us,
                        "build_cost_us": build_base_us,
                        "per_level_resident": per_level_res_base,
                    })

                    del bl
                    torch.cuda.empty_cache()

                    # ---- D_tree ----
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(DEVICE)

                    dl = build_tree_d(levels)
                    _sync()

                    resident_d = (torch.cuda.memory_allocated(DEVICE)
                                  if DEVICE.type == "cuda" else 0)

                    torch.cuda.reset_peak_memory_stats(DEVICE)
                    times_d = _timed_runs(
                        lambda: compute_tree_d(dl))
                    peak_d = (torch.cuda.max_memory_allocated(DEVICE)
                              if DEVICE.type == "cuda" else 0)

                    wall_d_us = float(np.median(times_d)) * 1e6

                    # Per-level resident for D
                    per_level_res_d = []
                    for packed_data, tile_map, _, n_nodes in dl:
                        nbytes = (packed_data.nelement() * packed_data.element_size()
                                  + tile_map.nelement() * tile_map.element_size())
                        per_level_res_d.append(int(nbytes))

                    # Cross-level lookup time: time to resolve parent slots only
                    def _cross_level_lookup():
                        for L in range(1, len(dl)):
                            _, _, parent_ids_L, _ = dl[L]
                            _, tile_map_parent, _, _ = dl[L - 1]
                            _ = tile_map_parent[parent_ids_L.long()]

                    times_lookup = _timed_runs(_cross_level_lookup)
                    cross_lookup_us = float(np.median(times_lookup)) * 1e6

                    build_d_us = _build_time(
                        lambda: build_tree_d(levels))

                    seed_records_d.append({
                        "resident_bytes": int(resident_d),
                        "peak_vram_bytes": int(peak_d),
                        "wall_clock_us": wall_d_us,
                        "build_cost_us": build_d_us,
                        "cross_level_lookup_us": cross_lookup_us,
                        "per_level_resident": per_level_res_d,
                    })

                    del dl
                    # Clean up level features from GPU
                    for lev in levels:
                        del lev["features"]
                    torch.cuda.empty_cache()

                    tree_count += 1

                # Aggregate across seeds (median for scalars)
                def _median_scalar_dict(records):
                    out = {}
                    for key in records[0]:
                        if key == "per_level_resident":
                            # Median per level
                            n_levels = len(records[0][key])
                            out[key] = [
                                float(np.median([r[key][i] for r in records]))
                                for i in range(n_levels)
                            ]
                        else:
                            out[key] = float(np.median([r[key] for r in records]))
                    return out

                base_agg = _median_scalar_dict(seed_records_base)
                d_agg = _median_scalar_dict(seed_records_d)

                time_overhead = ((d_agg["wall_clock_us"] - base_agg["wall_clock_us"])
                                 / max(base_agg["wall_clock_us"], 1e-9))

                contour_a = (d_agg["resident_bytes"] < base_agg["resident_bytes"]
                             and time_overhead < KILL_THRESH)
                contour_b = (d_agg["peak_vram_bytes"] < base_agg["peak_vram_bytes"]
                             and time_overhead < KILL_THRESH)

                total_res_base = sum(base_agg["per_level_resident"])
                total_res_d = sum(d_agg["per_level_resident"])

                rec = {
                    "space": "tree_hierarchy",
                    "max_depth": max_depth,
                    "branching": branching,
                    "feat_dim": feat_dim,
                    "sparsity": sparsity,
                    "tree_baseline": base_agg,
                    "D_tree": d_agg,
                    "total_resident_base": total_res_base,
                    "total_resident_d": total_res_d,
                    "time_overhead_frac": round(time_overhead, 4),
                    "contour_A": "PASS" if contour_a else "FAIL",
                    "contour_B": "PASS" if contour_b else "FAIL",
                }
                tree_results.append(rec)

                ca_tag = "PASS" if contour_a else "FAIL"
                cb_tag = "PASS" if contour_b else "FAIL"
                print(f"  [{tree_count}/{total_tree}] depth={max_depth} "
                      f"br={branching} fd={feat_dim} sp={sparsity} | "
                      f"A={ca_tag} B={cb_tag} "
                      f"time_oh={time_overhead:+.1%} "
                      f"res={d_agg['resident_bytes']:.0f}/"
                      f"{base_agg['resident_bytes']:.0f} "
                      f"peak={d_agg['peak_vram_bytes']:.0f}/"
                      f"{base_agg['peak_vram_bytes']:.0f}")

print(f"\nTree Hierarchy: {len(tree_results)} configs done.")
tr_pass_a = sum(1 for r in tree_results if r["contour_A"] == "PASS")
tr_pass_b = sum(1 for r in tree_results if r["contour_B"] == "PASS")
print(f"  Contour A: {tr_pass_a}/{len(tree_results)} PASS")
print(f"  Contour B: {tr_pass_b}/{len(tree_results)} PASS")


# #####################################################################
# OUTPUT: Summary JSON
# #####################################################################

print("\n" + "=" * 72)
print("Writing outputs...")
print("=" * 72)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

summary = {
    "experiment": "exp10h_cross_space",
    "device": str(DEVICE),
    "dtype": str(DTYPE),
    "n_seeds": N_SEEDS,
    "kill_threshold": KILL_THRESH,
    "vector_grid": {
        "config": {
            "sides": VG_SIDES,
            "channels": VG_CHANNELS,
            "sparsities": VG_SPARSITIES,
            "patterns": VG_PATTERNS,
            "tile_size": VG_TILE_SIZE,
        },
        "results": vg_results,
        "contour_A_pass": vg_pass_a,
        "contour_A_total": len(vg_results),
        "contour_B_pass": vg_pass_b,
        "contour_B_total": len(vg_results),
    },
    "tree_hierarchy": {
        "config": {
            "max_depths": TREE_MAX_DEPTHS,
            "branching_factors": TREE_BRANCHING,
            "feature_dims": TREE_FEAT_DIMS,
            "sparsities": TREE_SPARSITIES,
        },
        "results": tree_results,
        "contour_A_pass": tr_pass_a,
        "contour_A_total": len(tree_results),
        "contour_B_pass": tr_pass_b,
        "contour_B_total": len(tree_results),
    },
}

with open(RESULTS_DIR / "exp10h_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"  Wrote {RESULTS_DIR / 'exp10h_summary.json'}")


# #####################################################################
# OUTPUT: Plots
# #####################################################################

# =====================================================================
# Plot 1: Vector Grid -- time + VRAM comparison
# =====================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: wall clock ratio (D / grid) by sparsity, colored by channel count
ax = axes[0]
for C in VG_CHANNELS:
    subset = [r for r in vg_results if r["channels"] == C]
    sps = sorted(set(r["sparsity"] for r in subset))
    ratios = []
    for sp in sps:
        sp_recs = [r for r in subset if r["sparsity"] == sp]
        ratio = np.mean([r["D_direct"]["wall_clock_us"]
                         / max(r["grid_baseline"]["wall_clock_us"], 1e-9)
                         for r in sp_recs])
        ratios.append(ratio)
    ax.plot(sps, ratios, "o-", label=f"C={C}")
ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
ax.axhline(1.0 + KILL_THRESH, color="red", linestyle=":", alpha=0.5,
           label=f"+{KILL_THRESH:.0%} kill")
ax.set_xlabel("Sparsity (fraction active)")
ax.set_ylabel("Time ratio (D / grid)")
ax.set_title("Vector Grid: Wall Clock Ratio")
ax.legend()
ax.grid(True, alpha=0.3)

# Right: VRAM ratio (D / grid) by sparsity
ax = axes[1]
for C in VG_CHANNELS:
    subset = [r for r in vg_results if r["channels"] == C]
    sps = sorted(set(r["sparsity"] for r in subset))
    res_ratios = []
    peak_ratios = []
    for sp in sps:
        sp_recs = [r for r in subset if r["sparsity"] == sp]
        rr = np.mean([r["D_direct"]["resident_bytes"]
                       / max(r["grid_baseline"]["resident_bytes"], 1)
                       for r in sp_recs])
        pr = np.mean([r["D_direct"]["peak_vram_bytes"]
                       / max(r["grid_baseline"]["peak_vram_bytes"], 1)
                       for r in sp_recs])
        res_ratios.append(rr)
        peak_ratios.append(pr)
    ax.plot(sps, res_ratios, "o-", label=f"C={C} resident")
    ax.plot(sps, peak_ratios, "s--", label=f"C={C} peak", alpha=0.7)
ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("Sparsity (fraction active)")
ax.set_ylabel("VRAM ratio (D / grid)")
ax.set_title("Vector Grid: VRAM Ratio")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

fig.suptitle("exp10h: Vector Grid -- D_direct vs grid_baseline", fontsize=13)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "exp10h_vector_grid.png", dpi=150)
plt.close(fig)
print(f"  Wrote {RESULTS_DIR / 'exp10h_vector_grid.png'}")


# =====================================================================
# Plot 2: Tree Hierarchy -- time + VRAM by depth
# =====================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: wall clock ratio by sparsity, grouped by depth
ax = axes[0]
for depth in TREE_MAX_DEPTHS:
    subset = [r for r in tree_results if r["max_depth"] == depth]
    sps = sorted(set(r["sparsity"] for r in subset))
    ratios = []
    for sp in sps:
        sp_recs = [r for r in subset if r["sparsity"] == sp]
        ratio = np.mean([r["D_tree"]["wall_clock_us"]
                         / max(r["tree_baseline"]["wall_clock_us"], 1e-9)
                         for r in sp_recs])
        ratios.append(ratio)
    ax.plot(sps, ratios, "o-", label=f"depth={depth}")
ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
ax.axhline(1.0 + KILL_THRESH, color="red", linestyle=":", alpha=0.5,
           label=f"+{KILL_THRESH:.0%} kill")
ax.set_xlabel("Sparsity (fraction active)")
ax.set_ylabel("Time ratio (D / baseline)")
ax.set_title("Tree: Wall Clock Ratio")
ax.legend()
ax.grid(True, alpha=0.3)

# Right: VRAM ratio by sparsity, grouped by depth
ax = axes[1]
for depth in TREE_MAX_DEPTHS:
    subset = [r for r in tree_results if r["max_depth"] == depth]
    sps = sorted(set(r["sparsity"] for r in subset))
    res_ratios = []
    peak_ratios = []
    for sp in sps:
        sp_recs = [r for r in subset if r["sparsity"] == sp]
        rr = np.mean([r["D_tree"]["resident_bytes"]
                       / max(r["tree_baseline"]["resident_bytes"], 1)
                       for r in sp_recs])
        pr = np.mean([r["D_tree"]["peak_vram_bytes"]
                       / max(r["tree_baseline"]["peak_vram_bytes"], 1)
                       for r in sp_recs])
        res_ratios.append(rr)
        peak_ratios.append(pr)
    ax.plot(sps, res_ratios, "o-", label=f"d={depth} resident")
    ax.plot(sps, peak_ratios, "s--", label=f"d={depth} peak", alpha=0.7)
ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("Sparsity (fraction active)")
ax.set_ylabel("VRAM ratio (D / baseline)")
ax.set_title("Tree: VRAM Ratio")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

fig.suptitle("exp10h: Tree Hierarchy -- D_tree vs tree_baseline", fontsize=13)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "exp10h_tree.png", dpi=150)
plt.close(fig)
print(f"  Wrote {RESULTS_DIR / 'exp10h_tree.png'}")


# #####################################################################
# OUTPUT: Report
# #####################################################################

report_lines = [
    "# exp10h Cross-Space Validation Report",
    "",
    "## Vector Grid",
    "",
    f"Contour A (architectural): {vg_pass_a}/{len(vg_results)} PASS",
    f"Contour B (operational):   {vg_pass_b}/{len(vg_results)} PASS",
    "",
    "| side | C | sparsity | pattern | A | B | time_oh | res_D/res_grid | peak_D/peak_grid |",
    "|------|---|----------|---------|---|---|---------|----------------|------------------|",
]

for r in vg_results:
    g = r["grid_baseline"]
    d = r["D_direct"]
    res_ratio = d["resident_bytes"] / max(g["resident_bytes"], 1)
    peak_ratio = d["peak_vram_bytes"] / max(g["peak_vram_bytes"], 1)
    report_lines.append(
        f"| {r['side']} | {r['channels']} | {r['sparsity']} | {r['pattern']} "
        f"| {r['contour_A']} | {r['contour_B']} "
        f"| {r['time_overhead_frac']:+.1%} "
        f"| {res_ratio:.2f} | {peak_ratio:.2f} |"
    )

report_lines += [
    "",
    "## Tree Hierarchy",
    "",
    f"Contour A (architectural): {tr_pass_a}/{len(tree_results)} PASS",
    f"Contour B (operational):   {tr_pass_b}/{len(tree_results)} PASS",
    "",
    "| depth | br | feat | sparsity | A | B | time_oh | res_D/res_base | peak_D/peak_base |",
    "|-------|----|----- |----------|---|---|---------|----------------|------------------|",
]

for r in tree_results:
    g = r["tree_baseline"]
    d = r["D_tree"]
    res_ratio = d["resident_bytes"] / max(g["resident_bytes"], 1)
    peak_ratio = d["peak_vram_bytes"] / max(g["peak_vram_bytes"], 1)
    report_lines.append(
        f"| {r['max_depth']} | {r['branching']} | {r['feat_dim']} "
        f"| {r['sparsity']} "
        f"| {r['contour_A']} | {r['contour_B']} "
        f"| {r['time_overhead_frac']:+.1%} "
        f"| {res_ratio:.2f} | {peak_ratio:.2f} |"
    )

report_lines += [
    "",
    "## Verdict",
    "",
    f"Vector Grid:    A={vg_pass_a}/{len(vg_results)}  B={vg_pass_b}/{len(vg_results)}",
    f"Tree Hierarchy: A={tr_pass_a}/{len(tree_results)}  B={tr_pass_b}/{len(tree_results)}",
    "",
]

# Overall verdict
all_a = vg_pass_a + tr_pass_a
all_total = len(vg_results) + len(tree_results)
all_b = vg_pass_b + tr_pass_b

if all_a == all_total and all_b == all_total:
    report_lines.append("**OVERALL: PASS** -- D_direct generalizes to both spaces.")
elif all_a == all_total:
    report_lines.append("**OVERALL: PARTIAL** -- Contour A passes everywhere; "
                        "Contour B has failures (operator workspace overhead).")
else:
    report_lines.append("**OVERALL: MIXED** -- See per-config results above.")

report_text = "\n".join(report_lines) + "\n"
with open(RESULTS_DIR / "exp10h_report.md", "w") as f:
    f.write(report_text)
print(f"  Wrote {RESULTS_DIR / 'exp10h_report.md'}")

print("\n" + "=" * 72)
print("exp10h COMPLETE")
print("=" * 72)
print(f"\nVector Grid:    Contour A {vg_pass_a}/{len(vg_results)} | "
      f"Contour B {vg_pass_b}/{len(vg_results)}")
print(f"Tree Hierarchy: Contour A {tr_pass_a}/{len(tree_results)} | "
      f"Contour B {tr_pass_b}/{len(tree_results)}")
