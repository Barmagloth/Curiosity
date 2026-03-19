#!/usr/bin/env python3
"""
Curiosity -- Exp10g: Dual-Mode Benchmark (Layout vs Operator)

Motivation:
  Exp10f showed packed tiles + direct tile_map (candidate D) is 5x faster on
  compute and 5.5x smaller on resident memory, BUT fails the peak VRAM
  criterion because F.conv2d allocates temporary workspace buffers.

  This experiment separates layout cost from operator cost with two benchmark
  modes to produce orthogonal verdicts:

  Mode 1 (Stencil) -- Layout benchmark:
    Manual 3x3 stencil using torch.roll (grid/A_bitset) or tile_map neighbor
    lookup (D_direct).  NO F.conv2d, NO temporary workspace.
    Measures pure layout cost: addressing + memory access pattern.

  Mode 2 (Conv2d) -- Operator benchmark:
    F.conv2d with 3x3 kernel, same as exp10f.
    Shows real operational cost including workspace.

Candidates:
  grid_baseline: full data tensor [side*side] + bool mask
  A_bitset:      full grid + packed bitset mask (from exp10e)
  D_direct:      packed tiles + tile_map (from exp10f)

Memory measurement: THREE numbers per config:
  1. resident_bytes  -- memory after build, before compute
  2. peak_bytes      -- max_memory_allocated during compute
  3. workspace_bytes -- peak - resident (temporary overhead)

Dual kill criteria:
  Contour A (architectural viability, Mode 1 stencil):
    - resident_bytes: D must be < grid
    - wall_clock: D must not be >20% slower than grid
  Contour B (operational viability, Mode 2 conv2d):
    - peak_bytes: overhead >20% vs grid -> fail
    - wall_clock: same criterion

Outputs (in results/ subdirectory):
  exp10g_summary.json              -- all data + verdicts
  exp10g_report.md                 -- human-readable report with tables
  exp10g_resident_comparison.png   -- resident memory by config (Mode 1)
  exp10g_peak_comparison.png       -- peak memory by config (Mode 2)
  exp10g_workspace_comparison.png  -- workspace overhead both modes
  exp10g_time_comparison.png       -- wall-clock both modes
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =====================================================================
# Imports from prior experiments
# =====================================================================

# D_direct: PackedTileLayout from exp10f
sys.path.insert(0, str(Path(__file__).parent.parent / "exp10f_packed_lookup"))
from packed_tile_layout import PackedTileLayout

# A_bitset: GridBitsetLayout from exp10e
sys.path.insert(0, str(Path(__file__).parent.parent / "exp10e_tile_sparse"))
from candidate_a_bitset import GridBitsetLayout, BitsetMask

# =====================================================================
# Configuration
# =====================================================================

SIDES = [64, 128, 256]
SPARSITIES = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70]
PATTERNS = ["random", "clustered", "checkerboard"]
TILE_SIZE = 8
N_SEEDS = 10
N_WARMUP = 5
N_REPEAT = 20
SEED_BASE = 42
DTYPE = torch.float32
KILL_THRESH = 0.20  # 20%
MODES = ["stencil", "conv2d"]

STENCIL_WEIGHTS_2D = torch.tensor(
    [[0.05, 0.1, 0.05],
     [0.1,  0.4, 0.1],
     [0.05, 0.1, 0.05]],
    dtype=DTYPE,
)

CANDIDATE_NAMES = ["grid_baseline", "A_bitset", "D_direct"]

CAND_COLORS = {
    "grid_baseline": "#1f77b4",
    "A_bitset":      "#ff7f0e",
    "D_direct":      "#2ca02c",
}

# =====================================================================
# Mask generation (from exp10e/exp10f, extended)
# =====================================================================

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


# =====================================================================
# Timing helpers
# =====================================================================

def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed_runs(func, n_warmup: int, n_repeat: int,
                device: torch.device) -> np.ndarray:
    """Return array of wall-clock seconds for n_repeat runs after warmup."""
    for _ in range(n_warmup):
        func()
        _sync(device)

    times = []
    for _ in range(n_repeat):
        _sync(device)
        t0 = time.perf_counter()
        func()
        _sync(device)
        dt = time.perf_counter() - t0
        times.append(dt)
    return np.array(times)


# =====================================================================
# Three-number memory measurement
# =====================================================================

def measure_three_memory(compute_fn, device: torch.device
                         ) -> Tuple[int, int, int]:
    """Measure resident, peak, and workspace bytes around a compute call.

    IMPORTANT: caller must have already built the layout and call
    torch.cuda.reset_peak_memory_stats AFTER layout build, BEFORE
    calling this function.

    Returns: (resident_bytes, peak_bytes, workspace_bytes)
    """
    if device.type != "cuda":
        compute_fn()
        return (0, 0, 0)

    torch.cuda.reset_peak_memory_stats(device)
    _sync(device)

    # 1. Resident: memory after build, before compute
    resident_bytes = torch.cuda.memory_allocated(device)

    # 2. Run compute
    compute_fn()
    _sync(device)

    # 3. Peak step bytes
    peak_bytes = torch.cuda.max_memory_allocated(device)

    # 4. Workspace = peak - resident
    workspace_bytes = peak_bytes - resident_bytes

    return (resident_bytes, peak_bytes, workspace_bytes)


# =====================================================================
# Stencil compute functions (Mode 1 -- NO F.conv2d)
# =====================================================================

def stencil_grid(data_flat: torch.Tensor, mask_flat: torch.Tensor,
                 side: int, weights_2d: torch.Tensor) -> torch.Tensor:
    """Manual 3x3 stencil on grid layout using torch.roll.

    No F.conv2d, no temporaries from PyTorch operators beyond the rolls.
    For each active element, reads 8 neighbors + self via shifted views,
    computes weighted average.

    Args:
        data_flat: [side*side] float tensor (full grid data)
        mask_flat: [side*side] bool tensor
        side: grid side length
        weights_2d: [3, 3] stencil weights
    Returns:
        output_flat: [side*side] float tensor with stencil applied at active positions
    """
    data_2d = data_flat.reshape(side, side)

    # Use torch.roll for each of the 9 stencil positions
    # Roll offsets: (dy, dx) -> roll(data, shifts=(-dy, -dx), dims=(0, 1))
    accum = torch.zeros_like(data_2d)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            w = weights_2d[dy + 1, dx + 1]
            if w == 0.0:
                continue
            shifted = torch.roll(data_2d, shifts=(-dy, -dx), dims=(0, 1))
            accum = accum + w * shifted

    result_flat = accum.reshape(-1)
    output = data_flat.clone()
    output[mask_flat] = result_flat[mask_flat]
    return output


def stencil_a_bitset(layout: GridBitsetLayout, field: torch.Tensor,
                     weights_2d: torch.Tensor) -> torch.Tensor:
    """Manual 3x3 stencil on A_bitset layout using torch.roll.

    Same as stencil_grid but uses the layout's cached bool mask.
    """
    side = layout.side
    data_2d = field.reshape(side, side)

    accum = torch.zeros_like(data_2d)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            w = weights_2d[dy + 1, dx + 1]
            if w == 0.0:
                continue
            shifted = torch.roll(data_2d, shifts=(-dy, -dx), dims=(0, 1))
            accum = accum + w * shifted

    result_flat = accum.reshape(-1)
    output = field.clone()
    output[layout._bool_mask] = result_flat[layout._bool_mask]
    return output


def stencil_d_direct(layout: PackedTileLayout,
                     weights_2d: torch.Tensor) -> torch.Tensor:
    """Manual 3x3 stencil on packed tile layout using tile_map neighbor lookup.

    For each active tile, gather the 9 neighbor tiles (using neighbor_slots),
    apply element-wise weighted sum across the stencil.
    NO F.conv2d -- pure tile_map-based neighbor access.

    Args:
        layout: PackedTileLayout with tiles packed and neighbor_slots built
        weights_2d: [3, 3] stencil weights
    Returns:
        Updated tiles written back into layout.tiles
    """
    # Ensure neighbor table is built
    if layout.neighbor_slots is None:
        layout.build_neighbor_table()

    k = layout.k
    ts = layout.tile_size
    device = layout.device

    # layout.tiles: [k, ts, ts, C]
    # neighbor_slots: [k, 9] int32 -- slot index for each of the 9 neighbors
    #   (-1 = boundary/inactive -> zero pad)

    # Append a zero tile for boundary padding
    zero_tile = torch.zeros(1, ts, ts, layout.n_channels,
                            dtype=layout.dtype, device=device)
    tiles_padded = torch.cat([layout.tiles, zero_tile], dim=0)  # [k+1, ts, ts, C]

    # Remap -1 -> k (zero tile index)
    gather_idx = layout.neighbor_slots.long()  # [k, 9]
    gather_idx = torch.where(
        gather_idx < 0,
        torch.tensor(k, dtype=torch.int64, device=device),
        gather_idx,
    )

    # Gather all 9 neighbor tiles: [k, 9, ts, ts, C]
    neighbor_tiles = tiles_padded[gather_idx.reshape(-1)].reshape(
        k, 9, ts, ts, layout.n_channels
    )

    # Apply stencil weights: weights_2d is [3,3], flatten to [9]
    # For tile-level stencil: each neighbor tile corresponds to a tile-level
    # offset, but within each tile we need the element-level stencil.
    # Since tiles are spatially contiguous sub-blocks, the correct approach
    # for an intra-tile stencil is to work element-by-element within each tile.
    #
    # However, to keep this comparable to the grid stencil (which operates
    # element-wise across the whole grid), we apply the stencil at the
    # element level WITHIN each tile using torch.roll on the tile data,
    # then blend with neighbor tile borders for boundary elements.
    #
    # Simplified approach matching the grid stencil semantics:
    # For interior tile elements, torch.roll within the tile is correct.
    # For border tile elements, we need neighbor tile data.
    #
    # Implementation: Assemble a (ts+2) x (ts+2) padded tile by stitching
    # neighbor tile borders, then apply the 3x3 stencil via slicing.

    # Neighbor layout (9 neighbors indexed as):
    #   0=(-1,-1) 1=(-1,0) 2=(-1,+1)
    #   3=(0,-1)  4=(0,0)  5=(0,+1)
    #   6=(+1,-1) 7=(+1,0) 8=(+1,+1)

    C = layout.n_channels

    # Build padded tile: [k, ts+2, ts+2, C]
    padded = torch.zeros(k, ts + 2, ts + 2, C, dtype=layout.dtype, device=device)

    # Center: the tile itself (neighbor index 4 = self)
    padded[:, 1:ts+1, 1:ts+1, :] = neighbor_tiles[:, 4, :, :, :]

    # Top row (from neighbor 1 = (-1,0)): last row of top neighbor
    padded[:, 0, 1:ts+1, :] = neighbor_tiles[:, 1, ts-1, :, :]

    # Bottom row (from neighbor 7 = (+1,0)): first row of bottom neighbor
    padded[:, ts+1, 1:ts+1, :] = neighbor_tiles[:, 7, 0, :, :]

    # Left col (from neighbor 3 = (0,-1)): last col of left neighbor
    padded[:, 1:ts+1, 0, :] = neighbor_tiles[:, 3, :, ts-1, :]

    # Right col (from neighbor 5 = (0,+1)): first col of right neighbor
    padded[:, 1:ts+1, ts+1, :] = neighbor_tiles[:, 5, :, 0, :]

    # Top-left corner (from neighbor 0 = (-1,-1)): bottom-right element
    padded[:, 0, 0, :] = neighbor_tiles[:, 0, ts-1, ts-1, :]

    # Top-right corner (from neighbor 2 = (-1,+1)): bottom-left element
    padded[:, 0, ts+1, :] = neighbor_tiles[:, 2, ts-1, 0, :]

    # Bottom-left corner (from neighbor 6 = (+1,-1)): top-right element
    padded[:, ts+1, 0, :] = neighbor_tiles[:, 6, 0, ts-1, :]

    # Bottom-right corner (from neighbor 8 = (+1,+1)): top-left element
    padded[:, ts+1, ts+1, :] = neighbor_tiles[:, 8, 0, 0, :]

    # Apply 3x3 stencil via 9 shifted slices of the padded tile
    result = torch.zeros(k, ts, ts, C, dtype=layout.dtype, device=device)
    for dy in range(3):
        for dx in range(3):
            w = weights_2d[dy, dx]
            if w == 0.0:
                continue
            result = result + w * padded[:, dy:dy+ts, dx:dx+ts, :]

    layout.tiles = result
    return result


# =====================================================================
# Conv2d compute functions (Mode 2 -- with F.conv2d)
# =====================================================================

def conv2d_grid(data_flat: torch.Tensor, mask_flat: torch.Tensor,
                side: int, stencil_4d: torch.Tensor) -> torch.Tensor:
    """F.conv2d on grid layout."""
    field_2d = data_flat.reshape(1, 1, side, side)
    out_2d = F.conv2d(field_2d, stencil_4d, padding=1)
    result_flat = out_2d.reshape(-1)
    output = data_flat.clone()
    output[mask_flat] = result_flat[mask_flat]
    return output


def conv2d_a_bitset(layout: GridBitsetLayout, field: torch.Tensor,
                    stencil_4d: torch.Tensor) -> torch.Tensor:
    """F.conv2d on A_bitset layout (same as grid, uses cached bool mask)."""
    side = layout.side
    field_2d = field.reshape(1, 1, side, side)
    out_2d = F.conv2d(field_2d, stencil_4d, padding=1)
    result_flat = out_2d.reshape(-1)
    output = field.clone()
    output[layout._bool_mask] = result_flat[layout._bool_mask]
    return output


def conv2d_d_direct(layout: PackedTileLayout,
                    stencil_4d: torch.Tensor) -> torch.Tensor:
    """F.conv2d on packed tiles (each tile independently)."""
    # tiles: [k, ts, ts, C] -> [k, C, ts, ts] for conv2d
    t = layout.tiles.permute(0, 3, 1, 2)
    out = F.conv2d(t, stencil_4d, padding=1)
    layout.tiles = out.permute(0, 2, 3, 1)
    return layout.tiles


# =====================================================================
# Layout builders
# =====================================================================

def build_grid(mask_np: np.ndarray, side: int, device: torch.device
               ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build grid layout: data tensor + bool mask.

    Returns (data_flat, mask_flat).
    Build cost = 0 (mask is treated as input).
    """
    M = side * side
    mask_flat = torch.tensor(mask_np.ravel(), dtype=torch.bool, device=device)
    data_flat = torch.randn(M, dtype=DTYPE, device=device)
    return data_flat, mask_flat


def build_a_bitset(mask_np: np.ndarray, side: int, device: torch.device
                   ) -> Tuple[GridBitsetLayout, torch.Tensor]:
    """Build A_bitset layout.

    Returns (layout, field_flat).
    Build cost is measured externally.
    """
    M = side * side
    layout = GridBitsetLayout().build(mask_np, device=device)
    field = torch.randn(M, dtype=DTYPE, device=device)
    return layout, field


def build_d_direct(mask_np: np.ndarray, side: int, device: torch.device
                   ) -> PackedTileLayout:
    """Build D_direct (packed tile) layout.

    Returns layout with tiles packed from random data.
    Build cost is measured via layout.build_time_ms.
    """
    mask_t = torch.tensor(mask_np, dtype=torch.bool, device=device)
    grid_data = torch.randn(side, side, dtype=DTYPE, device=device)

    layout = PackedTileLayout(
        side=side,
        tile_size=TILE_SIZE,
        mask=mask_t,
        device=device,
        dtype=DTYPE,
        n_channels=1,
    )
    layout.pack_from_grid(grid_data)
    # Pre-build neighbor table for stencil mode
    layout.build_neighbor_table()
    return layout


# =====================================================================
# Single-config benchmark
# =====================================================================

def benchmark_single_config(
    side: int,
    sparsity: float,
    pattern: str,
    mode: str,
    device: torch.device,
) -> Dict[str, Dict[str, Any]]:
    """Run one (side, sparsity, pattern, mode) config across all candidates.

    Returns dict keyed by candidate name, each containing:
      - resident_bytes, peak_bytes, workspace_bytes
      - wall_clock_s (median of N_REPEAT)
      - wall_clock_std_s
      - build_cost_s
    """
    stencil_4d = STENCIL_WEIGHTS_2D.to(device=device).reshape(1, 1, 3, 3)
    weights_2d = STENCIL_WEIGHTS_2D.to(device=device)

    results = {}

    for seed_i in range(N_SEEDS):
        seed = SEED_BASE + seed_i
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        mask_np = make_mask(side, sparsity, pattern, rng)

        for cand_name in CANDIDATE_NAMES:
            if cand_name not in results:
                results[cand_name] = {
                    "resident_bytes_list": [],
                    "peak_bytes_list": [],
                    "workspace_bytes_list": [],
                    "wall_clock_s_list": [],
                    "build_cost_s_list": [],
                }

            # Clean slate
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                _sync(device)

            # --- Build layout (timed) ---
            _sync(device)
            t_build_start = time.perf_counter()

            if cand_name == "grid_baseline":
                data_flat, mask_flat = build_grid(mask_np, side, device)
                build_cost_s = 0.0  # mask is input, not counted
            elif cand_name == "A_bitset":
                layout_a, field_a = build_a_bitset(mask_np, side, device)
                _sync(device)
                build_cost_s = time.perf_counter() - t_build_start
            elif cand_name == "D_direct":
                layout_d = build_d_direct(mask_np, side, device)
                _sync(device)
                build_cost_s = time.perf_counter() - t_build_start
            else:
                raise ValueError(f"Unknown candidate: {cand_name}")

            if cand_name == "grid_baseline":
                _sync(device)
                build_cost_s = 0.0

            # --- Define compute function ---
            if mode == "stencil":
                if cand_name == "grid_baseline":
                    compute_fn = lambda: stencil_grid(
                        data_flat, mask_flat, side, weights_2d)
                elif cand_name == "A_bitset":
                    compute_fn = lambda: stencil_a_bitset(
                        layout_a, field_a, weights_2d)
                elif cand_name == "D_direct":
                    compute_fn = lambda: stencil_d_direct(
                        layout_d, weights_2d)
            elif mode == "conv2d":
                if cand_name == "grid_baseline":
                    compute_fn = lambda: conv2d_grid(
                        data_flat, mask_flat, side, stencil_4d)
                elif cand_name == "A_bitset":
                    compute_fn = lambda: conv2d_a_bitset(
                        layout_a, field_a, stencil_4d)
                elif cand_name == "D_direct":
                    compute_fn = lambda: conv2d_d_direct(
                        layout_d, stencil_4d)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            # --- Memory measurement ---
            resident, peak, workspace = measure_three_memory(
                compute_fn, device)

            # --- Wall-clock timing ---
            wall_times = _timed_runs(
                compute_fn, N_WARMUP, N_REPEAT, device)
            wall_median = float(np.median(wall_times))

            results[cand_name]["resident_bytes_list"].append(resident)
            results[cand_name]["peak_bytes_list"].append(peak)
            results[cand_name]["workspace_bytes_list"].append(workspace)
            results[cand_name]["wall_clock_s_list"].append(wall_median)
            results[cand_name]["build_cost_s_list"].append(build_cost_s)

            # Cleanup
            if cand_name == "grid_baseline":
                del data_flat, mask_flat
            elif cand_name == "A_bitset":
                del layout_a, field_a
            elif cand_name == "D_direct":
                del layout_d

            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Aggregate across seeds
    aggregated = {}
    for cand_name, cand_data in results.items():
        aggregated[cand_name] = {
            "resident_bytes": int(np.median(cand_data["resident_bytes_list"])),
            "peak_bytes": int(np.median(cand_data["peak_bytes_list"])),
            "workspace_bytes": int(np.median(cand_data["workspace_bytes_list"])),
            "wall_clock_s": float(np.median(cand_data["wall_clock_s_list"])),
            "wall_clock_std_s": float(np.std(cand_data["wall_clock_s_list"])),
            "build_cost_s": float(np.median(cand_data["build_cost_s_list"])),
            "per_seed": {
                "resident_bytes": cand_data["resident_bytes_list"],
                "peak_bytes": cand_data["peak_bytes_list"],
                "workspace_bytes": cand_data["workspace_bytes_list"],
                "wall_clock_s": cand_data["wall_clock_s_list"],
                "build_cost_s": cand_data["build_cost_s_list"],
            },
        }

    return aggregated


# =====================================================================
# Verdicts
# =====================================================================

def compute_verdicts(all_results: Dict[str, Dict[str, Dict]]
                     ) -> Dict[str, Any]:
    """Compute dual contour verdicts.

    Contour A (stencil mode):
      - resident_bytes: D must be < grid
      - wall_clock: D must not be >20% slower than grid

    Contour B (conv2d mode):
      - peak_bytes: overhead >20% vs grid -> fail
      - wall_clock: same criterion

    Returns dict with per-config and overall verdicts.
    """
    contour_a_results = []
    contour_b_results = []

    for config_key, mode_data in all_results.items():
        # Parse mode from config key
        parts = config_key.split("_mode=")
        mode = parts[1] if len(parts) > 1 else "unknown"
        config_part = parts[0]

        grid = mode_data.get("grid_baseline")
        d_direct = mode_data.get("D_direct")

        if grid is None or d_direct is None:
            continue

        if mode == "stencil":
            # Contour A
            resident_pass = d_direct["resident_bytes"] < grid["resident_bytes"]
            time_overhead = (
                (d_direct["wall_clock_s"] / grid["wall_clock_s"] - 1.0)
                if grid["wall_clock_s"] > 0 else 0.0
            )
            time_pass = time_overhead <= KILL_THRESH

            resident_ratio = (
                d_direct["resident_bytes"] / grid["resident_bytes"]
                if grid["resident_bytes"] > 0 else 0.0
            )

            contour_a_results.append({
                "config": config_part,
                "resident_pass": resident_pass,
                "resident_ratio": resident_ratio,
                "time_pass": time_pass,
                "time_overhead_pct": round(time_overhead * 100, 2),
                "verdict": "PASS" if (resident_pass and time_pass) else "FAIL",
            })

        elif mode == "conv2d":
            # Contour B
            peak_overhead = (
                (d_direct["peak_bytes"] / grid["peak_bytes"] - 1.0)
                if grid["peak_bytes"] > 0 else 0.0
            )
            peak_pass = peak_overhead <= KILL_THRESH

            time_overhead = (
                (d_direct["wall_clock_s"] / grid["wall_clock_s"] - 1.0)
                if grid["wall_clock_s"] > 0 else 0.0
            )
            time_pass = time_overhead <= KILL_THRESH

            contour_b_results.append({
                "config": config_part,
                "peak_pass": peak_pass,
                "peak_overhead_pct": round(peak_overhead * 100, 2),
                "time_pass": time_pass,
                "time_overhead_pct": round(time_overhead * 100, 2),
                "verdict": "PASS" if (peak_pass and time_pass) else "FAIL",
            })

    # Overall verdicts
    a_pass_count = sum(1 for r in contour_a_results if r["verdict"] == "PASS")
    a_total = len(contour_a_results)
    b_pass_count = sum(1 for r in contour_b_results if r["verdict"] == "PASS")
    b_total = len(contour_b_results)

    return {
        "contour_a": {
            "per_config": contour_a_results,
            "pass_count": a_pass_count,
            "total": a_total,
            "pass_rate": round(a_pass_count / max(1, a_total) * 100, 1),
            "overall": "PASS" if a_pass_count == a_total and a_total > 0 else "FAIL",
        },
        "contour_b": {
            "per_config": contour_b_results,
            "pass_count": b_pass_count,
            "total": b_total,
            "pass_rate": round(b_pass_count / max(1, b_total) * 100, 1),
            "overall": "PASS" if b_pass_count == b_total and b_total > 0 else "FAIL",
        },
    }


# =====================================================================
# Plotting
# =====================================================================

def _make_grouped_bar(data_by_cand: Dict[str, List[float]],
                      x_labels: List[str],
                      ylabel: str,
                      title: str,
                      out_path: Path) -> None:
    """Grouped bar chart: x = configs, bars = candidates."""
    fig, ax = plt.subplots(figsize=(14, 6))

    n_groups = len(x_labels)
    n_bars = len(CANDIDATE_NAMES)
    width = 0.8 / n_bars
    x = np.arange(n_groups)

    for i, cand in enumerate(CANDIDATE_NAMES):
        vals = data_by_cand.get(cand, [0.0] * n_groups)
        offset = (i - n_bars / 2 + 0.5) * width
        color = CAND_COLORS.get(cand, "#888888")
        ax.bar(x + offset, vals, width, label=cand, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_resident_comparison(all_results: Dict, out_dir: Path) -> None:
    """Resident memory by config for Mode 1 (stencil)."""
    configs = []
    data_by_cand: Dict[str, List[float]] = {c: [] for c in CANDIDATE_NAMES}

    for config_key in sorted(all_results.keys()):
        if "_mode=stencil" not in config_key:
            continue
        config_label = config_key.replace("_mode=stencil", "")
        configs.append(config_label)
        mode_data = all_results[config_key]
        for cand in CANDIDATE_NAMES:
            val = mode_data.get(cand, {}).get("resident_bytes", 0)
            data_by_cand[cand].append(val / 1024)  # KB

    if configs:
        _make_grouped_bar(
            data_by_cand, configs,
            "Resident Memory (KB)",
            "Exp10g Mode 1 (Stencil): Resident Memory",
            out_dir / "exp10g_resident_comparison.png",
        )


def plot_peak_comparison(all_results: Dict, out_dir: Path) -> None:
    """Peak memory by config for Mode 2 (conv2d)."""
    configs = []
    data_by_cand: Dict[str, List[float]] = {c: [] for c in CANDIDATE_NAMES}

    for config_key in sorted(all_results.keys()):
        if "_mode=conv2d" not in config_key:
            continue
        config_label = config_key.replace("_mode=conv2d", "")
        configs.append(config_label)
        mode_data = all_results[config_key]
        for cand in CANDIDATE_NAMES:
            val = mode_data.get(cand, {}).get("peak_bytes", 0)
            data_by_cand[cand].append(val / 1024)

    if configs:
        _make_grouped_bar(
            data_by_cand, configs,
            "Peak Memory (KB)",
            "Exp10g Mode 2 (Conv2d): Peak Memory",
            out_dir / "exp10g_peak_comparison.png",
        )


def plot_workspace_comparison(all_results: Dict, out_dir: Path) -> None:
    """Workspace overhead for both modes."""
    for mode in MODES:
        configs = []
        data_by_cand: Dict[str, List[float]] = {c: [] for c in CANDIDATE_NAMES}

        for config_key in sorted(all_results.keys()):
            if f"_mode={mode}" not in config_key:
                continue
            config_label = config_key.replace(f"_mode={mode}", "")
            configs.append(config_label)
            mode_data = all_results[config_key]
            for cand in CANDIDATE_NAMES:
                val = mode_data.get(cand, {}).get("workspace_bytes", 0)
                data_by_cand[cand].append(val / 1024)

        if configs:
            _make_grouped_bar(
                data_by_cand, configs,
                "Workspace Overhead (KB)",
                f"Exp10g Workspace Overhead -- Mode: {mode}",
                out_dir / f"exp10g_workspace_{mode}.png",
            )

    # Combined workspace plot: average across configs per candidate per mode
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax_idx, mode in enumerate(MODES):
        ax = axes[ax_idx]
        cand_means = {}
        for config_key in sorted(all_results.keys()):
            if f"_mode={mode}" not in config_key:
                continue
            mode_data = all_results[config_key]
            for cand in CANDIDATE_NAMES:
                val = mode_data.get(cand, {}).get("workspace_bytes", 0)
                if cand not in cand_means:
                    cand_means[cand] = []
                cand_means[cand].append(val / 1024)

        x = np.arange(len(CANDIDATE_NAMES))
        vals = [float(np.mean(cand_means.get(c, [0]))) for c in CANDIDATE_NAMES]
        colors = [CAND_COLORS.get(c, "#888") for c in CANDIDATE_NAMES]
        ax.bar(x, vals, color=colors, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(CANDIDATE_NAMES, fontsize=9)
        ax.set_ylabel("Mean Workspace (KB)")
        ax.set_title(f"Mode: {mode}", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Exp10g: Workspace Comparison", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "exp10g_workspace_comparison.png",
                dpi=150, bbox_inches="tight")
    plt.close()


def plot_time_comparison(all_results: Dict, out_dir: Path) -> None:
    """Wall-clock for both modes."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, mode in enumerate(MODES):
        ax = axes[ax_idx]
        configs = []
        data_by_cand: Dict[str, List[float]] = {c: [] for c in CANDIDATE_NAMES}

        for config_key in sorted(all_results.keys()):
            if f"_mode={mode}" not in config_key:
                continue
            config_label = config_key.replace(f"_mode={mode}", "")
            configs.append(config_label)
            mode_data = all_results[config_key]
            for cand in CANDIDATE_NAMES:
                val = mode_data.get(cand, {}).get("wall_clock_s", 0)
                data_by_cand[cand].append(val * 1e6)  # us

        if not configs:
            continue

        n_groups = len(configs)
        n_bars = len(CANDIDATE_NAMES)
        width = 0.8 / n_bars
        x = np.arange(n_groups)

        for i, cand in enumerate(CANDIDATE_NAMES):
            vals = data_by_cand.get(cand, [0.0] * n_groups)
            offset = (i - n_bars / 2 + 0.5) * width
            color = CAND_COLORS.get(cand, "#888888")
            ax.bar(x + offset, vals, width, label=cand, color=color, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=45, ha="right", fontsize=6)
        ax.set_ylabel("Wall-clock (us)")
        ax.set_title(f"Mode: {mode}", fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Exp10g: Wall-clock Comparison", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "exp10g_time_comparison.png",
                dpi=150, bbox_inches="tight")
    plt.close()


# =====================================================================
# Report generation
# =====================================================================

def generate_report(all_results: Dict, verdicts: Dict,
                    out_path: Path) -> None:
    """Generate human-readable markdown report."""
    lines = []
    lines.append("# Exp10g: Dual-Mode Benchmark Report")
    lines.append("")
    lines.append("## Experiment Design")
    lines.append("")
    lines.append("Two benchmark modes separate layout cost from operator cost:")
    lines.append("- **Mode 1 (Stencil):** Manual 3x3 stencil, no F.conv2d. "
                 "Isolates pure layout cost.")
    lines.append("- **Mode 2 (Conv2d):** F.conv2d with 3x3 kernel. "
                 "Shows real operator cost including workspace.")
    lines.append("")
    lines.append(f"- sides: {SIDES}")
    lines.append(f"- sparsities: {SPARSITIES}")
    lines.append(f"- patterns: {PATTERNS}")
    lines.append(f"- tile_size: {TILE_SIZE}")
    lines.append(f"- n_seeds: {N_SEEDS}, n_warmup: {N_WARMUP}, "
                 f"n_repeat: {N_REPEAT}")
    lines.append(f"- kill threshold: {KILL_THRESH:.0%}")
    lines.append("")

    # --- Contour A table ---
    lines.append("## Contour A (Architectural Viability -- Stencil Mode)")
    lines.append("")
    lines.append(f"Overall: **{verdicts['contour_a']['overall']}** "
                 f"({verdicts['contour_a']['pass_count']}"
                 f"/{verdicts['contour_a']['total']} configs pass)")
    lines.append("")
    lines.append("| Config | D Time vs grid | D Resident vs grid | "
                 "Contour A |")
    lines.append("|--------|---------------:|-------------------:|"
                 "-----------|")

    for config_key in sorted(all_results.keys()):
        if "_mode=stencil" not in config_key:
            continue
        config_label = config_key.replace("_mode=stencil", "")
        mode_data = all_results[config_key]
        grid = mode_data.get("grid_baseline", {})
        d = mode_data.get("D_direct", {})
        a = mode_data.get("A_bitset", {})

        grid_t = grid.get("wall_clock_s", 0)
        d_t = d.get("wall_clock_s", 0)
        a_t = a.get("wall_clock_s", 0)
        time_oh = (d_t / grid_t - 1.0) * 100 if grid_t > 0 else 0

        grid_r = grid.get("resident_bytes", 0)
        d_r = d.get("resident_bytes", 0)
        res_ratio = d_r / grid_r if grid_r > 0 else 0

        # Find matching verdict
        verdict_str = "N/A"
        for v in verdicts["contour_a"]["per_config"]:
            if v["config"] == config_label:
                verdict_str = v["verdict"]
                break

        lines.append(
            f"| {config_label} | {time_oh:+.1f}% | "
            f"{res_ratio:.2f}x | {verdict_str} |"
        )

    lines.append("")

    # --- Contour B table ---
    lines.append("## Contour B (Operational Viability -- Conv2d Mode)")
    lines.append("")
    lines.append(f"Overall: **{verdicts['contour_b']['overall']}** "
                 f"({verdicts['contour_b']['pass_count']}"
                 f"/{verdicts['contour_b']['total']} configs pass)")
    lines.append("")
    lines.append("| Config | D Time vs grid | D Peak vs grid | "
                 "D Workspace vs grid | Contour B |")
    lines.append("|--------|---------------:|---------------:|"
                 "-------------------:|-----------|")

    for config_key in sorted(all_results.keys()):
        if "_mode=conv2d" not in config_key:
            continue
        config_label = config_key.replace("_mode=conv2d", "")
        mode_data = all_results[config_key]
        grid = mode_data.get("grid_baseline", {})
        d = mode_data.get("D_direct", {})

        grid_t = grid.get("wall_clock_s", 0)
        d_t = d.get("wall_clock_s", 0)
        time_oh = (d_t / grid_t - 1.0) * 100 if grid_t > 0 else 0

        grid_p = grid.get("peak_bytes", 0)
        d_p = d.get("peak_bytes", 0)
        peak_oh = (d_p / grid_p - 1.0) * 100 if grid_p > 0 else 0

        grid_w = grid.get("workspace_bytes", 0)
        d_w = d.get("workspace_bytes", 0)
        ws_ratio = d_w / grid_w if grid_w > 0 else 0

        verdict_str = "N/A"
        for v in verdicts["contour_b"]["per_config"]:
            if v["config"] == config_label:
                verdict_str = v["verdict"]
                break

        lines.append(
            f"| {config_label} | {time_oh:+.1f}% | "
            f"{peak_oh:+.1f}% | {ws_ratio:.2f}x | {verdict_str} |"
        )

    lines.append("")

    # --- Summary comparison table (both modes, all candidates) ---
    lines.append("## Detailed Comparison (All Candidates)")
    lines.append("")

    lines.append("### Mode 1 (Stencil -- Layout Benchmark)")
    lines.append("")
    lines.append("| Candidate | Time vs grid | Resident vs grid | "
                 "Workspace vs grid | Contour A |")
    lines.append("|-----------|------------:|----------------:|"
                 "-----------------:|-----------|")

    # Aggregate across configs per candidate
    for cand in ["A_bitset", "D_direct"]:
        time_ohs = []
        res_ratios = []
        ws_ratios = []

        for config_key in sorted(all_results.keys()):
            if "_mode=stencil" not in config_key:
                continue
            mode_data = all_results[config_key]
            grid = mode_data.get("grid_baseline", {})
            c = mode_data.get(cand, {})

            gt = grid.get("wall_clock_s", 0)
            ct = c.get("wall_clock_s", 0)
            if gt > 0:
                time_ohs.append((ct / gt - 1.0) * 100)

            gr = grid.get("resident_bytes", 0)
            cr = c.get("resident_bytes", 0)
            if gr > 0:
                res_ratios.append(cr / gr)

            gw = grid.get("workspace_bytes", 0)
            cw = c.get("workspace_bytes", 0)
            if gw > 0:
                ws_ratios.append(cw / gw)

        med_time = float(np.median(time_ohs)) if time_ohs else 0
        med_res = float(np.median(res_ratios)) if res_ratios else 0
        med_ws = float(np.median(ws_ratios)) if ws_ratios else 0

        contour = "N/A"
        if cand == "D_direct":
            contour = verdicts["contour_a"]["overall"]

        lines.append(
            f"| {cand} | {med_time:+.1f}% | "
            f"{med_res:.2f}x | {med_ws:.2f}x | {contour} |"
        )

    lines.append("")

    lines.append("### Mode 2 (Conv2d -- Operator Benchmark)")
    lines.append("")
    lines.append("| Candidate | Time vs grid | Peak vs grid | "
                 "Workspace vs grid | Contour B |")
    lines.append("|-----------|------------:|------------:|"
                 "-----------------:|-----------|")

    for cand in ["A_bitset", "D_direct"]:
        time_ohs = []
        peak_ratios = []
        ws_ratios = []

        for config_key in sorted(all_results.keys()):
            if "_mode=conv2d" not in config_key:
                continue
            mode_data = all_results[config_key]
            grid = mode_data.get("grid_baseline", {})
            c = mode_data.get(cand, {})

            gt = grid.get("wall_clock_s", 0)
            ct = c.get("wall_clock_s", 0)
            if gt > 0:
                time_ohs.append((ct / gt - 1.0) * 100)

            gp = grid.get("peak_bytes", 0)
            cp = c.get("peak_bytes", 0)
            if gp > 0:
                peak_ratios.append(cp / gp)

            gw = grid.get("workspace_bytes", 0)
            cw = c.get("workspace_bytes", 0)
            if gw > 0:
                ws_ratios.append(cw / gw)

        med_time = float(np.median(time_ohs)) if time_ohs else 0
        med_peak = float(np.median(peak_ratios)) if peak_ratios else 0
        med_ws = float(np.median(ws_ratios)) if ws_ratios else 0

        contour = "N/A"
        if cand == "D_direct":
            contour = verdicts["contour_b"]["overall"]

        lines.append(
            f"| {cand} | {med_time:+.1f}% | "
            f"{med_peak:.2f}x | {med_ws:.2f}x | {contour} |"
        )

    lines.append("")

    # --- Build cost ---
    lines.append("## Build Cost")
    lines.append("")
    lines.append("| Config | grid | A_bitset | D_direct |")
    lines.append("|--------|-----:|---------:|---------:|")

    # Pick stencil mode configs (build cost is mode-independent)
    for config_key in sorted(all_results.keys()):
        if "_mode=stencil" not in config_key:
            continue
        config_label = config_key.replace("_mode=stencil", "")
        mode_data = all_results[config_key]

        grid_bc = mode_data.get("grid_baseline", {}).get("build_cost_s", 0)
        a_bc = mode_data.get("A_bitset", {}).get("build_cost_s", 0)
        d_bc = mode_data.get("D_direct", {}).get("build_cost_s", 0)

        lines.append(
            f"| {config_label} | {grid_bc*1e3:.2f}ms | "
            f"{a_bc*1e3:.2f}ms | {d_bc*1e3:.2f}ms |"
        )

    lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


# =====================================================================
# Main sweep
# =====================================================================

def run_all() -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Exp10g: Dual-Mode Benchmark (Layout vs Operator)")
    print("=" * 70)
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(device)}")
        props = torch.cuda.get_device_properties(device)
        print(f"  VRAM: {props.total_memory / 1e9:.1f} GB")
    print(f"  Device: {device}")
    print(f"  Sides: {SIDES}")
    print(f"  Sparsities: {SPARSITIES}")
    print(f"  Patterns: {PATTERNS}")
    print(f"  Tile size: {TILE_SIZE}")
    print(f"  Seeds: {N_SEEDS}, Warmup: {N_WARMUP}, Repeats: {N_REPEAT}")
    print(f"  Kill threshold: {KILL_THRESH:.0%}")
    print(f"  Modes: {MODES}")
    print(f"  Candidates: {CANDIDATE_NAMES}")
    print("=" * 70)

    # Results keyed by "side={s}_sp={sp}_pat={pat}_mode={mode}"
    all_results: Dict[str, Dict[str, Dict]] = {}

    total = len(SIDES) * len(SPARSITIES) * len(PATTERNS) * len(MODES)
    idx = 0

    for side in SIDES:
        for sparsity in SPARSITIES:
            for pattern in PATTERNS:
                for mode in MODES:
                    idx += 1
                    config_key = (f"side={side}_sp={sparsity}"
                                  f"_pat={pattern}_mode={mode}")
                    print(f"\n  [{idx}/{total}] {config_key}")

                    try:
                        result = benchmark_single_config(
                            side=side,
                            sparsity=sparsity,
                            pattern=pattern,
                            mode=mode,
                            device=device,
                        )
                        all_results[config_key] = result

                        # Print summary
                        grid = result.get("grid_baseline", {})
                        d = result.get("D_direct", {})

                        gt = grid.get("wall_clock_s", 0)
                        dt = d.get("wall_clock_s", 0)
                        t_oh = (dt / gt - 1.0) * 100 if gt > 0 else 0

                        print(f"    grid: {gt*1e6:.1f}us  "
                              f"D: {dt*1e6:.1f}us  "
                              f"time_oh={t_oh:+.1f}%")
                        print(f"    grid_res={grid.get('resident_bytes', 0)/1024:.0f}KB  "
                              f"D_res={d.get('resident_bytes', 0)/1024:.0f}KB  "
                              f"grid_peak={grid.get('peak_bytes', 0)/1024:.0f}KB  "
                              f"D_peak={d.get('peak_bytes', 0)/1024:.0f}KB")

                    except RuntimeError as e:
                        print(f"    SKIPPED ({e})")

    # ================================================================
    # Verdicts
    # ================================================================
    print("\n" + "=" * 70)
    print("VERDICTS")
    print("=" * 70)

    verdicts = compute_verdicts(all_results)

    print(f"\n  Contour A (Stencil -- architectural viability): "
          f"{verdicts['contour_a']['overall']}  "
          f"({verdicts['contour_a']['pass_count']}"
          f"/{verdicts['contour_a']['total']})")
    for v in verdicts["contour_a"]["per_config"]:
        flag = "" if v["verdict"] == "PASS" else " ***"
        print(f"    {v['config']}: {v['verdict']}  "
              f"time={v['time_overhead_pct']:+.1f}%  "
              f"res={v['resident_ratio']:.2f}x{flag}")

    print(f"\n  Contour B (Conv2d -- operational viability): "
          f"{verdicts['contour_b']['overall']}  "
          f"({verdicts['contour_b']['pass_count']}"
          f"/{verdicts['contour_b']['total']})")
    for v in verdicts["contour_b"]["per_config"]:
        flag = "" if v["verdict"] == "PASS" else " ***"
        print(f"    {v['config']}: {v['verdict']}  "
              f"time={v['time_overhead_pct']:+.1f}%  "
              f"peak={v['peak_overhead_pct']:+.1f}%{flag}")

    # ================================================================
    # Save JSON
    # ================================================================
    # Strip per-seed lists for JSON (keep median aggregates)
    json_results = {}
    for config_key, mode_data in all_results.items():
        json_results[config_key] = {}
        for cand_name, cand_data in mode_data.items():
            json_entry = {k: v for k, v in cand_data.items()
                          if k != "per_seed"}
            json_results[config_key][cand_name] = json_entry

    summary = {
        "config": {
            "sides": SIDES,
            "sparsities": SPARSITIES,
            "patterns": PATTERNS,
            "tile_size": TILE_SIZE,
            "n_seeds": N_SEEDS,
            "n_warmup": N_WARMUP,
            "n_repeat": N_REPEAT,
            "kill_threshold": KILL_THRESH,
            "modes": MODES,
            "candidates": CANDIDATE_NAMES,
            "device": str(device),
            "gpu_name": (torch.cuda.get_device_name(device)
                         if device.type == "cuda" else "N/A"),
        },
        "results": json_results,
        "verdicts": verdicts,
    }

    json_path = out_dir / "exp10g_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved: {json_path}")

    # ================================================================
    # Plots
    # ================================================================
    plot_resident_comparison(all_results, out_dir)
    print(f"Saved: {out_dir / 'exp10g_resident_comparison.png'}")

    plot_peak_comparison(all_results, out_dir)
    print(f"Saved: {out_dir / 'exp10g_peak_comparison.png'}")

    plot_workspace_comparison(all_results, out_dir)
    print(f"Saved: {out_dir / 'exp10g_workspace_comparison.png'}")

    plot_time_comparison(all_results, out_dir)
    print(f"Saved: {out_dir / 'exp10g_time_comparison.png'}")

    # ================================================================
    # Report
    # ================================================================
    report_path = out_dir / "exp10g_report.md"
    generate_report(all_results, verdicts, report_path)
    print(f"Saved: {report_path}")

    return summary


if __name__ == "__main__":
    t0 = time.time()
    summary = run_all()
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
