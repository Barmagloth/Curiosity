#!/usr/bin/env python3
"""
Curiosity -- Exp10f: Packed Tile Storage with Direct Tile-Level Index Lookup

Motivation:
  Exp10e showed packed tile storage saves VRAM (-30% at low occupancy) but
  binary search (torch.searchsorted on Morton keys) killed performance
  (+1700%).  This experiment replaces binary search with a direct
  tile_map[tile_id] -> slot array.

  For a 256x256 grid with 8x8 tiles: 32x32 = 1024 tiles = 4 KB for
  tile_map.  Pocket change compared to the binary-search overhead.

Data structures:
  tiles[k, Ht, Wt, C]       -- only active tiles, packed contiguously
  tile_map[n_tiles] int32    -- tile_id -> slot (-1 = inactive)
  active_tile_ids[k] int32   -- slot -> tile_id (canonical order)

# TODO: if n_tiles < 65535, store tile_map as int16 with -1 sentinel
#       (currently int32 everywhere)
"""

import time
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# =====================================================================
# Configuration
# =====================================================================

N_WARMUP = 5
N_REPEAT = 20
SEED_BASE = 42
DTYPE = torch.float32

# 3x3 stencil weights (same as exp10 SyntheticKernel / exp10e baseline)
STENCIL_WEIGHTS = torch.tensor(
    [[0.05, 0.1, 0.05],
     [0.1,  0.4, 0.1],
     [0.05, 0.1, 0.05]],
    dtype=DTYPE,
).reshape(1, 1, 3, 3)

# 3x3 neighbor offsets (row, col) for stencil / halo
NEIGHBOR_OFFSETS_3x3 = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1), ( 0, 0), ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
]


# =====================================================================
# Mask generation (ported from exp10e)
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


# =====================================================================
# PackedTileLayout
# =====================================================================

class PackedTileLayout:
    """Packed active tiles with direct tile-level index lookup.

    Storage: tiles[k, Ht, Wt, C] -- only active tiles, packed contiguously
    Index:   tile_map[n_tiles] int32 -- tile_id -> slot (-1 = inactive)
    Reverse: active_tile_ids[k] int32 -- slot -> tile_id (canonical order)
    """

    def __init__(
        self,
        side: int,
        tile_size: int,
        mask: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        n_channels: int = 1,
    ):
        """Build packed tile layout from element-level mask.

        Args:
            side: grid side length (e.g. 256).
            tile_size: tile dimensions (e.g. 8).
            mask: bool tensor [side, side] of active elements.
            device: torch device.
            dtype: data type for tile storage.
            n_channels: number of channels per element.
        """
        assert mask.shape == (side, side), f"mask shape {mask.shape} != ({side}, {side})"
        assert side % tile_size == 0, f"side {side} not divisible by tile_size {tile_size}"

        self.side = side
        self.tile_size = tile_size
        self.device = device
        self.dtype = dtype
        self.n_channels = n_channels

        self.n_tiles_y = side // tile_size
        self.n_tiles_x = side // tile_size
        self.n_tiles = self.n_tiles_y * self.n_tiles_x

        # -- Build (fully tensorized, timed separately) --
        t0 = time.perf_counter()
        self._build(mask)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self._build_time_ms = (time.perf_counter() - t0) * 1e3

        # Neighbor table (built on demand)
        self.neighbor_slots: Optional[torch.Tensor] = None  # [k, n_offsets] int32

    def _build(self, mask: torch.Tensor) -> None:
        """Fully tensorized builder -- NO Python loops."""
        mask_dev = mask.to(device=self.device, dtype=torch.bool)

        # Step 1: Compute tile activity.
        # Reshape [side, side] -> [n_tiles_y, tile_size, n_tiles_x, tile_size]
        mask_tiled = mask_dev.reshape(
            self.n_tiles_y, self.tile_size,
            self.n_tiles_x, self.tile_size,
        )
        # A tile is active if ANY element in it is active.
        tile_active = mask_tiled.any(dim=3).any(dim=1)  # [n_tiles_y, n_tiles_x] bool

        # Step 2: Get active tile linear indices (canonical order = row-major).
        # flatten -> nonzero -> squeeze to get 1D tensor of active tile ids.
        active_ids = tile_active.flatten().nonzero(as_tuple=False).squeeze(1)
        self.k = active_ids.shape[0]
        self.active_tile_ids = active_ids.to(dtype=torch.int32)  # [k]

        # Step 3: Build tile_map: tile_id -> slot, -1 = inactive.
        self.tile_map = torch.full(
            (self.n_tiles,), -1, dtype=torch.int32, device=self.device,
        )
        slots = torch.arange(self.k, dtype=torch.int32, device=self.device)
        self.tile_map[active_ids] = slots

        # Step 4: Pack tile data via advanced indexing (NO Python loops).
        # Reshape the full grid to [n_tiles_y, n_tiles_x, tile_size, tile_size]
        # (channel dimension added later if needed).
        # For now, initialize tile data to zeros; caller fills via pack_from_grid.
        self.tiles = torch.zeros(
            (self.k, self.tile_size, self.tile_size, self.n_channels),
            dtype=self.dtype, device=self.device,
        )

        # Precompute tile-row and tile-col for each active tile (for scatter/gather).
        self._tile_rows = (active_ids // self.n_tiles_x).to(torch.int64)  # [k]
        self._tile_cols = (active_ids % self.n_tiles_x).to(torch.int64)   # [k]

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

    @property
    def build_time_ms(self) -> float:
        """Construction time in milliseconds (tile_map + packing)."""
        return self._build_time_ms

    # -----------------------------------------------------------------
    # Pack / unpack
    # -----------------------------------------------------------------

    def pack_from_grid(self, grid: torch.Tensor) -> None:
        """Pack active tile data from a full grid into self.tiles.

        Args:
            grid: tensor of shape [side, side] or [side, side, C].
                  Must be on self.device.
        """
        if grid.dim() == 2:
            grid = grid.unsqueeze(-1)  # [side, side, 1]

        # Reshape to [n_tiles_y, tile_size, n_tiles_x, tile_size, C]
        g = grid.reshape(
            self.n_tiles_y, self.tile_size,
            self.n_tiles_x, self.tile_size,
            grid.shape[-1],
        )
        # Permute to [n_tiles_y, n_tiles_x, tile_size, tile_size, C]
        g = g.permute(0, 2, 1, 3, 4)
        # Flatten tile dimensions: [n_tiles, tile_size, tile_size, C]
        g = g.reshape(self.n_tiles, self.tile_size, self.tile_size, grid.shape[-1])
        # Gather active tiles via advanced indexing.
        self.tiles = g[self.active_tile_ids.long()]  # [k, Ht, Wt, C]

    def scatter_back(self, grid: torch.Tensor) -> torch.Tensor:
        """Write packed tiles back to full grid positions.

        Args:
            grid: tensor [side, side] or [side, side, C] to write into
                  (will be cloned).
        Returns:
            Modified grid with active tile data written back.
        """
        squeezed = False
        if grid.dim() == 2:
            grid = grid.unsqueeze(-1)
            squeezed = True

        out = grid.clone()
        C = grid.shape[-1]

        # Reshape to [n_tiles_y, n_tiles_x, tile_size, tile_size, C]
        out_tiled = out.reshape(
            self.n_tiles_y, self.tile_size,
            self.n_tiles_x, self.tile_size, C,
        ).permute(0, 2, 1, 3, 4).reshape(
            self.n_tiles, self.tile_size, self.tile_size, C,
        )

        # Scatter via advanced indexing.
        out_tiled[self.active_tile_ids.long()] = self.tiles

        # Reshape back.
        result = out_tiled.reshape(
            self.n_tiles_y, self.n_tiles_x,
            self.tile_size, self.tile_size, C,
        ).permute(0, 2, 1, 3, 4).reshape(
            self.side, self.side, C,
        )
        if squeezed:
            result = result.squeeze(-1)
        return result

    # -----------------------------------------------------------------
    # Lookup
    # -----------------------------------------------------------------

    def lookup(self, neighbor_tile_id: torch.Tensor) -> torch.Tensor:
        """Look up slot index for tile id(s). Returns -1 for inactive tiles.

        This is a simple tensor index -- no binary search.

        Args:
            neighbor_tile_id: int tensor of tile ids.
        Returns:
            int32 tensor of slot indices (-1 = inactive / boundary).
        """
        # Clamp to valid range first, then mask out-of-bounds to -1.
        valid = (neighbor_tile_id >= 0) & (neighbor_tile_id < self.n_tiles)
        clamped = neighbor_tile_id.clamp(0, self.n_tiles - 1)
        slots = self.tile_map[clamped.long()]
        # Out-of-bounds tile ids -> slot = -1.
        slots = torch.where(valid, slots, torch.tensor(-1, dtype=torch.int32, device=self.device))
        return slots

    # -----------------------------------------------------------------
    # Neighbor table (prebuilt)
    # -----------------------------------------------------------------

    def build_neighbor_table(
        self,
        offsets: Optional[List[Tuple[int, int]]] = None,
    ) -> torch.Tensor:
        """Precompute neighbor_slots[k, n_offsets] for each active tile.

        For each active tile, for each offset in the stencil:
          - compute neighbor tile_id via row/col arithmetic
          - look up slot via tile_map
        Fully tensorized -- uses broadcasting, no Python loops.

        Args:
            offsets: list of (dy, dx) tile offsets. Defaults to 3x3 stencil
                     (NEIGHBOR_OFFSETS_3x3).
        Returns:
            neighbor_slots tensor [k, n_offsets] int32.
        """
        if offsets is None:
            offsets = NEIGHBOR_OFFSETS_3x3

        # Convert offsets to tensors: [n_offsets]
        dy = torch.tensor([o[0] for o in offsets], dtype=torch.int64, device=self.device)
        dx = torch.tensor([o[1] for o in offsets], dtype=torch.int64, device=self.device)

        # Active tile row/col: [k]
        tile_r = self._tile_rows  # [k] int64
        tile_c = self._tile_cols  # [k] int64

        # Broadcast: neighbor coords [k, n_offsets]
        nr = tile_r.unsqueeze(1) + dy.unsqueeze(0)  # [k, n_offsets]
        nc = tile_c.unsqueeze(1) + dx.unsqueeze(0)  # [k, n_offsets]

        # Boundary check: neighbor outside grid -> invalid.
        in_bounds = (
            (nr >= 0) & (nr < self.n_tiles_y) &
            (nc >= 0) & (nc < self.n_tiles_x)
        )  # [k, n_offsets] bool

        # Compute linear tile id (clamp for safe indexing, then mask).
        nr_safe = nr.clamp(0, self.n_tiles_y - 1)
        nc_safe = nc.clamp(0, self.n_tiles_x - 1)
        neighbor_tid = nr_safe * self.n_tiles_x + nc_safe  # [k, n_offsets]

        # Lookup via tile_map: O(1) per entry.
        slots = self.tile_map[neighbor_tid]  # [k, n_offsets] int32

        # Out-of-bounds -> -1.
        slots = torch.where(
            in_bounds, slots,
            torch.tensor(-1, dtype=torch.int32, device=self.device),
        )

        self.neighbor_slots = slots
        return slots

    # -----------------------------------------------------------------
    # Halo gather
    # -----------------------------------------------------------------

    def gather_halo(
        self,
        halo_w: int = 1,
        use_prebuilt: bool = False,
    ) -> torch.Tensor:
        """Gather neighbor tile data for each active tile.

        Two modes:
          a) on_the_fly: compute neighbor tile_ids, lookup via tile_map, gather.
          b) prebuilt: use pre-computed neighbor_slots table.

        Boundary handling: neighbor outside grid -> slot = -1 -> zero-pad.

        Args:
            halo_w: halo width in tiles (1 = 3x3 stencil).
            use_prebuilt: if True, use self.neighbor_slots (must be built).
        Returns:
            Gathered halo data [k, n_neighbors, tile_size, tile_size, C].
        """
        # Build offsets for the halo stencil.
        offsets = [
            (dy, dx)
            for dy in range(-halo_w, halo_w + 1)
            for dx in range(-halo_w, halo_w + 1)
        ]
        n_offsets = len(offsets)

        if use_prebuilt:
            assert self.neighbor_slots is not None, (
                "Call build_neighbor_table() before using prebuilt mode."
            )
            slots = self.neighbor_slots  # [k, n_offsets] int32
        else:
            # On-the-fly: compute neighbor slots (fully tensorized).
            dy = torch.tensor([o[0] for o in offsets], dtype=torch.int64, device=self.device)
            dx = torch.tensor([o[1] for o in offsets], dtype=torch.int64, device=self.device)

            nr = self._tile_rows.unsqueeze(1) + dy.unsqueeze(0)  # [k, n_offsets]
            nc = self._tile_cols.unsqueeze(1) + dx.unsqueeze(0)  # [k, n_offsets]

            in_bounds = (
                (nr >= 0) & (nr < self.n_tiles_y) &
                (nc >= 0) & (nc < self.n_tiles_x)
            )
            nr_safe = nr.clamp(0, self.n_tiles_y - 1)
            nc_safe = nc.clamp(0, self.n_tiles_x - 1)
            neighbor_tid = nr_safe * self.n_tiles_x + nc_safe

            slots = self.tile_map[neighbor_tid]
            slots = torch.where(
                in_bounds, slots,
                torch.tensor(-1, dtype=torch.int32, device=self.device),
            )

        # Gather tile data from packed storage.
        # self.tiles: [k, Ht, Wt, C]
        # For slots == -1, we gather from a zero tile.
        # Strategy: append a zero tile at index k, remap -1 -> k.
        zero_tile = torch.zeros(
            1, self.tile_size, self.tile_size, self.n_channels,
            dtype=self.dtype, device=self.device,
        )
        tiles_padded = torch.cat([self.tiles, zero_tile], dim=0)  # [k+1, Ht, Wt, C]

        # Remap -1 -> k (the zero-tile index).
        gather_idx = slots.long()
        gather_idx = torch.where(
            gather_idx < 0,
            torch.tensor(self.k, dtype=torch.int64, device=self.device),
            gather_idx,
        )  # [k, n_offsets]

        # Gather: tiles_padded[gather_idx] -> [k, n_offsets, Ht, Wt, C]
        halo_data = tiles_padded[gather_idx.reshape(-1)].reshape(
            self.k, n_offsets, self.tile_size, self.tile_size, self.n_channels,
        )

        return halo_data

    # -----------------------------------------------------------------
    # Memory measurement
    # -----------------------------------------------------------------

    def measure_vram(self) -> Dict[str, int]:
        """Dict of component sizes in bytes."""
        tile_data = self.tiles.nelement() * self.tiles.element_size()
        tile_map = self.tile_map.nelement() * self.tile_map.element_size()
        active_ids = self.active_tile_ids.nelement() * self.active_tile_ids.element_size()
        ns = 0
        if self.neighbor_slots is not None:
            ns = self.neighbor_slots.nelement() * self.neighbor_slots.element_size()
        return {
            "tile_data": tile_data,
            "tile_map": tile_map,
            "active_tile_ids": active_ids,
            "neighbor_slots": ns,
            "total": tile_data + tile_map + active_ids + ns,
        }

    def memory_breakdown(self) -> Dict[str, Any]:
        """Detailed memory breakdown with sizes and metadata."""
        vram = self.measure_vram()
        return {
            **vram,
            "k": self.k,
            "n_tiles": self.n_tiles,
            "tile_size": self.tile_size,
            "n_channels": self.n_channels,
            "tiles_shape": list(self.tiles.shape),
            "tile_map_dtype": str(self.tile_map.dtype),
            "occupancy": self.k / max(1, self.n_tiles),
        }


# =====================================================================
# Timing helpers
# =====================================================================

def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed_median_ms(
    func,
    device: torch.device,
    n_warmup: int = N_WARMUP,
    n_repeat: int = N_REPEAT,
) -> float:
    """Return median wall-clock in ms over n_repeat runs (after warmup).

    Uses torch.cuda.Event for GPU timing when available.
    """
    # Warmup.
    for _ in range(n_warmup):
        func()
        _sync(device)

    if device.type == "cuda":
        # Use CUDA events for precise GPU timing.
        times_ms = []
        for _ in range(n_repeat):
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            func()
            end_ev.record()
            torch.cuda.synchronize(device)
            times_ms.append(start_ev.elapsed_time(end_ev))
        return float(np.median(times_ms))
    else:
        times_ms = []
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            func()
            dt = (time.perf_counter() - t0) * 1e3
            times_ms.append(dt)
        return float(np.median(times_ms))


# =====================================================================
# Benchmark
# =====================================================================

def benchmark_packed_direct(
    side: int,
    sparsity: float,
    pattern: str,
    tile_size: int = 8,
    n_seeds: int = 10,
    device: torch.device = torch.device("cuda"),
    use_prebuilt_neighbors: bool = False,
) -> Dict[str, Any]:
    """Benchmark packed tile layout with direct tile_map lookup.

    Returns dict with:
      - wall_clock_ms: median compute time (20 reps, 5 warmup)
      - peak_vram_bytes: peak VRAM during compute
      - halo_access_ms: halo gather time
      - build_time_ms: tile_map + packing construction time (SEPARATE)
      - memory_breakdown: dict of component sizes
      - n_active_tiles: how many tiles were active
      - n_total_tiles: total tile universe size
    """
    assert side % tile_size == 0

    all_wall_ms: List[float] = []
    all_halo_ms: List[float] = []
    all_build_ms: List[float] = []
    all_peak_vram: List[int] = []
    all_memory: List[Dict[str, Any]] = []
    all_n_active: List[int] = []

    n_tiles_total = (side // tile_size) ** 2

    # Stencil kernel for F.conv2d on packed tiles.
    # tiles are [k, C, Ht, Wt] for conv2d (we'll permute).
    stencil = STENCIL_WEIGHTS.to(device=device, dtype=DTYPE)

    for seed_i in range(n_seeds):
        seed = SEED_BASE + seed_i
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        # Generate element-level mask.
        mask_np = make_mask(side, sparsity, pattern, rng)
        mask_t = torch.tensor(mask_np, dtype=torch.bool, device=device)

        # Create source grid data.
        grid_data = torch.randn(side, side, dtype=DTYPE, device=device)

        # ---- Build layout (timed separately) ----
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

        layout = PackedTileLayout(
            side=side,
            tile_size=tile_size,
            mask=mask_t,
            device=device,
            dtype=DTYPE,
            n_channels=1,
        )
        layout.pack_from_grid(grid_data)

        if layout.k == 0:
            continue

        all_build_ms.append(layout.build_time_ms)
        all_n_active.append(layout.k)

        # Optionally precompute neighbor table.
        if use_prebuilt_neighbors:
            layout.build_neighbor_table()

        # ---- Memory breakdown ----
        all_memory.append(layout.memory_breakdown())

        # ---- Compute kernel: halo gather -> conv2d -> scatter ----
        # Tiles are [k, Ht, Wt, C]; conv2d needs [N, C, H, W].
        def compute_step():
            # Permute tiles to [k, C, Ht, Wt] for conv2d.
            t = layout.tiles.permute(0, 3, 1, 2)  # [k, C, Ht, Wt]
            # Apply 3x3 conv with padding=1 (acts on each tile independently).
            out = F.conv2d(t, stencil, padding=1)  # [k, 1, Ht, Wt]
            # Write result back into layout tiles.
            layout.tiles = out.permute(0, 2, 3, 1)  # [k, Ht, Wt, C]
            return out

        wall_ms = _timed_median_ms(compute_step, device)
        all_wall_ms.append(wall_ms)

        # ---- Halo access timing (separate) ----
        def halo_step():
            return layout.gather_halo(
                halo_w=1, use_prebuilt=use_prebuilt_neighbors,
            )

        halo_ms = _timed_median_ms(halo_step, device)
        all_halo_ms.append(halo_ms)

        # ---- Peak VRAM ----
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(device)
            all_peak_vram.append(peak)

        # Cleanup.
        del layout, grid_data, mask_t
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- Aggregate across seeds ----
    if not all_wall_ms:
        return {
            "side": side,
            "tile_size": tile_size,
            "sparsity": sparsity,
            "pattern": pattern,
            "n_seeds": n_seeds,
            "wall_clock_ms": 0.0,
            "peak_vram_bytes": 0,
            "halo_access_ms": 0.0,
            "build_time_ms": 0.0,
            "memory_breakdown": {},
            "n_active_tiles": 0,
            "n_total_tiles": n_tiles_total,
            "use_prebuilt_neighbors": use_prebuilt_neighbors,
        }

    # Aggregate memory breakdown (median per component).
    agg_memory: Dict[str, Any] = {}
    for key in all_memory[0]:
        vals = [m[key] for m in all_memory]
        if isinstance(vals[0], (int, float)):
            agg_memory[key] = float(np.median(vals))
        elif isinstance(vals[0], list):
            agg_memory[key] = vals[0]  # shape is constant across seeds
        else:
            agg_memory[key] = vals[0]

    return {
        "side": side,
        "tile_size": tile_size,
        "sparsity": sparsity,
        "pattern": pattern,
        "n_seeds": n_seeds,
        "wall_clock_ms": float(np.median(all_wall_ms)),
        "wall_clock_p95_ms": float(np.percentile(all_wall_ms, 95)),
        "peak_vram_bytes": int(np.median(all_peak_vram)) if all_peak_vram else 0,
        "halo_access_ms": float(np.median(all_halo_ms)),
        "halo_access_p95_ms": float(np.percentile(all_halo_ms, 95)),
        "build_time_ms": float(np.median(all_build_ms)),
        "build_time_p95_ms": float(np.percentile(all_build_ms, 95)),
        "memory_breakdown": agg_memory,
        "n_active_tiles": int(np.median(all_n_active)),
        "n_total_tiles": n_tiles_total,
        "use_prebuilt_neighbors": use_prebuilt_neighbors,
    }
