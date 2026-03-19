#!/usr/bin/env python3
"""
Curiosity -- Exp10f: Packed Tile Storage with GPU Hash Table Lookup

Motivation:
  Alternative to packed_tile_layout.py's direct tile_map[tile_id] -> slot array.
  When the tile universe becomes large or irregular (e.g., multi-resolution,
  sparse octrees), a direct-mapped array wastes memory.  An open-addressing
  hash table keeps O(k) storage regardless of the tile-id range.

  For current scales (1024 tiles) hash is overkill, but we test it for
  completeness and to have a reference implementation for future work.

Data structures:
  tiles[k, Ht, Wt, C]         -- only active tiles, packed contiguously
  hash_keys[capacity] int32    -- tile keys (-1 = empty slot)
  hash_vals[capacity] int32    -- packed slot indices (-1 = empty)
  active_tile_ids[k] int32     -- slot -> tile_id (canonical order)

Hash function: h(key) = (key * 2654435761) >> shift   (Knuth multiplicative)
Collision resolution: linear probing
Load factor: 0.5 (capacity = next_power_of_2(2 * k), minimum 16)
"""

import time
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# =====================================================================
# Configuration (shared with packed_tile_layout.py)
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

# Knuth multiplicative hash constant (golden-ratio derived, 32-bit).
_HASH_MULTIPLIER = 2654435761


# =====================================================================
# Mask generation (identical to packed_tile_layout.py)
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
# Hash helpers
# =====================================================================

def _next_power_of_2(n: int) -> int:
    """Smallest power of 2 >= n."""
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


def _hash_fn(keys: torch.Tensor, shift: int) -> torch.Tensor:
    """Knuth multiplicative hash: h(key) = ((key * M) & 0xFFFFFFFF) >> shift.

    Operates on int64 to avoid overflow issues with int32 multiplication.
    Returns int64 indices into the hash table.
    """
    # Work in int64 to avoid sign issues.  Mask to 32-bit before shift.
    prod = keys.long() * _HASH_MULTIPLIER
    return (prod & 0xFFFFFFFF) >> shift


def _compute_shift(capacity: int) -> int:
    """Compute right-shift so hash output is in [0, capacity).

    capacity must be a power of 2.
    """
    # shift = 32 - log2(capacity)
    return 32 - int(np.log2(capacity))


# =====================================================================
# HashTileLayout
# =====================================================================

class HashTileLayout:
    """Packed active tiles with open-addressing hash table lookup.

    Storage: tiles[k, Ht, Wt, C] -- only active tiles
    Index:   hash_keys[capacity] int32 -- tile keys (-1 = empty)
             hash_vals[capacity] int32 -- slot indices
    Reverse: active_tile_ids[k] int32 -- slot -> tile_id

    Hash function: h(key) = (key * 2654435761) >> shift  (Knuth multiplicative)
    Collision: linear probing
    Load factor: 0.5 (capacity = 2 * k, rounded up to power of 2)
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

        # -- Build (timed separately) --
        t0 = time.perf_counter()
        self._build(mask)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self._build_time_ms = (time.perf_counter() - t0) * 1e3

        # Neighbor table (built on demand)
        self.neighbor_slots: Optional[torch.Tensor] = None  # [k, n_offsets] int32

    def _build(self, mask: torch.Tensor) -> None:
        """Build hash table and pack tiles."""
        mask_dev = mask.to(device=self.device, dtype=torch.bool)

        # Step 1: Compute tile activity.
        mask_tiled = mask_dev.reshape(
            self.n_tiles_y, self.tile_size,
            self.n_tiles_x, self.tile_size,
        )
        tile_active = mask_tiled.any(dim=3).any(dim=1)  # [n_tiles_y, n_tiles_x]

        # Step 2: Get active tile ids (row-major canonical order).
        active_ids = tile_active.flatten().nonzero(as_tuple=False).squeeze(1)
        self.k = active_ids.shape[0]
        self.active_tile_ids = active_ids.to(dtype=torch.int32)  # [k]

        # Step 3: Build open-addressing hash table.
        self._build_hash_table()

        # Step 4: Pack tile data (initialized to zeros; caller fills via pack_from_grid).
        self.tiles = torch.zeros(
            (self.k, self.tile_size, self.tile_size, self.n_channels),
            dtype=self.dtype, device=self.device,
        )

        # Precompute tile-row and tile-col for neighbor computation.
        self._tile_rows = (active_ids // self.n_tiles_x).to(torch.int64)
        self._tile_cols = (active_ids % self.n_tiles_x).to(torch.int64)

    def _build_hash_table(self) -> None:
        """Build the open-addressing hash table from active_tile_ids.

        Uses sorted insertion order for determinism.  Attempts vectorized
        insertion with a Python loop only for collision rounds.
        """
        if self.k == 0:
            self.capacity = 16
            self.shift = _compute_shift(16)
            self.hash_keys = torch.full(
                (16,), -1, dtype=torch.int32, device=self.device,
            )
            self.hash_vals = torch.full(
                (16,), -1, dtype=torch.int32, device=self.device,
            )
            return

        # Capacity: next power of 2 >= 2*k, minimum 16.
        raw_cap = max(16, 2 * self.k)
        self.capacity = _next_power_of_2(raw_cap)
        self.shift = _compute_shift(self.capacity)
        cap_mask = self.capacity - 1  # for fast modulo (capacity is power of 2)

        self.hash_keys = torch.full(
            (self.capacity,), -1, dtype=torch.int32, device=self.device,
        )
        self.hash_vals = torch.full(
            (self.capacity,), -1, dtype=torch.int32, device=self.device,
        )

        # Keys to insert (already sorted since active_tile_ids comes from nonzero).
        keys = self.active_tile_ids.long()        # [k] int64
        slots = torch.arange(self.k, dtype=torch.int32, device=self.device)

        # Vectorized insertion with collision rounds.
        # Each round: compute hash positions for remaining keys, insert where
        # the slot is empty, advance colliders by linear probing.
        remaining_mask = torch.ones(self.k, dtype=torch.bool, device=self.device)
        probe_offsets = torch.zeros(self.k, dtype=torch.int64, device=self.device)

        # Maximum probes = capacity (guaranteed to terminate at load < 1).
        for _ in range(self.capacity):
            if not remaining_mask.any():
                break

            # Current keys and slots still needing insertion.
            cur_keys = keys[remaining_mask]            # [r]
            cur_slots = slots[remaining_mask]           # [r]
            cur_offsets = probe_offsets[remaining_mask]  # [r]

            # Compute target positions: (hash(key) + offset) & cap_mask
            h = _hash_fn(cur_keys, self.shift)
            pos = ((h + cur_offsets) & cap_mask).long()  # [r]

            # Check which positions are empty.
            target_keys = self.hash_keys[pos]  # [r] int32
            is_empty = (target_keys == -1)     # [r] bool

            # For positions with collisions among cur_keys themselves (duplicate
            # target positions), only the first occurrence wins this round.
            # Use scatter to detect duplicates: mark first-seen positions.
            if is_empty.any():
                empty_pos = pos[is_empty]                 # positions that are empty
                empty_keys = cur_keys[is_empty].int()     # keys to insert
                empty_slots = cur_slots[is_empty]         # slot values

                # Handle duplicate target positions within this batch:
                # only the first key mapping to each position gets inserted.
                unique_pos, inverse = torch.unique(empty_pos, return_inverse=True)
                # For each unique position, pick the first occurrence.
                # scatter_: last write wins, but keys are sorted so this is
                # deterministic (largest key at each position wins).
                # Instead, use a min-based approach for determinism.
                first_idx = torch.full(
                    (unique_pos.shape[0],), empty_pos.shape[0],
                    dtype=torch.int64, device=self.device,
                )
                # scatter_reduce with min to find first index per unique position.
                src_indices = torch.arange(
                    empty_pos.shape[0], dtype=torch.int64, device=self.device,
                )
                first_idx.scatter_reduce_(
                    0, inverse, src_indices, reduce="amin",
                    include_self=False,
                )

                # Insert only the winners.
                win_pos = empty_pos[first_idx]
                win_keys = empty_keys[first_idx]
                win_slots = empty_slots[first_idx]

                self.hash_keys[win_pos.long()] = win_keys
                self.hash_vals[win_pos.long()] = win_slots

                # Mark winners as done in the full remaining set.
                # Map back: which indices in the remaining set were winners?
                remaining_indices = torch.where(remaining_mask)[0]  # [r]
                empty_in_remaining = torch.where(is_empty)[0]       # indices in [r]
                winner_in_remaining = empty_in_remaining[first_idx]  # indices in [r]
                winner_in_full = remaining_indices[winner_in_remaining]
                remaining_mask[winner_in_full] = False

            # Advance probe offset for all still-remaining keys.
            probe_offsets[remaining_mask] += 1

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

    @property
    def build_time_ms(self) -> float:
        """Construction time in milliseconds (hash table + packing)."""
        return self._build_time_ms

    # -----------------------------------------------------------------
    # Pack / unpack
    # -----------------------------------------------------------------

    def pack_from_grid(self, grid: torch.Tensor) -> None:
        """Pack active tile data from a full grid into self.tiles.

        Args:
            grid: tensor of shape [side, side] or [side, side, C].
        """
        if grid.dim() == 2:
            grid = grid.unsqueeze(-1)

        g = grid.reshape(
            self.n_tiles_y, self.tile_size,
            self.n_tiles_x, self.tile_size,
            grid.shape[-1],
        )
        g = g.permute(0, 2, 1, 3, 4)
        g = g.reshape(self.n_tiles, self.tile_size, self.tile_size, grid.shape[-1])
        self.tiles = g[self.active_tile_ids.long()]

    def scatter_back(self, grid: torch.Tensor) -> torch.Tensor:
        """Write packed tiles back to full grid positions.

        Args:
            grid: tensor [side, side] or [side, side, C] to write into (cloned).
        Returns:
            Modified grid with active tile data written back.
        """
        squeezed = False
        if grid.dim() == 2:
            grid = grid.unsqueeze(-1)
            squeezed = True

        out = grid.clone()
        C = grid.shape[-1]

        out_tiled = out.reshape(
            self.n_tiles_y, self.tile_size,
            self.n_tiles_x, self.tile_size, C,
        ).permute(0, 2, 1, 3, 4).reshape(
            self.n_tiles, self.tile_size, self.tile_size, C,
        )

        out_tiled[self.active_tile_ids.long()] = self.tiles

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
    # Lookup (vectorized hash probe -- hot path)
    # -----------------------------------------------------------------

    def lookup(self, tile_ids: torch.Tensor) -> torch.Tensor:
        """Look up slot indices for tile id(s) via hash probing.

        Fully vectorized: fixed max_probes iterations, no Python data-dependent
        branching in the hot path.

        Args:
            tile_ids: int tensor of tile ids (any shape).
        Returns:
            int32 tensor of slot indices (-1 = not found), same shape.
        """
        if self.k == 0:
            return torch.full_like(tile_ids, -1, dtype=torch.int32)

        orig_shape = tile_ids.shape
        flat_ids = tile_ids.reshape(-1).long()  # [N]
        cap_mask = self.capacity - 1
        max_probes = min(32, self.capacity)

        # Initial hash positions.
        h = _hash_fn(flat_ids, self.shift)         # [N]
        result = torch.full(
            (flat_ids.shape[0],), -1, dtype=torch.int32, device=self.device,
        )
        still_searching = torch.ones(
            flat_ids.shape[0], dtype=torch.bool, device=self.device,
        )

        for probe in range(max_probes):
            if not still_searching.any():
                break

            pos = ((h + probe) & cap_mask).long()  # [N]

            # Only probe for elements still searching.
            cur_pos = pos[still_searching]
            cur_ids = flat_ids[still_searching]

            table_keys = self.hash_keys[cur_pos]   # int32
            table_vals = self.hash_vals[cur_pos]   # int32

            # Found: key matches.
            found = (table_keys.long() == cur_ids)

            # Empty slot: key is -1, meaning the key was never inserted.
            empty = (table_keys == -1)

            # Update results for found keys.
            if found.any():
                active_indices = torch.where(still_searching)[0]
                found_global = active_indices[found]
                result[found_global] = table_vals[found]

            # Stop searching for found or empty-slot keys.
            done = found | empty
            if done.any():
                active_indices = torch.where(still_searching)[0]
                done_global = active_indices[done]
                still_searching[done_global] = False

        return result.reshape(orig_shape)

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
          - look up slot via hash table
        Fully tensorized via broadcasting + vectorized hash lookup.

        Args:
            offsets: list of (dy, dx) tile offsets. Defaults to 3x3 stencil.
        Returns:
            neighbor_slots tensor [k, n_offsets] int32.
        """
        if offsets is None:
            offsets = NEIGHBOR_OFFSETS_3x3

        dy = torch.tensor([o[0] for o in offsets], dtype=torch.int64, device=self.device)
        dx = torch.tensor([o[1] for o in offsets], dtype=torch.int64, device=self.device)

        tile_r = self._tile_rows  # [k]
        tile_c = self._tile_cols  # [k]

        # Broadcast: neighbor coords [k, n_offsets]
        nr = tile_r.unsqueeze(1) + dy.unsqueeze(0)
        nc = tile_c.unsqueeze(1) + dx.unsqueeze(0)

        # Boundary check.
        in_bounds = (
            (nr >= 0) & (nr < self.n_tiles_y) &
            (nc >= 0) & (nc < self.n_tiles_x)
        )

        # Compute linear tile id (clamp for safe indexing, then mask).
        nr_safe = nr.clamp(0, self.n_tiles_y - 1)
        nc_safe = nc.clamp(0, self.n_tiles_x - 1)
        neighbor_tid = (nr_safe * self.n_tiles_x + nc_safe).int()  # [k, n_offsets]

        # Lookup via hash table (vectorized).
        slots = self.lookup(neighbor_tid)  # [k, n_offsets] int32

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
          a) on_the_fly: compute neighbor tile_ids, lookup via hash, gather.
          b) prebuilt: use pre-computed neighbor_slots table.

        Boundary handling: neighbor outside grid -> slot = -1 -> zero-pad.

        Args:
            halo_w: halo width in tiles (1 = 3x3 stencil).
            use_prebuilt: if True, use self.neighbor_slots (must be built).
        Returns:
            Gathered halo data [k, n_neighbors, tile_size, tile_size, C].
        """
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
            # On-the-fly: compute neighbor slots via hash lookup.
            dy = torch.tensor([o[0] for o in offsets], dtype=torch.int64, device=self.device)
            dx = torch.tensor([o[1] for o in offsets], dtype=torch.int64, device=self.device)

            nr = self._tile_rows.unsqueeze(1) + dy.unsqueeze(0)
            nc = self._tile_cols.unsqueeze(1) + dx.unsqueeze(0)

            in_bounds = (
                (nr >= 0) & (nr < self.n_tiles_y) &
                (nc >= 0) & (nc < self.n_tiles_x)
            )
            nr_safe = nr.clamp(0, self.n_tiles_y - 1)
            nc_safe = nc.clamp(0, self.n_tiles_x - 1)
            neighbor_tid = (nr_safe * self.n_tiles_x + nc_safe).int()

            slots = self.lookup(neighbor_tid)
            slots = torch.where(
                in_bounds, slots,
                torch.tensor(-1, dtype=torch.int32, device=self.device),
            )

        # Gather tile data from packed storage.
        # Append a zero tile at index k; remap -1 -> k.
        zero_tile = torch.zeros(
            1, self.tile_size, self.tile_size, self.n_channels,
            dtype=self.dtype, device=self.device,
        )
        tiles_padded = torch.cat([self.tiles, zero_tile], dim=0)  # [k+1, ...]

        gather_idx = slots.long()
        gather_idx = torch.where(
            gather_idx < 0,
            torch.tensor(self.k, dtype=torch.int64, device=self.device),
            gather_idx,
        )

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
        hash_keys = self.hash_keys.nelement() * self.hash_keys.element_size()
        hash_vals = self.hash_vals.nelement() * self.hash_vals.element_size()
        active_ids = self.active_tile_ids.nelement() * self.active_tile_ids.element_size()
        ns = 0
        if self.neighbor_slots is not None:
            ns = self.neighbor_slots.nelement() * self.neighbor_slots.element_size()
        return {
            "tile_data": tile_data,
            "hash_keys": hash_keys,
            "hash_vals": hash_vals,
            "active_tile_ids": active_ids,
            "neighbor_slots": ns,
            "total": tile_data + hash_keys + hash_vals + active_ids + ns,
        }

    def memory_breakdown(self) -> Dict[str, Any]:
        """Detailed memory breakdown with sizes and metadata."""
        vram = self.measure_vram()
        return {
            **vram,
            "k": self.k,
            "n_tiles": self.n_tiles,
            "capacity": self.capacity,
            "tile_size": self.tile_size,
            "n_channels": self.n_channels,
            "tiles_shape": list(self.tiles.shape),
            "hash_keys_dtype": str(self.hash_keys.dtype),
            "load_factor": self.k / max(1, self.capacity),
            "occupancy": self.k / max(1, self.n_tiles),
        }


# =====================================================================
# Timing helpers (same as packed_tile_layout.py)
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
    for _ in range(n_warmup):
        func()
        _sync(device)

    if device.type == "cuda":
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

def benchmark_hash_lookup(
    side: int,
    sparsity: float,
    pattern: str,
    tile_size: int = 8,
    n_seeds: int = 10,
    device: torch.device = torch.device("cuda"),
    use_prebuilt_neighbors: bool = False,
) -> Dict[str, Any]:
    """Benchmark packed tile layout with hash table lookup.

    Same interface as benchmark_packed_direct but with hash table.

    Returns dict with:
      - wall_clock_ms: median compute time (20 reps, 5 warmup)
      - peak_vram_bytes: peak VRAM during compute
      - halo_access_ms: halo gather time
      - build_time_ms: hash table + packing construction time (SEPARATE)
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

        layout = HashTileLayout(
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

        # ---- Compute kernel: conv2d on packed tiles ----
        def compute_step():
            t = layout.tiles.permute(0, 3, 1, 2)  # [k, C, Ht, Wt]
            out = F.conv2d(t, stencil, padding=1)  # [k, 1, Ht, Wt]
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
            agg_memory[key] = vals[0]
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
