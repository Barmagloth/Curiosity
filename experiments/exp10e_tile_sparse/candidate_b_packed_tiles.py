#!/usr/bin/env python3
"""
Curiosity -- Exp10e, Candidate B: Packed Active Tiles + Sorted Morton Keys

Concept:
  Store ONLY active tiles in a packed array.  Address them via Morton-coded
  tile keys.  Lookup via binary search on sorted keys (torch.searchsorted).
  NO global reverse_map.  O(k) memory, not O(M).

Data structures:
  tile_keys[k]   -- Morton-coded coordinates of active tiles (int64), sorted
  tile_data[k]   -- float values for each active tile

For the 3x3 stencil halo, neighbors are found by:
  1. morton_encode(r+dr, c+dc) for each direction
  2. binary search in sorted tile_keys to check if neighbor is active
  This is O(log k) per lookup vs O(1) for reverse_map, but O(k) memory vs O(M).

For non-grid spaces (graph, tree):
  Use node index as key instead of Morton code; binary search still works.
"""

import time
import json
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch

# =====================================================================
# Configuration
# =====================================================================

SIDES = [32, 64, 128, 256, 512]
SPARSITIES = [0.1, 0.3, 0.5, 0.7, 0.9]
PATTERNS = ["random", "clustered", "checkerboard"]
N_SEEDS = 10
N_WARMUP = 5
N_REPEAT = 20
SEED_BASE = 42
DTYPE = torch.float32

# =====================================================================
# Morton encoding (2D)  -- ported from exp09a, adapted for torch
# =====================================================================

def _part1by1_np(n: np.ndarray) -> np.ndarray:
    """Spread bits of n for 2D Morton interleave (up to 16-bit coords)."""
    n = n.astype(np.uint32)
    n = (n | (n << 8)) & np.uint32(0x00FF00FF)
    n = (n | (n << 4)) & np.uint32(0x0F0F0F0F)
    n = (n | (n << 2)) & np.uint32(0x33333333)
    n = (n | (n << 1)) & np.uint32(0x55555555)
    return n


def morton_encode_2d_np(row: np.ndarray, col: np.ndarray) -> np.ndarray:
    """Return Morton (Z-order) code for 2D coordinates (numpy, uint32)."""
    return _part1by1_np(col) | (_part1by1_np(row) << np.uint32(1))


def _part1by1_torch(n: torch.Tensor) -> torch.Tensor:
    """Spread bits of n for 2D Morton interleave (torch int64)."""
    n = n.long()
    n = (n | (n << 8)) & 0x00FF00FF
    n = (n | (n << 4)) & 0x0F0F0F0F
    n = (n | (n << 2)) & 0x33333333
    n = (n | (n << 1)) & 0x55555555
    return n


def morton_encode_2d_torch(row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
    """Return Morton (Z-order) code for 2D coordinates (torch int64)."""
    return _part1by1_torch(col) | (_part1by1_torch(row) << 1)


# =====================================================================
# Mask generation  (ported from exp09a / exp10, with checkerboard)
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
# PackedTilesLayout
# =====================================================================

class PackedTilesLayout:
    """
    Packed active tiles addressed by sorted Morton keys.

    Memory: O(k) -- tile_keys (int64) + tile_data (float32).
    Lookup: O(log k) via torch.searchsorted on sorted keys.
    No global reverse_map.
    """

    def __init__(self):
        self.tile_keys: Optional[torch.Tensor] = None   # [k] int64, sorted
        self.tile_data: Optional[torch.Tensor] = None    # [k] float32
        self.side: int = 0
        self.k: int = 0
        self.M: int = 0
        self.device: torch.device = torch.device("cpu")
        # Flat indices in row-major order corresponding to sorted Morton keys
        self._flat_idx: Optional[torch.Tensor] = None    # [k] int64

    def build(self, mask_2d: np.ndarray, device: torch.device) -> "PackedTilesLayout":
        """
        Build packed layout from a 2D boolean mask.

        Steps:
          1. Find active elements (row, col)
          2. Compute Morton keys
          3. Sort by Morton key
          4. Store: tile_keys (sorted int64), tile_data (float zeros)
          NO reverse_map.  Total memory = O(k).
        """
        self.side = mask_2d.shape[0]
        self.M = self.side * self.side
        self.device = device

        # Active element coordinates
        active_coords = np.argwhere(mask_2d)  # [k, 2]  (row, col)
        rows_np = active_coords[:, 0]
        cols_np = active_coords[:, 1]
        self.k = len(rows_np)

        # Morton codes (numpy uint32 -> int64 for torch compatibility)
        morton_np = morton_encode_2d_np(rows_np, cols_np).astype(np.int64)

        # Sort by Morton key
        sort_order = np.argsort(morton_np)
        morton_sorted = morton_np[sort_order]
        rows_sorted = rows_np[sort_order]
        cols_sorted = cols_np[sort_order]

        # Flat indices for gather/scatter (row-major)
        flat_idx_np = (rows_sorted * self.side + cols_sorted).astype(np.int64)

        # Transfer to device
        self.tile_keys = torch.tensor(morton_sorted, dtype=torch.int64, device=device)
        self.tile_data = torch.zeros(self.k, dtype=DTYPE, device=device)
        self._flat_idx = torch.tensor(flat_idx_np, dtype=torch.int64, device=device)

        return self

    def lookup(self, query_keys: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Look up Morton keys via binary search.

        Returns:
          indices: positions in tile_data (clamped to [0, k-1])
          found:   bool mask -- True where query_key exists in tile_keys
        """
        # searchsorted gives insertion point; key is present iff
        # tile_keys[pos] == query_key and pos < k
        pos = torch.searchsorted(self.tile_keys, query_keys)
        pos_clamped = pos.clamp(max=self.k - 1)
        found = (pos < self.k) & (self.tile_keys[pos_clamped] == query_keys)
        return pos_clamped, found

    def gather(self, full_data: torch.Tensor) -> torch.Tensor:
        """
        Extract active values from the full grid using precomputed flat
        indices (in Morton-sorted order).
        """
        return full_data.view(-1)[self._flat_idx]

    def compute(self, gathered: torch.Tensor) -> torch.Tensor:
        """
        3x3 conv stencil on packed tiles.

        For each active tile, find its 8 neighbors via Morton neighbor
        calculation + binary search lookup.  Accumulate weighted sum.

        Stencil weights (same as exp10 SyntheticKernel):
          [[0.05, 0.1, 0.05],
           [0.1,  0.4, 0.1 ],
           [0.05, 0.1, 0.05]]
        """
        weights = [
            (-1, -1, 0.05), (-1, 0, 0.1), (-1, 1, 0.05),
            ( 0, -1, 0.1 ), ( 0, 0, 0.4), ( 0, 1, 0.1 ),
            ( 1, -1, 0.05), ( 1, 0, 0.1), ( 1, 1, 0.05),
        ]

        # Decode Morton keys back to (row, col) for neighbor computation.
        # We stored flat_idx = row * side + col, so recover from that.
        rows = self._flat_idx // self.side
        cols = self._flat_idx % self.side

        result = torch.zeros_like(gathered)

        for dr, dc, w in weights:
            if dr == 0 and dc == 0:
                # Center: always present (it's the tile itself)
                result += w * gathered
                continue

            nr = rows + dr
            nc = cols + dc
            # Clamp to grid bounds
            nr = nr.clamp(0, self.side - 1)
            nc = nc.clamp(0, self.side - 1)

            # Morton code for neighbor
            neighbor_keys = morton_encode_2d_torch(nr, nc)

            # Binary search lookup
            pos, found = self.lookup(neighbor_keys)

            # Gather neighbor values (0 where not found)
            neighbor_vals = torch.where(found, gathered[pos], torch.zeros(1, dtype=DTYPE, device=self.device))
            result += w * neighbor_vals

        return result

    def scatter(self, result: torch.Tensor, full_data: torch.Tensor) -> torch.Tensor:
        """Write computed results back to the full grid at active positions."""
        output = full_data.clone()
        output.view(-1)[self._flat_idx] = result
        return output

    def halo_access(self, full_data: torch.Tensor, halo_w: int = 1) -> torch.Tensor:
        """
        For each active tile, gather halo neighbors within a
        (2*halo_w+1) x (2*halo_w+1) stencil via Morton-key binary search.

        Returns sum of all accessed halo values (for benchmarking).
        """
        rows = self._flat_idx // self.side
        cols = self._flat_idx % self.side
        gathered = self.tile_data  # current tile data

        total = torch.zeros(self.k, dtype=DTYPE, device=self.device)

        for dr in range(-halo_w, halo_w + 1):
            for dc in range(-halo_w, halo_w + 1):
                nr = (rows + dr).clamp(0, self.side - 1)
                nc = (cols + dc).clamp(0, self.side - 1)

                neighbor_keys = morton_encode_2d_torch(nr, nc)
                pos, found = self.lookup(neighbor_keys)

                # Read from tile_data for active neighbors; 0 otherwise
                vals = torch.where(
                    found,
                    self.tile_data[pos],
                    torch.zeros(1, dtype=DTYPE, device=self.device),
                )
                total += vals

        return total

    def measure_vram(self) -> int:
        """Peak VRAM allocated (CUDA only)."""
        if self.device.type != "cuda":
            return 0
        return torch.cuda.max_memory_allocated(self.device)

    def memory_bytes(self) -> Dict[str, int]:
        """
        Breakdown of layout memory.
        Only tile_keys + tile_data (+ _flat_idx as auxiliary).
        No reverse_map.
        """
        keys_bytes = self.tile_keys.nelement() * self.tile_keys.element_size()
        data_bytes = self.tile_data.nelement() * self.tile_data.element_size()
        flat_bytes = self._flat_idx.nelement() * self._flat_idx.element_size()
        return {
            "tile_keys": keys_bytes,
            "tile_data": data_bytes,
            "flat_idx": flat_bytes,
            "total": keys_bytes + data_bytes + flat_bytes,
            "k": self.k,
            "M": self.M,
            "ratio_vs_full": (keys_bytes + data_bytes + flat_bytes)
                             / max(1, self.M * 4 + self.M),  # float32 + bool mask
        }


# =====================================================================
# Timing / VRAM helpers
# =====================================================================

def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_runs(func, n_warmup: int, n_repeat: int,
               device: torch.device) -> np.ndarray:
    """Return array of wall-clock seconds for *n_repeat* runs after warmup."""
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
# Benchmark
# =====================================================================

def benchmark_candidate_b(
    side: int,
    sparsity: float,
    pattern: str,
    n_seeds: int = N_SEEDS,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """
    Benchmark Candidate B (packed tiles + sorted Morton keys).

    Measures:
      - build time
      - gather/compute/scatter wall-clock (median of N_REPEAT, N_WARMUP warmup)
      - halo access time
      - peak VRAM (CUDA only)
      - memory breakdown

    Returns dict with all metrics.
    """
    M = side * side
    all_build_ms = []
    all_gcs_ms = []
    all_halo_ms = []
    all_peak_vram = []
    all_mem_breakdown = []

    for seed_i in range(n_seeds):
        seed = SEED_BASE + seed_i
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        mask_2d = make_mask(side, sparsity, pattern, rng)
        k = int(mask_2d.sum())
        if k == 0:
            continue

        # -- Build layout --
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

        t0 = time.perf_counter()
        layout = PackedTilesLayout().build(mask_2d, device)
        _sync(device)
        build_s = time.perf_counter() - t0
        all_build_ms.append(build_s * 1e3)

        # -- Memory breakdown --
        mem = layout.memory_bytes()
        all_mem_breakdown.append(mem)

        # -- Create full-grid field --
        field = torch.randn(M, dtype=DTYPE, device=device)

        # -- Gather / compute / scatter timing --
        def gcs_step():
            gathered = layout.gather(field)
            result = layout.compute(gathered)
            return layout.scatter(result, field)

        gcs_times = timed_runs(gcs_step, N_WARMUP, N_REPEAT, device)
        all_gcs_ms.append(float(np.median(gcs_times) * 1e3))

        # -- Halo access timing --
        # Fill tile_data so halo_access reads real values
        layout.tile_data = layout.gather(field)

        def halo_step():
            return layout.halo_access(field, halo_w=1)

        halo_times = timed_runs(halo_step, N_WARMUP, N_REPEAT, device)
        all_halo_ms.append(float(np.median(halo_times) * 1e3))

        # -- Peak VRAM --
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(device)
            all_peak_vram.append(peak)

        # Cleanup
        del layout, field
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # -- Aggregate across seeds --
    result = {
        "side": side,
        "M": M,
        "sparsity": sparsity,
        "pattern": pattern,
        "n_seeds": n_seeds,
        "build_ms": {
            "median": float(np.median(all_build_ms)),
            "p95": float(np.percentile(all_build_ms, 95)),
        },
        "gcs_ms": {
            "median": float(np.median(all_gcs_ms)),
            "p95": float(np.percentile(all_gcs_ms, 95)),
        },
        "halo_ms": {
            "median": float(np.median(all_halo_ms)),
            "p95": float(np.percentile(all_halo_ms, 95)),
        },
        "memory": {
            "tile_keys_bytes": int(np.median([m["tile_keys"] for m in all_mem_breakdown])),
            "tile_data_bytes": int(np.median([m["tile_data"] for m in all_mem_breakdown])),
            "flat_idx_bytes": int(np.median([m["flat_idx"] for m in all_mem_breakdown])),
            "total_bytes": int(np.median([m["total"] for m in all_mem_breakdown])),
            "k_median": int(np.median([m["k"] for m in all_mem_breakdown])),
            "ratio_vs_full": float(np.median([m["ratio_vs_full"] for m in all_mem_breakdown])),
        },
    }

    # VRAM (CUDA only)
    if all_peak_vram:
        result["peak_vram"] = {
            "median_bytes": int(np.median(all_peak_vram)),
            "max_bytes": int(np.max(all_peak_vram)),
        }

    # Reference: what a full grid + bool mask would cost
    full_grid_bytes = M * 4 + M  # float32 data + bool mask
    result["reference_full_grid_bytes"] = full_grid_bytes
    result["memory_saving_vs_full"] = 1.0 - result["memory"]["total_bytes"] / max(1, full_grid_bytes)

    return result


# =====================================================================
# Runner
# =====================================================================

def run_all(device_str: str = "cuda"):
    """Run candidate B benchmark across the full sweep."""
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(__file__).parent
    all_results = {
        "candidate": "B_packed_tiles",
        "device": str(device),
        "config": {
            "sides": SIDES,
            "sparsities": SPARSITIES,
            "patterns": PATTERNS,
            "n_seeds": N_SEEDS,
            "n_warmup": N_WARMUP,
            "n_repeat": N_REPEAT,
            "seed_base": SEED_BASE,
        },
        "runs": [],
    }

    for side in SIDES:
        for sp in SPARSITIES:
            for pat in PATTERNS:
                print(f"  side={side:4d}  sp={sp:.1f}  pat={pat:13s} ...", end="", flush=True)
                res = benchmark_candidate_b(side, sp, pat, N_SEEDS, device)
                all_results["runs"].append(res)

                mem_ratio = res["memory"]["ratio_vs_full"]
                gcs = res["gcs_ms"]["median"]
                halo = res["halo_ms"]["median"]
                print(f"  gcs={gcs:8.3f}ms  halo={halo:8.3f}ms  "
                      f"mem_ratio={mem_ratio:.3f}  k={res['memory']['k_median']}")

    # Save
    json_path = output_dir / "candidate_b_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # Summary
    _print_summary(all_results)
    return all_results


def _print_summary(results: dict):
    """Print condensed summary."""
    print("\n" + "=" * 70)
    print("CANDIDATE B SUMMARY: Packed Active Tiles + Sorted Morton Keys")
    print("=" * 70)

    runs = results["runs"]

    # Memory savings by sparsity
    print("\n-- Memory savings vs full grid (1 - total/full) --")
    for sp in SPARSITIES:
        entries = [r for r in runs if r["sparsity"] == sp]
        savings = [r["memory_saving_vs_full"] for r in entries]
        print(f"  sparsity={sp:.1f}: median saving = {np.median(savings)*100:.1f}%, "
              f"min = {np.min(savings)*100:.1f}%")

    # GCS time by side
    print("\n-- Gather/Compute/Scatter time (median ms) --")
    for side in SIDES:
        entries = [r for r in runs if r["side"] == side]
        times = [r["gcs_ms"]["median"] for r in entries]
        print(f"  side={side:4d}: median = {np.median(times):.3f}ms, "
              f"p95 = {np.percentile(times, 95):.3f}ms")

    # Halo access by side
    print("\n-- Halo access time (median ms) --")
    for side in SIDES:
        entries = [r for r in runs if r["side"] == side]
        times = [r["halo_ms"]["median"] for r in entries]
        print(f"  side={side:4d}: median = {np.median(times):.3f}ms, "
              f"p95 = {np.percentile(times, 95):.3f}ms")

    # Key insight: memory ratio
    print("\n-- Key insight: O(k) memory, no reverse_map --")
    for sp in SPARSITIES:
        entries = [r for r in runs if r["sparsity"] == sp]
        ratios = [r["memory"]["ratio_vs_full"] for r in entries]
        print(f"  sparsity={sp:.1f}: layout/full = {np.median(ratios):.3f}x")


if __name__ == "__main__":
    import sys
    device_str = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    run_all(device_str)
