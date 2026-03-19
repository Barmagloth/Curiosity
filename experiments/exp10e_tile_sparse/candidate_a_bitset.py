#!/usr/bin/env python3
"""
Curiosity -- Exp10e, Candidate A: Dense Grid + Bitset Mask

Improved baseline for the tile-sparse layout comparison.  Same as exp10's
grid layout but replaces the byte-per-element bool mask with a packed
uint8 bitset (1 bit per element = 8x smaller mask).  Everything else
(data tensor, 3x3 conv stencil, gather/scatter logic) is identical to
the exp10 grid path.

Classes:
    BitsetMask       -- packed uint8 bitset with set/get/pack/unpack/popcount
    GridBitsetLayout -- full-grid layout using BitsetMask instead of bool mask

Entry point:
    benchmark_candidate_a(side, sparsity, pattern, n_seeds, device)
"""

import math
import time
from typing import Any, Dict

import numpy as np
import torch


# ======================================================================
# Configuration
# ======================================================================

N_WARMUP = 5
N_REPEAT = 20
SEED_BASE = 42
DTYPE = torch.float32


# ======================================================================
# Mask generation  (from exp09a / exp10, extended with checkerboard)
# ======================================================================

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


# ======================================================================
# BitsetMask
# ======================================================================

class BitsetMask:
    """Packed bitset stored as a uint8 tensor (1 bit per element).

    Parameters
    ----------
    shape : int
        Total number of elements (flat).
    device : torch.device or str
        Where to allocate the backing tensor.
    """

    def __init__(self, shape: int,
                 device: torch.device | str = "cpu") -> None:
        self.n = shape
        self.n_bytes = math.ceil(shape / 8)
        self.device = torch.device(device)
        self.data = torch.zeros(self.n_bytes, dtype=torch.uint8,
                                device=self.device)

    # -- single-element ops -------------------------------------------

    def set_bit(self, idx: int) -> None:
        """Set bit at flat index *idx* to 1."""
        byte_idx = idx >> 3
        bit_idx = idx & 7
        self.data[byte_idx] = self.data[byte_idx] | torch.tensor(
            1 << bit_idx, dtype=torch.uint8, device=self.device)

    def clear_bit(self, idx: int) -> None:
        """Set bit at flat index *idx* to 0."""
        byte_idx = idx >> 3
        bit_idx = idx & 7
        self.data[byte_idx] = self.data[byte_idx] & torch.tensor(
            ~(1 << bit_idx) & 0xFF, dtype=torch.uint8, device=self.device)

    def get_bit(self, idx: int) -> bool:
        """Return True if bit at flat index *idx* is set."""
        byte_idx = idx >> 3
        bit_idx = idx & 7
        return bool((self.data[byte_idx].item() >> bit_idx) & 1)

    # -- bulk conversions ---------------------------------------------

    @classmethod
    def from_bool_mask(cls, bool_mask: torch.Tensor,
                       device: torch.device | str | None = None
                       ) -> "BitsetMask":
        """Pack a flat bool tensor into a BitsetMask.

        The packing is vectorised: groups of 8 bools are combined into
        one uint8 using positional bit-shifts.
        """
        flat = bool_mask.reshape(-1)
        n = flat.shape[0]
        target_device = torch.device(device) if device is not None else flat.device

        bs = cls.__new__(cls)
        bs.n = n
        bs.n_bytes = math.ceil(n / 8)
        bs.device = target_device

        # Pad to a multiple of 8 so reshape works
        padded_len = bs.n_bytes * 8
        if padded_len != n:
            padded = torch.zeros(padded_len, dtype=torch.uint8,
                                 device=flat.device)
            padded[:n] = flat.to(torch.uint8)
        else:
            padded = flat.to(torch.uint8)

        groups = padded.reshape(-1, 8)  # [n_bytes, 8]
        shifts = torch.arange(8, dtype=torch.uint8, device=flat.device)
        packed = (groups << shifts).sum(dim=1).to(torch.uint8)
        bs.data = packed.to(target_device)
        return bs

    def to_bool_mask(self) -> torch.Tensor:
        """Unpack bitset back to a flat bool tensor of length *self.n*."""
        shifts = torch.arange(8, dtype=torch.uint8, device=self.device)
        # Broadcast: [n_bytes, 1] >> [8] -> [n_bytes, 8]
        unpacked = ((self.data.unsqueeze(1) >> shifts) & 1).reshape(-1)
        return unpacked[:self.n].to(torch.bool)

    # -- analytics ----------------------------------------------------

    def count_active(self) -> int:
        """Popcount: number of set bits."""
        # Unpack and sum (simple, portable, no custom CUDA kernel needed)
        return int(self.to_bool_mask().sum().item())

    def memory_bytes(self) -> int:
        """Actual memory consumed by the backing uint8 tensor."""
        return self.data.nelement() * self.data.element_size()


# ======================================================================
# GridBitsetLayout
# ======================================================================

class GridBitsetLayout:
    """Full-grid layout with a packed bitset mask.

    Allocates a data tensor of size M (full grid) and a bitset mask of
    size ceil(M/8).  The key difference from exp10's GridLayout is that
    the mask is 8x smaller.
    """

    def __init__(self) -> None:
        self.bitset: BitsetMask | None = None
        self.data: torch.Tensor | None = None
        self.side: int = 0
        self.M: int = 0
        self.device: torch.device = torch.device("cpu")
        # Cached bool expansion (lazily created)
        self._bool_mask: torch.Tensor | None = None
        # 3x3 stencil kernel
        self._stencil: torch.Tensor | None = None

    def build(self, mask_2d: np.ndarray,
              device: torch.device | str = "cpu") -> "GridBitsetLayout":
        """Construct the layout from a 2-D numpy bool mask."""
        self.device = torch.device(device)
        self.side = mask_2d.shape[0]
        self.M = self.side * self.side

        mask_flat_t = torch.tensor(mask_2d.ravel(), dtype=torch.bool,
                                   device=self.device)
        self.bitset = BitsetMask.from_bool_mask(mask_flat_t, device=self.device)
        self.data = torch.zeros(self.M, dtype=DTYPE, device=self.device)

        # Pre-build stencil for compute step
        self._stencil = torch.tensor(
            [[0.05, 0.1, 0.05],
             [0.1,  0.4, 0.1],
             [0.05, 0.1, 0.05]],
            dtype=DTYPE, device=self.device,
        ).reshape(1, 1, 3, 3)

        # Cache the bool expansion so gather/scatter don't pay unpack cost
        # every iteration during benchmarks (the mask is static).
        self._bool_mask = self.bitset.to_bool_mask()

        return self

    # -- gather / compute / scatter -----------------------------------

    def gather(self, data: torch.Tensor) -> torch.Tensor:
        """Return values at active positions (using bool expansion)."""
        return data[self._bool_mask]

    def compute(self, gathered: torch.Tensor) -> torch.Tensor:
        """Apply 3x3 conv stencil (same synthetic kernel as exp10).

        Because the stencil needs spatial neighbours we scatter the
        gathered values back to a full grid, convolve, then re-gather.
        This faithfully mirrors the exp10 grid compute path.
        """
        # Scatter gathered values into a full-size temporary
        tmp = torch.zeros(self.M, dtype=DTYPE, device=self.device)
        tmp[self._bool_mask] = gathered
        tmp_2d = tmp.reshape(1, 1, self.side, self.side)
        out_2d = torch.nn.functional.conv2d(tmp_2d, self._stencil, padding=1)
        result_full = out_2d.reshape(self.M)
        # Re-gather active outputs
        return result_full[self._bool_mask]

    def scatter(self, result: torch.Tensor,
                data: torch.Tensor) -> torch.Tensor:
        """Write *result* (active-only) back into a clone of *data*."""
        output = data.clone()
        output[self._bool_mask] = result
        return output

    # -- full gather-compute-scatter pass -----------------------------

    def run(self, field: torch.Tensor) -> torch.Tensor:
        """Execute a complete gather -> compute -> scatter pass."""
        self.data.copy_(field)
        field_2d = self.data.reshape(1, 1, self.side, self.side)
        out_2d = torch.nn.functional.conv2d(field_2d, self._stencil, padding=1)
        result = out_2d.reshape(self.M)
        output = field.clone()
        output[self._bool_mask] = result[self._bool_mask]
        return output

    # -- memory -------------------------------------------------------

    def measure_vram(self) -> Dict[str, int]:
        """Return a breakdown of GPU memory used by this layout."""
        mask_bytes = self.bitset.memory_bytes()
        data_bytes = self.data.nelement() * self.data.element_size()
        stencil_bytes = self._stencil.nelement() * self._stencil.element_size()
        bool_cache_bytes = (self._bool_mask.nelement()
                            * self._bool_mask.element_size())
        return {
            "bitset_mask_bytes": mask_bytes,
            "data_bytes": data_bytes,
            "stencil_bytes": stencil_bytes,
            "bool_cache_bytes": bool_cache_bytes,
            "total_bytes": mask_bytes + data_bytes + stencil_bytes
                           + bool_cache_bytes,
        }

    # -- halo access cost ---------------------------------------------

    def halo_access(self, data: torch.Tensor, active_idx: torch.Tensor,
                    halo_w: int = 1) -> torch.Tensor:
        """Measure halo neighbour fetch cost.

        For each active element, reads a (2*halo_w+1)^2 stencil from
        *data* using arithmetic grid indexing (row/col +/- halo_w,
        clamped).  Returns the summed halo values per active element.
        """
        side = self.side
        rows = active_idx // side
        cols = active_idx % side
        accum = torch.zeros(active_idx.shape[0], dtype=DTYPE,
                            device=self.device)
        for dy in range(-halo_w, halo_w + 1):
            for dx in range(-halo_w, halo_w + 1):
                nr = (rows + dy).clamp(0, side - 1)
                nc = (cols + dx).clamp(0, side - 1)
                nidx = nr * side + nc
                accum += data[nidx]
        return accum


# ======================================================================
# Timing helpers
# ======================================================================

def _timed_runs(func, n_warmup: int, n_repeat: int,
                device: torch.device) -> np.ndarray:
    """Return array of wall-clock seconds for *n_repeat* runs after warmup."""
    is_cuda = device.type == "cuda"
    for _ in range(n_warmup):
        func()
        if is_cuda:
            torch.cuda.synchronize(device)

    times = []
    for _ in range(n_repeat):
        if is_cuda:
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        func()
        if is_cuda:
            torch.cuda.synchronize(device)
        dt = time.perf_counter() - t0
        times.append(dt)
    return np.array(times)


def _measure_peak_vram(func, device: torch.device) -> int:
    """Run *func* and return peak VRAM in bytes.  Returns 0 on CPU."""
    if device.type != "cuda":
        func()
        return 0
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    func()
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device)


# ======================================================================
# Benchmark entry point
# ======================================================================

def benchmark_candidate_a(
    side: int = 128,
    sparsity: float = 0.3,
    pattern: str = "random",
    n_seeds: int = 5,
    device: torch.device | str = "cpu",
) -> Dict[str, Any]:
    """Run the full Candidate-A benchmark and return a results dict.

    Metrics collected (per seed, then aggregated):
        - wall_clock_s   : median of *N_REPEAT* reps (after *N_WARMUP*)
        - peak_vram_bytes : peak VRAM during a single run
        - halo_time_s     : median halo-access time
        - memory_breakdown: bitset, data, stencil, bool-cache sizes

    Returns
    -------
    dict with keys: config, per_seed, aggregated, memory_breakdown
    """
    device = torch.device(device)
    M = side * side

    per_seed: list[Dict[str, Any]] = []

    for s in range(n_seeds):
        seed = SEED_BASE + s
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        mask_np = make_mask(side, sparsity, pattern, rng)

        # Build layout
        layout = GridBitsetLayout().build(mask_np, device=device)
        field = torch.randn(M, dtype=DTYPE, device=device)

        # -- Wall-clock --
        wall_times = _timed_runs(
            lambda: layout.run(field), N_WARMUP, N_REPEAT, device,
        )
        wall_median = float(np.median(wall_times))

        # -- Peak VRAM --
        if device.type == "cuda":
            del layout, field
            torch.cuda.empty_cache()
            # Rebuild for clean VRAM measurement
            layout = GridBitsetLayout().build(mask_np, device=device)
            field = torch.randn(M, dtype=DTYPE, device=device)
        peak_vram = _measure_peak_vram(lambda: layout.run(field), device)

        # -- Memory breakdown --
        mem = layout.measure_vram()

        # -- Halo access --
        active_idx = torch.where(layout._bool_mask)[0].to(device)
        halo_times = _timed_runs(
            lambda: layout.halo_access(field, active_idx, halo_w=1),
            N_WARMUP, N_REPEAT, device,
        )
        halo_median = float(np.median(halo_times))

        n_active = int(layout.bitset.count_active())

        per_seed.append({
            "seed": seed,
            "n_active": n_active,
            "wall_clock_median_s": wall_median,
            "peak_vram_bytes": peak_vram,
            "halo_time_median_s": halo_median,
            "memory_breakdown": mem,
        })

        # Cleanup for next seed
        del layout, field, active_idx
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # -- Aggregate across seeds --
    wall_clocks = np.array([r["wall_clock_median_s"] for r in per_seed])
    peak_vrams = np.array([r["peak_vram_bytes"] for r in per_seed])
    halo_times = np.array([r["halo_time_median_s"] for r in per_seed])

    # Use first seed's memory breakdown (identical across seeds for same config)
    mem_breakdown = per_seed[0]["memory_breakdown"]

    aggregated = {
        "wall_clock_median_s": float(np.median(wall_clocks)),
        "wall_clock_iqr_s": float(
            np.subtract(*np.percentile(wall_clocks, [75, 25]))),
        "peak_vram_median_bytes": int(np.median(peak_vrams)),
        "halo_time_median_s": float(np.median(halo_times)),
        "halo_time_iqr_s": float(
            np.subtract(*np.percentile(halo_times, [75, 25]))),
        "n_active_median": int(np.median([r["n_active"] for r in per_seed])),
    }

    return {
        "candidate": "A_bitset",
        "config": {
            "side": side,
            "M": M,
            "sparsity": sparsity,
            "pattern": pattern,
            "n_seeds": n_seeds,
            "n_warmup": N_WARMUP,
            "n_repeat": N_REPEAT,
            "device": str(device),
            "dtype": str(DTYPE),
        },
        "per_seed": per_seed,
        "aggregated": aggregated,
        "memory_breakdown": mem_breakdown,
    }


# ======================================================================
# CLI
# ======================================================================

if __name__ == "__main__":
    import json

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Candidate A (bitset) -- device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(device)}")

    sides = [64, 128, 256]
    sparsities = [0.1, 0.3, 0.5, 0.7, 0.9]
    patterns = ["random", "clustered", "checkerboard"]

    all_results = []
    total = len(sides) * len(sparsities) * len(patterns)
    idx = 0

    for side in sides:
        for sp in sparsities:
            for pat in patterns:
                idx += 1
                print(f"  [{idx}/{total}] side={side} sp={sp:.1f} "
                      f"pat={pat}", end="", flush=True)
                r = benchmark_candidate_a(side, sp, pat, n_seeds=5,
                                          device=device)
                agg = r["aggregated"]
                mem = r["memory_breakdown"]
                print(f"  t={agg['wall_clock_median_s']*1e6:.1f}us"
                      f"  vram={agg['peak_vram_median_bytes']/1024:.1f}KB"
                      f"  mask={mem['bitset_mask_bytes']}B"
                      f"  halo={agg['halo_time_median_s']*1e6:.1f}us")
                all_results.append(r)

    from pathlib import Path
    out_dir = Path(__file__).parent
    json_path = out_dir / "candidate_a_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {json_path}")
