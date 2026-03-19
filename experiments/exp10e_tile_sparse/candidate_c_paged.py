#!/usr/bin/env python3
"""
Curiosity — exp10e, Candidate C: Paged Sparse Tiles

Two-level addressing for sparse 2D grids.  Elements are grouped into
square pages (macroblocks) of Ph x Pw.  A page is *active* if ANY element
in it is active.  Only active pages are stored, with dense data + a bool
mask indicating which elements within the page are truly active.

Data structures
---------------
page_data  [n_active_pages, Ph, Pw]   — dense values per active page
page_keys  [n_active_pages]            — packed (py, px) as int32
page_masks [n_active_pages, Ph, Pw]    — bool: which elements are truly active

Key trade-offs
--------------
+ Intra-page halo is FREE (dense block → arithmetic indexing)
+ For clustered patterns, n_active_pages << n_pages_total → big VRAM win
+ GPU-friendly: each page is one contiguous memory chunk
- Page fragmentation on random/checkerboard patterns (many pages, few active)
- Partially-active pages waste memory (full dense page stored)
- Cross-page boundary handling adds lookup complexity
"""

import time
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# ─────────────────────────────────────────────
# Mask generation (shared convention)
# ─────────────────────────────────────────────

def make_mask(side: int, sparsity: float, pattern: str,
              rng: np.random.Generator) -> np.ndarray:
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


# ─────────────────────────────────────────────
# PagedSparseLayout
# ─────────────────────────────────────────────

class PagedSparseLayout:
    """
    Two-level sparse layout: pages (macroblocks) of page_size x page_size.

    After build():
      .page_data   — [n_active_pages, Ph, Pw] float32, dense values
      .page_keys   — [n_active_pages] int32, packed page coords (py * n_pages_x + px)
      .page_masks  — [n_active_pages, Ph, Pw] bool, intra-page activity
      .page_lookup — [n_pages_total] int32, maps page linear index → slot
                     in page_data (-1 if page inactive)
      .side        — grid side length
      .page_size   — Ph = Pw
      .n_pages_x   — number of pages along x axis
      .n_pages_y   — number of pages along y axis
    """

    def __init__(self, page_size: int = 8):
        self.page_size = page_size
        self.page_data: Optional[torch.Tensor] = None
        self.page_keys: Optional[torch.Tensor] = None
        self.page_masks: Optional[torch.Tensor] = None
        self.page_lookup: Optional[torch.Tensor] = None
        self.side: int = 0
        self.n_pages_x: int = 0
        self.n_pages_y: int = 0
        self.n_active_pages: int = 0
        self.n_pages_total: int = 0
        self.build_time_s: float = 0.0

    # ── build ────────────────────────────────

    def build(self, mask_2d: np.ndarray, device: torch.device) -> "PagedSparseLayout":
        """
        Analyse *mask_2d* (side x side bool), identify active pages,
        and allocate page_data / page_keys / page_masks / page_lookup
        on *device*.
        """
        t0 = time.perf_counter()

        side = mask_2d.shape[0]
        assert mask_2d.shape == (side, side)
        ps = self.page_size
        self.side = side
        self.n_pages_y = (side + ps - 1) // ps
        self.n_pages_x = (side + ps - 1) // ps
        self.n_pages_total = self.n_pages_y * self.n_pages_x

        # Determine which pages are active ---------------------------------
        active_page_indices: List[int] = []
        page_mask_list: List[np.ndarray] = []

        for py in range(self.n_pages_y):
            for px in range(self.n_pages_x):
                y0 = py * ps
                x0 = px * ps
                y1 = min(y0 + ps, side)
                x1 = min(x0 + ps, side)
                block = mask_2d[y0:y1, x0:x1]
                if block.any():
                    page_idx = py * self.n_pages_x + px
                    active_page_indices.append(page_idx)
                    # Pad to full page_size x page_size (boundary pages may
                    # be smaller than ps when side % ps != 0)
                    pmask = np.zeros((ps, ps), dtype=bool)
                    pmask[:block.shape[0], :block.shape[1]] = block
                    page_mask_list.append(pmask)

        self.n_active_pages = len(active_page_indices)

        # page_keys --------------------------------------------------------
        keys_np = np.array(active_page_indices, dtype=np.int32)
        self.page_keys = torch.from_numpy(keys_np).to(device)

        # page_masks -------------------------------------------------------
        if self.n_active_pages > 0:
            masks_np = np.stack(page_mask_list, axis=0)  # [N, ps, ps]
        else:
            masks_np = np.zeros((0, ps, ps), dtype=bool)
        self.page_masks = torch.from_numpy(masks_np).to(device)

        # page_data (initialise to zeros; filled by gather) ----------------
        self.page_data = torch.zeros(
            (self.n_active_pages, ps, ps),
            dtype=torch.float32, device=device,
        )

        # page_lookup: full → slot mapping ---------------------------------
        lookup_np = np.full(self.n_pages_total, -1, dtype=np.int32)
        for slot, pidx in enumerate(active_page_indices):
            lookup_np[pidx] = slot
        self.page_lookup = torch.from_numpy(lookup_np).to(device)

        self.build_time_s = time.perf_counter() - t0
        return self

    # ── gather ───────────────────────────────

    def gather(self, full_data: torch.Tensor) -> torch.Tensor:
        """
        Copy Ph x Pw blocks from *full_data* (side x side) into page_data
        for every active page.  Returns the filled page_data tensor.
        """
        ps = self.page_size
        side = self.side
        for slot in range(self.n_active_pages):
            pidx = self.page_keys[slot].item()
            py = pidx // self.n_pages_x
            px = pidx % self.n_pages_x
            y0 = py * ps
            x0 = px * ps
            y1 = min(y0 + ps, side)
            x1 = min(x0 + ps, side)
            self.page_data[slot, :y1 - y0, :x1 - x0] = full_data[y0:y1, x0:x1]
        return self.page_data

    # ── compute (intra-page 3x3 conv) ────────

    def compute(self, gathered: torch.Tensor) -> torch.Tensor:
        """
        Apply a 3x3 averaging convolution *within* each active page.
        Intra-page halo is free (dense data).  Cross-page boundary is
        NOT handled here (treated as zero-padded at page edges).

        Returns tensor of same shape as gathered.
        """
        # gathered: [n_active_pages, Ph, Pw]
        # Use F.conv2d with groups or simple loop
        if gathered.shape[0] == 0:
            return gathered.clone()

        inp = gathered.unsqueeze(1)  # [N, 1, Ph, Pw]
        weight = torch.ones(1, 1, 3, 3, device=gathered.device,
                            dtype=gathered.dtype) / 9.0
        result = torch.nn.functional.conv2d(inp, weight, padding=1)
        return result.squeeze(1)  # [N, Ph, Pw]

    # ── scatter ──────────────────────────────

    def scatter(self, result: torch.Tensor, full_data: torch.Tensor) -> torch.Tensor:
        """
        Write active-page results back to *full_data* (side x side),
        only at truly active elements (respecting page_masks).
        Returns the updated full_data.
        """
        ps = self.page_size
        side = self.side
        out = full_data.clone()
        for slot in range(self.n_active_pages):
            pidx = self.page_keys[slot].item()
            py = pidx // self.n_pages_x
            px = pidx % self.n_pages_x
            y0 = py * ps
            x0 = px * ps
            y1 = min(y0 + ps, side)
            x1 = min(x0 + ps, side)
            pmask = self.page_masks[slot, :y1 - y0, :x1 - x0]
            block = out[y0:y1, x0:x1]
            patch = result[slot, :y1 - y0, :x1 - x0]
            block[pmask] = patch[pmask]
            out[y0:y1, x0:x1] = block
        return out

    # ── halo access ──────────────────────────

    def halo_access(self, full_data: torch.Tensor,
                    halo_w: int = 1) -> Dict[str, float]:
        """
        Measure cost of reading halo neighbours for every active element.

        Intra-page halo: neighbours that fall within the same page
            → free (dense arithmetic indexing).
        Cross-page halo: neighbours that fall in a different page
            → require page_lookup to find the neighbour page slot.

        Returns dict with wall-clock times for intra and cross components
        (in seconds) plus counts.
        """
        ps = self.page_size
        side = self.side

        # ── Intra-page halo ──────────────────
        t0 = time.perf_counter()
        intra_total = 0.0
        for slot in range(self.n_active_pages):
            pmask = self.page_masks[slot]  # [ps, ps]
            data = self.page_data[slot]    # [ps, ps]
            active_yx = torch.nonzero(pmask, as_tuple=False)  # [K, 2]
            for k in range(active_yx.shape[0]):
                ly, lx = active_yx[k, 0].item(), active_yx[k, 1].item()
                for dy in range(-halo_w, halo_w + 1):
                    for dx in range(-halo_w, halo_w + 1):
                        ny, nx = ly + dy, lx + dx
                        if 0 <= ny < ps and 0 <= nx < ps:
                            intra_total += data[ny, nx].item()
        t_intra = time.perf_counter() - t0

        # ── Cross-page halo ──────────────────
        t0 = time.perf_counter()
        cross_total = 0.0
        cross_lookups = 0
        for slot in range(self.n_active_pages):
            pidx = self.page_keys[slot].item()
            page_py = pidx // self.n_pages_x
            page_px = pidx % self.n_pages_x
            pmask = self.page_masks[slot]
            active_yx = torch.nonzero(pmask, as_tuple=False)
            for k in range(active_yx.shape[0]):
                ly, lx = active_yx[k, 0].item(), active_yx[k, 1].item()
                gy = page_py * ps + ly  # global coords
                gx = page_px * ps + lx
                for dy in range(-halo_w, halo_w + 1):
                    for dx in range(-halo_w, halo_w + 1):
                        ny, nx = ly + dy, lx + dx
                        # Skip if within page (already counted above)
                        if 0 <= ny < ps and 0 <= nx < ps:
                            continue
                        # Global neighbour coords
                        gny = gy + dy
                        gnx = gx + dx
                        if gny < 0 or gny >= side or gnx < 0 or gnx >= side:
                            continue  # out of grid
                        # Which page does the neighbour belong to?
                        npy = gny // ps
                        npx = gnx // ps
                        neighbour_pidx = npy * self.n_pages_x + npx
                        neighbour_slot = self.page_lookup[neighbour_pidx].item()
                        cross_lookups += 1
                        if neighbour_slot >= 0:
                            # Read from neighbour page
                            local_ny = gny - npy * ps
                            local_nx = gnx - npx * ps
                            cross_total += self.page_data[
                                neighbour_slot, local_ny, local_nx
                            ].item()
                        else:
                            # Neighbour page is inactive → read from full_data
                            cross_total += full_data[gny, gnx].item()
        t_cross = time.perf_counter() - t0

        return {
            "intra_s": t_intra,
            "cross_s": t_cross,
            "intra_sum": intra_total,
            "cross_sum": cross_total,
            "cross_lookups": cross_lookups,
        }

    # ── memory accounting ────────────────────

    def memory_bytes(self) -> Dict[str, int]:
        """Itemised memory breakdown (bytes) of stored tensors."""
        pd = self.page_data.nelement() * self.page_data.element_size() if self.page_data is not None else 0
        pk = self.page_keys.nelement() * self.page_keys.element_size() if self.page_keys is not None else 0
        pm = self.page_masks.nelement() * self.page_masks.element_size() if self.page_masks is not None else 0
        pl = self.page_lookup.nelement() * self.page_lookup.element_size() if self.page_lookup is not None else 0
        return {
            "page_data": pd,
            "page_keys": pk,
            "page_masks": pm,
            "page_lookup": pl,
            "total": pd + pk + pm + pl,
        }

    def measure_vram(self, device: torch.device) -> int:
        """
        Peak VRAM currently allocated on *device* (bytes).
        Returns 0 for CPU.
        """
        if device.type == "cuda":
            return torch.cuda.max_memory_allocated(device)
        return 0


# ─────────────────────────────────────────────
# Benchmark driver
# ─────────────────────────────────────────────

def benchmark_candidate_c(
    side: int = 256,
    sparsity: float = 0.15,
    pattern: str = "clustered",
    n_seeds: int = 3,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Benchmark the paged-sparse layout across multiple page sizes.

    Sweeps page_sizes [4, 8, 16] and for each measures:
      - build time
      - gather / compute / scatter wall-clock
      - halo access (intra + cross-page)
      - peak VRAM
      - memory breakdown

    Returns a dict keyed by page_size with all metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    page_sizes = [4, 8, 16]
    all_results: Dict[str, object] = {
        "config": {
            "side": side,
            "sparsity": sparsity,
            "pattern": pattern,
            "n_seeds": n_seeds,
            "device": str(device),
            "page_sizes": page_sizes,
        },
        "per_page_size": {},
    }

    for ps in page_sizes:
        seed_results: List[Dict] = []

        for seed_idx in range(n_seeds):
            seed = 42 + seed_idx
            rng = np.random.default_rng(seed)
            torch.manual_seed(seed)

            mask_2d = make_mask(side, sparsity, pattern, rng)
            n_active = int(mask_2d.sum())
            full_data = torch.randn(side, side, device=device, dtype=torch.float32)

            # Reset peak VRAM tracker
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            # Build
            layout = PagedSparseLayout(page_size=ps)
            layout.build(mask_2d, device)

            # Gather
            t0 = time.perf_counter()
            gathered = layout.gather(full_data)
            t_gather = time.perf_counter() - t0

            # Compute (intra-page 3x3 conv)
            t0 = time.perf_counter()
            computed = layout.compute(gathered)
            t_compute = time.perf_counter() - t0

            # Scatter
            t0 = time.perf_counter()
            out = layout.scatter(computed, full_data)
            t_scatter = time.perf_counter() - t0

            # Halo access
            halo_metrics = layout.halo_access(full_data, halo_w=1)

            # Memory
            mem = layout.memory_bytes()
            peak_vram = layout.measure_vram(device)

            n_pages_total = layout.n_pages_total
            n_active_pages = layout.n_active_pages
            fill_ratio = (n_active / (n_active_pages * ps * ps)
                          if n_active_pages > 0 else 0.0)

            seed_results.append({
                "seed": seed,
                "n_active": n_active,
                "n_pages_total": n_pages_total,
                "n_active_pages": n_active_pages,
                "page_fill_ratio": fill_ratio,
                "build_time_s": layout.build_time_s,
                "gather_s": t_gather,
                "compute_s": t_compute,
                "scatter_s": t_scatter,
                "halo_intra_s": halo_metrics["intra_s"],
                "halo_cross_s": halo_metrics["cross_s"],
                "halo_cross_lookups": halo_metrics["cross_lookups"],
                "memory": mem,
                "peak_vram_bytes": peak_vram,
            })

        # Aggregate across seeds -----------------------------------------
        def _median(key):
            return float(np.median([r[key] for r in seed_results]))

        agg = {
            "page_size": ps,
            "n_active_median": int(np.median([r["n_active"] for r in seed_results])),
            "n_pages_total": seed_results[0]["n_pages_total"],
            "n_active_pages_median": int(np.median([r["n_active_pages"]
                                                     for r in seed_results])),
            "page_fill_ratio_median": _median("page_fill_ratio"),
            "build_time_s": _median("build_time_s"),
            "gather_s": _median("gather_s"),
            "compute_s": _median("compute_s"),
            "scatter_s": _median("scatter_s"),
            "halo_intra_s": _median("halo_intra_s"),
            "halo_cross_s": _median("halo_cross_s"),
            "halo_cross_lookups_median": int(np.median([r["halo_cross_lookups"]
                                                         for r in seed_results])),
            "memory_bytes": seed_results[0]["memory"],
            "peak_vram_bytes": int(np.median([r["peak_vram_bytes"]
                                               for r in seed_results])),
            "per_seed": seed_results,
        }
        all_results["per_page_size"][ps] = agg

    return all_results


# ─────────────────────────────────────────────
# Pretty-print
# ─────────────────────────────────────────────

def print_results(results: Dict) -> None:
    """Print a compact summary table."""
    cfg = results["config"]
    print("=" * 72)
    print(f"Candidate C — Paged Sparse  |  side={cfg['side']}  "
          f"sparsity={cfg['sparsity']}  pattern={cfg['pattern']}  "
          f"device={cfg['device']}")
    print("=" * 72)

    header = (f"{'PS':>4s}  {'ActPg':>6s} / {'TotPg':>6s}  "
              f"{'Fill%':>5s}  {'Build':>8s}  {'Gather':>8s}  "
              f"{'Compute':>8s}  {'Scatter':>8s}  "
              f"{'HaloIn':>8s}  {'HaloCr':>8s}  "
              f"{'Mem(KB)':>9s}  {'VRAM(KB)':>9s}")
    print(header)
    print("-" * len(header))

    for ps in sorted(results["per_page_size"]):
        a = results["per_page_size"][ps]
        fill_pct = a["page_fill_ratio_median"] * 100
        mem_kb = a["memory_bytes"]["total"] / 1024
        vram_kb = a["peak_vram_bytes"] / 1024

        def _us(key):
            return a[key] * 1e6

        print(f"{ps:4d}  {a['n_active_pages_median']:6d} / {a['n_pages_total']:6d}  "
              f"{fill_pct:5.1f}  "
              f"{_us('build_time_s'):8.0f}  {_us('gather_s'):8.0f}  "
              f"{_us('compute_s'):8.0f}  {_us('scatter_s'):8.0f}  "
              f"{_us('halo_intra_s'):8.0f}  {_us('halo_cross_s'):8.0f}  "
              f"{mem_kb:9.1f}  {vram_kb:9.1f}")
    print()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print()

    configs = [
        # (side, sparsity, pattern)
        (128, 0.10, "clustered"),
        (128, 0.10, "random"),
        (128, 0.30, "clustered"),
        (128, 0.30, "random"),
        (256, 0.10, "clustered"),
        (256, 0.10, "random"),
        (256, 0.30, "clustered"),
        (256, 0.30, "random"),
    ]

    all_runs = []
    for side, sp, pat in configs:
        res = benchmark_candidate_c(
            side=side, sparsity=sp, pattern=pat,
            n_seeds=3, device=device,
        )
        print_results(res)
        all_runs.append(res)

    # ── Summary: page fill ratio (clustered vs random) ──
    print("=" * 72)
    print("SUMMARY: page fill ratio tells us how well pages match the pattern")
    print("=" * 72)
    for res in all_runs:
        cfg = res["config"]
        for ps in sorted(res["per_page_size"]):
            a = res["per_page_size"][ps]
            print(f"  side={cfg['side']:4d}  sp={cfg['sparsity']:.2f}  "
                  f"{cfg['pattern']:9s}  ps={ps:2d}  →  "
                  f"active_pages={a['n_active_pages_median']:5d}/"
                  f"{a['n_pages_total']:5d}  "
                  f"fill={a['page_fill_ratio_median']*100:5.1f}%  "
                  f"mem={a['memory_bytes']['total']/1024:8.1f}KB")
    print()
