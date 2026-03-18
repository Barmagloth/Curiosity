#!/usr/bin/env python3
"""
P2a — Sensitivity Sweep for Two-Stage Gate Thresholds.

Sweeps instability_threshold and FSR_threshold across:
  - 5 scene types: clean, noise, blur, spatial_variation, jpeg_artifacts
  - 4 space types: scalar_grid, vector_grid, irregular_graph, tree_hierarchy

For each (scene x space x threshold_pair): runs adaptive refinement,
measures quality metric, SeamScore, computational cost over 10 seeds.

Computes ridge width: range of thresholds within which performance
degrades by <5% from optimal.

Kill criterion:
  ridge_width > 30% of sweep range -> manual thresholds ok (PASS)
  ridge_width < 10% of sweep range -> P2b needed (FAIL)
  10%-30% -> INCONCLUSIVE

Roadmap: P2a (experiment_hierarchy.md step 4, future exp12)
"""

import numpy as np
import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm


# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

DEFAULTS = {
    "instability_threshold": 0.25,
    "FSR_threshold": 0.20,
}

N_GRID = 10
INSTAB_MULTS = np.geomspace(0.1, 10.0, N_GRID)
FSR_MULTS = np.geomspace(0.1, 10.0, N_GRID)

N_SEEDS = 10
BUDGET_FRACTION = 0.30
DEGRADE_5PCT = 0.05

SCENE_TYPES = ["clean", "noise", "blur", "spatial_variation", "jpeg_artifacts"]
SPACE_TYPES = ["scalar_grid", "vector_grid", "irregular_graph", "tree_hierarchy"]


# ═══════════════════════════════════════════════════════════════════════
# Scene generators (5 types)
# ═══════════════════════════════════════════════════════════════════════

def _base_field(N, seed):
    rng = np.random.RandomState(seed)
    x = np.linspace(0, 1, N, endpoint=False)
    xx, yy = np.meshgrid(x, x)
    field = np.zeros((N, N))
    for _ in range(5):
        cx, cy = rng.uniform(0.1, 0.9, 2)
        sigma = rng.uniform(0.05, 0.2)
        field += rng.uniform(0.3, 1.0) * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
    for _ in range(8):
        cx, cy = rng.uniform(0.05, 0.95, 2)
        sigma = rng.uniform(0.01, 0.05)
        field += rng.uniform(0.1, 0.6) * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
    # discontinuity
    field += 0.5 * (xx > rng.uniform(0.3, 0.7))
    return field


def scene_clean(N, seed):
    return _base_field(N, seed)


def scene_noise(N, seed):
    rng = np.random.RandomState(seed + 7000)
    return _base_field(N, seed) + rng.randn(N, N) * 0.15


def scene_blur(N, seed):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(_base_field(N, seed), sigma=3.0)


def scene_spatial_variation(N, seed):
    rng = np.random.RandomState(seed + 8000)
    field = _base_field(N, seed)
    mid = N // 2
    field[:, :mid] += rng.randn(N, mid) * 0.15
    from scipy.ndimage import gaussian_filter
    field[:, mid:] = gaussian_filter(field[:, mid:], sigma=2.5)
    return field


def scene_jpeg_artifacts(N, seed):
    rng = np.random.RandomState(seed + 9000)
    field = _base_field(N, seed)
    # simulate block-8 quantization artifacts
    bsz = 8
    ny, nx = N // bsz, N // bsz
    for bi in range(ny):
        for bj in range(nx):
            sl = (slice(bi * bsz, (bi + 1) * bsz), slice(bj * bsz, (bj + 1) * bsz))
            block = field[sl]
            # coarse quantization of DCT-like operation
            mean_val = block.mean()
            field[sl] = mean_val + (block - mean_val) * 0.6
            field[sl] += rng.randn(bsz, bsz) * 0.02
    return field


SCENE_GENERATORS = {
    "clean": scene_clean,
    "noise": scene_noise,
    "blur": scene_blur,
    "spatial_variation": scene_spatial_variation,
    "jpeg_artifacts": scene_jpeg_artifacts,
}


# ═══════════════════════════════════════════════════════════════════════
# Space type adapters
#
# Each adapter wraps one of the 4 space types from phase2_probe_seam
# and provides a uniform interface for the sweep.
# ═══════════════════════════════════════════════════════════════════════

EPS = 1e-10


def robust_jump(diffs):
    if len(diffs) == 0:
        return 0.0
    return float(np.median(diffs))


# --- Scalar Grid ---

class ScalarGridAdapter:
    name = "scalar_grid"

    def __init__(self, N=64, tile=8, seed=42):
        self.N = N
        self.T = tile
        self.NT = N // tile

    def setup(self, gt):
        self.gt = gt
        c = np.zeros_like(gt)
        for ti in range(self.NT):
            for tj in range(self.NT):
                s = slice(ti * self.T, (ti + 1) * self.T)
                cs = slice(tj * self.T, (tj + 1) * self.T)
                c[s, cs] = gt[s, cs].mean()
        self.coarse = c

    def get_units(self):
        return [(i, j) for i in range(self.NT) for j in range(self.NT)]

    def unit_error(self, state, unit):
        ti, tj = unit
        s = slice(ti * self.T, (ti + 1) * self.T)
        cs = slice(tj * self.T, (tj + 1) * self.T)
        return float(np.mean((self.gt[s, cs] - state[s, cs]) ** 2))

    def refine_unit(self, state, unit, halo=2):
        ti, tj = unit
        r0, c0 = ti * self.T, tj * self.T
        er0 = max(r0 - halo, 0)
        er1 = min(r0 + self.T + halo, self.N)
        ec0 = max(c0 - halo, 0)
        ec1 = min(c0 + self.T + halo, self.N)
        delta = self.gt[er0:er1, ec0:ec1] - state[er0:er1, ec0:ec1]
        h, w = delta.shape
        mask = np.ones((h, w))
        if halo > 0:
            for i in range(min(halo, h)):
                f = 0.5 * (1 - np.cos(np.pi * (i + 0.5) / halo))
                mask[i, :] *= f
                if h - 1 - i != i:
                    mask[h - 1 - i, :] *= f
            for j in range(min(halo, w)):
                f = 0.5 * (1 - np.cos(np.pi * (j + 0.5) / halo))
                mask[:, j] *= f
                if w - 1 - j != j:
                    mask[:, w - 1 - j] *= f
        out = state.copy()
        out[er0:er1, ec0:ec1] += delta * mask
        return out

    def quality_metric(self, state):
        mse = float(np.mean((self.gt - state) ** 2))
        if mse < 1e-15:
            return 100.0
        return 10 * np.log10(np.max(self.gt) ** 2 / mse)

    def seam_score(self, state, unit, halo=2):
        ti, tj = unit
        r0, c0 = ti * self.T, tj * self.T
        hr0 = max(r0 - halo, 0)
        hr1 = min(r0 + self.T + halo, self.N)
        hc0 = max(c0 - halo, 0)
        hc1 = min(c0 + self.T + halo, self.N)
        bp, ip = [], []
        if hr0 > 0:
            for c in range(hc0, hc1):
                bp.append(((hr0, c), (hr0 - 1, c)))
                if hr0 + 1 < hr1:
                    ip.append(((hr0 + 1, c), (hr0, c)))
        if hr1 < self.N:
            for c in range(hc0, hc1):
                bp.append(((hr1 - 1, c), (hr1, c)))
                if hr1 - 2 >= hr0:
                    ip.append(((hr1 - 2, c), (hr1 - 1, c)))
        if hc0 > 0:
            for r in range(hr0, hr1):
                bp.append(((r, hc0), (r, hc0 - 1)))
                if hc0 + 1 < hc1:
                    ip.append(((r, hc0 + 1), (r, hc0)))
        if hc1 < self.N:
            for r in range(hr0, hr1):
                bp.append(((r, hc1 - 1), (r, hc1)))
                if hc1 - 2 >= hc0:
                    ip.append(((r, hc1 - 2), (r, hc1 - 1)))
        d_out = [abs(state[a] - state[b]) for a, b in bp]
        d_in = [abs(state[a] - state[b]) for a, b in ip] if ip else []
        jo = robust_jump(d_out)
        ji = robust_jump(d_in) if d_in else 0.0
        return jo / (ji + EPS)


# --- Vector Grid ---

class VectorGridAdapter:
    name = "vector_grid"

    def __init__(self, N=32, tile=8, D=16, seed=42):
        self.N = N
        self.T = tile
        self.NT = N // tile
        self.D = D

    def setup(self, gt_2d):
        rng = np.random.RandomState(42)
        N = self.N
        D = self.D
        x = np.linspace(0, 1, N, endpoint=False)
        xx, yy = np.meshgrid(x, x)
        self.gt = np.zeros((N, N, D))
        # Use gt_2d as first channel, generate rest
        from scipy.ndimage import zoom
        if gt_2d.shape[0] != N:
            gt_resized = zoom(gt_2d, N / gt_2d.shape[0], order=1)[:N, :N]
        else:
            gt_resized = gt_2d
        self.gt[:, :, 0] = gt_resized
        for d in range(1, D):
            freq = 2 + d * 0.5
            phase = rng.uniform(0, 2 * np.pi)
            amp = 0.5 / (1 + d * 0.1)
            self.gt[:, :, d] = amp * np.sin(2 * np.pi * freq * xx + phase) * np.cos(
                2 * np.pi * (freq * 0.7) * yy)
        self.gt += rng.randn(N, N, D) * 0.02
        # coarse
        self.coarse = np.zeros_like(self.gt)
        for ti in range(self.NT):
            for tj in range(self.NT):
                s = slice(ti * self.T, (ti + 1) * self.T)
                cs = slice(tj * self.T, (tj + 1) * self.T)
                self.coarse[s, cs, :] = self.gt[s, cs, :].mean(axis=(0, 1), keepdims=True)

    def get_units(self):
        return [(i, j) for i in range(self.NT) for j in range(self.NT)]

    def unit_error(self, state, unit):
        ti, tj = unit
        s = slice(ti * self.T, (ti + 1) * self.T)
        cs = slice(tj * self.T, (tj + 1) * self.T)
        return float(np.mean((self.gt[s, cs, :] - state[s, cs, :]) ** 2))

    def refine_unit(self, state, unit, halo=2):
        ti, tj = unit
        r0, c0 = ti * self.T, tj * self.T
        er0 = max(r0 - halo, 0)
        er1 = min(r0 + self.T + halo, self.N)
        ec0 = max(c0 - halo, 0)
        ec1 = min(c0 + self.T + halo, self.N)
        delta = self.gt[er0:er1, ec0:ec1, :] - state[er0:er1, ec0:ec1, :]
        h, w = delta.shape[:2]
        mask = np.ones((h, w, 1))
        if halo > 0:
            for i in range(min(halo, h)):
                f = 0.5 * (1 - np.cos(np.pi * (i + 0.5) / halo))
                mask[i, :, :] *= f
                if h - 1 - i != i:
                    mask[h - 1 - i, :, :] *= f
            for j in range(min(halo, w)):
                f = 0.5 * (1 - np.cos(np.pi * (j + 0.5) / halo))
                mask[:, j, :] *= f
                if w - 1 - j != j:
                    mask[:, w - 1 - j, :] *= f
        out = state.copy()
        out[er0:er1, ec0:ec1, :] += delta * mask
        return out

    def quality_metric(self, state):
        mse = float(np.mean((self.gt - state) ** 2))
        if mse < 1e-15:
            return 100.0
        rng = np.max(self.gt) - np.min(self.gt)
        if rng < 1e-15:
            return 100.0
        return 10 * np.log10(rng ** 2 / mse)

    def seam_score(self, state, unit, halo=2):
        ti, tj = unit
        r0, c0 = ti * self.T, tj * self.T
        hr0 = max(r0 - halo, 0)
        hr1 = min(r0 + self.T + halo, self.N)
        hc0 = max(c0 - halo, 0)
        hc1 = min(c0 + self.T + halo, self.N)
        bp, ip = [], []
        if hr0 > 0:
            for c in range(hc0, hc1):
                bp.append(((hr0, c), (hr0 - 1, c)))
                if hr0 + 1 < hr1:
                    ip.append(((hr0 + 1, c), (hr0, c)))
        if hr1 < self.N:
            for c in range(hc0, hc1):
                bp.append(((hr1 - 1, c), (hr1, c)))
                if hr1 - 2 >= hr0:
                    ip.append(((hr1 - 2, c), (hr1 - 1, c)))
        if hc0 > 0:
            for r in range(hr0, hr1):
                bp.append(((r, hc0), (r, hc0 - 1)))
                if hc0 + 1 < hc1:
                    ip.append(((r, hc0 + 1), (r, hc0)))
        if hc1 < self.N:
            for r in range(hr0, hr1):
                bp.append(((r, hc1 - 1), (r, hc1)))
                if hc1 - 2 >= hc0:
                    ip.append(((r, hc1 - 2), (r, hc1 - 1)))
        d_out = [np.linalg.norm(state[a[0], a[1], :] - state[b[0], b[1], :])
                 for a, b in bp]
        d_in = [np.linalg.norm(state[a[0], a[1], :] - state[b[0], b[1], :])
                for a, b in ip] if ip else []
        jo = robust_jump(d_out)
        ji = robust_jump(d_in) if d_in else 0.0
        return jo / (ji + EPS)


# --- Irregular Graph ---

class IrregularGraphAdapter:
    name = "irregular_graph"

    def __init__(self, n_points=200, k=6, n_clusters=10, seed=42):
        self.n_pts = n_points
        self.k = k
        self.n_clusters = n_clusters
        self._base_seed = seed

    def setup(self, gt_2d):
        from scipy.spatial import cKDTree
        from scipy.cluster.vq import kmeans2

        seed = self._base_seed
        rng = np.random.RandomState(seed)
        self.pos = rng.rand(self.n_pts, 2)
        tree = cKDTree(self.pos)
        _, idx = tree.query(self.pos, k=self.k + 1)
        self.neighbors = {i: set(idx[i, 1:]) for i in range(self.n_pts)}

        # Sample GT values from the 2D field
        N = gt_2d.shape[0]
        pix = (self.pos * (N - 1)).astype(int).clip(0, N - 1)
        self.gt = gt_2d[pix[:, 1], pix[:, 0]] + rng.randn(self.n_pts) * 0.03

        _, self.labels = kmeans2(self.pos, self.n_clusters, minit='points', seed=seed)
        self.coarse = np.zeros(self.n_pts)
        for c in range(self.n_clusters):
            mask = self.labels == c
            if mask.any():
                self.coarse[mask] = self.gt[mask].mean()

    def get_units(self):
        return list(range(self.n_clusters))

    def _cluster_pts(self, cid):
        return set(np.where(self.labels == cid)[0])

    def _halo(self, cluster, hops=1):
        halo = set()
        frontier = set(cluster)
        for _ in range(hops):
            nf = set()
            for p in frontier:
                for n in self.neighbors[p]:
                    if n not in cluster and n not in halo:
                        nf.add(n)
                        halo.add(n)
            frontier = nf
        return halo

    def unit_error(self, state, unit):
        pts = self._cluster_pts(unit)
        if not pts:
            return 0.0
        return float(np.mean([(self.gt[p] - state[p]) ** 2 for p in pts]))

    def refine_unit(self, state, unit, halo_hops=1):
        cluster = self._cluster_pts(unit)
        halo = self._halo(cluster, halo_hops)
        out = state.copy()
        for p in cluster:
            out[p] = state[p] + (self.gt[p] - state[p])
        for p in halo:
            out[p] = state[p] + (self.gt[p] - state[p]) * 0.3
        return out

    def quality_metric(self, state):
        mse = float(np.mean((self.gt - state) ** 2))
        if mse < 1e-15:
            return 100.0
        rng_val = max(abs(self.gt.max()), abs(self.gt.min()))
        if rng_val < 1e-15:
            rng_val = 1.0
        return 10 * np.log10(rng_val ** 2 / mse)

    def seam_score(self, state, unit, halo_hops=1):
        cluster = self._cluster_pts(unit)
        halo = self._halo(cluster, halo_hops)
        outside = set(range(self.n_pts)) - cluster - halo
        bp, ip = [], []
        for p in halo:
            for n in self.neighbors[p]:
                if n in outside:
                    bp.append((p, n))
                if n in cluster:
                    ip.append((n, p))
        d_out = [abs(state[a] - state[b]) for a, b in bp]
        d_in = [abs(state[a] - state[b]) for a, b in ip] if ip else []
        jo = robust_jump(d_out)
        ji = robust_jump(d_in) if d_in else 0.0
        return jo / (ji + EPS)


# --- Tree Hierarchy ---

class TreeAdapter:
    name = "tree_hierarchy"

    def __init__(self, depth=6, seed=42):
        self.depth = depth
        self.n = 2 ** depth - 1
        self._base_seed = seed

    def setup(self, gt_2d):
        rng = np.random.RandomState(self._base_seed)
        self.gt = np.zeros(self.n)
        N = gt_2d.shape[0]
        for i in range(self.n):
            d = int(np.log2(i + 1))
            # Sample position deterministically from node index
            t = (i + 1) / (self.n + 1)
            px = int(t * (N - 1)) % N
            py = int((t * 7.3) * (N - 1)) % N
            self.gt[i] = gt_2d[py, px] / (1 + d * 0.2) + rng.randn() * 0.05
            if d >= 3:
                self.gt[i] += 0.3 * ((i % 7) > 3)
        # Coarse
        self.coarse_depth = 3
        self.coarse = np.zeros(self.n)
        for i in range(self.n):
            d = int(np.log2(i + 1))
            if d < self.coarse_depth:
                self.coarse[i] = self.gt[i]
            else:
                ancestor = i
                while int(np.log2(ancestor + 1)) >= self.coarse_depth:
                    ancestor = (ancestor - 1) // 2
                subtree = self._subtree(ancestor)
                self.coarse[i] = np.mean([self.gt[j] for j in subtree])

    def _children(self, i):
        left = 2 * i + 1
        right = 2 * i + 2
        c = []
        if left < self.n:
            c.append(left)
        if right < self.n:
            c.append(right)
        return c

    def _parent(self, i):
        return (i - 1) // 2 if i > 0 else None

    def _subtree(self, i):
        nodes = [i]
        q = [i]
        while q:
            curr = q.pop()
            for c in self._children(curr):
                nodes.append(c)
                q.append(c)
        return nodes

    def _neighbors(self, i):
        n = set()
        p = self._parent(i)
        if p is not None:
            n.add(p)
            for c in self._children(p):
                if c != i:
                    n.add(c)
        for c in self._children(i):
            n.add(c)
        return n

    def get_units(self):
        # Subtrees rooted at depth 3
        return [i for i in range(self.n) if int(np.log2(i + 1)) == 3]

    def unit_error(self, state, unit):
        pts = self._subtree(unit)
        return float(np.mean([(self.gt[p] - state[p]) ** 2 for p in pts]))

    def refine_unit(self, state, unit, halo_hops=1):
        core = set(self._subtree(unit))
        halo = set()
        frontier = set(core)
        for _ in range(halo_hops):
            nf = set()
            for p in frontier:
                for n in self._neighbors(p):
                    if n not in core and n not in halo:
                        halo.add(n)
                        nf.add(n)
            frontier = nf
        out = state.copy()
        for p in core:
            out[p] = state[p] + (self.gt[p] - state[p])
        for p in halo:
            out[p] = state[p] + (self.gt[p] - state[p]) * 0.3
        return out

    def quality_metric(self, state):
        mse = float(np.mean((self.gt - state) ** 2))
        if mse < 1e-15:
            return 100.0
        rng_val = max(abs(self.gt.max()), abs(self.gt.min()))
        if rng_val < 1e-15:
            rng_val = 1.0
        return 10 * np.log10(rng_val ** 2 / mse)

    def seam_score(self, state, unit, halo_hops=1):
        core = set(self._subtree(unit))
        halo = set()
        frontier = set(core)
        for _ in range(halo_hops):
            nf = set()
            for p in frontier:
                for n in self._neighbors(p):
                    if n not in core and n not in halo:
                        halo.add(n)
                        nf.add(n)
            frontier = nf
        outside = set(range(self.n)) - core - halo
        bp, ip = [], []
        for p in halo:
            for n in self._neighbors(p):
                if n in outside:
                    bp.append((p, n))
                if n in core:
                    ip.append((n, p))
        d_out = [abs(state[a] - state[b]) for a, b in bp]
        d_in = [abs(state[a] - state[b]) for a, b in ip] if ip else []
        jo = robust_jump(d_out)
        ji = robust_jump(d_in) if d_in else 0.0
        return jo / (ji + EPS)


SPACE_FACTORIES = {
    "scalar_grid": lambda: ScalarGridAdapter(N=64, tile=8),
    "vector_grid": lambda: VectorGridAdapter(N=32, tile=8, D=16),
    "irregular_graph": lambda: IrregularGraphAdapter(n_points=200, k=6, n_clusters=10),
    "tree_hierarchy": lambda: TreeAdapter(depth=6),
}


# ═══════════════════════════════════════════════════════════════════════
# Two-stage gate (self-contained, mirrors exp07b logic)
# ═══════════════════════════════════════════════════════════════════════

def compute_expert_scores(space, state, units):
    errors = np.array([space.unit_error(state, u) for u in units])
    if errors.max() - errors.min() < 1e-12:
        return errors, errors
    from scipy.stats import rankdata
    ranked = rankdata(errors) / len(errors)
    return errors, ranked


def run_probe_diagnostics(space, state, units, probe_mask, seed):
    probe_units = [u for u, m in zip(units, probe_mask) if m]
    if not probe_units:
        return {"fsr": 1.0, "instability": 1.0}

    # Compute gain on probe tiles
    gains = []
    for u in probe_units:
        before = state.copy() if isinstance(state, np.ndarray) else state.copy()
        after = space.refine_unit(before, u)
        g = space.unit_error(before, u) - space.unit_error(after, u)
        gains.append(max(g, 0.0))
    gains = np.array(gains)

    # FSR: fraction of probe tiles with negative or negligible gain
    delta_false = np.median(gains) * 0.1 if len(gains) > 0 and np.median(gains) > 0 else 1e-8
    fsr = float(np.mean(gains < delta_false))

    # Instability: jitter the state, check if top units flip
    rng = np.random.RandomState(seed + 5555)
    errors_orig = np.array([space.unit_error(state, u) for u in units])
    n_units = len(units)
    k_top = max(1, int(0.3 * n_units))

    top_orig = set(np.argsort(errors_orig)[-k_top:])

    if isinstance(state, np.ndarray):
        jittered = state + rng.randn(*state.shape) * 0.005 * (state.max() - state.min() + 1e-10)
    else:
        jittered = state.copy()

    errors_jit = np.array([space.unit_error(jittered, u) for u in units])
    top_jit = set(np.argsort(errors_jit)[-k_top:])

    flipped = len(top_orig.symmetric_difference(top_jit))
    instability = flipped / (2 * k_top + 1e-12)

    return {"fsr": fsr, "instability": instability}


def two_stage_gate_decision(fsr, instability, instab_thresh, fsr_thresh):
    residual_healthy = (fsr <= fsr_thresh) and (instability <= instab_thresh)
    if residual_healthy:
        return "stage1_healthy", 1.0
    else:
        # Stage 2: utility-based — in this simplified version we compute
        # a "gate_quality" that modulates how much of the budget goes to
        # exploitation vs exploration-like behavior
        penalty = 2.0 * fsr + 1.0 * instability
        gate_quality = max(0.2, 1.0 - penalty * 0.3)
        return "stage2_utility", gate_quality


# ═══════════════════════════════════════════════════════════════════════
# Single configuration run
# ═══════════════════════════════════════════════════════════════════════

def run_single(space, scene_field, seed, instab_thresh, fsr_thresh, budget_frac):
    rng = np.random.RandomState(seed)

    # Generate scene-dependent field size
    N_field = scene_field.shape[0]
    space.setup(scene_field)

    if hasattr(space, 'coarse'):
        state = space.coarse.copy()
    else:
        state = np.zeros_like(space.gt)

    units = space.get_units()
    n_units = len(units)
    budget = max(1, int(budget_frac * n_units))

    # Probe: 15% of units
    n_probe = max(2, int(0.15 * n_units))
    probe_indices = rng.choice(n_units, size=min(n_probe, n_units), replace=False)
    probe_mask = np.zeros(n_units, dtype=bool)
    probe_mask[probe_indices] = True

    # Run probe diagnostics
    diag = run_probe_diagnostics(space, state, units, probe_mask, seed)

    # Two-stage gate decision
    stage, gate_quality = two_stage_gate_decision(
        diag["fsr"], diag["instability"], instab_thresh, fsr_thresh)

    # Select and refine units
    errors = np.array([space.unit_error(state, u) for u in units])
    order = np.argsort(errors)[::-1]  # highest error first

    n_refine = min(budget, n_units)
    # gate_quality modulates effective budget utilization
    n_effective = max(1, int(n_refine * gate_quality))

    refined_count = 0
    seam_scores = []
    for idx in order[:n_effective]:
        u = units[idx]
        ss_before = space.seam_score(state, u)
        state = space.refine_unit(state, u)
        ss_after = space.seam_score(state, u)
        seam_scores.append(ss_after)
        refined_count += 1

    quality = space.quality_metric(state)
    avg_seam = float(np.mean(seam_scores)) if seam_scores else 0.0

    return {
        "quality": quality,
        "seam_score": avg_seam,
        "cost": refined_count,
        "stage": stage,
        "gate_quality": gate_quality,
        "fsr": diag["fsr"],
        "instability": diag["instability"],
    }


# ═══════════════════════════════════════════════════════════════════════
# Sweep engine
# ═══════════════════════════════════════════════════════════════════════

def compute_ridge_width(perf_grid, threshold_values, axis, degrade_pct=DEGRADE_5PCT):
    """Compute ridge width along one axis of the performance grid.

    perf_grid: 2D array (n_instab x n_fsr), mean quality values
    threshold_values: 1D array of threshold multipliers for the axis
    axis: 0 = instability axis, 1 = FSR axis
    degrade_pct: fraction below optimum that is still acceptable

    Returns ridge_width as fraction of sweep range [0, 1].
    """
    optimal = np.nanmax(perf_grid)
    if optimal <= 0 or np.isnan(optimal):
        return 0.0

    cutoff = optimal * (1 - degrade_pct)

    # For each slice along the other axis, find the width of acceptable region
    widths = []
    n_slices = perf_grid.shape[1 - axis]
    for i in range(n_slices):
        if axis == 0:
            slc = perf_grid[:, i]
        else:
            slc = perf_grid[i, :]
        acceptable = slc >= cutoff
        if not acceptable.any():
            widths.append(0.0)
            continue
        indices = np.where(acceptable)[0]
        lo, hi = indices[0], indices[-1]
        width = (np.log10(threshold_values[hi]) - np.log10(threshold_values[lo]))
        total_range = np.log10(threshold_values[-1]) - np.log10(threshold_values[0])
        widths.append(width / total_range if total_range > 0 else 0.0)

    return float(np.median(widths))


def compute_ridge_2d(perf_grid, instab_mults, fsr_mults, degrade_pct=DEGRADE_5PCT):
    """Compute 2D ridge width: fraction of (instab x fsr) grid within 5% of optimal."""
    optimal = np.nanmax(perf_grid)
    if optimal <= 0 or np.isnan(optimal):
        return 0.0
    cutoff = optimal * (1 - degrade_pct)
    n_acceptable = np.sum(perf_grid >= cutoff)
    n_total = perf_grid.size
    return float(n_acceptable / n_total)


def run_sweep():
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    instab_values = DEFAULTS["instability_threshold"] * INSTAB_MULTS
    fsr_values = DEFAULTS["FSR_threshold"] * FSR_MULTS

    all_results = {}
    ridge_summary = {}

    total_configs = len(SCENE_TYPES) * len(SPACE_TYPES) * N_GRID * N_GRID * N_SEEDS
    print(f"P2a Sensitivity Sweep")
    print(f"  Scenes: {len(SCENE_TYPES)}, Spaces: {len(SPACE_TYPES)}")
    print(f"  Grid: {N_GRID}x{N_GRID} thresholds, {N_SEEDS} seeds")
    print(f"  Total configurations: {total_configs}")
    print(f"  Instability range: [{instab_values[0]:.4f}, {instab_values[-1]:.4f}]")
    print(f"  FSR range: [{fsr_values[0]:.4f}, {fsr_values[-1]:.4f}]")
    print("=" * 70)

    for scene_name in SCENE_TYPES:
        all_results[scene_name] = {}
        ridge_summary[scene_name] = {}
        scene_gen = SCENE_GENERATORS[scene_name]

        for space_name in SPACE_TYPES:
            print(f"\n[{scene_name} x {space_name}]", end="", flush=True)
            space_factory = SPACE_FACTORIES[space_name]

            # Performance grids: (n_instab, n_fsr)
            quality_grid = np.zeros((N_GRID, N_GRID))
            seam_grid = np.zeros((N_GRID, N_GRID))
            cost_grid = np.zeros((N_GRID, N_GRID))
            quality_std_grid = np.zeros((N_GRID, N_GRID))

            for ii, instab_mult in enumerate(INSTAB_MULTS):
                for fi, fsr_mult in enumerate(FSR_MULTS):
                    it = DEFAULTS["instability_threshold"] * instab_mult
                    ft = DEFAULTS["FSR_threshold"] * fsr_mult

                    quals, seams, costs = [], [], []
                    for seed in range(N_SEEDS):
                        space = space_factory()
                        # Use 64x64 field for all scenes
                        field = scene_gen(64, seed)
                        result = run_single(space, field, seed, it, ft, BUDGET_FRACTION)
                        quals.append(result["quality"])
                        seams.append(result["seam_score"])
                        costs.append(result["cost"])

                    quality_grid[ii, fi] = np.mean(quals)
                    quality_std_grid[ii, fi] = np.std(quals)
                    seam_grid[ii, fi] = np.mean(seams)
                    cost_grid[ii, fi] = np.mean(costs)

                print(".", end="", flush=True)

            # Compute ridge widths
            rw_instab = compute_ridge_width(quality_grid, INSTAB_MULTS, axis=0)
            rw_fsr = compute_ridge_width(quality_grid, FSR_MULTS, axis=1)
            rw_2d = compute_ridge_2d(quality_grid, INSTAB_MULTS, FSR_MULTS)

            ridge_summary[scene_name][space_name] = {
                "ridge_instab": rw_instab,
                "ridge_fsr": rw_fsr,
                "ridge_2d": rw_2d,
                "optimal_quality": float(np.nanmax(quality_grid)),
                "quality_at_default": float(quality_grid[N_GRID // 2, N_GRID // 2]),
            }

            all_results[scene_name][space_name] = {
                "quality_grid": quality_grid.tolist(),
                "quality_std_grid": quality_std_grid.tolist(),
                "seam_grid": seam_grid.tolist(),
                "cost_grid": cost_grid.tolist(),
                "instab_values": instab_values.tolist(),
                "fsr_values": fsr_values.tolist(),
            }

            print(f" rw_i={rw_instab:.2%} rw_f={rw_fsr:.2%} rw_2d={rw_2d:.2%}")

    # ── Analysis ──────────────────────────────────────────────────────

    print("\n" + "=" * 90)
    print("RIDGE WIDTH SUMMARY")
    print("=" * 90)
    print(f"{'Scene':<20s} {'Space':<20s} {'RW_instab':>10s} {'RW_fsr':>10s} "
          f"{'RW_2d':>10s} {'Verdict':>12s}")
    print("-" * 90)

    verdicts = []
    for scene_name in SCENE_TYPES:
        for space_name in SPACE_TYPES:
            r = ridge_summary[scene_name][space_name]
            rw = r["ridge_2d"]
            if rw > 0.30:
                verdict = "PASS (wide)"
            elif rw < 0.10:
                verdict = "FAIL (narrow)"
            else:
                verdict = "INCONCLUSIVE"
            verdicts.append({
                "scene": scene_name, "space": space_name,
                "rw_2d": rw, "verdict": verdict,
            })
            print(f"{scene_name:<20s} {space_name:<20s} "
                  f"{r['ridge_instab']:>10.2%} {r['ridge_fsr']:>10.2%} "
                  f"{r['ridge_2d']:>10.2%} {verdict:>12s}")

    # Check for cross-space divergence
    print("\n" + "=" * 90)
    print("CROSS-SPACE DIVERGENCE (per scene)")
    print("=" * 90)

    divergence_flags = []
    for scene_name in SCENE_TYPES:
        rws = [ridge_summary[scene_name][sp]["ridge_2d"] for sp in SPACE_TYPES]
        rw_range = max(rws) - min(rws)
        flag = rw_range > 0.15
        divergence_flags.append({
            "scene": scene_name,
            "rw_min": min(rws), "rw_max": max(rws),
            "rw_range": rw_range,
            "divergent": flag,
        })
        status = "DIVERGENT" if flag else "ok"
        print(f"  {scene_name:<20s} range=[{min(rws):.2%}, {max(rws):.2%}] "
              f"spread={rw_range:.2%} {status}")

    # Overall verdict
    n_pass = sum(1 for v in verdicts if v["verdict"].startswith("PASS"))
    n_fail = sum(1 for v in verdicts if v["verdict"].startswith("FAIL"))
    n_inc = sum(1 for v in verdicts if v["verdict"].startswith("INC"))
    n_total = len(verdicts)
    n_divergent = sum(1 for d in divergence_flags if d["divergent"])

    print("\n" + "=" * 90)
    print("FINAL VERDICT")
    print("=" * 90)
    print(f"  Pass: {n_pass}/{n_total}, Fail: {n_fail}/{n_total}, "
          f"Inconclusive: {n_inc}/{n_total}")
    print(f"  Cross-space divergent scenes: {n_divergent}/{len(SCENE_TYPES)}")

    if n_fail == 0 and n_pass >= n_total * 0.6:
        overall = "MANUAL_OK"
        print("  -> Ridge is wide. Manual thresholds are robust. P2b NOT needed.")
    elif n_fail >= n_total * 0.3:
        overall = "P2B_NEEDED"
        print("  -> Ridge is narrow. Auto-tuning (P2b) IS needed.")
    else:
        overall = "INCONCLUSIVE"
        print("  -> Mixed results. Consider targeted P2b for narrow cases.")

    if n_divergent > 0:
        print(f"  WARNING: {n_divergent} scene(s) show significant cross-space divergence.")
        print("  This is an architectural signal: thresholds may need to be space-aware.")

    # ── Save ──────────────────────────────────────────────────────────

    summary = {
        "config": {
            "defaults": DEFAULTS,
            "n_grid": N_GRID,
            "n_seeds": N_SEEDS,
            "budget_fraction": BUDGET_FRACTION,
            "instab_multipliers": INSTAB_MULTS.tolist(),
            "fsr_multipliers": FSR_MULTS.tolist(),
        },
        "ridge_summary": ridge_summary,
        "verdicts": verdicts,
        "divergence_flags": divergence_flags,
        "overall_verdict": overall,
    }

    with open(out_dir / "p2a_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    with open(out_dir / "p2a_full_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # ── Plots ─────────────────────────────────────────────────────────

    plot_heatmaps(all_results, ridge_summary, out_dir)
    plot_ridge_comparison(ridge_summary, out_dir)

    print(f"\nResults saved to {out_dir}/")
    return summary


# ═══════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════

def plot_heatmaps(all_results, ridge_summary, out_dir):
    for scene_name in SCENE_TYPES:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f"P2a Threshold Sensitivity: {scene_name}", fontsize=14, fontweight="bold")

        for idx, space_name in enumerate(SPACE_TYPES):
            ax = axes[idx // 2, idx % 2]
            data = all_results[scene_name][space_name]
            grid = np.array(data["quality_grid"])

            instab_labels = [f"{v:.3f}" for v in data["instab_values"]]
            fsr_labels = [f"{v:.3f}" for v in data["fsr_values"]]

            im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
            ax.set_xticks(range(N_GRID))
            ax.set_xticklabels([f"{m:.1f}x" for m in FSR_MULTS], rotation=45, fontsize=7)
            ax.set_yticks(range(N_GRID))
            ax.set_yticklabels([f"{m:.1f}x" for m in INSTAB_MULTS], fontsize=7)
            ax.set_xlabel("FSR threshold multiplier")
            ax.set_ylabel("Instability threshold multiplier")

            rw = ridge_summary[scene_name][space_name]
            ax.set_title(f"{space_name}\nRW_2d={rw['ridge_2d']:.1%}", fontsize=10)
            plt.colorbar(im, ax=ax, label="Quality (dB)")

            # Mark the default point
            ax.plot(N_GRID // 2, N_GRID // 2, "r*", markersize=12, label="default")

            # Draw 5% contour
            optimal = np.nanmax(grid)
            if optimal > 0:
                cutoff = optimal * (1 - DEGRADE_5PCT)
                ax.contour(grid, levels=[cutoff], colors=["red"], linewidths=1.5,
                           linestyles=["--"])

        plt.tight_layout()
        fig.savefig(out_dir / f"p2a_heatmap_{scene_name}.png", dpi=150, bbox_inches="tight")
        plt.close()


def plot_ridge_comparison(ridge_summary, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("P2a Ridge Width Comparison", fontsize=14, fontweight="bold")

    metrics = [
        ("ridge_instab", "Ridge Width (Instability axis)"),
        ("ridge_fsr", "Ridge Width (FSR axis)"),
        ("ridge_2d", "Ridge Width (2D area)"),
    ]

    for ax_idx, (metric, title) in enumerate(metrics):
        ax = axes[ax_idx]
        x = np.arange(len(SCENE_TYPES))
        width = 0.18
        colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]

        for si, space_name in enumerate(SPACE_TYPES):
            vals = [ridge_summary[sc][space_name][metric] for sc in SCENE_TYPES]
            ax.bar(x + si * width, vals, width, label=space_name, color=colors[si], alpha=0.8)

        ax.axhline(0.30, color="green", ls="--", alpha=0.5, label="Pass (>30%)")
        ax.axhline(0.10, color="red", ls="--", alpha=0.5, label="Fail (<10%)")
        ax.set_xticks(x + 1.5 * width)
        ax.set_xticklabels(SCENE_TYPES, rotation=30, ha="right", fontsize=8)
        ax.set_title(title)
        ax.set_ylabel("Ridge width (fraction)")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(out_dir / "p2a_ridge_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t0 = time.time()
    summary = run_sweep()
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
