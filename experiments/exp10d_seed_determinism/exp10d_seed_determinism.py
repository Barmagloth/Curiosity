#!/usr/bin/env python3
"""
exp10d — Seed Determinism (DET-1)

Verifies bitwise determinism: two runs with identical inputs + seed
must produce identical results.

Three determinism components:
  1. Canonical traversal order (Z-order / Morton tie-break when rho equal)
  2. Deterministic probe (seed = f(coordinates, level, global_seed))
  3. Governor isolation (EMA update only after full step)

Tests ALL four space types x {CPU, GPU} x 10 seeds x 3 budgets.
Kill criterion: ANY divergence = FAIL.

Roadmap: DET-1 (concept_v1.8.md section 8A)
"""

import os
import hashlib
import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import torch


# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

N_SEEDS = 10
BUDGET_LEVELS = {"low": 0.10, "medium": 0.30, "high": 0.60}
SPACE_TYPES = ["scalar_grid", "vector_grid", "irregular_graph", "tree_hierarchy"]

# Grid / space defaults
GRID_N = 64
GRID_TILE = 8
VECTOR_DIM = 16
GRAPH_POINTS = 200
GRAPH_K = 6
GRAPH_CLUSTERS = 10
TREE_DEPTH = 6

# Governor EMA
EMA_ALPHA = 0.1


# ═══════════════════════════════════════════════════════════════════════
# Morton encoding (2D, from exp09a)
# ═══════════════════════════════════════════════════════════════════════

def _part1by1(n: np.ndarray) -> np.ndarray:
    """Spread bits of n for 2D Morton interleave."""
    n = n.astype(np.uint32)
    n = (n | (n << 8)) & np.uint32(0x00FF00FF)
    n = (n | (n << 4)) & np.uint32(0x0F0F0F0F)
    n = (n | (n << 2)) & np.uint32(0x33333333)
    n = (n | (n << 1)) & np.uint32(0x55555555)
    return n


def morton_encode_2d(row: np.ndarray, col: np.ndarray) -> np.ndarray:
    """Return Morton (Z-order) code for 2D coordinates."""
    return _part1by1(col) | (_part1by1(row) << np.uint32(1))


# ═══════════════════════════════════════════════════════════════════════
# Component 1: CanonicalTraversal
# ═══════════════════════════════════════════════════════════════════════

class CanonicalTraversal:
    """Produces a deterministic processing order for units.

    Primary sort: descending rho (highest priority first).
    Tie-break: ascending Morton code (grids) or node index (graph/tree).
    """

    @staticmethod
    def _morton_key_grid(unit: Tuple[int, int]) -> int:
        """Morton code from (row, col) for grid units."""
        row, col = unit
        r = np.array([row], dtype=np.int32)
        c = np.array([col], dtype=np.int32)
        return int(morton_encode_2d(r, c)[0])

    @staticmethod
    def _index_key(unit: int) -> int:
        """Direct node index for graph/tree units."""
        return int(unit)

    @classmethod
    def sort_units(cls, units: list, rho_values: np.ndarray,
                   space_type: str) -> list:
        """Sort units by descending rho with deterministic tie-breaking.

        Args:
            units: list of unit identifiers (tuples for grids, ints for graph/tree)
            rho_values: array of rho values, same length as units
            space_type: one of scalar_grid, vector_grid, irregular_graph, tree_hierarchy

        Returns:
            Ordered list of units.
        """
        assert len(units) == len(rho_values)

        if space_type in ("scalar_grid", "vector_grid"):
            key_fn = cls._morton_key_grid
        else:
            key_fn = cls._index_key

        # Build (neg_rho, tie_break_key, original_index) for stable sort
        decorated = []
        for i, (unit, rho) in enumerate(zip(units, rho_values)):
            decorated.append((-rho, key_fn(unit), i))

        # Sort: primary by -rho (descending rho), secondary by morton/index (ascending)
        decorated.sort()

        return [units[d[2]] for d in decorated]


# ═══════════════════════════════════════════════════════════════════════
# Component 2: DeterministicProbe
# ═══════════════════════════════════════════════════════════════════════

class DeterministicProbe:
    """Deterministic probe unit selection based on hashed seeds."""

    @staticmethod
    def probe_seed(coords: tuple, level: int, global_seed: int) -> int:
        """Compute a deterministic probe seed from coordinates + level + global seed.

        Uses SHA-256 hash truncated to 32-bit integer for reproducibility.
        """
        payload = coords + (level, global_seed)
        h = hashlib.sha256(str(payload).encode("utf-8")).digest()
        return int.from_bytes(h[:4], byteorder="little")

    @classmethod
    def select_probe_units(cls, all_units: list, probe_fraction: float,
                           coords_list: list, level: int,
                           global_seed: int) -> list:
        """Select a deterministic subset of units for probing.

        Args:
            all_units: list of all available units
            probe_fraction: fraction of units to probe (0, 1]
            coords_list: list of coordinate tuples, one per unit
            level: refinement level
            global_seed: global seed for this run

        Returns:
            List of selected probe units (deterministic given identical inputs).
        """
        n_total = len(all_units)
        n_probe = max(1, int(probe_fraction * n_total))

        # Compute a score for each unit from its probe seed
        scores = []
        for i, (unit, coords) in enumerate(zip(all_units, coords_list)):
            s = cls.probe_seed(coords, level, global_seed)
            scores.append((s, i))

        # Sort by seed value (deterministic), take first n_probe
        scores.sort()
        selected_indices = [s[1] for s in scores[:n_probe]]
        return [all_units[i] for i in selected_indices]


# ═══════════════════════════════════════════════════════════════════════
# Component 3: GovernorIsolation
# ═══════════════════════════════════════════════════════════════════════

class GovernorIsolation:
    """EMA-based governor with step-level isolation.

    Key invariant: EMA is updated ONLY after a full step completes,
    never mid-step. This ensures that the order in which branches
    are processed within a step does not affect the EMA state.
    """

    def __init__(self, alpha: float = EMA_ALPHA, initial_ema: float = 0.0):
        self.alpha = alpha
        self.ema = initial_ema
        self._buffer: List[float] = []
        self._step_count = 0

    def accumulate(self, cost: float):
        """Buffer a cost value during the current step."""
        self._buffer.append(cost)

    def commit_step(self) -> float:
        """Apply all buffered costs at once, update EMA.

        Returns the new EMA value after the update.
        """
        if not self._buffer:
            self._step_count += 1
            return self.ema

        # Aggregate: sum of all costs in this step
        step_cost = sum(self._buffer)
        self._buffer.clear()

        # EMA update
        self.ema = self.alpha * step_cost + (1.0 - self.alpha) * self.ema
        self._step_count += 1
        return self.ema

    def get_ema(self) -> float:
        return self.ema

    def get_step_count(self) -> int:
        return self._step_count

    def reset(self, initial_ema: float = 0.0):
        self.ema = initial_ema
        self._buffer.clear()
        self._step_count = 0


# ═══════════════════════════════════════════════════════════════════════
# Space adapters (simplified from p2a / seam experiments)
# ═══════════════════════════════════════════════════════════════════════

class ScalarGridSpace:
    """T1: 2D scalar grid with tile-based units."""
    name = "scalar_grid"

    def __init__(self, N=GRID_N, tile=GRID_TILE, seed=42):
        self.N = N
        self.T = tile
        self.NT = N // tile
        self._seed = seed

    def setup(self, seed: int):
        rng = np.random.default_rng(seed)
        x = np.linspace(0, 1, self.N, endpoint=False)
        xx, yy = np.meshgrid(x, x)
        self.gt = np.zeros((self.N, self.N))
        for _ in range(5):
            cx, cy = rng.uniform(0.1, 0.9, 2)
            sigma = rng.uniform(0.05, 0.2)
            self.gt += rng.uniform(0.3, 1.0) * np.exp(
                -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
        self.gt += 0.5 * (xx > 0.5)

        self.coarse = np.zeros_like(self.gt)
        for ti in range(self.NT):
            for tj in range(self.NT):
                s = slice(ti * self.T, (ti + 1) * self.T)
                cs = slice(tj * self.T, (tj + 1) * self.T)
                self.coarse[s, cs] = self.gt[s, cs].mean()

    def get_units(self) -> list:
        return [(i, j) for i in range(self.NT) for j in range(self.NT)]

    def get_coords(self, units: list) -> list:
        return [u for u in units]  # (row, col) tuples

    def unit_rho(self, state: np.ndarray, unit: tuple) -> float:
        ti, tj = unit
        s = slice(ti * self.T, (ti + 1) * self.T)
        cs = slice(tj * self.T, (tj + 1) * self.T)
        return float(np.mean((self.gt[s, cs] - state[s, cs]) ** 2))

    def refine_unit(self, state: np.ndarray, unit: tuple,
                    halo: int = 2) -> np.ndarray:
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

    def state_to_tensor(self, state: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(state.copy())

    def get_initial_state(self) -> np.ndarray:
        return self.coarse.copy()


class VectorGridSpace:
    """T2: 2D vector-valued grid."""
    name = "vector_grid"

    def __init__(self, N=32, tile=GRID_TILE, D=VECTOR_DIM, seed=42):
        self.N = N
        self.T = tile
        self.NT = N // tile
        self.D = D

    def setup(self, seed: int):
        rng = np.random.default_rng(seed)
        x = np.linspace(0, 1, self.N, endpoint=False)
        xx, yy = np.meshgrid(x, x)
        self.gt = np.zeros((self.N, self.N, self.D))
        for d in range(self.D):
            freq = 2 + d * 0.5
            phase = rng.uniform(0, 2 * np.pi)
            amp = 0.5 / (1 + d * 0.1)
            self.gt[:, :, d] = amp * np.sin(
                2 * np.pi * freq * xx + phase) * np.cos(
                2 * np.pi * (freq * 0.7) * yy)
        self.gt += rng.standard_normal((self.N, self.N, self.D)) * 0.02

        self.coarse = np.zeros_like(self.gt)
        for ti in range(self.NT):
            for tj in range(self.NT):
                s = slice(ti * self.T, (ti + 1) * self.T)
                cs = slice(tj * self.T, (tj + 1) * self.T)
                self.coarse[s, cs, :] = self.gt[s, cs, :].mean(
                    axis=(0, 1), keepdims=True)

    def get_units(self) -> list:
        return [(i, j) for i in range(self.NT) for j in range(self.NT)]

    def get_coords(self, units: list) -> list:
        return [u for u in units]

    def unit_rho(self, state: np.ndarray, unit: tuple) -> float:
        ti, tj = unit
        s = slice(ti * self.T, (ti + 1) * self.T)
        cs = slice(tj * self.T, (tj + 1) * self.T)
        return float(np.mean((self.gt[s, cs, :] - state[s, cs, :]) ** 2))

    def refine_unit(self, state: np.ndarray, unit: tuple,
                    halo: int = 2) -> np.ndarray:
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

    def state_to_tensor(self, state: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(state.copy())

    def get_initial_state(self) -> np.ndarray:
        return self.coarse.copy()


class IrregularGraphSpace:
    """T3: Irregular graph (k-NN point cloud)."""
    name = "irregular_graph"

    def __init__(self, n_points=GRAPH_POINTS, k=GRAPH_K):
        self.n_pts = n_points
        self.k = k
        self.n_clusters = None  # set by community detection in setup()

    def setup(self, seed: int):
        from scipy.spatial import cKDTree

        rng = np.random.default_rng(seed)

        self.pos = rng.random((self.n_pts, 2))
        tree = cKDTree(self.pos)
        _, idx = tree.query(self.pos, k=self.k + 1)
        self.neighbors = {i: set(idx[i, 1:]) for i in range(self.n_pts)}

        self.gt = (0.5 * np.sin(4 * np.pi * self.pos[:, 0])
                   * np.cos(3 * np.pi * self.pos[:, 1])
                   + 0.7 * (self.pos[:, 0] > 0.4).astype(float)
                   + rng.standard_normal(self.n_pts) * 0.03)

        # ------------------------------------------------------------------
        # Community detection: topology-only clustering.
        # Leiden preferred (connected guaranteed).  Louvain+CC fallback
        # for environments where igraph C-extensions won't compile (ARM etc).
        # Both paths produce identical semantics: connected communities.
        # ------------------------------------------------------------------
        self.labels, self.n_clusters, self._cluster_backend = \
            self._community_detect(seed)

        self.coarse = np.zeros(self.n_pts)
        for c in range(self.n_clusters):
            mask = self.labels == c
            if mask.any():
                self.coarse[mask] = self.gt[mask].mean()

    def _community_detect(self, seed: int):
        """Topology-only community detection.

        Strategy:
          1. Leiden (igraph + leidenalg) — native connected communities.
          2. Fallback: NetworkX Louvain + connected_components post-fix.
             Same semantics, zero C-dependencies.

        Not a performance decision — both are O(N log N).
        This is deployment resilience: igraph needs a C compiler,
        networkx doesn't.
        """
        try:
            import igraph as ig
            import leidenalg
            return self._cluster_leiden(seed, ig, leidenalg)
        except ImportError:
            return self._cluster_louvain_cc(seed)

    def _cluster_leiden(self, seed, ig, leidenalg):
        edges = [(i, j)
                 for i, nbrs in self.neighbors.items()
                 for j in nbrs if j > i]
        G = ig.Graph(n=self.n_pts, edges=edges, directed=False)
        partition = leidenalg.find_partition(
            G, leidenalg.ModularityVertexPartition, seed=seed)
        labels = np.array(partition.membership)
        return labels, len(set(labels)), "leiden"

    def _cluster_louvain_cc(self, seed):
        """Louvain + connected_components post-fix.

        Louvain may produce disconnected clusters.
        CC splits them in O(V+E).  Result: same connected-community
        guarantee as Leiden, zero C-dependencies.
        """
        import networkx as nx

        G = nx.Graph()
        G.add_nodes_from(range(self.n_pts))
        for i, nbrs in self.neighbors.items():
            for j in nbrs:
                G.add_edge(i, j)

        communities = nx.algorithms.community.louvain_communities(
            G, seed=seed, resolution=1.0)

        labels = np.zeros(self.n_pts, dtype=int)
        for cid, nodes in enumerate(communities):
            for n in nodes:
                labels[n] = cid

        # CC post-fix: split disconnected clusters
        next_id = len(communities)
        for cid, nodes in enumerate(communities):
            subg = G.subgraph(nodes)
            components = list(nx.connected_components(subg))
            if len(components) > 1:
                for comp in components[1:]:
                    for node in comp:
                        labels[node] = next_id
                    next_id += 1

        # Re-pack to 0..N-1 (no gaps after splits)
        unique = sorted(set(labels))
        remap = {old: new for new, old in enumerate(unique)}
        labels = np.array([remap[l] for l in labels])
        return labels, len(unique), "louvain+cc"

    def get_units(self) -> list:
        return list(range(self.n_clusters))

    def get_coords(self, units: list) -> list:
        # For graph: use cluster centroid as coords
        coords = []
        for u in units:
            pts = np.where(self.labels == u)[0]
            if len(pts) > 0:
                centroid = tuple(self.pos[pts].mean(axis=0).tolist())
            else:
                centroid = (0.0, 0.0)
            coords.append((u,) + centroid)
        return coords

    def _cluster_pts(self, cid: int) -> set:
        return set(np.where(self.labels == cid)[0])

    def _halo(self, cluster: set, hops: int = 1) -> set:
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

    def unit_rho(self, state: np.ndarray, unit: int) -> float:
        pts = self._cluster_pts(unit)
        if not pts:
            return 0.0
        return float(np.mean([(self.gt[p] - state[p]) ** 2 for p in pts]))

    def refine_unit(self, state: np.ndarray, unit: int,
                    halo_hops: int = 1) -> np.ndarray:
        cluster = self._cluster_pts(unit)
        halo = self._halo(cluster, halo_hops)
        out = state.copy()
        for p in sorted(cluster):  # sorted for determinism
            out[p] = state[p] + (self.gt[p] - state[p])
        for p in sorted(halo):  # sorted for determinism
            out[p] = state[p] + (self.gt[p] - state[p]) * 0.3
        return out

    def state_to_tensor(self, state: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(state.copy())

    def get_initial_state(self) -> np.ndarray:
        return self.coarse.copy()


class TreeHierarchySpace:
    """T4: Binary tree hierarchy."""
    name = "tree_hierarchy"

    def __init__(self, depth=TREE_DEPTH):
        self.depth = depth
        self.n = 2 ** depth - 1

    def setup(self, seed: int):
        rng = np.random.default_rng(seed)
        self.gt = np.zeros(self.n)
        for i in range(self.n):
            d = int(np.log2(i + 1))
            self.gt[i] = (0.5 * np.sin(i * 0.3) / (1 + d * 0.2)
                          + rng.standard_normal() * 0.05)
            if d >= 3:
                self.gt[i] += 0.3 * ((i % 7) > 3)

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

    def _children(self, i: int) -> list:
        left = 2 * i + 1
        right = 2 * i + 2
        c = []
        if left < self.n:
            c.append(left)
        if right < self.n:
            c.append(right)
        return c

    def _parent(self, i: int) -> Optional[int]:
        return (i - 1) // 2 if i > 0 else None

    def _subtree(self, i: int) -> list:
        nodes = [i]
        q = [i]
        while q:
            curr = q.pop()
            for c in self._children(curr):
                nodes.append(c)
                q.append(c)
        return nodes

    def _neighbors(self, i: int) -> set:
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

    def get_units(self) -> list:
        # Subtrees rooted at depth 3
        return [i for i in range(self.n) if int(np.log2(i + 1)) == 3]

    def get_coords(self, units: list) -> list:
        # BFS index as coordinate
        return [(u,) for u in units]

    def unit_rho(self, state: np.ndarray, unit: int) -> float:
        pts = self._subtree(unit)
        return float(np.mean([(self.gt[p] - state[p]) ** 2 for p in pts]))

    def refine_unit(self, state: np.ndarray, unit: int,
                    halo_hops: int = 1) -> np.ndarray:
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
        for p in sorted(core):  # sorted for determinism
            out[p] = state[p] + (self.gt[p] - state[p])
        for p in sorted(halo):  # sorted for determinism
            out[p] = state[p] + (self.gt[p] - state[p]) * 0.3
        return out

    def state_to_tensor(self, state: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(state.copy())

    def get_initial_state(self) -> np.ndarray:
        return self.coarse.copy()


SPACE_FACTORIES = {
    "scalar_grid": lambda: ScalarGridSpace(N=GRID_N, tile=GRID_TILE),
    "vector_grid": lambda: VectorGridSpace(N=32, tile=GRID_TILE, D=VECTOR_DIM),
    "irregular_graph": lambda: IrregularGraphSpace(
        n_points=GRAPH_POINTS, k=GRAPH_K),
    "tree_hierarchy": lambda: TreeHierarchySpace(depth=TREE_DEPTH),
}


# ═══════════════════════════════════════════════════════════════════════
# AdaptivePipeline — combines all three determinism components
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TreeResult:
    """Full result of a pipeline run, used for bitwise comparison."""
    node_values: np.ndarray            # final state (all values)
    split_decisions: List[bool]        # which units were refined
    refinement_deltas: List[np.ndarray] # delta at each step
    governor_ema_history: List[float]   # EMA after each step
    traversal_order: list               # canonical order of units
    probe_set: list                     # selected probe units


class AdaptivePipeline:
    """Adaptive refinement pipeline with deterministic guarantees.

    Combines:
      - CanonicalTraversal for processing order
      - DeterministicProbe for probe selection
      - GovernorIsolation for step-isolated EMA
    """

    def __init__(self, probe_fraction: float = 0.15):
        self.probe_fraction = probe_fraction

    def run(self, space, seed: int, budget_fraction: float) -> TreeResult:
        """Run the adaptive pipeline.

        Args:
            space: a space adapter (setup() must have been called)
            seed: global seed for this run
            budget_fraction: fraction of units to refine

        Returns:
            TreeResult with all state for bitwise comparison.
        """
        # Seed all RNGs deterministically
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        state = space.get_initial_state()
        units = space.get_units()
        coords_list = space.get_coords(units)
        n_units = len(units)
        budget = max(1, int(budget_fraction * n_units))

        # --- Deterministic probe selection ---
        probe_set = DeterministicProbe.select_probe_units(
            units, self.probe_fraction, coords_list, level=0,
            global_seed=seed)

        # --- Compute rho for all units ---
        rho_values = np.array([space.unit_rho(state, u) for u in units])

        # --- Canonical traversal order ---
        ordered_units = CanonicalTraversal.sort_units(
            units, rho_values, space.name)

        # --- Governor-isolated refinement ---
        governor = GovernorIsolation(alpha=EMA_ALPHA)
        split_decisions = []
        refinement_deltas = []
        ema_history = []

        refined_count = 0
        for unit in ordered_units:
            if refined_count >= budget:
                split_decisions.append(False)
                continue

            # Decide: refine if rho is above governor threshold
            rho = space.unit_rho(state, unit)
            threshold = governor.get_ema() * 0.5  # simple threshold rule

            if rho > threshold or refined_count < max(1, budget // 3):
                # Refine
                old_state = state.copy()
                state = space.refine_unit(state, unit)
                delta = state - old_state if isinstance(state, np.ndarray) else None

                # Accumulate cost for governor (cost = 1 per refinement)
                governor.accumulate(1.0)

                split_decisions.append(True)
                refinement_deltas.append(
                    delta.copy() if delta is not None else np.array([0.0]))
                refined_count += 1
            else:
                split_decisions.append(False)

            # Commit governor step after each unit (step = one unit decision)
            ema_val = governor.commit_step()
            ema_history.append(ema_val)

        return TreeResult(
            node_values=state.copy(),
            split_decisions=split_decisions,
            refinement_deltas=refinement_deltas,
            governor_ema_history=ema_history,
            traversal_order=ordered_units,
            probe_set=probe_set,
        )


# ═══════════════════════════════════════════════════════════════════════
# Bitwise comparison utilities
# ═══════════════════════════════════════════════════════════════════════

def compare_results_bitwise(r1: TreeResult, r2: TreeResult) -> Dict[str, Any]:
    """Compare two TreeResults for bitwise identity.

    Returns dict with per-field pass/fail and divergence details.
    """
    report = {}

    # Node values
    vals_equal = np.array_equal(r1.node_values, r2.node_values)
    report["node_values"] = {
        "equal": vals_equal,
        "max_diff": float(np.max(np.abs(
            r1.node_values - r2.node_values))) if not vals_equal else 0.0,
        "n_different": int(np.sum(
            r1.node_values != r2.node_values)) if not vals_equal else 0,
    }

    # Split decisions
    splits_equal = (r1.split_decisions == r2.split_decisions)
    report["split_decisions"] = {"equal": splits_equal}

    # Refinement deltas
    deltas_equal = True
    if len(r1.refinement_deltas) != len(r2.refinement_deltas):
        deltas_equal = False
    else:
        for d1, d2 in zip(r1.refinement_deltas, r2.refinement_deltas):
            if not np.array_equal(d1, d2):
                deltas_equal = False
                break
    report["refinement_deltas"] = {"equal": deltas_equal}

    # Governor EMA history
    ema_equal = (r1.governor_ema_history == r2.governor_ema_history)
    report["governor_ema_history"] = {"equal": ema_equal}
    if not ema_equal:
        diffs = [abs(a - b) for a, b in zip(
            r1.governor_ema_history, r2.governor_ema_history)]
        report["governor_ema_history"]["max_diff"] = max(diffs) if diffs else 0.0

    # Traversal order
    order_equal = (r1.traversal_order == r2.traversal_order)
    report["traversal_order"] = {"equal": order_equal}

    # Probe set
    probe_equal = (r1.probe_set == r2.probe_set)
    report["probe_set"] = {"equal": probe_equal}

    # Overall
    report["all_equal"] = all(
        report[k]["equal"]
        for k in ["node_values", "split_decisions", "refinement_deltas",
                   "governor_ema_history", "traversal_order", "probe_set"]
    )

    return report


def compare_tensors_bitwise(t1: torch.Tensor, t2: torch.Tensor) -> Dict[str, Any]:
    """Compare two torch tensors for bitwise equality."""
    equal = torch.equal(t1, t2)
    result = {"equal": equal}
    if not equal:
        diff = (t1 - t2).abs()
        result["max_diff"] = float(diff.max().item())
        result["n_different"] = int((t1 != t2).sum().item())
    return result


# ═══════════════════════════════════════════════════════════════════════
# GPU determinism setup
# ═══════════════════════════════════════════════════════════════════════

def setup_gpu_determinism():
    """Configure PyTorch for deterministic GPU operations.

    Sets:
      - torch.use_deterministic_algorithms(True)
      - CUBLAS_WORKSPACE_CONFIG=:4096:8
    """
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        torch.use_deterministic_algorithms(True)
        return {"success": True, "error": None}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════
# Main experiment runner
# ═══════════════════════════════════════════════════════════════════════

def run_determinism_test(space_type: str, device: str, seed: int,
                         budget_name: str, budget_frac: float) -> Dict[str, Any]:
    """Run two identical pipeline executions and compare bitwise.

    Returns dict with pass/fail, comparison details, and timing.
    """
    space1 = SPACE_FACTORIES[space_type]()
    space2 = SPACE_FACTORIES[space_type]()

    # Identical setup with same seed
    space1.setup(seed)
    space2.setup(seed)

    pipeline = AdaptivePipeline(probe_fraction=0.15)

    # Run 1
    t0 = time.perf_counter()
    result1 = pipeline.run(space1, seed, budget_frac)
    t1 = time.perf_counter()

    # Run 2 (identical)
    result2 = pipeline.run(space2, seed, budget_frac)
    t2 = time.perf_counter()

    # Bitwise comparison of numpy state
    comparison = compare_results_bitwise(result1, result2)

    # Also verify torch tensor comparison
    tensor1 = space1.state_to_tensor(result1.node_values)
    tensor2 = space2.state_to_tensor(result2.node_values)

    if device == "cuda" and torch.cuda.is_available():
        tensor1 = tensor1.to("cuda")
        tensor2 = tensor2.to("cuda")

    tensor_cmp = compare_tensors_bitwise(tensor1, tensor2)
    comparison["tensor_comparison"] = tensor_cmp

    # Update overall pass
    comparison["all_equal"] = comparison["all_equal"] and tensor_cmp["equal"]

    return {
        "space": space_type,
        "device": device,
        "seed": seed,
        "budget": budget_name,
        "budget_frac": budget_frac,
        "pass": comparison["all_equal"],
        "comparison": comparison,
        "time_run1_s": t1 - t0,
        "time_run2_s": t2 - t1,
    }


def run_experiment():
    """Run full determinism experiment across all configurations."""
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("exp10d — Seed Determinism (DET-1)")
    print("=" * 70)
    print(f"  Spaces: {SPACE_TYPES}")
    print(f"  Seeds: {N_SEEDS}")
    print(f"  Budgets: {list(BUDGET_LEVELS.keys())}")
    print()

    # Determine available devices
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
        gpu_setup = setup_gpu_determinism()
        print(f"  GPU determinism setup: {gpu_setup}")
    else:
        print("  GPU not available, skipping CUDA tests.")
    print()

    all_results = []
    n_pass = 0
    n_fail = 0
    n_total = 0

    for space_type in SPACE_TYPES:
        for device in devices:
            for budget_name, budget_frac in BUDGET_LEVELS.items():
                for seed in range(N_SEEDS):
                    n_total += 1
                    try:
                        result = run_determinism_test(
                            space_type, device, seed, budget_name, budget_frac)
                        all_results.append(result)

                        if result["pass"]:
                            n_pass += 1
                            status = "PASS"
                        else:
                            n_fail += 1
                            status = "FAIL"

                    except Exception as e:
                        n_fail += 1
                        status = f"ERROR: {e}"
                        all_results.append({
                            "space": space_type,
                            "device": device,
                            "seed": seed,
                            "budget": budget_name,
                            "budget_frac": budget_frac,
                            "pass": False,
                            "error": str(e),
                        })

                print(f"  [{space_type:18s} {device:4s} {budget_name:6s}] "
                      f"{N_SEEDS} seeds: "
                      f"{sum(1 for r in all_results[-N_SEEDS:] if r.get('pass', False))}"
                      f"/{N_SEEDS} pass")

    # ── Summary ──────────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total tests: {n_total}")
    print(f"  Pass: {n_pass}")
    print(f"  Fail: {n_fail}")

    overall_pass = (n_fail == 0)

    if overall_pass:
        verdict = "PASS — bitwise determinism verified"
    else:
        verdict = "FAIL — divergence detected"

    print(f"\n  VERDICT: {verdict}")

    # Failure details
    failures = [r for r in all_results if not r.get("pass", False)]
    if failures:
        print(f"\n  FAILURE DETAILS ({len(failures)} failures):")
        for f in failures[:20]:  # cap output
            err = f.get("error", "")
            cmp = f.get("comparison", {})
            details = []
            for field_name in ["node_values", "split_decisions",
                               "refinement_deltas", "governor_ema_history",
                               "traversal_order", "probe_set",
                               "tensor_comparison"]:
                if field_name in cmp and not cmp[field_name].get("equal", True):
                    details.append(field_name)
            detail_str = ", ".join(details) if details else err
            print(f"    {f['space']:18s} {f['device']:4s} seed={f['seed']} "
                  f"budget={f['budget']:6s}: {detail_str}")

    # ── Save results ─────────────────────────────────────────────────

    # Prepare JSON-serializable results
    json_results = []
    for r in all_results:
        jr = {k: v for k, v in r.items() if k != "comparison"}
        if "comparison" in r:
            # Flatten comparison for JSON
            cmp_flat = {}
            for field_name, field_val in r["comparison"].items():
                if isinstance(field_val, dict):
                    for k2, v2 in field_val.items():
                        cmp_flat[f"{field_name}_{k2}"] = v2
                else:
                    cmp_flat[field_name] = field_val
            jr["comparison"] = cmp_flat
        json_results.append(jr)

    results_path = out_dir / "exp10d_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "config": {
                "n_seeds": N_SEEDS,
                "budget_levels": BUDGET_LEVELS,
                "space_types": SPACE_TYPES,
                "devices": devices,
            },
            "verdict": verdict,
            "n_total": n_total,
            "n_pass": n_pass,
            "n_fail": n_fail,
            "results": json_results,
        }, f, indent=2, default=str)

    # ── Generate report ──────────────────────────────────────────────

    report_lines = [
        "# exp10d — Seed Determinism Report",
        "",
        f"**Verdict:** {verdict}",
        f"**Total tests:** {n_total}",
        f"**Pass:** {n_pass}",
        f"**Fail:** {n_fail}",
        "",
        "## Configuration",
        "",
        f"- Seeds: {N_SEEDS} (0..{N_SEEDS - 1})",
        f"- Budgets: {list(BUDGET_LEVELS.items())}",
        f"- Spaces: {SPACE_TYPES}",
        f"- Devices: {devices}",
        "",
        "## Results by space x device",
        "",
        "| Space | Device | Budget | Pass | Fail |",
        "|-------|--------|--------|------|------|",
    ]

    for space_type in SPACE_TYPES:
        for device in devices:
            for budget_name in BUDGET_LEVELS:
                matching = [
                    r for r in all_results
                    if r["space"] == space_type
                    and r["device"] == device
                    and r["budget"] == budget_name
                ]
                np_ = sum(1 for r in matching if r.get("pass", False))
                nf_ = len(matching) - np_
                report_lines.append(
                    f"| {space_type} | {device} | {budget_name} | "
                    f"{np_}/{len(matching)} | {nf_}/{len(matching)} |")

    if failures:
        report_lines.extend([
            "",
            "## Divergence details",
            "",
        ])
        for f in failures[:20]:
            err = f.get("error", "")
            cmp = f.get("comparison", {})
            details = []
            for field_name in ["node_values", "split_decisions",
                               "refinement_deltas", "governor_ema_history",
                               "traversal_order", "probe_set",
                               "tensor_comparison"]:
                if field_name in cmp and not cmp[field_name].get("equal", True):
                    info = cmp[field_name]
                    detail = f"{field_name}"
                    if "max_diff" in info:
                        detail += f" (max_diff={info['max_diff']:.2e})"
                    if "n_different" in info:
                        detail += f" (n_diff={info['n_different']})"
                    details.append(detail)
            detail_str = ", ".join(details) if details else err
            report_lines.append(
                f"- **{f['space']}** {f['device']} seed={f['seed']} "
                f"budget={f['budget']}: {detail_str}")

    report_lines.extend(["", "## Determinism components tested", ""])
    report_lines.append(
        "1. **Canonical traversal**: Z-order (Morton) tie-break "
        "when rho values are equal")
    report_lines.append(
        "2. **Deterministic probe**: SHA-256 hash of "
        "(coords, level, global_seed)")
    report_lines.append(
        "3. **Governor isolation**: EMA update only after "
        "full step commit")

    report_path = out_dir / "exp10d_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"\n  Results: {results_path}")
    print(f"  Report:  {report_path}")

    return {
        "verdict": verdict,
        "n_total": n_total,
        "n_pass": n_pass,
        "n_fail": n_fail,
    }


# ═══════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t0 = time.time()
    summary = run_experiment()
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
