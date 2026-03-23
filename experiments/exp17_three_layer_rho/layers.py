"""Three-Layer rho decomposition.

Layer 0: Topology   — "How is the space structured?" (data-independent)
Layer 1: Presence   — "Where is there data?"          (data-dependent, query-independent)
Layer 2: Query      — "Where does THIS metric need refinement?" (task-specific)

Each layer narrows the working set for the next.
The tree from L0+L1 is a reusable index.
"""
import sys
import time
import hashlib
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Callable

# Add pipeline imports
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent / 'exp_phase2_pipeline'))
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent / 'exp10d_seed_determinism'))

from exp10d_seed_determinism import SPACE_FACTORIES, CanonicalTraversal
from topo_features import extract_topo_features, topo_adjusted_rho, TopoFeatures
from config import PipelineConfig


# =====================================================================
# Data classes for layer results
# =====================================================================

@dataclass
class L0Result:
    """Layer 0 output: topological structure scores."""
    topo_scores: np.ndarray       # per-unit topology importance [0, 1]
    topo_features: Optional[Any]  # TopoFeatures for graph, None otherwise
    zone: str                     # GREEN/YELLOW/RED/n_a
    unit_mask: np.ndarray         # bool: units that pass L0 threshold
    cluster_ids: np.ndarray       # int: cluster assignment per unit (from L0 topology)
    n_clusters: int               # number of distinct clusters
    n_total: int
    n_surviving: int
    computation_ms: float


@dataclass
class L1Result:
    """Layer 1 output: data presence scores."""
    presence_scores: np.ndarray   # per-unit data presence [0, inf)
    active_mask: np.ndarray       # bool: units that pass L1 threshold
    n_input: int                  # units from L0
    n_surviving: int              # units after L1 gate
    n_pruned_l0: int
    n_pruned_l1: int
    computation_ms: float


@dataclass
class L2Result:
    """Layer 2 output: query-specific refinement result."""
    query_fn: str
    rho_values: np.ndarray        # per-surviving-unit rho
    ordered_units: list           # units sorted by rho descending
    n_refined: int
    psnr_final: float
    mse_final: float
    computation_ms: float


@dataclass
class FrozenTree:
    """Serializable snapshot of Layers 0+1."""
    space_type: str
    seed: int
    l0_scores: np.ndarray
    l1_scores: np.ndarray
    active_units: list            # units that passed both gates
    all_units: list               # all units in the space
    zone: str
    l0_ms: float
    l1_ms: float
    memory_bytes: int

    def to_dict(self) -> dict:
        return {
            'space_type': self.space_type,
            'seed': self.seed,
            'l0_scores': self.l0_scores.tolist(),
            'l1_scores': self.l1_scores.tolist(),
            'active_units': [str(u) for u in self.active_units],
            'n_active': len(self.active_units),
            'n_total': len(self.all_units),
            'zone': self.zone,
            'l0_ms': self.l0_ms,
            'l1_ms': self.l1_ms,
            'memory_bytes': self.memory_bytes,
        }


# =====================================================================
# Layer 0: Topology (data-independent)
# =====================================================================

class Layer0_Topology:
    """Score units by structural importance of the space.

    Does NOT look at data — only at space geometry.
    """

    def compute(self, space, units, space_type: str,
                l0_threshold: float = 0.1) -> L0Result:
        t0 = time.perf_counter()

        n_total = len(units)

        if space_type == "irregular_graph":
            scores, topo, zone, cluster_ids = self._compute_graph(space, units)
        elif space_type == "tree_hierarchy":
            scores, topo, zone, cluster_ids = self._compute_tree(space, units)
        else:
            # Grid spaces: topology is trivial (uniform lattice)
            # Spatial quadrant clustering: group tiles into NQ x NQ blocks
            scores = np.ones(n_total, dtype=np.float64)
            topo = None
            zone = "n/a"
            cluster_ids = self._grid_spatial_clusters(units, space)

        # Normalize scores to [0, 1]
        smax = scores.max()
        if smax > 0:
            scores = scores / smax

        # Apply threshold gate
        # For grids: threshold=0 effectively (all pass)
        effective_threshold = 0.0 if space_type in ("scalar_grid", "vector_grid") else l0_threshold
        mask = scores >= effective_threshold

        n_clusters = len(np.unique(cluster_ids[mask]))
        dt_ms = (time.perf_counter() - t0) * 1000.0

        return L0Result(
            topo_scores=scores,
            topo_features=topo,
            zone=zone,
            unit_mask=mask,
            cluster_ids=cluster_ids,
            n_clusters=max(1, n_clusters),
            n_total=n_total,
            n_surviving=int(mask.sum()),
            computation_ms=dt_ms,
        )

    def _grid_spatial_clusters(self, units, space):
        """Assign grid tiles to spatial quadrant clusters.

        Divides the grid into ~4-16 spatial blocks so L1 can
        guarantee representation from each region.
        """
        import math
        NT = space.NT
        # Target ~4 blocks for small grids, ~16 for large
        n_blocks = max(2, min(4, int(math.sqrt(NT))))
        block_size = max(1, NT // n_blocks)

        cluster_ids = np.zeros(len(units), dtype=np.int32)
        for i, (ti, tj) in enumerate(units):
            bi = min(ti // block_size, n_blocks - 1)
            bj = min(tj // block_size, n_blocks - 1)
            cluster_ids[i] = bi * n_blocks + bj
        return cluster_ids

    def _compute_graph(self, space, units):
        """Full topo feature extraction for graph spaces."""
        import networkx as nx
        # Build networkx graph from space's neighbor dict
        G = nx.Graph()
        G.add_nodes_from(range(len(space.pos)))
        for node, neighbors in space.neighbors.items():
            for nb in neighbors:
                G.add_edge(node, nb)
        labels = space.labels  # cluster assignments
        n_clusters = len(units)

        topo = extract_topo_features(G, labels,
                                     topo_budget_ms=50.0)

        # Composite structural importance score per cluster:
        # curvature heterogeneity (bridges) + hub concentration + boundary anomaly
        curv_std = topo.cluster_curvature_std
        pr_max = topo.cluster_pagerank_max
        boundary = np.clip(-topo.cluster_boundary_curvature, 0, None)

        # Normalize each component
        def _norm(x):
            mx = x.max()
            return x / mx if mx > 0 else x

        scores = (0.3 * _norm(curv_std) +
                  0.2 * _norm(pr_max) +
                  0.5 * _norm(boundary))

        # Cluster IDs: Leiden clusters ARE the natural L0 clusters
        # Each unit index IS a cluster ID already
        cluster_ids = np.arange(len(units), dtype=np.int32)

        return scores, topo, topo.topo_zone, cluster_ids

    def _compute_tree(self, space, units):
        """Depth-based structural scoring for tree spaces."""
        n = space.n
        scores = np.zeros(len(units), dtype=np.float64)

        for i, u in enumerate(units):
            depth = int(np.log2(u + 1))
            # Subtree size (number of descendants)
            subtree_size = len(space._subtree(u))
            # Boundary nodes: those with parent in a different unit
            parent = (u - 1) // 2 if u > 0 else -1
            is_root_subtree = (parent < 0 or int(np.log2(parent + 1)) < 3)

            # Score: larger subtrees and root subtrees are more important
            scores[i] = subtree_size / n + (0.3 if is_root_subtree else 0.0)

        # Cluster by depth band: units at similar depths form natural groups
        cluster_ids = np.zeros(len(units), dtype=np.int32)
        for i, u in enumerate(units):
            depth = int(np.log2(u + 1))
            cluster_ids[i] = depth
        return scores, None, "n/a", cluster_ids


# =====================================================================
# Layer 1: Presence (data-dependent, query-independent)
# =====================================================================

class Layer1_Presence:
    """Score units by data presence / activity.

    Answers: "Is there non-trivial data here worth computing on?"
    NOT the same as unit_rho (which is task-specific MSE).

    Uses CASCADE QUOTAS from L0: each L0 cluster guarantees
    a minimum number of surviving units proportional to its size.
    This prevents any topological region from being wiped out.
    """

    def compute(self, space, state, units, unit_mask: np.ndarray,
                cluster_ids: np.ndarray,
                n_clusters: int,
                min_survival_ratio: float = 0.3,
                sparsity_mask: Optional[np.ndarray] = None) -> L1Result:
        """Filter units by data presence with per-cluster quotas.

        Args:
            min_survival_ratio: minimum fraction of L0-surviving units
                to keep per cluster. Tied to budget_fraction so L1
                never prunes below what L2 can consume.
        """
        t0 = time.perf_counter()

        surviving_indices = np.where(unit_mask)[0]
        n_pruned_l0 = len(units) - len(surviving_indices)

        # Step 1: compute presence scores for all L0-surviving units
        presence = np.zeros(len(units), dtype=np.float64)
        for idx in surviving_indices:
            u = units[idx]
            presence[idx] = self._unit_presence(space, u, sparsity_mask)

        # Step 2: per-cluster adaptive quota
        # Each L0 cluster keeps top-K units by presence,
        # where K = max(1, ceil(cluster_size * min_survival_ratio))
        active = np.zeros(len(units), dtype=bool)

        # Group surviving units by their L0 cluster
        cluster_groups: Dict[int, List[int]] = {}
        for idx in surviving_indices:
            cid = int(cluster_ids[idx])
            cluster_groups.setdefault(cid, []).append(idx)

        for cid, member_indices in cluster_groups.items():
            cluster_size = len(member_indices)
            # Quota: proportional to cluster size, minimum 1
            quota = max(1, int(np.ceil(cluster_size * min_survival_ratio)))

            # Sort members by presence score descending, keep top-quota
            scored = [(idx, presence[idx]) for idx in member_indices]
            scored.sort(key=lambda x: -x[1])

            for rank, (idx, _score) in enumerate(scored):
                if rank < quota:
                    active[idx] = True
                # Units below quota are pruned (active stays False)

        n_pruned_l1 = int(unit_mask.sum()) - int(active.sum())
        dt_ms = (time.perf_counter() - t0) * 1000.0

        return L1Result(
            presence_scores=presence,
            active_mask=active,
            n_input=int(unit_mask.sum()),
            n_surviving=int(active.sum()),
            n_pruned_l0=n_pruned_l0,
            n_pruned_l1=n_pruned_l1,
            computation_ms=dt_ms,
        )

    def _unit_presence(self, space, unit, sparsity_mask=None):
        """Compute data presence score for a unit.

        Uses variance of ground truth (not state) — measures
        whether there is structure worth computing on.
        """
        space_type = type(space).__name__

        if "ScalarGrid" in space_type:
            ti, tj = unit
            T = space.T
            s = slice(ti * T, (ti + 1) * T)
            cs = slice(tj * T, (tj + 1) * T)
            region = space.gt[s, cs]
            if sparsity_mask is not None and sparsity_mask[ti, tj]:
                return 0.0
            return float(np.var(region))

        elif "VectorGrid" in space_type:
            ti, tj = unit
            T = space.T
            s = slice(ti * T, (ti + 1) * T)
            cs = slice(tj * T, (tj + 1) * T)
            region = space.gt[s, cs, :]
            if sparsity_mask is not None and sparsity_mask[ti, tj]:
                return 0.0
            return float(np.var(region))

        elif "IrregularGraph" in space_type:
            pts = space._cluster_pts(unit)
            if not pts:
                return 0.0
            vals = [space.gt[p] for p in pts]
            return float(np.var(vals))

        elif "TreeHierarchy" in space_type:
            pts = space._subtree(unit)
            if not pts:
                return 0.0
            vals = [space.gt[p] for p in pts]
            return float(np.var(vals))

        return 0.0


# =====================================================================
# Layer 2: Query (task-specific)
# =====================================================================

class Layer2_Query:
    """Task-specific refinement on surviving units.

    Supports multiple query functions on the same frozen tree.
    """

    QUERY_FUNCTIONS = {
        "mse": "_query_mse",
        "max_abs": "_query_max_abs",
        "hf_residual": "_query_hf_residual",
    }

    def compute(self, space, state, units, active_mask: np.ndarray,
                query_fn: str = "mse",
                topo_features=None,
                budget_fraction: float = 0.30,
                halo_width: int = 2,
                halo_hops: int = 1,
                n_total_units: int = None) -> L2Result:
        """Run query on surviving units.

        budget_fraction is applied to n_total_units (all units in space),
        not just surviving — so pruning saves compute, not budget.
        """
        t0 = time.perf_counter()

        surviving = [units[i] for i in range(len(units)) if active_mask[i]]
        if n_total_units is None:
            n_total_units = len(units)

        if not surviving:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            psnr = self._compute_psnr(space, state)
            mse_val = self._compute_mse(space, state)
            return L2Result(query_fn=query_fn, rho_values=np.array([]),
                            ordered_units=[], n_refined=0,
                            psnr_final=psnr, mse_final=mse_val,
                            computation_ms=dt_ms)

        # Compute query-specific rho
        fn = getattr(self, self.QUERY_FUNCTIONS[query_fn])
        rho_values = np.array([fn(space, state, u) for u in surviving])

        # Topo adjustment (if available, for graph spaces)
        if topo_features is not None and len(rho_values) == len(surviving):
            # Build subset topo scores for surviving units only
            surviving_indices = [i for i in range(len(units)) if active_mask[i]]
            if hasattr(topo_features, 'cluster_curvature_std'):
                subset_topo = self._subset_topo_boost(
                    rho_values, topo_features, surviving_indices)
                rho_values = subset_topo

        # Sort by rho descending (canonical traversal)
        space_name = type(space).__name__.lower()
        if "grid" in space_name:
            sname = "scalar_grid"
        elif "graph" in space_name:
            sname = "irregular_graph"
        else:
            sname = "tree_hierarchy"
        ordered = CanonicalTraversal.sort_units(surviving, rho_values, sname)

        # Budget is absolute (fraction of TOTAL units, not surviving)
        # This way pruning saves compute, not budget allocation
        n_refine = max(1, int(n_total_units * budget_fraction))
        n_refine = min(n_refine, len(surviving))  # can't refine more than surviving
        new_state = state.copy()
        n_refined = 0

        for unit in ordered[:n_refine]:
            if "Grid" in type(space).__name__:
                new_state = space.refine_unit(new_state, unit, halo=halo_width)
            else:
                new_state = space.refine_unit(new_state, unit,
                                              halo_hops=halo_hops)
            n_refined += 1

        psnr = self._compute_psnr(space, new_state)
        mse_val = self._compute_mse(space, new_state)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        return L2Result(
            query_fn=query_fn,
            rho_values=rho_values,
            ordered_units=ordered[:n_refine],
            n_refined=n_refined,
            psnr_final=psnr,
            mse_final=mse_val,
            computation_ms=dt_ms,
        )

    # --- Query functions ---

    def _query_mse(self, space, state, unit):
        """Standard MSE residual (same as unit_rho)."""
        return space.unit_rho(state, unit)

    def _query_max_abs(self, space, state, unit):
        """Maximum absolute error in unit region."""
        space_type = type(space).__name__

        if "ScalarGrid" in space_type:
            ti, tj = unit
            T = space.T
            s = slice(ti * T, (ti + 1) * T)
            cs = slice(tj * T, (tj + 1) * T)
            return float(np.max(np.abs(space.gt[s, cs] - state[s, cs])))

        elif "VectorGrid" in space_type:
            ti, tj = unit
            T = space.T
            s = slice(ti * T, (ti + 1) * T)
            cs = slice(tj * T, (tj + 1) * T)
            return float(np.max(np.abs(space.gt[s, cs, :] - state[s, cs, :])))

        elif "IrregularGraph" in space_type:
            pts = space._cluster_pts(unit)
            if not pts:
                return 0.0
            return float(max(abs(space.gt[p] - state[p]) for p in pts))

        elif "TreeHierarchy" in space_type:
            pts = space._subtree(unit)
            return float(max(abs(space.gt[p] - state[p]) for p in pts))

        return 0.0

    def _query_hf_residual(self, space, state, unit):
        """High-frequency residual: Laplacian of (gt - state)."""
        space_type = type(space).__name__

        if "ScalarGrid" in space_type:
            ti, tj = unit
            T = space.T
            s = slice(ti * T, (ti + 1) * T)
            cs = slice(tj * T, (tj + 1) * T)
            diff = space.gt[s, cs] - state[s, cs]
            # Simple discrete Laplacian
            lap = np.zeros_like(diff)
            lap[1:-1, 1:-1] = (diff[:-2, 1:-1] + diff[2:, 1:-1] +
                                diff[1:-1, :-2] + diff[1:-1, 2:] -
                                4 * diff[1:-1, 1:-1])
            return float(np.mean(lap ** 2))

        elif "VectorGrid" in space_type:
            ti, tj = unit
            T = space.T
            s = slice(ti * T, (ti + 1) * T)
            cs = slice(tj * T, (tj + 1) * T)
            diff = space.gt[s, cs, :] - state[s, cs, :]
            # Per-channel Laplacian, average across channels
            hf_total = 0.0
            for c in range(diff.shape[2]):
                d = diff[:, :, c]
                lap = np.zeros_like(d)
                lap[1:-1, 1:-1] = (d[:-2, 1:-1] + d[2:, 1:-1] +
                                    d[1:-1, :-2] + d[1:-1, 2:] -
                                    4 * d[1:-1, 1:-1])
                hf_total += float(np.mean(lap ** 2))
            return hf_total / diff.shape[2]

        else:
            # For graph/tree: use difference from neighbors as HF proxy
            return space.unit_rho(state, unit)  # fallback to MSE

    # --- Helpers ---

    def _subset_topo_boost(self, rho_values, topo, surviving_indices):
        """Apply topo boost to subset of units."""
        curv_std = topo.cluster_curvature_std[surviving_indices]
        pr_max = topo.cluster_pagerank_max[surviving_indices]
        boundary = np.clip(-topo.cluster_boundary_curvature[surviving_indices], 0, None)

        def _norm(x):
            mx = x.max()
            return x / mx if mx > 0 else x

        signal = 0.3 * _norm(curv_std) + 0.2 * _norm(pr_max) + 0.5 * _norm(boundary)
        return rho_values * (1.0 + signal)

    def _compute_psnr(self, space, state):
        """Compute PSNR vs ground truth."""
        mse = self._compute_mse(space, state)
        if mse < 1e-15:
            return 100.0
        gt_range = float(np.max(space.gt) - np.min(space.gt))
        if gt_range < 1e-15:
            gt_range = 1.0
        return float(10 * np.log10(gt_range ** 2 / mse))

    def _compute_mse(self, space, state):
        """Compute MSE vs ground truth."""
        return float(np.mean((space.gt - state) ** 2))


# =====================================================================
# Industry Baselines
# =====================================================================

class IndustryBaselines:
    """Standard industry approaches for spatial search + refinement."""

    @staticmethod
    def run_kdtree(space, state, units, budget_fraction=0.30,
                   halo_width=2, halo_hops=1):
        """scipy.spatial.cKDTree: build index + NN query."""
        from scipy.spatial import cKDTree
        t0 = time.perf_counter()

        # Build: create k-d tree from unit coordinates
        coords = space.get_coords(units)
        coords_array = np.array([list(c) if hasattr(c, '__len__') else [c]
                                 for c in coords], dtype=np.float64)
        build_t0 = time.perf_counter()
        tree = cKDTree(coords_array)
        build_ms = (time.perf_counter() - build_t0) * 1000.0

        # Query: find units with highest residual (using the index for ordering)
        query_t0 = time.perf_counter()
        rho_values = np.array([space.unit_rho(state, u) for u in units])
        # Sort by rho descending
        order = np.argsort(-rho_values)
        n_refine = max(1, int(len(units) * budget_fraction))

        new_state = state.copy()
        n_refined = 0
        for idx in order[:n_refine]:
            u = units[idx]
            if "Grid" in type(space).__name__:
                new_state = space.refine_unit(new_state, u, halo=halo_width)
            else:
                new_state = space.refine_unit(new_state, u,
                                              halo_hops=halo_hops)
            n_refined += 1

        query_ms = (time.perf_counter() - query_t0) * 1000.0
        total_ms = (time.perf_counter() - t0) * 1000.0

        mse = float(np.mean((space.gt - new_state) ** 2))
        gt_range = float(np.max(space.gt) - np.min(space.gt))
        psnr = float(10 * np.log10(gt_range ** 2 / max(mse, 1e-15)))

        return {
            'build_ms': build_ms,
            'query_ms': query_ms,
            'total_ms': total_ms,
            'psnr': psnr,
            'mse': mse,
            'n_refined': n_refined,
            'index_bytes': coords_array.nbytes + 64,  # approximate
        }

    @staticmethod
    def run_quadtree(space, state, units, budget_fraction=0.30,
                     halo_width=2):
        """Quadtree spatial index for grid spaces."""
        t0 = time.perf_counter()

        # Build quadtree-like spatial index
        # For grids: simple recursive subdivision by variance
        build_t0 = time.perf_counter()

        coords = space.get_coords(units)
        rho_values = np.array([space.unit_rho(state, u) for u in units])

        # Quadtree is implicit: we just subdivide by spatial quadrants
        # and track which quadrant has highest total rho
        NT = space.NT
        quadrant_rho = {}
        for i, (ti, tj) in enumerate(units):
            qi, qj = ti // max(1, NT // 2), tj // max(1, NT // 2)
            key = (qi, qj)
            quadrant_rho.setdefault(key, []).append((rho_values[i], i))

        # Sort quadrants by total rho, then units within
        quad_order = sorted(quadrant_rho.keys(),
                            key=lambda k: sum(r for r, _ in quadrant_rho[k]),
                            reverse=True)
        ordered_indices = []
        for qk in quad_order:
            sub = sorted(quadrant_rho[qk], key=lambda x: -x[0])
            ordered_indices.extend(idx for _, idx in sub)

        build_ms = (time.perf_counter() - build_t0) * 1000.0

        # Query: refine in quadtree order
        query_t0 = time.perf_counter()
        n_refine = max(1, int(len(units) * budget_fraction))
        new_state = state.copy()
        n_refined = 0

        for idx in ordered_indices[:n_refine]:
            u = units[idx]
            new_state = space.refine_unit(new_state, u, halo=halo_width)
            n_refined += 1

        query_ms = (time.perf_counter() - query_t0) * 1000.0
        total_ms = (time.perf_counter() - t0) * 1000.0

        mse = float(np.mean((space.gt - new_state) ** 2))
        gt_range = float(np.max(space.gt) - np.min(space.gt))
        psnr = float(10 * np.log10(gt_range ** 2 / max(mse, 1e-15)))

        return {
            'build_ms': build_ms,
            'query_ms': query_ms,
            'total_ms': total_ms,
            'psnr': psnr,
            'mse': mse,
            'n_refined': n_refined,
            'index_bytes': 8 * len(quadrant_rho),
        }

    @staticmethod
    def run_leiden_bf(space, state, units, budget_fraction=0.30,
                      halo_hops=1):
        """Leiden community detection + brute force within communities.

        This is already what our pipeline does for graphs,
        but without the gate/governor/SC overhead.
        """
        t0 = time.perf_counter()

        # Build: Leiden clustering is already done in space.setup()
        build_t0 = time.perf_counter()
        # The community structure IS the unit structure
        build_ms = (time.perf_counter() - build_t0) * 1000.0

        # Query: brute-force rho within each community, sort globally
        query_t0 = time.perf_counter()
        rho_values = np.array([space.unit_rho(state, u) for u in units])
        order = np.argsort(-rho_values)
        n_refine = max(1, int(len(units) * budget_fraction))

        new_state = state.copy()
        n_refined = 0
        for idx in order[:n_refine]:
            u = units[idx]
            new_state = space.refine_unit(new_state, u, halo_hops=halo_hops)
            n_refined += 1

        query_ms = (time.perf_counter() - query_t0) * 1000.0
        total_ms = (time.perf_counter() - t0) * 1000.0

        mse = float(np.mean((space.gt - new_state) ** 2))
        gt_range = float(np.max(space.gt) - np.min(space.gt))
        psnr = float(10 * np.log10(gt_range ** 2 / max(mse, 1e-15)))

        return {
            'build_ms': build_ms,
            'query_ms': query_ms,
            'total_ms': total_ms,
            'psnr': psnr,
            'mse': mse,
            'n_refined': n_refined,
            'index_bytes': 0,  # Leiden structure is part of space
        }

    @staticmethod
    def run_wavelet(space, state, units, budget_fraction=0.30,
                    halo_width=2):
        """Wavelet decomposition: detail coefficients as saliency map.

        Grid spaces only.
        """
        import pywt
        t0 = time.perf_counter()

        # Build: wavelet decomposition of ground truth
        build_t0 = time.perf_counter()
        gt_2d = space.gt if space.gt.ndim == 2 else space.gt[:, :, 0]
        coeffs = pywt.dwt2(gt_2d, 'haar')
        cA, (cH, cV, cD) = coeffs
        # Detail energy per tile
        T = space.T
        NT = space.NT
        detail_energy = np.zeros(NT * NT)
        for i, (ti, tj) in enumerate(units):
            # Map tile to wavelet coefficient region
            wi = ti * T // 2
            wj = tj * T // 2
            we = min(wi + T // 2, cH.shape[0])
            wf = min(wj + T // 2, cH.shape[1])
            if we > wi and wf > wj:
                detail_energy[i] = float(
                    np.sum(cH[wi:we, wj:wf] ** 2) +
                    np.sum(cV[wi:we, wj:wf] ** 2) +
                    np.sum(cD[wi:we, wj:wf] ** 2))
        build_ms = (time.perf_counter() - build_t0) * 1000.0

        # Query: refine tiles with highest detail energy
        query_t0 = time.perf_counter()
        order = np.argsort(-detail_energy)
        n_refine = max(1, int(len(units) * budget_fraction))

        new_state = state.copy()
        n_refined = 0
        for idx in order[:n_refine]:
            u = units[idx]
            new_state = space.refine_unit(new_state, u, halo=halo_width)
            n_refined += 1

        query_ms = (time.perf_counter() - query_t0) * 1000.0
        total_ms = (time.perf_counter() - t0) * 1000.0

        mse = float(np.mean((space.gt - new_state) ** 2))
        gt_range = float(np.max(space.gt) - np.min(space.gt))
        psnr = float(10 * np.log10(gt_range ** 2 / max(mse, 1e-15)))

        return {
            'build_ms': build_ms,
            'query_ms': query_ms,
            'total_ms': total_ms,
            'psnr': psnr,
            'mse': mse,
            'n_refined': n_refined,
            'index_bytes': sum(c.nbytes for c in [cA, cH, cV, cD]),
        }


# =====================================================================
# Three-Layer Pipeline Orchestrator
# =====================================================================

class ThreeLayerPipeline:
    """Orchestrates the three-layer rho decomposition."""

    def __init__(self):
        self.l0 = Layer0_Topology()
        self.l1 = Layer1_Presence()
        self.l2 = Layer2_Query()
        self.industry = IndustryBaselines()

    def build_frozen_tree(self, space, state, units, space_type: str,
                          seed: int,
                          l0_threshold: float = 0.1,
                          l1_min_survival: float = 0.3,
                          sparsity_mask=None) -> FrozenTree:
        """Build and freeze Layers 0+1."""
        l0_result = self.l0.compute(space, units, space_type, l0_threshold)
        l1_result = self.l1.compute(space, state, units, l0_result.unit_mask,
                                     cluster_ids=l0_result.cluster_ids,
                                     n_clusters=l0_result.n_clusters,
                                     min_survival_ratio=l1_min_survival,
                                     sparsity_mask=sparsity_mask)

        active_units = [units[i] for i in range(len(units))
                        if l1_result.active_mask[i]]

        # Estimate memory
        mem = (l0_result.topo_scores.nbytes +
               l1_result.presence_scores.nbytes +
               sys.getsizeof(active_units))

        return FrozenTree(
            space_type=space_type,
            seed=seed,
            l0_scores=l0_result.topo_scores,
            l1_scores=l1_result.presence_scores,
            active_units=active_units,
            all_units=list(units),
            zone=l0_result.zone,
            l0_ms=l0_result.computation_ms,
            l1_ms=l1_result.computation_ms,
            memory_bytes=mem,
        ), l0_result, l1_result

    def run_query_on_frozen(self, space, state, units,
                             frozen: FrozenTree, l1_result: L1Result,
                             l0_result: L0Result,
                             query_fn: str = "mse",
                             budget_fraction: float = 0.30,
                             halo_width: int = 2,
                             halo_hops: int = 1) -> L2Result:
        """Run a Layer 2 query on a frozen tree."""
        return self.l2.compute(
            space, state, units, l1_result.active_mask,
            query_fn=query_fn,
            topo_features=l0_result.topo_features,
            budget_fraction=budget_fraction,
            halo_width=halo_width,
            halo_hops=halo_hops,
            n_total_units=len(units),
        )

    def run_streaming(self, space, state, units, space_type: str,
                       seed: int,
                       query_fn: str = "mse",
                       budget_fraction: float = 0.30,
                       l0_threshold: float = 0.1,
                       l1_min_survival: float = 0.3,
                       halo_width: int = 2,
                       halo_hops: int = 1,
                       sparsity_mask=None):
        """Streaming three-layer pipeline: process cluster-by-cluster.

        Instead of L0(all) -> L1(all) -> L2(all), processes each L0 cluster
        through all three layers before moving to the next:

            Cluster 0: [L0] -> [L1] -> [L2 refine]
            Cluster 1:        [L0] -> [L1] -> [L2 refine]
            ...

        Two key advantages:
        1. Latency: first refinements appear after 1 cluster, not all.
        2. Budget efficiency: L2 budget is per-cluster proportional to
           cluster size. L1 pruning directly reduces refinement count
           (not just scoring count).
        """
        t0 = time.perf_counter()

        # Step 0: L0 topology pass (must be global — needs full graph)
        l0_result = self.l0.compute(space, units, space_type, l0_threshold)

        # Group units by L0 cluster
        cluster_groups: Dict[int, List[int]] = {}
        for idx in range(len(units)):
            if l0_result.unit_mask[idx]:
                cid = int(l0_result.cluster_ids[idx])
                cluster_groups.setdefault(cid, []).append(idx)

        n_total = len(units)
        total_surviving_l0 = sum(len(v) for v in cluster_groups.values())
        global_budget = max(1, int(budget_fraction * n_total))
        new_state = state.copy()
        n_refined_total = 0
        n_surviving_l1_total = 0
        n_pruned_l1_total = 0
        cluster_details = []

        is_grid = "grid" in space_type

        # Sort clusters by mean L0 score descending (most important first)
        # This ensures budget goes to structurally important regions
        cluster_priority = []
        for cid, members in cluster_groups.items():
            mean_l0 = np.mean([l0_result.topo_scores[idx] for idx in members])
            cluster_priority.append((cid, mean_l0))
        cluster_priority.sort(key=lambda x: -x[1])

        # Stream: process each cluster through L1 -> L2
        for cid, _priority in cluster_priority:
            # Global budget exhausted — stop early
            if n_refined_total >= global_budget:
                break

            member_indices = cluster_groups[cid]
            cluster_size = len(member_indices)

            # --- L1 per cluster: presence scoring + quota ---
            l1_quota = max(1, int(np.ceil(cluster_size * l1_min_survival)))
            scored = []
            for idx in member_indices:
                u = units[idx]
                presence = self.l1._unit_presence(space, u, sparsity_mask)
                scored.append((idx, u, presence))
            scored.sort(key=lambda x: -x[2])
            survivors = scored[:l1_quota]
            n_surviving_l1_total += len(survivors)
            n_pruned_l1_total += cluster_size - len(survivors)

            # --- L2 per cluster: query + refine ---
            # Budget proportional to cluster's share of total, capped by global remaining
            cluster_budget = max(1, int(
                budget_fraction * n_total * cluster_size / max(total_surviving_l0, 1)))
            cluster_budget = min(cluster_budget, len(survivors))
            cluster_budget = min(cluster_budget, global_budget - n_refined_total)

            if not survivors:
                continue

            # Score surviving units with query function
            query_fn_method = getattr(self.l2, self.l2.QUERY_FUNCTIONS[query_fn])
            query_scores = [(idx, u, query_fn_method(space, new_state, u))
                            for idx, u, _ in survivors]
            query_scores.sort(key=lambda x: -x[2])

            # Refine top-budget units
            for rank, (idx, u, score) in enumerate(query_scores[:cluster_budget]):
                if is_grid:
                    new_state = space.refine_unit(new_state, u, halo=halo_width)
                else:
                    new_state = space.refine_unit(new_state, u, halo_hops=halo_hops)
                n_refined_total += 1

            cluster_details.append({
                'cluster_id': cid,
                'size': cluster_size,
                'l1_survivors': len(survivors),
                'l2_refined': min(cluster_budget, len(survivors)),
            })

        total_ms = (time.perf_counter() - t0) * 1000.0
        mse = float(np.mean((space.gt - new_state) ** 2))
        gt_range = float(np.max(space.gt) - np.min(space.gt))
        psnr = float(10 * np.log10(gt_range ** 2 / max(mse, 1e-15)))

        return {
            'total_ms': total_ms,
            'l0_ms': l0_result.computation_ms,
            'psnr': psnr,
            'mse': mse,
            'n_refined': n_refined_total,
            'n_after_l0': total_surviving_l0,
            'n_after_l1': n_surviving_l1_total,
            'n_clusters': len(cluster_groups),
            'cluster_details': cluster_details,
            'zone': l0_result.zone,
        }

    def run_single_pass(self, space, state, units, space_type: str,
                         budget_fraction=0.30, halo_width=2, halo_hops=1,
                         use_topo=True):
        """Current monolithic rho approach."""
        t0 = time.perf_counter()

        # Compute base rho
        rho_values = np.array([space.unit_rho(state, u) for u in units])

        # Topo boost (if applicable)
        topo = None
        if use_topo and space_type == "irregular_graph":
            try:
                topo = extract_topo_features(
                    space.G, space.labels, len(units), topo_budget_ms=50.0)
                rho_values = topo_adjusted_rho(rho_values, topo)
            except Exception:
                pass

        # Sort and refine
        ordered = CanonicalTraversal.sort_units(units, rho_values, space_type)
        n_refine = max(1, int(len(units) * budget_fraction))

        new_state = state.copy()
        n_refined = 0
        for unit in ordered[:n_refine]:
            if "grid" in space_type:
                new_state = space.refine_unit(new_state, unit, halo=halo_width)
            else:
                new_state = space.refine_unit(new_state, unit,
                                              halo_hops=halo_hops)
            n_refined += 1

        dt_ms = (time.perf_counter() - t0) * 1000.0
        mse = float(np.mean((space.gt - new_state) ** 2))
        gt_range = float(np.max(space.gt) - np.min(space.gt))
        psnr = float(10 * np.log10(gt_range ** 2 / max(mse, 1e-15)))

        return {
            'total_ms': dt_ms,
            'psnr': psnr,
            'mse': mse,
            'n_refined': n_refined,
        }
