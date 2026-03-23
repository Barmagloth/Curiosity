"""Exp17 configuration: Three-Layer rho Architecture."""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Exp17Config:
    """Configuration for three-layer rho experiment."""

    # --- Experiment matrix ---
    space_types: List[str] = field(default_factory=lambda: [
        "scalar_grid", "vector_grid", "irregular_graph", "tree_hierarchy"])
    n_seeds: int = 20
    base_seed: int = 3000

    # --- Approaches ---
    approaches: List[str] = field(default_factory=lambda: [
        "naive_full",           # no topo, no filtering, just unit_rho sort
        "single_pass",          # current monolithic rho (residual * topo_boost)
        "three_layer",          # L0 -> L1 -> L2 (single query)
        "three_layer_reuse",    # L0+L1 once -> 3 different L2 queries
        "industry_kdtree",      # scipy cKDTree build + query
        "industry_quadtree",    # quadtree spatial index (grid only)
        "industry_leiden_bf",   # Leiden + brute force within (graph only)
        "industry_wavelet",     # wavelet detail coefficients (grid only)
    ])

    # --- Layer thresholds ---
    l0_threshold: float = 0.1      # bottom 10% topology score pruned
    l1_min_survival: float = 0.3   # cascade quota: each L0 cluster keeps this
                                    # fraction of its units (tied to budget_fraction)
    sparsity_fraction: float = 0.3  # fraction of units zeroed for L1 test

    # --- Query functions for reuse test ---
    query_functions: List[str] = field(default_factory=lambda: [
        "mse", "max_abs", "hf_residual"])

    # --- Budget ---
    budget_fraction: float = 0.30

    # --- Scaling ---
    scale_levels: List[int] = field(default_factory=lambda: [100, 1000, 10000])

    # Grid scaling: N = tile_size * sqrt(n_units)
    tile_size: int = 8

    # Graph scaling: n_points ≈ n_units * 10, k=6-10
    graph_k: int = 6

    # Tree scaling: depth = ceil(log2(n_units * 8)) + 1
    # (units are subtrees at depth 3)

    # --- Drift test ---
    drift_n_steps: int = 20
    drift_n_seeds: int = 10

    # --- Pipeline config passthrough ---
    enforce_enabled: bool = True
    topo_profiling_enabled: bool = True
    enox_journal_enabled: bool = True
    halo_width: int = 2
    halo_hops: int = 1

    # --- Execution ---
    save_every: int = 20  # incremental save interval


# Scale parametrization helpers
def grid_N_for_units(n_units: int, tile_size: int = 8) -> int:
    """Grid side length to produce ~n_units tiles."""
    import math
    nt = int(math.ceil(math.sqrt(n_units)))
    return nt * tile_size


def graph_points_for_units(n_units: int) -> int:
    """Approximate number of points to get ~n_units Leiden clusters."""
    return n_units * 10  # empirical: ~10 points per cluster


def tree_depth_for_units(n_units: int) -> int:
    """Tree depth to produce ~n_units subtrees at depth 3."""
    import math
    # Units are at depth 3: 2^3 = 8 subtrees for depth 6
    # For n_units subtrees at depth 3: need depth = 3 + ceil(log2(n_units))
    # But actually: units at depth d have 2^d nodes
    # depth 3 -> 8 subtrees. depth 4 -> 16. depth 7 -> 128. depth 10 -> 1024.
    d = 3 + max(0, int(math.ceil(math.log2(max(n_units, 1)))))
    return min(d, 14)  # cap at depth 14 (16K nodes)
