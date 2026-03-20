"""
Curiosity -- Phase 2 Space Registry.

Imports the four space adapters from exp10d, wires up R/Up operators from
sc_baseline and exp12a, and defines per-space-type policy tables (halo,
layout, SC operators).
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Add experiment paths so cross-experiment imports resolve
# ---------------------------------------------------------------------------
EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENTS_DIR))
sys.path.insert(0, str(EXPERIMENTS_DIR / "exp10d_seed_determinism"))
sys.path.insert(0, str(EXPERIMENTS_DIR / "sc_baseline"))
sys.path.insert(0, str(EXPERIMENTS_DIR / "exp12a_tau_parent"))
sys.path.insert(0, str(EXPERIMENTS_DIR / "exp14a_sc_enforce"))

# ---------------------------------------------------------------------------
# Space adapters from exp10d
# ---------------------------------------------------------------------------
from exp10d_seed_determinism import (
    ScalarGridSpace, VectorGridSpace, IrregularGraphSpace, TreeHierarchySpace,
    CanonicalTraversal, DeterministicProbe, GovernorIsolation,
    SPACE_FACTORIES,
)

# ---------------------------------------------------------------------------
# R/Up operators (scalar grid from operators_v2, others from exp12a)
# ---------------------------------------------------------------------------
from operators_v2 import make_restrict_gaussian, prolong_gaussian
from exp12a_tau_parent import make_vector_ops, GraphOps, TreeOps

# ---------------------------------------------------------------------------
# SC-enforce module
# ---------------------------------------------------------------------------
from sc_enforce import SCEnforcer, load_thresholds, StrictnessTracker, WasteBudget

# ---------------------------------------------------------------------------
# Segment compression (guard + tree builder)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(EXPERIMENTS_DIR / "exp13_segment_compression"))
from segment_compress import RefinementTree, should_compress, N_CRITICAL_D2


# ===================================================================
# Halo applicability policy
# ===================================================================

HALO_POLICY = {
    "scalar_grid": {
        "enabled": True,
        "reason": "boundary parallelism >= 3",
    },
    "vector_grid": {
        "enabled": True,
        "reason": "boundary parallelism >= 3",
    },
    "irregular_graph": {
        "enabled": True,
        "reason": "k-NN, min_cut >> 3",
    },
    "tree_hierarchy": {
        "enabled": False,
        "reason": "min_cut=1, context leakage",
    },
}


# ===================================================================
# Layout policy dispatch table
# ===================================================================

LAYOUT_POLICY = {
    "scalar_grid": "D_direct",
    "vector_grid": "D_direct",
    "tree_hierarchy": "hybrid_D_direct_A_bitset",
    "irregular_graph": "D_blocked_conditional",
}


# ===================================================================
# SC operator factory per space type
# ===================================================================

def make_sc_operators(space_type, space):
    """Return (R_fn, Up_fn) for the given space type and adapter instance.

    Args:
        space_type: One of 'scalar_grid', 'vector_grid', 'irregular_graph',
                    'tree_hierarchy'.
        space:      The space adapter instance (already set up).

    Returns:
        Tuple of (restrict_fn, prolong_fn) compatible with SCEnforcer.
    """
    if space_type == "scalar_grid":
        R_fn = make_restrict_gaussian(sigma=3.0)
        Up_fn = prolong_gaussian
        return R_fn, Up_fn

    if space_type == "vector_grid":
        R_fn, Up_fn = make_vector_ops(sigma=3.0)
        return R_fn, Up_fn

    if space_type == "irregular_graph":
        ops = GraphOps(space.labels, space.n_clusters, space.n_pts)
        R_fn = ops.restrict
        Up_fn = lambda xc, tgt, _ops=ops: _ops.prolong(xc, tgt)
        return R_fn, Up_fn

    if space_type == "tree_hierarchy":
        ops = TreeOps(space.n, space.coarse_depth)
        R_fn = ops.restrict
        Up_fn = lambda xc, tgt, _ops=ops: _ops.prolong(xc, tgt)
        return R_fn, Up_fn

    raise ValueError(f"Unknown space_type: {space_type}")


# ===================================================================
# Threshold file path
# ===================================================================

THRESHOLD_PATH = (
    EXPERIMENTS_DIR / "exp12a_tau_parent" / "results" / "exp12a_thresholds.json"
)
