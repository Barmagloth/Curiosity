#!/usr/bin/env python3
"""
Curiosity -- exp13_compression: Segment Compression experiment runner (P1-B1).

Tests whether degree-2 chains in adaptive refinement trees can be
compressed >50% using dirty signature stability, with per-step
overhead < 10%.

Generates 4 space types (binary trees of varying depth, and a
quadtree), simulates a refinement process with controlled signature
stability patterns, and measures compression ratio, merge/split events,
and chain statistics.

Kill criteria:
  - Compression ratio > 50% of degree-2 nodes  -> PASS
  - Per-step overhead < 10%                     -> PASS
  - Otherwise                                   -> FAIL / investigate

Outputs (saved incrementally to results/):
  - exp13_results.json         -- full results
  - exp13_per_step.json        -- per-step compression time series
  - exp13_chain_stats.json     -- chain statistics per space type

Usage:
    python exp13_compression.py
"""

from __future__ import annotations

import json
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

from segment_compress import (
    RefinementTree, SegmentTree, SignatureStabilityChecker,
    Segment, hamming12, component_diff, unpack_signature,
    should_compress,
)

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

N_SEEDS = 10
N_STEPS = 40          # simulation steps per trial
STABILITY_WINDOW = 3
MAX_SEGMENT_LENGTH = 8


# ═══════════════════════════════════════════════════════════════════════
# Tree generators
# ═══════════════════════════════════════════════════════════════════════

def make_binary_tree(depth: int) -> RefinementTree:
    """Create a complete binary tree of given depth.

    Node IDs are 0-based level-order indices:
      parent[i] = (i-1)//2 for i > 0
      children[i] = [2i+1, 2i+2] if they exist

    Parameters
    ----------
    depth : int
        Tree depth (root is depth 0, leaves at depth-1).

    Returns
    -------
    RefinementTree
    """
    n_nodes = 2**depth - 1
    parent = {}
    children = {}
    for i in range(n_nodes):
        children[i] = []
        if i > 0:
            parent[i] = (i - 1) // 2
        else:
            parent[i] = None
        left = 2 * i + 1
        right = 2 * i + 2
        if left < n_nodes:
            children[i].append(left)
        if right < n_nodes:
            children[i].append(right)
    return RefinementTree(parent, children)


def make_partial_binary_tree(depth: int, n_chains_target: int,
                             rng: np.random.Generator) -> RefinementTree:
    """Create a binary tree with explicit degree-2 chains.

    Strategy: start with a full binary tree.  Select n_chains_target
    non-overlapping root-to-leaf paths.  Along each path, choose a
    contiguous sub-path of 3-6 levels and prune the sibling subtree
    at each level.  Crucially, we prune ONLY the immediate sibling
    (not an entire deep subtree) -- the sibling's own subtree is
    preserved by re-parenting it.  Actually, since this is a binary
    tree and siblings have their own subtrees, we DO prune the
    sibling's entire subtree.  To keep the tree large, we limit the
    chain depth range to the LOWER half of the tree (closer to leaves)
    where subtrees are small.

    Parameters
    ----------
    depth : int
        Full tree depth before pruning.
    n_chains_target : int
        Number of degree-2 chains to create.
    rng : numpy Generator
        Random number generator.

    Returns
    -------
    RefinementTree
    """
    n_nodes = 2**depth - 1
    parent = {}
    children = {}

    for i in range(n_nodes):
        children[i] = []
        parent[i] = (i - 1) // 2 if i > 0 else None
        left = 2 * i + 1
        right = 2 * i + 2
        if left < n_nodes:
            children[i].append(left)
        if right < n_nodes:
            children[i].append(right)

    n_leaves = 2**(depth - 1)
    leaf_start = 2**(depth - 1) - 1

    # Select leaf paths from DIFFERENT top-level subtrees to avoid
    # overlap in the chain region.  The root has 2 children, each
    # with 2 children, etc.  We spread chains across different
    # subtrees at depth ~chain_start to avoid collisions.
    chain_end = depth - 1
    chain_len_default = min(4, depth - 2)
    chain_start_default = max(1, chain_end - chain_len_default)

    # Partition leaves by their ancestor at chain_start level.
    # Each ancestor at chain_start has 2^(depth-1-chain_start) leaves.
    n_ancestors = 2**chain_start_default
    leaves_per_ancestor = max(1, n_leaves // n_ancestors)

    # Pick at most one leaf per ancestor subtree
    n_chains = min(n_chains_target, n_ancestors)
    ancestor_indices = rng.choice(n_ancestors, size=n_chains, replace=False)

    removed = set()
    for ai in ancestor_indices:
        # Pick a random leaf under this ancestor
        leaf_range_start = ai * leaves_per_ancestor
        leaf_range_end = min((ai + 1) * leaves_per_ancestor, n_leaves)
        if leaf_range_start >= leaf_range_end:
            continue
        pi = rng.integers(leaf_range_start, leaf_range_end)
        leaf = leaf_start + pi

        # Build path from root to leaf
        path = []
        node = leaf
        while node is not None and node >= 0:
            path.append(node)
            node = parent.get(node)
        path.reverse()  # path[0]=root, path[-1]=leaf

        # Chain range: bottom 3-4 levels only (small subtrees pruned)
        max_chain_len = min(4, depth - 2)
        min_chain_len = min(3, max_chain_len)
        if max_chain_len < 2:
            continue
        chain_len = rng.integers(min_chain_len, max_chain_len + 1)
        cs = max(1, chain_end - chain_len)
        # Ensure chain_start is near bottom (at least depth-4)
        cs = max(cs, depth - 5)

        # Prune sibling subtrees along the chain range
        for d in range(cs, chain_end):
            if d >= len(path) - 1:
                break
            p = path[d]
            if p in removed:
                break
            kept = path[d + 1]
            for sib in children.get(p, []):
                if sib != kept and sib not in removed:
                    _collect_subtree(sib, children, removed)

    # Rebuild without removed nodes
    new_parent = {}
    new_children = {}
    for i in range(n_nodes):
        if i in removed:
            continue
        new_children[i] = [c for c in children[i] if c not in removed]
        p = parent[i]
        new_parent[i] = p if (p is not None and p not in removed) else None

    return RefinementTree(new_parent, new_children)


def _collect_subtree(root: int, children: dict, result: set) -> None:
    """Recursively collect all nodes in a subtree."""
    result.add(root)
    for c in children.get(root, []):
        _collect_subtree(c, children, result)


def make_quadtree(depth: int) -> RefinementTree:
    """Create a complete quadtree (4 children per internal node).

    Used as the grid analog.  Node IDs are level-order.

    Parameters
    ----------
    depth : int
        Tree depth (root at 0, leaves at depth-1).

    Returns
    -------
    RefinementTree
    """
    # Total nodes in a complete 4-ary tree of given depth
    n_nodes = sum(4**d for d in range(depth))
    parent = {}
    children = {}

    for i in range(n_nodes):
        children[i] = []
        if i == 0:
            parent[i] = None
        else:
            parent[i] = (i - 1) // 4

        for k in range(4):
            child = 4 * i + 1 + k
            if child < n_nodes:
                children[i].append(child)

    return RefinementTree(parent, children)


def make_partial_quadtree(depth: int, n_chains_target: int,
                          rng: np.random.Generator) -> RefinementTree:
    """Create a quadtree with degree-2 chains from path pruning.

    Strategy: select paths from root to leaf.  Along each path in
    the bottom levels, prune all 3 sibling subtrees at each node,
    leaving exactly 1 child.  This creates a chain of degree-2 nodes.

    Parameters
    ----------
    depth : int
        Full quadtree depth.
    n_chains_target : int
        Number of degree-2 chains to create.
    rng : numpy Generator
        Random number generator.

    Returns
    -------
    RefinementTree
    """
    n_nodes = sum(4**d for d in range(depth))
    parent = {}
    children = {}

    for i in range(n_nodes):
        children[i] = []
        parent[i] = (i - 1) // 4 if i > 0 else None
        for k in range(4):
            child = 4 * i + 1 + k
            if child < n_nodes:
                children[i].append(child)

    if depth < 3:
        return RefinementTree(parent, children)

    # Enumerate leaves
    leaf_start = sum(4**d for d in range(depth - 1))
    n_leaves = 4**(depth - 1)

    # Spread chains across different ancestor subtrees at depth 1
    # (root's 4 children), picking from different quadrants.
    n_quad = 4  # root's children
    leaves_per_quad = n_leaves // n_quad
    n_chains = min(n_chains_target, n_leaves)

    # Assign chains to quadrants round-robin
    selected_leaves = []
    for ci in range(n_chains):
        quad = ci % n_quad
        start = quad * leaves_per_quad
        end = min(start + leaves_per_quad, n_leaves)
        if start < end:
            pi = rng.integers(start, end)
            selected_leaves.append(leaf_start + pi)

    removed = set()
    for leaf in selected_leaves:
        # Build path from root to leaf
        path = []
        node = leaf
        while node is not None and node >= 0:
            path.append(node)
            node = parent.get(node)
        path.reverse()

        # Chain in bottom 2-3 levels (where subtrees are small)
        chain_len = min(depth - 2, rng.integers(2, 4))
        cs = max(1, depth - 1 - chain_len)

        for d in range(cs, depth - 1):
            if d >= len(path) - 1:
                break
            p = path[d]
            if p in removed:
                break
            kept = path[d + 1]
            for sib in children.get(p, []):
                if sib != kept and sib not in removed:
                    _collect_subtree(sib, children, removed)

    # Rebuild
    new_parent = {}
    new_children = {}
    for i in range(n_nodes):
        if i in removed:
            continue
        new_children[i] = [c for c in children[i] if c not in removed]
        p = parent[i]
        new_parent[i] = p if (p is not None and p not in removed) else None

    return RefinementTree(new_parent, new_children)


# ═══════════════════════════════════════════════════════════════════════
# Space configurations
# ═══════════════════════════════════════════════════════════════════════

SPACE_CONFIGS = {
    "binary_d6_3chains": {
        "description": "Binary tree depth 6, 3 chains",
        "factory": lambda rng: make_partial_binary_tree(6, 3, rng),
    },
    "binary_d7_4chains": {
        "description": "Binary tree depth 7, 4 chains",
        "factory": lambda rng: make_partial_binary_tree(7, 4, rng),
    },
    "binary_d8_6chains": {
        "description": "Binary tree depth 8, 6 chains",
        "factory": lambda rng: make_partial_binary_tree(8, 6, rng),
    },
    "quadtree_d5_4chains": {
        "description": "Quadtree depth 5, 4 chains",
        "factory": lambda rng: make_partial_quadtree(5, 4, rng),
    },
}


# ═══════════════════════════════════════════════════════════════════════
# Signature generation with controlled stability
# ═══════════════════════════════════════════════════════════════════════

def generate_signatures(nodes, step: int, rng: np.random.Generator,
                        stable_fraction: float = 0.7,
                        stable_nodes: set = None) -> dict:
    """Generate dirty signatures for all nodes with controlled stability.

    A fraction of nodes get stable signatures (small perturbations
    step-to-step); the rest get volatile signatures.

    Parameters
    ----------
    nodes : iterable of int
        Node IDs.
    step : int
        Current simulation step.
    rng : numpy Generator
        Random number generator.
    stable_fraction : float
        Fraction of nodes that should have stable signatures.
    stable_nodes : set or None
        Pre-determined set of stable nodes.  If None, chosen randomly.

    Returns
    -------
    dict : node_id -> 12-bit signature
    """
    node_list = sorted(nodes)

    if stable_nodes is None:
        n_stable = int(len(node_list) * stable_fraction)
        stable_nodes = set(rng.choice(node_list,
                                      size=min(n_stable, len(node_list)),
                                      replace=False))

    signatures = {}
    for nid in node_list:
        if nid in stable_nodes:
            # Stable: low-noise signature based on node position.
            # Jitter is at most +/-1 from the base, but we only perturb
            # one component per step (keeps Hamming distance <= 1-2 bits
            # from baseline, well under the threshold of 3).
            base_seam = (nid * 3) % 16
            base_uncert = (nid * 7) % 16
            base_mass = (nid * 11) % 16
            # Perturb at most one component by at most 1
            which = rng.integers(0, 4)  # 0=seam, 1=uncert, 2=mass, 3=none
            delta = rng.choice([-1, 0, 1])
            seam = base_seam + (delta if which == 0 else 0)
            uncert = base_uncert + (delta if which == 1 else 0)
            mass = base_mass + (delta if which == 2 else 0)
            seam = max(0, min(15, seam))
            uncert = max(0, min(15, uncert))
            mass = max(0, min(15, mass))
        else:
            # Volatile: random signature each step
            seam = rng.integers(0, 16)
            uncert = rng.integers(0, 16)
            mass = rng.integers(0, 16)

        signatures[nid] = (seam << 8) | (uncert << 4) | mass

    return signatures


def generate_signatures_with_event(nodes, step: int,
                                   rng: np.random.Generator,
                                   stable_nodes: set,
                                   event_step: int,
                                   event_nodes: set) -> dict:
    """Generate signatures with a destabilization event.

    Before event_step, event_nodes are stable.  At and after event_step,
    they become volatile (simulating a structural change).

    Parameters
    ----------
    nodes : iterable of int
        All node IDs.
    step : int
        Current step.
    rng : numpy Generator
    stable_nodes : set
        Nodes that are always stable.
    event_step : int
        Step at which destabilization occurs.
    event_nodes : set
        Nodes affected by the event.

    Returns
    -------
    dict : node_id -> 12-bit signature
    """
    if step < event_step:
        # Before event: event_nodes are also stable
        effective_stable = stable_nodes | event_nodes
    else:
        # After event: event_nodes become volatile
        effective_stable = stable_nodes - event_nodes

    return generate_signatures(nodes, step, rng,
                               stable_nodes=effective_stable)


# ═══════════════════════════════════════════════════════════════════════
# Single trial runner
# ═══════════════════════════════════════════════════════════════════════

def run_trial(space_name: str, seed: int,
              budget_fraction: float = 0.15) -> dict:
    """Run a single compression trial for one space type and seed.

    Parameters
    ----------
    space_name : str
        Key into SPACE_CONFIGS.
    seed : int
        Random seed.
    budget_fraction : float
        Fraction of active nodes the governor plans to refine this step.
        Used by the thermodynamic guards to decide viability.

    Returns
    -------
    dict with trial results.
    """
    rng = np.random.default_rng(seed)
    config = SPACE_CONFIGS[space_name]
    tree = config["factory"](rng)

    # Chain statistics
    chain_stats = tree.chain_statistics()

    if chain_stats["degree2_nodes"] == 0:
        # No degree-2 nodes to compress
        return {
            "space": space_name,
            "seed": seed,
            "budget_fraction": budget_fraction,
            "chain_stats": chain_stats,
            "compression_ratios": [0.0] * N_STEPS,
            "overhead_estimates": [0.0] * N_STEPS,
            "merge_count": 0,
            "split_count": 0,
            "final_compression": 0.0,
            "final_overhead": 0.0,
            "n_segments_over_time": [0] * N_STEPS,
            "skipped": True,
            "guard_blocked": False,
            "guard_reason": "",
        }

    # Determine stable vs volatile nodes.
    # In a well-refined tree, most nodes are stable; only a small
    # fraction near active refinement fronts are volatile.
    all_nodes = sorted(tree.nodes)
    d2_nodes = sorted(n for n in all_nodes if tree.degree(n) == 2)
    n_stable = int(len(all_nodes) * 0.85)
    stable_set = set(rng.choice(all_nodes,
                                size=min(n_stable, len(all_nodes)),
                                replace=False))

    # Thermodynamic guard check: use pipeline-available metrics
    n_active = len(all_nodes)
    budget_step = int(n_active * budget_fraction)
    # Stable degree-2 nodes: approximate from the stable_set overlap
    n_stable_d2 = len([n for n in d2_nodes if n in stable_set])

    viable, guard_reason = should_compress(
        n_active=n_active,
        budget_step=budget_step,
        n_stable_d2=n_stable_d2,
    )

    if not viable:
        return {
            "space": space_name,
            "seed": seed,
            "budget_fraction": budget_fraction,
            "chain_stats": chain_stats,
            "compression_ratios": [0.0] * N_STEPS,
            "overhead_estimates": [0.0] * N_STEPS,
            "merge_count": 0,
            "split_count": 0,
            "final_compression": 0.0,
            "final_overhead": 0.0,
            "n_segments_over_time": [0] * N_STEPS,
            "skipped": True,
            "guard_blocked": True,
            "guard_reason": guard_reason,
        }

    # Event: at step N_STEPS//3, destabilize ~5% of degree-2 nodes
    # (simulates a local refinement event)
    event_step = N_STEPS // 3
    n_event = max(1, int(len(d2_nodes) * 0.05))
    if len(d2_nodes) > 0:
        event_nodes = set(rng.choice(d2_nodes,
                                     size=min(n_event, len(d2_nodes)),
                                     replace=False))
    else:
        event_nodes = set()

    # Initialize segment tree
    seg_tree = SegmentTree(tree, max_length=MAX_SEGMENT_LENGTH,
                           stability_window=STABILITY_WINDOW)

    compression_ratios = []
    overhead_estimates = []
    n_segments_over_time = []

    for step in range(N_STEPS):
        sigs = generate_signatures_with_event(
            all_nodes, step, rng, stable_set, event_step, event_nodes)

        seg_tree.update_step(sigs, step)

        compression_ratios.append(seg_tree.compression_ratio())
        overhead_estimates.append(seg_tree.net_overhead())
        n_segments_over_time.append(len(seg_tree.segments))

    return {
        "space": space_name,
        "seed": seed,
        "budget_fraction": budget_fraction,
        "chain_stats": chain_stats,
        "compression_ratios": compression_ratios,
        "overhead_estimates": overhead_estimates,
        "merge_count": len(seg_tree.merge_events),
        "split_count": len(seg_tree.split_events),
        "final_compression": compression_ratios[-1],
        "final_overhead": overhead_estimates[-1],
        "n_segments_over_time": n_segments_over_time,
        "skipped": False,
        "guard_blocked": False,
        "guard_reason": "",
    }


# ═══════════════════════════════════════════════════════════════════════
# Results aggregation
# ═══════════════════════════════════════════════════════════════════════

def aggregate_results(all_trials: list) -> dict:
    """Aggregate trial results into summary statistics.

    Parameters
    ----------
    all_trials : list of dict
        Results from run_trial.

    Returns
    -------
    dict with aggregated results per space type.
    """
    by_space = defaultdict(list)
    for trial in all_trials:
        by_space[trial["space"]].append(trial)

    summary = {}
    for space_name, trials in by_space.items():
        active_trials = [t for t in trials if not t.get("skipped", False)]
        if not active_trials:
            summary[space_name] = {
                "n_trials": len(trials),
                "n_active": 0,
                "mean_compression": 0.0,
                "std_compression": 0.0,
                "mean_overhead": 0.0,
                "std_overhead": 0.0,
                "mean_merges": 0.0,
                "mean_splits": 0.0,
                "chain_stats_mean": {},
                "pass_compression": False,
                "pass_overhead": True,
            }
            continue

        final_comps = [t["final_compression"] for t in active_trials]
        final_ohs = [t["final_overhead"] for t in active_trials]
        merges = [t["merge_count"] for t in active_trials]
        splits = [t["split_count"] for t in active_trials]

        # Average chain statistics
        chain_keys = ["total_nodes", "degree2_nodes", "num_chains",
                      "max_chain", "mean_chain", "fraction_degree2"]
        chain_means = {}
        for key in chain_keys:
            vals = [t["chain_stats"][key] for t in active_trials]
            chain_means[key] = float(np.mean(vals))

        mean_comp = float(np.mean(final_comps))
        mean_oh = float(np.mean(final_ohs))

        summary[space_name] = {
            "n_trials": len(trials),
            "n_active": len(active_trials),
            "mean_compression": mean_comp,
            "std_compression": float(np.std(final_comps)),
            "min_compression": float(np.min(final_comps)),
            "max_compression": float(np.max(final_comps)),
            "mean_overhead": mean_oh,
            "std_overhead": float(np.std(final_ohs)),
            "mean_merges": float(np.mean(merges)),
            "mean_splits": float(np.mean(splits)),
            "chain_stats_mean": chain_means,
            "pass_compression": mean_comp > 0.50,
            "pass_overhead": mean_oh < 0.10,  # net_overhead: negative = profitable
        }

    return summary


# ═══════════════════════════════════════════════════════════════════════
# Incremental save
# ═══════════════════════════════════════════════════════════════════════

def save_incremental(out_dir: Path, space_name: str, seed: int,
                     trial: dict) -> None:
    """Save a single trial result incrementally."""
    fname = out_dir / f"trial_{space_name}_seed{seed:02d}.json"
    # Convert numpy types for JSON serialization
    clean = _json_safe(trial)
    with open(fname, "w") as f:
        json.dump(clean, f, indent=2)


def _json_safe(obj):
    """Recursively convert numpy types to Python native for JSON."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ═══════════════════════════════════════════════════════════════════════
# Main experiment driver
# ═══════════════════════════════════════════════════════════════════════

def main():
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Budget fractions to sweep: low (guards allow), high (guards block)
    BUDGET_FRACTIONS = [0.10, 0.30, 0.60]

    print("=" * 70)
    print("exp13 -- Segment Compression (P1-B1) + Thermodynamic Guards")
    print("=" * 70)
    print(f"Spaces: {list(SPACE_CONFIGS.keys())}")
    print(f"Seeds: {N_SEEDS}, Steps: {N_STEPS}")
    print(f"Budget fractions: {BUDGET_FRACTIONS}")
    print(f"Max segment length: {MAX_SEGMENT_LENGTH}, "
          f"Stability window: {STABILITY_WINDOW}")
    print()

    all_trials = []
    all_chain_stats = defaultdict(list)
    guard_log = []

    for space_name in SPACE_CONFIGS:
        desc = SPACE_CONFIGS[space_name]["description"]

        for bf in BUDGET_FRACTIONS:
            tag = f"{space_name}_bf{int(bf*100):02d}"
            print(f"[{tag}] {desc}, budget={bf:.0%}", end="", flush=True)

            for seed in range(N_SEEDS):
                trial = run_trial(space_name, seed, budget_fraction=bf)
                all_trials.append(trial)
                all_chain_stats[space_name].append(trial["chain_stats"])

                if trial.get("guard_blocked"):
                    guard_log.append({
                        "space": space_name,
                        "seed": seed,
                        "budget_fraction": bf,
                        "reason": trial["guard_reason"],
                        "total_nodes": trial["chain_stats"]["total_nodes"],
                        "degree2_nodes": trial["chain_stats"]["degree2_nodes"],
                    })

                # Incremental save
                save_incremental(out_dir, space_name, seed, trial)
                print(".", end="", flush=True)

            print(" done")

    # ── Guard summary ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("THERMODYNAMIC GUARD REPORT")
    print("=" * 70)
    n_total = len(all_trials)
    n_blocked = sum(1 for t in all_trials if t.get("guard_blocked"))
    n_ran = sum(1 for t in all_trials if not t.get("skipped"))
    n_skipped_no_d2 = sum(1 for t in all_trials
                          if t.get("skipped") and not t.get("guard_blocked"))
    print(f"  Total trials:     {n_total}")
    print(f"  Guard blocked:    {n_blocked}")
    print(f"  Ran compression:  {n_ran}")
    print(f"  Skipped (no d2):  {n_skipped_no_d2}")

    if guard_log:
        # Group by reason
        by_reason = defaultdict(list)
        for g in guard_log:
            by_reason[g["reason"]].append(g)
        for reason, entries in by_reason.items():
            spaces_hit = set(e["space"] for e in entries)
            budgets_hit = sorted(set(e["budget_fraction"] for e in entries))
            print(f"\n  [{reason}] {len(entries)} trials blocked")
            print(f"    Spaces:  {sorted(spaces_hit)}")
            print(f"    Budgets: {budgets_hit}")
            for e in entries[:3]:
                print(f"      {e['space']} seed={e['seed']} bf={e['budget_fraction']}"
                      f" nodes={e['total_nodes']} d2={e['degree2_nodes']}")
            if len(entries) > 3:
                print(f"      ... and {len(entries)-3} more")

    # ── Aggregate (only non-guard-blocked, non-skipped) ───────────
    # Group by (space, budget_fraction) for detailed view
    from itertools import groupby
    print("\n" + "=" * 70)
    print("RESULTS BY SPACE x BUDGET")
    print("=" * 70)

    print(f"\n{'Space+Budget':<35s} {'Comp%':>7s} {'OH%':>7s} "
          f"{'D2%':>7s} {'Guard':>7s} {'Verdict':>8s}")
    print("-" * 75)

    overall_pass = True
    for space_name in SPACE_CONFIGS:
        for bf in BUDGET_FRACTIONS:
            trials_here = [t for t in all_trials
                           if t["space"] == space_name
                           and t.get("budget_fraction", 0.15) == bf]
            active = [t for t in trials_here
                      if not t.get("skipped") and not t.get("guard_blocked")]
            blocked = [t for t in trials_here if t.get("guard_blocked")]

            tag = f"{space_name} bf={bf:.0%}"

            if blocked and not active:
                # All blocked by guard — overhead = 0%, this is correct
                print(f"{tag:<35s} {'---':>7s} {'0.0%':>7s} "
                      f"{'---':>7s} {len(blocked):>5d}bl {'GUARD':>8s}")
                continue

            if not active:
                print(f"{tag:<35s} {'---':>7s} {'---':>7s} "
                      f"{'---':>7s} {'---':>7s} {'SKIP':>8s}")
                continue

            comps = [t["final_compression"] for t in active]
            ohs = [t["final_overhead"] for t in active]
            mean_comp = float(np.mean(comps))
            mean_oh = float(np.mean(ohs))
            cs_vals = [t["chain_stats"]["fraction_degree2"] for t in active]
            mean_d2 = float(np.mean(cs_vals))

            pass_comp = mean_comp > 0.50
            pass_oh = mean_oh < 0.10
            verdict = "PASS" if (pass_comp and pass_oh) else "FAIL"
            if not pass_comp and pass_oh:
                verdict = "INVEST"
            if not pass_oh:
                overall_pass = False

            n_bl = len(blocked)
            guard_str = f"{n_bl}bl" if n_bl else "0"

            print(f"{tag:<35s} {mean_comp*100:>6.1f}% {mean_oh*100:>6.1f}% "
                  f"{mean_d2*100:>6.1f}% {guard_str:>7s} {verdict:>8s}")

    # ── Legacy aggregate (budget=0.10 only, for KC comparison) ────
    low_budget_trials = [t for t in all_trials
                         if t.get("budget_fraction", 0.15) == 0.10]
    summary = aggregate_results(low_budget_trials)

    # ── Chain statistics detail ────────────────────────────────────
    print("\n" + "-" * 70)
    print("Chain Statistics (per space, averaged over seeds, bf=10%):")
    print("-" * 70)
    for space_name in SPACE_CONFIGS:
        if space_name not in summary:
            continue
        s = summary[space_name]
        cs = s["chain_stats_mean"]
        if not cs:
            continue
        print(f"  {space_name}:")
        print(f"    total_nodes:    {cs.get('total_nodes', 0):.1f}")
        print(f"    degree2_nodes:  {cs.get('degree2_nodes', 0):.1f}")
        print(f"    num_chains:     {cs.get('num_chains', 0):.1f}")
        print(f"    max_chain:      {cs.get('max_chain', 0):.1f}")
        print(f"    mean_chain:     {cs.get('mean_chain', 0):.2f}")
        print(f"    fraction_d2:    {cs.get('fraction_degree2', 0):.3f}")

    # ── Kill criteria check ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("KILL CRITERIA CHECK (with thermodynamic guards)")
    print("=" * 70)

    # KC: for any trial that actually ran, compression > 50% and OH < 10%
    # Guard-blocked trials have OH = 0% by definition (no work done)
    ran_trials = [t for t in all_trials if not t.get("skipped")]
    if ran_trials:
        comps_all = [t["final_compression"] for t in ran_trials]
        ohs_all = [t["final_overhead"] for t in ran_trials]
        print(f"  Trials that ran compression: {len(ran_trials)}")
        print(f"    Mean compression: {np.mean(comps_all):.3f} "
              f"({'> 50% PASS' if np.mean(comps_all) > 0.50 else '< 50% INVEST'})")
        print(f"    Mean overhead:    {np.mean(ohs_all):.3f} "
              f"({'< 10% PASS' if np.mean(ohs_all) < 0.10 else '> 10% FAIL'})")
        kc_oh_pass = all(oh < 0.10 for oh in ohs_all)
        print(f"    Max overhead:     {max(ohs_all):.3f} "
              f"({'< 10% PASS' if kc_oh_pass else '> 10% FAIL'})")
    else:
        kc_oh_pass = True
        print("  No trials ran compression (all guard-blocked or no d2)")

    print(f"\n  Guard-blocked trials: {n_blocked} "
          f"(overhead = 0% by construction)")
    print(f"  Guard effectiveness: eliminated {n_blocked}/{n_total} "
          f"non-viable trials")

    overall_verdict = "PASS" if (overall_pass and kc_oh_pass) else "FAIL"
    print(f"\n  Overall: {overall_verdict}")

    # ── Save final results ─────────────────────────────────────────
    results_json = {
        "verdict": overall_verdict,
        "summary": _json_safe(summary),
        "guard_report": {
            "total_trials": n_total,
            "guard_blocked": n_blocked,
            "ran_compression": n_ran,
            "guard_log_sample": _json_safe(guard_log[:20]),
        },
        "config": {
            "n_seeds": N_SEEDS,
            "n_steps": N_STEPS,
            "budget_fractions": BUDGET_FRACTIONS,
            "max_segment_length": MAX_SEGMENT_LENGTH,
            "stability_window": STABILITY_WINDOW,
            "hamming_threshold": 3,
            "component_threshold": 4,
            "spaces": {k: v["description"] for k, v in SPACE_CONFIGS.items()},
        },
    }

    with open(out_dir / "exp13_results.json", "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nSaved: {out_dir / 'exp13_results.json'}")

    # Per-step data for non-blocked trials
    per_step_data = {}
    for space_name in SPACE_CONFIGS:
        space_trials = [t for t in all_trials
                        if t["space"] == space_name and not t.get("skipped")]
        if not space_trials:
            continue
        comp_arr = np.array([t["compression_ratios"] for t in space_trials])
        oh_arr = np.array([t["overhead_estimates"] for t in space_trials])
        seg_arr = np.array([t["n_segments_over_time"] for t in space_trials])
        per_step_data[space_name] = {
            "mean_compression": comp_arr.mean(axis=0).tolist(),
            "std_compression": comp_arr.std(axis=0).tolist(),
            "mean_overhead": oh_arr.mean(axis=0).tolist(),
            "mean_n_segments": seg_arr.mean(axis=0).tolist(),
        }

    with open(out_dir / "exp13_per_step.json", "w") as f:
        json.dump(_json_safe(per_step_data), f, indent=2)
    print(f"Saved: {out_dir / 'exp13_per_step.json'}")

    chain_stats_json = {
        space: [_json_safe(cs) for cs in stats_list]
        for space, stats_list in all_chain_stats.items()
    }
    with open(out_dir / "exp13_chain_stats.json", "w") as f:
        json.dump(chain_stats_json, f, indent=2)
    print(f"Saved: {out_dir / 'exp13_chain_stats.json'}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    return results_json


# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
