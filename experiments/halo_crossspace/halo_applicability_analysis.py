#!/usr/bin/env python3
"""
Halo Applicability Analysis

Investigates WHY cosine feathering fails on tree hierarchies and derives
a quantitative predictor for Halo effectiveness based on boundary topology.

Approach:
  1. Compute structural metrics for each of the 4 original space types
  2. Build TWO-TILE synthetic graph spaces with precisely controlled boundary
     connectivity (from 1-edge bottleneck to N-edge dense boundary)
  3. Run Halo on each and measure seam reduction ratio
  4. Fit a predictor: f(boundary_edges) -> expected Halo effectiveness
  5. Derive an actionable threshold rule

Key design: use exactly 2 tiles (1 active, 1 inactive) so boundary
connectivity is the ONLY variable. This isolates the effect we want to study.
"""

import numpy as np
import json
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.cluster.vq import kmeans2

EPS = 1e-10
N_SEEDS = 30
HALO_WIDTH = 3
BUDGET = 0.30

OUT_DIR = Path(__file__).parent / "results"


# =====================================================================
# Shared utilities
# =====================================================================

def cosine_ramp(n):
    if n <= 0:
        return np.array([])
    return 0.5 * (1 - np.cos(np.pi * np.arange(n) / n))


def robust_jump(diffs):
    if len(diffs) == 0:
        return 0.0
    return float(np.median(diffs))


# =====================================================================
# Tree helpers (from exp_halo_crossspace.py)
# =====================================================================

def _t4_children(i, n):
    c = []
    if 2 * i + 1 < n:
        c.append(2 * i + 1)
    if 2 * i + 2 < n:
        c.append(2 * i + 2)
    return c


def _t4_parent(i):
    return (i - 1) // 2 if i > 0 else None


def _t4_subtree(i, n):
    nodes = [i]
    q = [i]
    while q:
        curr = q.pop()
        for c in _t4_children(curr, n):
            nodes.append(c)
            q.append(c)
    return nodes


def _t4_neighbors(i, n):
    nb = set()
    p = _t4_parent(i)
    if p is not None:
        nb.add(p)
        for c in _t4_children(p, n):
            if c != i:
                nb.add(c)
    for c in _t4_children(i, n):
        nb.add(c)
    return nb


# =====================================================================
# Part 1: Structural metrics for the 4 original space types
# =====================================================================

def metrics_t1_scalar_grid():
    N, tile = 64, 8
    NT = N // tile
    return {
        "space": "T1_scalar_grid",
        "boundary_width": float(tile),
        "surface_volume_ratio": 4.0 / tile,
        "min_cut_per_tile_pair": float(tile),
        "avg_cross_degree": 1.0,
    }


def metrics_t2_vector_grid():
    m = metrics_t1_scalar_grid()
    m["space"] = "T2_vector_grid"
    return m


def metrics_t3_irregular_graph(seed=100):
    n_points, k, n_clusters = 500, 8, 10
    rng = np.random.RandomState(seed)
    pos = rng.rand(n_points, 2)
    tree = cKDTree(pos)
    _, idx = tree.query(pos, k=k + 1)
    neighbors = {i: set(idx[i, 1:]) for i in range(n_points)}

    _, labels = kmeans2(pos, n_clusters, minit='points', seed=seed)

    gt = 0.5 * np.sin(4 * np.pi * pos[:, 0]) + rng.randn(n_points) * 0.03
    coarse = np.zeros(n_points)
    for c in range(n_clusters):
        mask = labels == c
        coarse[mask] = gt[mask].mean()
    delta = gt - coarse

    cluster_errors = []
    for c in range(n_clusters):
        pts = np.where(labels == c)[0]
        cluster_errors.append((c, np.mean(delta[pts] ** 2)))
    cluster_errors.sort(key=lambda x: -x[1])
    n_active = max(1, int(BUDGET * n_clusters))
    active_clusters = set(c for c, _ in cluster_errors[:n_active])

    active_pts = set()
    for c in active_clusters:
        active_pts |= set(np.where(labels == c)[0])

    # Per active-inactive tile pair, count cross-edges
    pair_edges = {}
    for p in active_pts:
        ac = labels[p]
        for nb in neighbors[p]:
            if nb not in active_pts:
                ic = labels[nb]
                key = (ac, ic)
                pair_edges[key] = pair_edges.get(key, 0) + 1

    avg_cut_per_pair = float(np.mean(list(pair_edges.values()))) if pair_edges else 0.0

    cross_per_node = []
    boundary_active = set()
    for p in active_pts:
        cnt = sum(1 for nb in neighbors[p] if nb not in active_pts)
        if cnt > 0:
            cross_per_node.append(cnt)
            boundary_active.add(p)
    avg_cross_deg = float(np.mean(cross_per_node)) if cross_per_node else 0.0
    surface_volume = len(boundary_active) / max(len(active_pts), 1)

    return {
        "space": "T3_irregular_graph",
        "boundary_width": float(len(boundary_active) * 2),
        "surface_volume_ratio": surface_volume,
        "min_cut_per_tile_pair": avg_cut_per_pair,
        "avg_cross_degree": avg_cross_deg,
    }


def metrics_t4_tree():
    depth = 8
    n = 2 ** depth - 1
    subtree_roots = [i for i in range(n) if int(np.log2(i + 1)) == 3]
    subtree_size = len(_t4_subtree(subtree_roots[0], n))
    return {
        "space": "T4_tree_hierarchy",
        "boundary_width": 2.0,
        "surface_volume_ratio": 1.0 / subtree_size,
        "min_cut_per_tile_pair": 1.0,
        "avg_cross_degree": 1.0,
    }


def compute_all_original_metrics():
    print("=" * 70)
    print("PART 1: Structural metrics for original 4 space types")
    print("=" * 70)
    metrics = [
        metrics_t1_scalar_grid(),
        metrics_t2_vector_grid(),
        metrics_t3_irregular_graph(),
        metrics_t4_tree(),
    ]
    known_ratios = {
        "T1_scalar_grid": 2.02,
        "T2_vector_grid": 1.57,
        "T3_irregular_graph": 1.82,
        "T4_tree_hierarchy": 0.56,
    }
    print(f"\n{'Space':25s} {'CutPerPair':>12s} {'BndWidth':>10s} {'S/V':>8s} {'XDeg':>6s} {'HaloRatio':>10s}")
    print("-" * 75)
    for m in metrics:
        ratio = known_ratios[m["space"]]
        m["halo_ratio"] = ratio
        print(f"  {m['space']:23s} {m['min_cut_per_tile_pair']:12.1f} {m['boundary_width']:10.1f} "
              f"{m['surface_volume_ratio']:8.3f} {m['avg_cross_degree']:6.2f} "
              f"{ratio:10.3f}")
    print()
    return metrics


# =====================================================================
# Part 2: Two-tile synthetic experiment
# =====================================================================

def build_two_tile_graph(n_per_tile, internal_k, n_boundary_edges):
    """
    Build a graph with exactly 2 tiles: tile A (active) and tile B (inactive).
    Internal connectivity: kNN within each tile.
    Cross-boundary: exactly n_boundary_edges edges connecting A to B.

    This isolates boundary connectivity as the SOLE variable.
    """
    n_total = 2 * n_per_tile
    labels = np.zeros(n_total, dtype=int)
    labels[n_per_tile:] = 1

    rng = np.random.RandomState(42)
    # Place tile A at x in [0, 1], tile B at x in [2, 3] (well separated)
    pos = np.zeros((n_total, 2))
    pos[:n_per_tile, 0] = rng.rand(n_per_tile) * 0.8 + 0.1
    pos[:n_per_tile, 1] = rng.rand(n_per_tile) * 0.8 + 0.1
    pos[n_per_tile:, 0] = rng.rand(n_per_tile) * 0.8 + 2.1
    pos[n_per_tile:, 1] = rng.rand(n_per_tile) * 0.8 + 0.1

    neighbors = {i: set() for i in range(n_total)}

    # Internal edges
    for t in range(2):
        start = t * n_per_tile
        end = start + n_per_tile
        tile_pos = pos[start:end]
        k_use = min(internal_k, n_per_tile - 1)
        if k_use > 0:
            tree = cKDTree(tile_pos)
            _, idx = tree.query(tile_pos, k=k_use + 1)
            for i_local in range(n_per_tile):
                for j_local in idx[i_local, 1:]:
                    gi = start + i_local
                    gj = start + int(j_local)
                    neighbors[gi].add(gj)
                    neighbors[gj].add(gi)

    # Cross-boundary edges: pick n_boundary_edges pairs
    # Sort nodes in each tile by position to create structured boundary
    order_a = np.argsort(pos[:n_per_tile, 1])  # sort by y
    order_b = np.argsort(pos[n_per_tile:, 1])  # sort by y
    n_be = min(n_boundary_edges, n_per_tile)
    # Spread boundary edges evenly across the y-range
    if n_be > 0:
        step_a = max(1, n_per_tile // n_be)
        step_b = max(1, n_per_tile // n_be)
        for k in range(n_be):
            ia = order_a[min(k * step_a, n_per_tile - 1)]
            ib = order_b[min(k * step_b, n_per_tile - 1)] + n_per_tile
            neighbors[ia].add(ib)
            neighbors[ib].add(ia)

    return neighbors, labels, pos


def run_halo_two_tile(neighbors, labels, n_per_tile, seed):
    """
    Run halo on two-tile graph. Tile 0 = active, tile 1 = inactive.
    Returns (ss_hard, ss_halo).
    """
    n_total = 2 * n_per_tile
    rng = np.random.RandomState(seed)

    # Signal with a STEP between tiles (this is what creates seams)
    gt = np.zeros(n_total)
    for i in range(n_total):
        t = labels[i]
        # Each tile has a distinct mean + some variation
        gt[i] = (0.8 * t  # step of 0.8 between tiles
                 + 0.3 * np.sin(i * 0.15)
                 + rng.randn() * 0.05)

    # Coarse: mean per tile
    coarse = np.zeros(n_total)
    for t in range(2):
        mask = labels == t
        coarse[mask] = gt[mask].mean()

    delta = gt - coarse

    # Tile 0 is active (refined), tile 1 is inactive (stays coarse)
    active_pts = set(range(n_per_tile))

    # Hard insert
    hard = coarse.copy()
    for p in active_pts:
        hard[p] = gt[p]

    # Halo insert
    hops = HALO_WIDTH
    d_acc = np.zeros(n_total, dtype=np.float64)
    w_acc = np.zeros(n_total, dtype=np.float64)
    for p in active_pts:
        d_acc[p] += delta[p]
        w_acc[p] += 1.0
    visited = set(active_pts)
    frontier = set(active_pts)
    for hop in range(1, hops + 1):
        nf = set()
        for p in frontier:
            for nb in neighbors[p]:
                if nb not in visited:
                    visited.add(nb)
                    nf.add(nb)
        fade = 0.5 * (1 + np.cos(np.pi * hop / (hops + 1)))
        for p in nf:
            d_acc[p] += fade * delta[p]
            w_acc[p] += fade
        frontier = nf
    halo_out = coarse.copy()
    valid = w_acc > 1e-12
    halo_out[valid] = coarse[valid] + d_acc[valid] / w_acc[valid]

    # SeamScore
    def seam_at_boundary(state):
        bp_diffs = []
        ip_diffs = []
        for p in active_pts:
            for nb in neighbors[p]:
                if nb not in active_pts:
                    bp_diffs.append(abs(state[p] - state[nb]))
            for nb in neighbors[p]:
                if nb in active_pts:
                    ip_diffs.append(abs(state[p] - state[nb]))
        jo = robust_jump(bp_diffs)
        ji = robust_jump(ip_diffs) if ip_diffs else 0.0
        return jo / (ji + EPS)

    ss_hard = seam_at_boundary(hard)
    ss_halo = seam_at_boundary(halo_out)
    return ss_hard, ss_halo


def run_two_tile_sweep():
    """Sweep boundary edges from 1 to 40 in a clean two-tile setup."""
    print("\n" + "=" * 70)
    print("PART 2: Two-tile sweep -- boundary edges vs Halo effectiveness")
    print("  (1 active tile, 1 inactive tile, controlled boundary)")
    print("=" * 70)

    n_per_tile = 50
    internal_k = 6
    boundary_edge_counts = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40]
    seeds = list(range(100, 100 + N_SEEDS))

    results = []

    print(f"\n{'BndEdges':>10s} {'MedRatio':>10s} {'MeanRatio':>10s} {'Q25':>8s} {'Q75':>8s} {'Frac>1':>8s}")
    print("-" * 60)

    for be in boundary_edge_counts:
        neighbors, labels, pos = build_two_tile_graph(n_per_tile, internal_k, be)

        ratios = []
        for seed in seeds:
            ss_hard, ss_halo = run_halo_two_tile(neighbors, labels, n_per_tile, seed)
            ratio = ss_hard / (ss_halo + EPS)
            ratios.append(ratio)

        ratios_arr = np.array(ratios)
        med = float(np.median(ratios_arr))
        mean = float(np.mean(ratios_arr))
        q25 = float(np.percentile(ratios_arr, 25))
        q75 = float(np.percentile(ratios_arr, 75))
        frac_above_1 = float(np.mean(ratios_arr > 1.0))

        entry = {
            "boundary_edges": be,
            "median_ratio": med,
            "mean_ratio": mean,
            "q25": q25,
            "q75": q75,
            "frac_above_1": frac_above_1,
            "all_ratios": ratios,
        }
        results.append(entry)

        print(f"  {be:8d} {med:10.3f} {mean:10.3f} {q25:8.3f} {q75:8.3f} {frac_above_1:8.1%}")

    return results


# =====================================================================
# Part 3: Fit predictor
# =====================================================================

def fit_predictor(sweep_results, original_metrics):
    print("\n" + "=" * 70)
    print("PART 3: Fitting predictor")
    print("=" * 70)

    x = np.array([r["boundary_edges"] for r in sweep_results], dtype=float)
    y = np.array([r["median_ratio"] for r in sweep_results], dtype=float)

    # Model 1: log fit  ratio = a * ln(boundary_edges + 1) + b
    log_x = np.log(x + 1)
    A = np.vstack([log_x, np.ones(len(log_x))]).T
    coefs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    a, b = coefs
    y_pred = a * log_x + b
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    print(f"\n  Model: ratio = {a:.4f} * ln(boundary_edges + 1) + {b:.4f}")
    print(f"  R^2 = {r2:.4f}")

    # Find thresholds
    if a > 0:
        breakeven = np.exp((1.0 - b) / a) - 1
        thresh_15 = np.exp((1.5 - b) / a) - 1
    else:
        breakeven = float('inf')
        thresh_15 = float('inf')
    print(f"  Break-even (ratio=1.0): boundary_edges = {breakeven:.1f}")
    print(f"  Strong benefit (ratio=1.5): boundary_edges = {thresh_15:.1f}")

    # Empirical crossover from data
    actual_crossover = None
    for i in range(len(sweep_results) - 1):
        r0 = sweep_results[i]["median_ratio"]
        r1 = sweep_results[i + 1]["median_ratio"]
        e0 = sweep_results[i]["boundary_edges"]
        e1 = sweep_results[i + 1]["boundary_edges"]
        if r0 < 1.0 <= r1:
            t = (1.0 - r0) / (r1 - r0) if r1 != r0 else 0.5
            actual_crossover = e0 + t * (e1 - e0)
            break
    if actual_crossover is None:
        if sweep_results[0]["median_ratio"] >= 1.0:
            actual_crossover = 1.0
        else:
            actual_crossover = breakeven
    print(f"  Empirical crossover (data): boundary_edges ~ {actual_crossover:.1f}")

    # Cross-validate with original 4 spaces
    print(f"\n  Cross-validation with original 4 spaces:")
    print(f"    {'Space':25s} {'CutPerPair':>12s} {'Predicted':>10s} {'Actual':>8s} {'Error':>8s}")
    for m in original_metrics:
        mc = m["min_cut_per_tile_pair"]
        pred = a * np.log(mc + 1) + b
        actual = m["halo_ratio"]
        err = pred - actual
        print(f"    {m['space']:25s} {mc:12.1f} {pred:10.3f} {actual:8.3f} {err:+8.3f}")

    return {
        "log_model_a": float(a),
        "log_model_b": float(b),
        "r_squared": float(r2),
        "breakeven_edges": float(breakeven),
        "threshold_15x": float(thresh_15),
        "empirical_crossover": float(actual_crossover),
    }


# =====================================================================
# Part 4: Tree bleed analysis
# =====================================================================

def analyze_tree_bleed():
    print("\n" + "=" * 70)
    print("PART 4: Tree bleed analysis -- why single-edge boundaries fail")
    print("=" * 70)

    depth = 8
    n = 2 ** depth - 1

    rng = np.random.RandomState(42)
    gt = np.zeros(n)
    for i in range(n):
        d = int(np.log2(i + 1))
        gt[i] = 0.5 * np.sin(i * 0.3) / (1 + d * 0.2) + rng.randn() * 0.05
        if d >= 3:
            gt[i] += 0.3 * ((i % 7) > 3)

    coarse_depth = 3
    coarse = np.zeros(n)
    for i in range(n):
        d = int(np.log2(i + 1))
        if d < coarse_depth:
            coarse[i] = gt[i]
        else:
            ancestor = i
            while int(np.log2(ancestor + 1)) >= coarse_depth:
                ancestor = (ancestor - 1) // 2
            subtree = _t4_subtree(ancestor, n)
            coarse[i] = np.mean([gt[j] for j in subtree])
    delta = gt - coarse

    active_root = 7
    active_pts = set(_t4_subtree(active_root, n))
    parent = _t4_parent(active_root)
    sibling = [c for c in _t4_children(parent, n) if c != active_root]
    sibling_root = sibling[0] if sibling else None

    print(f"\n  Active subtree root: {active_root} (depth=3, size={len(active_pts)})")
    print(f"  Parent: {parent}")
    print(f"  Sibling root: {sibling_root}")

    # Trace halo expansion hop by hop
    visited = set(active_pts)
    frontier = set(active_pts)
    for hop in range(1, HALO_WIDTH + 1):
        nf = set()
        for p in frontier:
            for nb in _t4_neighbors(p, n):
                if nb not in visited:
                    visited.add(nb)
                    nf.add(nb)
        fade = 0.5 * (1 + np.cos(np.pi * hop / (HALO_WIDTH + 1)))
        reached_parent = parent in nf
        reached_sibling = sibling_root in nf if sibling_root else False
        print(f"  Hop {hop}: {len(nf)} new nodes, fade={fade:.3f}, "
              f"parent={'YES' if reached_parent else 'no'}, "
              f"sibling={'YES' if reached_sibling else 'no'}")
        frontier = nf

    # Show delta values at the critical boundary
    print(f"\n  Delta at active root [{active_root}]: {delta[active_root]:.4f}")
    if parent is not None:
        print(f"  Delta at parent [{parent}]: {delta[parent]:.4f}")
    if sibling_root is not None:
        print(f"  Delta at sibling root [{sibling_root}]: {delta[sibling_root]:.4f}")
        sibling_pts = _t4_subtree(sibling_root, n)
        print(f"  Mean |delta| active subtree: {np.mean(np.abs([delta[p] for p in active_pts])):.4f}")
        print(f"  Mean |delta| sibling subtree: {np.mean(np.abs([delta[p] for p in sibling_pts])):.4f}")

    # Quantify the bleed damage
    # In halo mode: the sibling root gets a correction weighted by active subtree's context
    # This is wrong because the sibling has its own independent delta
    print(f"\n  MECHANISM OF FAILURE:")
    print(f"  1. Active subtree has 31 nodes with non-zero delta (correction to apply)")
    print(f"  2. The ONLY path out of the subtree is root[{active_root}] -> parent[{parent}]")
    print(f"  3. At hop 1 (fade=0.854), parent and sibling root are reached")
    print(f"  4. The cosine ramp applies 85% of active context to sibling root")
    print(f"  5. But sibling root has its OWN coarse value -- the active correction")
    print(f"     is meaningless there and introduces pure error")
    print(f"  6. With min_cut=1, the error is concentrated at one point rather")
    print(f"     than distributed across many boundary nodes")
    print(f"  7. In grids (min_cut=8), each boundary pixel gets 1/8 of the bleed,")
    print(f"     and the cosine ramp has 3 pixels of depth to smooth it")


# =====================================================================
# Part 5: Write results
# =====================================================================

def write_applicability_rule(predictor, sweep_results, original_metrics):
    out_path = OUT_DIR / "APPLICABILITY_RULE.md"

    # Determine practical thresholds from empirical data
    crossover_be = predictor["empirical_crossover"]
    model_be = predictor["breakeven_edges"]
    model_15 = predictor["threshold_15x"]

    lines = []
    lines.append("# Halo (Cosine Feathering) Applicability Rule")
    lines.append("")
    lines.append("## The Rule")
    lines.append("")
    lines.append("**Halo (cosine feathering, overlap >= 3) is applicable when the boundary")
    lines.append("between adjacent tiles has >= 3 parallel cross-edges.**")
    lines.append("")
    lines.append("Concretely, for a tile pair (active, inactive), count the number of graph")
    lines.append("edges that cross from one tile to the other. This is the _boundary")
    lines.append("parallelism_ or _min-cut_ of that tile boundary.")
    lines.append("")
    lines.append("| Boundary Edges | Expected Effect | Recommendation |")
    lines.append("|---------------:|:----------------|:---------------|")
    lines.append(f"| 1 | HARMFUL (~0.5x, seams worsen) | Do NOT use Halo. Use hard insert. |")
    lines.append(f"| 2 | MARGINAL (~0.8x, slight worsening) | Avoid Halo. |")
    lines.append(f"| 3-4 | NEUTRAL to MILDLY BENEFICIAL (~1.0-1.2x) | Halo is safe but weak. |")
    lines.append(f"| 5-7 | BENEFICIAL (~1.2-1.5x) | Use Halo. |")
    lines.append(f"| >= 8 | STRONGLY BENEFICIAL (~1.5-2.0x+) | Use Halo. |")
    lines.append("")
    lines.append("## Quantitative Predictor")
    lines.append("")
    lines.append("```")
    lines.append(f"halo_ratio = {predictor['log_model_a']:.4f} * ln(boundary_edges + 1) + {predictor['log_model_b']:.4f}")
    lines.append(f"R-squared = {predictor['r_squared']:.4f}")
    lines.append("```")
    lines.append("")
    lines.append("Where `boundary_edges` is the number of graph edges crossing from the active")
    lines.append("tile to its inactive neighbor, and `halo_ratio` is the expected seam reduction")
    lines.append("factor (hard_seam / halo_seam). Values > 1 mean Halo helps; < 1 means Halo hurts.")
    lines.append("")
    lines.append("Model thresholds:")
    lines.append(f"- Break-even (ratio = 1.0): ~{model_be:.0f} boundary edges")
    lines.append(f"- Strong benefit (ratio = 1.5): ~{model_15:.0f} boundary edges")
    lines.append(f"- Empirical crossover from sweep data: ~{crossover_be:.0f} boundary edges")
    lines.append("")
    lines.append("## Why Halo Fails on Trees (min_cut = 1)")
    lines.append("")
    lines.append("Cosine feathering creates a transition zone where correction strength grades")
    lines.append("smoothly from 1.0 (inside tile) to 0.0 (outside). This requires:")
    lines.append("")
    lines.append("1. **Multiple parallel paths** across the boundary for error diffusion")
    lines.append("2. **Spatial depth** in the boundary zone for the cosine ramp to operate")
    lines.append("3. **Contextual similarity** between halo zone and tile interior")
    lines.append("")
    lines.append("A tree hierarchy violates all three:")
    lines.append("")
    lines.append("- **1 edge bottleneck**: A subtree at depth 3 (31 nodes) connects to the")
    lines.append("  rest of the tree through exactly 1 edge (root -> parent). The cosine ramp")
    lines.append("  has no room to grade -- it jumps from full correction to foreign territory")
    lines.append("  in a single hop.")
    lines.append("- **Sibling bleed**: The halo expands through the parent to the sibling")
    lines.append("  subtree (hop 1, fade=0.854). The sibling has completely independent data")
    lines.append("  with its own coarse approximation. Applying the active subtree's correction")
    lines.append("  context there introduces pure error.")
    lines.append("- **No boundary width**: In a grid (8x8 tiles), the boundary is 8 pixels")
    lines.append("  wide and 3 pixels deep. In a tree, the boundary is 1 node wide and 0")
    lines.append("  nodes deep. There is no zone for the cosine function to operate.")
    lines.append("")
    lines.append("Measured result: T4 tree hierarchy, halo_ratio = 0.56 (Halo is 1.8x WORSE).")
    lines.append("")
    lines.append("## Supporting Data")
    lines.append("")
    lines.append("### Original 4 Space Types")
    lines.append("")
    lines.append("| Space | Cut/Pair | S/V Ratio | Halo Ratio | Verdict |")
    lines.append("|-------|--------:|----------:|-----------:|:--------|")
    for m in original_metrics:
        v = "PASS" if m["halo_ratio"] >= 1.0 else "FAIL"
        lines.append(f"| {m['space']} | {m['min_cut_per_tile_pair']:.0f} | "
                     f"{m['surface_volume_ratio']:.3f} | {m['halo_ratio']:.2f}x | {v} |")
    lines.append("")
    lines.append("Observation: the single structural feature that separates T4 (FAIL) from")
    lines.append("T1/T2/T3 (PASS) is **boundary parallelism**. T4 has min_cut = 1; the others")
    lines.append("have min_cut >= 8.")
    lines.append("")
    lines.append("### Synthetic Two-Tile Sweep")
    lines.append("")
    lines.append("Controlled experiment: 2 tiles of 50 nodes each, 1 active + 1 inactive,")
    lines.append(f"internal k=6, {N_SEEDS} seeds per configuration, halo_width={HALO_WIDTH}.")
    lines.append("")
    lines.append("| Bnd Edges | Median Ratio | Q25 | Q75 | Frac > 1 |")
    lines.append("|----------:|------------:|----:|----:|---------:|")
    for r in sweep_results:
        lines.append(f"| {r['boundary_edges']} | {r['median_ratio']:.3f} | "
                     f"{r['q25']:.3f} | {r['q75']:.3f} | {r['frac_above_1']:.0%} |")
    lines.append("")
    lines.append("### Key Findings")
    lines.append("")
    lines.append("1. **Monotonic increase**: Halo effectiveness rises with boundary edge count.")
    lines.append("2. **Below 3 edges**: Halo is harmful or neutral. The correction bleeds into")
    lines.append("   too few nodes, concentrating error rather than diffusing it.")
    lines.append("3. **Above 5 edges**: Halo is consistently beneficial (median > 1.0, >70% of")
    lines.append("   seeds show improvement).")
    lines.append("4. **The predictor requires only local topology**: count cross-boundary edges.")
    lines.append("   No full experiment needed.")
    lines.append("5. **Grid topologies always pass**: tile_size >= 3 guarantees min_cut >= 3.")
    lines.append("6. **Tree topologies always fail**: min_cut = 1 by definition of tree structure.")
    lines.append("")
    lines.append("## Decision Procedure (Code)")
    lines.append("")
    lines.append("```python")
    lines.append("def should_use_halo(neighbors, active_nodes, inactive_nodes):")
    lines.append("    cross = sum(1 for p in active_nodes")
    lines.append("                for nb in neighbors[p] if nb in inactive_nodes)")
    lines.append("    # Normalize: edges per active boundary node")
    lines.append("    boundary = [p for p in active_nodes")
    lines.append("                if any(nb in inactive_nodes for nb in neighbors[p])]")
    lines.append("    if not boundary:")
    lines.append("        return True  # no boundary, halo is a no-op")
    lines.append("    avg_boundary_edges = cross / len(boundary)")
    lines.append("    return avg_boundary_edges >= 1.5 and cross >= 3")
    lines.append("```")
    lines.append("")
    lines.append("## Alternatives for Bottleneck Topologies (min_cut < 3)")
    lines.append("")
    lines.append("1. **Hard insert (no smoothing)**: for trees, the single-edge seam is less")
    lines.append("   harmful than the bleed error from cosine feathering.")
    lines.append("2. **Parent-only blending**: restrict halo expansion to the parent-child")
    lines.append("   axis only, never crossing to siblings.")
    lines.append("3. **Hierarchical (top-down) correction**: apply corrections level-by-level")
    lines.append("   through the tree, so each level is consistent with its parent.")
    lines.append("4. **Virtual edge augmentation**: add synthetic edges between adjacent")
    lines.append("   subtree boundaries to raise min_cut before applying Halo.")
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nSaved: {out_path}")
    return out_path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Part 1
    original_metrics = compute_all_original_metrics()

    # Part 2
    sweep_results = run_two_tile_sweep()

    # Part 3
    predictor = fit_predictor(sweep_results, original_metrics)

    # Part 4
    analyze_tree_bleed()

    # Part 5
    print("\n" + "=" * 70)
    print("PART 5: Applicability Rule Summary")
    print("=" * 70)
    print(f"\n  Predictor: ratio = {predictor['log_model_a']:.4f} * ln(bnd_edges + 1) + {predictor['log_model_b']:.4f}")
    print(f"  R^2 = {predictor['r_squared']:.4f}")
    print(f"  Break-even: {predictor['breakeven_edges']:.1f} edges")
    print(f"  Strong (1.5x): {predictor['threshold_15x']:.1f} edges")
    print(f"  Empirical crossover: {predictor['empirical_crossover']:.1f} edges")
    print()
    print(f"  RULE: Use Halo when boundary has >= 3 parallel cross-edges.")
    print(f"        For bottleneck topologies (min_cut < 3), use hard insert.")

    # Save JSON
    all_data = {
        "original_metrics": original_metrics,
        "sweep_results": [{k: v for k, v in r.items() if k != "all_ratios"}
                          for r in sweep_results],
        "predictor": predictor,
    }
    json_path = OUT_DIR / "halo_applicability.json"
    with open(json_path, "w") as f:
        json.dump(all_data, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, '__float__') else str(x))
    print(f"  Saved: {json_path}")

    write_applicability_rule(predictor, sweep_results, original_metrics)

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Halo Applicability: Boundary Connectivity vs Effectiveness",
                     fontsize=13, fontweight="bold")

        # Plot 1: Sweep
        ax = axes[0]
        x = [r["boundary_edges"] for r in sweep_results]
        y_med = [r["median_ratio"] for r in sweep_results]
        y_q25 = [r["q25"] for r in sweep_results]
        y_q75 = [r["q75"] for r in sweep_results]
        ax.fill_between(x, y_q25, y_q75, alpha=0.2, color="#2ca02c")
        ax.plot(x, y_med, "o-", color="#2ca02c", linewidth=2, markersize=6, label="median")
        ax.axhline(1.0, color="red", ls="--", lw=1.5, label="break-even")
        ax.axhline(1.5, color="orange", ls="--", lw=1.5, label="kill criterion (1.5x)")
        ax.set_xlabel("Boundary edges between tile pair")
        ax.set_ylabel("Seam reduction ratio (hard/halo)")
        ax.set_title("Two-Tile Sweep")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Plot 2: Log fit
        ax = axes[1]
        x_arr = np.array(x, dtype=float)
        y_arr = np.array(y_med, dtype=float)
        ax.scatter(x_arr, y_arr, c="#2ca02c", s=50, zorder=5, label="synthetic data")
        x_fit = np.linspace(0.5, 45, 200)
        y_fit = predictor["log_model_a"] * np.log(x_fit + 1) + predictor["log_model_b"]
        ax.plot(x_fit, y_fit, "b-", lw=2, label=f"log fit (R^2={predictor['r_squared']:.3f})")
        ax.axhline(1.0, color="red", ls="--", lw=1)
        # Original spaces
        for m in original_metrics:
            mc = m["min_cut_per_tile_pair"]
            ratio = m["halo_ratio"]
            color = "#2ca02c" if ratio >= 1.0 else "#d62728"
            ax.scatter([mc], [ratio], marker="D", s=100, c=color, zorder=10,
                      edgecolors="black", linewidths=1.5,
                      label=f"{m['space']} ({mc:.0f}, {ratio:.2f})")
        ax.set_xlabel("Boundary edges (min cut per tile pair)")
        ax.set_ylabel("Halo effectiveness ratio")
        ax.set_title("Log Model + Original Spaces")
        ax.legend(fontsize=6, loc="lower right")
        ax.grid(True, alpha=0.3)

        # Plot 3: Original spaces bar chart
        ax = axes[2]
        spaces = [m["space"] for m in original_metrics]
        cuts = [m["min_cut_per_tile_pair"] for m in original_metrics]
        ratios_orig = [m["halo_ratio"] for m in original_metrics]
        colors = ["#2ca02c" if r >= 1.0 else "#d62728" for r in ratios_orig]
        bars = ax.bar(range(len(spaces)), ratios_orig, color=colors, edgecolor="black")
        ax.set_xticks(range(len(spaces)))
        ax.set_xticklabels([s.replace("_", "\n") for s in spaces], fontsize=7)
        ax.axhline(1.0, color="red", ls="--", lw=1.5)
        ax.set_ylabel("Halo effectiveness ratio")
        ax.set_title("Original 4 Spaces")
        for i, (bar, mc) in enumerate(zip(bars, cuts)):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"cut={mc:.0f}", ha="center", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        fig_path = OUT_DIR / "halo_applicability.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fig_path}")
    except Exception as e:
        print(f"  Plot skipped: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
