#!/usr/bin/env python3
"""
exp15 — P3a: LCA-distance vs Feature Similarity.

For each space type, runs the pipeline, extracts the refinement tree
(real or virtual), and measures correlation between:
  - LCA-distance: tree distance through Lowest Common Ancestor
  - Feature distance: ||feature_i - feature_j|| in refined state

Virtual hierarchy construction:
  - tree_hierarchy: direct binary tree (parent = (i-1)//2)
  - scalar_grid / vector_grid: quadtree from tile → super-tile → root
  - irregular_graph: community hierarchy (Leiden levels)

Kill criteria: Spearman r > 0.3 → tree is semantic (not just a journal).

4 spaces × 20 seeds = 80 runs. 1000 random pairs per run.

Usage:
  python exp15_lca_distance.py                 # run all
  python exp15_lca_distance.py --chunk 0 2     # run chunk 0 of 2
"""

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from scipy import stats as sp_stats

# --- path setup ---
EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENTS_DIR / "exp_phase2_pipeline"))
sys.path.insert(0, str(EXPERIMENTS_DIR / "exp10d_seed_determinism"))

from config import PipelineConfig
from pipeline import CuriosityPipeline
from space_registry import SPACE_FACTORIES

# ===================================================================
# Constants
# ===================================================================

SPACE_TYPES = ["scalar_grid", "vector_grid", "irregular_graph", "tree_hierarchy"]
N_SEEDS = 20
N_PAIRS = 1000  # random pairs per run
BUDGET = 0.30

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ===================================================================
# LCA computation for different space types
# ===================================================================

def lca_tree_hierarchy(u: int, v: int) -> Tuple[int, int]:
    """LCA in binary tree. Returns (lca_node, distance).

    Distance = depth(u) + depth(v) - 2*depth(lca).
    """
    def _depth(x):
        return int(np.log2(x + 1))

    a, b = u, v
    da, db = _depth(a), _depth(b)

    # Bring to same depth
    while da > db:
        a = (a - 1) // 2
        da -= 1
    while db > da:
        b = (b - 1) // 2
        db -= 1

    # Walk up until they meet
    while a != b:
        a = (a - 1) // 2
        b = (b - 1) // 2

    lca_node = a
    dist = _depth(u) + _depth(v) - 2 * _depth(lca_node)
    return lca_node, dist


def build_grid_quadtree(NT: int) -> Dict[str, int]:
    """Build virtual quadtree for grid tiles.

    Each tile (i,j) gets a path through super-tiles.
    LCA distance = number of levels until paths converge.
    Returns mapping: "(i,j)" -> morton_path_int for hierarchical distance.
    """
    # Use Morton code as a proxy for quadtree hierarchy.
    # LCA distance ≈ number of leading bits that differ in Morton codes.
    paths = {}
    for i in range(NT):
        for j in range(NT):
            # Simple Morton-like encoding for hierarchy
            path = 0
            for bit in range(8):  # up to 8 levels
                path |= ((i >> bit) & 1) << (2 * bit + 1)
                path |= ((j >> bit) & 1) << (2 * bit)
            paths[f"({i}, {j})"] = path
    return paths


def lca_distance_grid(u_str: str, v_str: str, morton_paths: Dict[str, int]) -> int:
    """LCA distance for grid tiles via Morton codes.

    Distance = index of highest differing bit pair (quadtree level).
    """
    mu = morton_paths.get(u_str, 0)
    mv = morton_paths.get(v_str, 0)
    xor = mu ^ mv
    if xor == 0:
        return 0
    # Count levels: each 2-bit group is one quadtree level
    level = 0
    while xor > 0:
        level += 1
        xor >>= 2
    return level


def build_graph_hierarchy(space) -> Dict[int, List[int]]:
    """Build community hierarchy for graph space.

    Returns parent map: cluster_id -> parent_cluster_id at coarser level.
    Uses Leiden community labels as the base level, then aggregates
    into super-clusters by spatial proximity.
    """
    n_clusters = space.n_clusters
    labels = space.labels
    pos = space.pos

    # Compute cluster centroids
    centroids = np.zeros((n_clusters, 2))
    for c in range(n_clusters):
        pts = np.where(labels == c)[0]
        if len(pts) > 0:
            centroids[c] = pos[pts].mean(axis=0)

    # Build 2-level hierarchy: clusters -> super-clusters (quadrant-based)
    # Level 0: individual clusters
    # Level 1: quadrant groups (NW, NE, SW, SE)
    parent = {}
    for c in range(n_clusters):
        cx, cy = centroids[c]
        # Assign to quadrant
        quadrant = (0 if cx < 0.5 else 1) + (0 if cy < 0.5 else 2)
        parent[c] = n_clusters + quadrant  # super-cluster IDs start at n_clusters

    # Level 2: root
    for q in range(4):
        parent[n_clusters + q] = n_clusters + 4  # root

    return parent


def lca_distance_graph(u: int, v: int, parent: Dict[int, int]) -> int:
    """LCA distance in graph community hierarchy."""
    # Walk both up to root, find meeting point
    path_u = [u]
    p = u
    while p in parent:
        p = parent[p]
        path_u.append(p)

    path_v = [v]
    p = v
    while p in parent:
        p = parent[p]
        path_v.append(p)

    # Find LCA
    set_v = set(path_v)
    lca = None
    lca_depth_u = 0
    for i, node in enumerate(path_u):
        if node in set_v:
            lca = node
            lca_depth_u = i
            break

    if lca is None:
        return len(path_u) + len(path_v)  # disconnected

    lca_depth_v = path_v.index(lca)
    return lca_depth_u + lca_depth_v


# ===================================================================
# Feature distance extraction
# ===================================================================

def extract_leaf_features(space, state: np.ndarray, units: list) -> Dict[str, np.ndarray]:
    """Extract feature vectors for each unit from the refined state."""
    features = {}
    for u in units:
        if hasattr(u, '__len__'):
            # Grid
            ti, tj = u
            T = space.T
            s = slice(ti * T, (ti + 1) * T)
            cs = slice(tj * T, (tj + 1) * T)
            if state.ndim == 3:
                feat = state[s, cs, :].ravel()
            else:
                feat = state[s, cs].ravel()
        elif isinstance(u, int):
            if hasattr(space, 'labels'):
                # Graph: cluster mean feature
                pts = np.where(space.labels == u)[0]
                feat = state[pts] if len(pts) > 0 else np.array([0.0])
            else:
                # Tree: subtree feature
                feat = np.array([state[u]])
        else:
            feat = np.array([0.0])
        features[str(u)] = feat
    return features


# ===================================================================
# Single run
# ===================================================================

@dataclass
class LCARunResult:
    space_type: str
    seed: int
    n_units: int
    n_pairs: int
    spearman_r: float
    spearman_p: float
    pearson_r: float
    pearson_p: float
    per_depth_spearman: Dict[int, float]
    mean_lca_dist: float
    mean_feat_dist: float
    wall_seconds: float


def run_lca_analysis(space_type: str, seed: int) -> LCARunResult:
    """Run one LCA analysis."""
    t0 = time.time()

    cfg = PipelineConfig(
        budget_fraction=BUDGET,
        enox_journal_enabled=True,
        enox_include_uri_map=True,
    )
    pipe = CuriosityPipeline(config=cfg)
    result = pipe.run(space_type, seed=seed, budget_fraction=BUDGET)

    # Get space adapter for feature extraction
    space = SPACE_FACTORIES[space_type]()
    space.setup(seed)
    units = space.get_units()
    n_units = len(units)

    # Extract features
    features = extract_leaf_features(space, result.final_state, units)
    unit_strs = [str(u) for u in units]

    # Build LCA infrastructure
    if space_type == "tree_hierarchy":
        pass  # use lca_tree_hierarchy directly
    elif space_type in ("scalar_grid", "vector_grid"):
        morton_paths = build_grid_quadtree(space.NT)
    elif space_type == "irregular_graph":
        graph_parent = build_graph_hierarchy(space)

    # Sample random pairs
    rng = np.random.default_rng(seed + 999)
    n_pairs = min(N_PAIRS, n_units * (n_units - 1) // 2)
    pair_indices = set()
    while len(pair_indices) < n_pairs:
        i, j = rng.integers(0, n_units, size=2)
        if i != j:
            pair_indices.add((min(i, j), max(i, j)))

    # Compute distances
    lca_dists = []
    feat_dists = []
    depth_pairs = {}  # depth -> [(lca_d, feat_d), ...]

    for i, j in pair_indices:
        u, v = units[i], units[j]
        u_str, v_str = str(u), str(v)

        # LCA distance
        if space_type == "tree_hierarchy":
            _, lca_d = lca_tree_hierarchy(u, v)
        elif space_type in ("scalar_grid", "vector_grid"):
            lca_d = lca_distance_grid(u_str, v_str, morton_paths)
        elif space_type == "irregular_graph":
            lca_d = lca_distance_graph(u, v, graph_parent)
        else:
            lca_d = 0

        # Feature distance
        f_i = features.get(u_str, np.array([0.0]))
        f_j = features.get(v_str, np.array([0.0]))
        # Align lengths
        min_len = min(len(f_i), len(f_j))
        feat_d = float(np.linalg.norm(f_i[:min_len] - f_j[:min_len]))

        lca_dists.append(lca_d)
        feat_dists.append(feat_d)

        # Track per-depth (for tree)
        if space_type == "tree_hierarchy":
            depth = lca_d
            if depth not in depth_pairs:
                depth_pairs[depth] = []
            depth_pairs[depth].append((lca_d, feat_d))

    lca_arr = np.array(lca_dists)
    feat_arr = np.array(feat_dists)

    # Correlation
    if len(lca_arr) > 2 and np.std(lca_arr) > 0 and np.std(feat_arr) > 0:
        sr, sp = sp_stats.spearmanr(lca_arr, feat_arr)
        pr, pp = sp_stats.pearsonr(lca_arr, feat_arr)
    else:
        sr, sp, pr, pp = 0.0, 1.0, 0.0, 1.0

    # Per-depth Spearman
    per_depth = {}
    for depth, pairs in depth_pairs.items():
        if len(pairs) > 5:
            ld = [p[0] for p in pairs]
            fd = [p[1] for p in pairs]
            if np.std(ld) > 0 and np.std(fd) > 0:
                r, _ = sp_stats.spearmanr(ld, fd)
                per_depth[depth] = float(r)

    wall = time.time() - t0

    return LCARunResult(
        space_type=space_type,
        seed=seed,
        n_units=n_units,
        n_pairs=n_pairs,
        spearman_r=float(sr),
        spearman_p=float(sp),
        pearson_r=float(pr),
        pearson_p=float(pp),
        per_depth_spearman=per_depth,
        mean_lca_dist=float(lca_arr.mean()),
        mean_feat_dist=float(feat_arr.mean()),
        wall_seconds=wall,
    )


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="exp15: LCA-distance vs Feature Similarity")
    parser.add_argument("--chunk", nargs=2, type=int, default=None,
                        help="Chunk index and total chunks")
    args = parser.parse_args()

    configs = [(st, 42 + s) for st in SPACE_TYPES for s in range(N_SEEDS)]
    print(f"Total configs: {len(configs)}")

    if args.chunk is not None:
        chunk_idx, n_chunks = args.chunk
        chunk_size = (len(configs) + n_chunks - 1) // n_chunks
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, len(configs))
        configs = configs[start:end]
        print(f"Running chunk {chunk_idx}/{n_chunks}: configs [{start}:{end}]")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for i, (st, seed) in enumerate(configs):
        print(f"  [{i+1}/{len(configs)}] {st} seed={seed}", end=" ", flush=True)
        try:
            r = run_lca_analysis(st, seed)
            results.append(asdict(r))
            print(f"spearman={r.spearman_r:.3f} pearson={r.pearson_r:.3f} "
                  f"({r.wall_seconds:.1f}s)")
        except Exception as e:
            print(f"FAILED: {e}")
            results.append({
                "space_type": st, "seed": seed, "error": str(e),
            })

        # Incremental save
        if (i + 1) % 10 == 0 or i == len(configs) - 1:
            suffix = f"_chunk{args.chunk[0]}" if args.chunk else ""
            out_path = RESULTS_DIR / f"exp15_results{suffix}.json"
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("exp15 Summary — LCA-distance ↔ Feature Similarity")
    print("=" * 60)
    for st in SPACE_TYPES:
        subset = [r for r in results if r.get("space_type") == st and "error" not in r]
        if subset:
            sr_vals = [r["spearman_r"] for r in subset]
            pr_vals = [r["pearson_r"] for r in subset]
            print(f"  {st:20s}: spearman={np.mean(sr_vals):.3f}±{np.std(sr_vals):.3f}  "
                  f"pearson={np.mean(pr_vals):.3f}±{np.std(pr_vals):.3f}  "
                  f"PASS={np.mean(sr_vals) > 0.3}")
    print("=" * 60)
    overall = [r["spearman_r"] for r in results if "error" not in r]
    if overall:
        print(f"  Overall: spearman_mean={np.mean(overall):.3f}  "
              f"Kill criteria (r>0.3): {'PASS' if np.mean(overall) > 0.3 else 'FAIL'}")


if __name__ == "__main__":
    main()
