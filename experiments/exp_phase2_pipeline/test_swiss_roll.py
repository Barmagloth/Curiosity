#!/usr/bin/env python3
"""
Stress test: Swiss Roll manifold — k-means vs spectral clustering.

Swiss Roll is a classic pathological case where Euclidean proximity ≠
manifold connectivity.  Points close in 3D may be far apart on the
unrolled 2D manifold, separated by layers of the roll.

We build a k-NN graph on the Swiss Roll, then compare:
  1. k-means on 3D coordinates (current IrregularGraphSpace approach)
  2. Spectral clustering on the graph Laplacian (topology-aware)

Metrics:
  - Edge cut ratio: fraction of k-NN edges that cross cluster boundaries.
    Low = clusters respect connectivity.  High = clusters shred the graph.
  - D_parent (lf_frac): scale-consistency metric from SC-enforce.
    Tests whether R/Up operators built from these clusters produce
    sensible coarsening.
  - Quality (PSNR): refinement quality after one pass.
"""

import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENTS_DIR / "sc_baseline"))
sys.path.insert(0, str(EXPERIMENTS_DIR / "exp12a_tau_parent"))
sys.path.insert(0, str(EXPERIMENTS_DIR / "exp14a_sc_enforce"))

from exp12a_tau_parent import GraphOps
from sc_enforce import d_parent_lf_frac


# ===================================================================
# Swiss Roll generator
# ===================================================================

def make_swiss_roll(n_points: int, noise: float = 0.3,
                    seed: int = 42) -> tuple:
    """Generate Swiss Roll point cloud in 3D.

    Returns (pos_3d, t) where t is the unrolled 1D parameter.
    """
    rng = np.random.default_rng(seed)
    t = 1.5 * np.pi * (1 + 2 * rng.random(n_points))
    height = 10 * rng.random(n_points)

    x = t * np.cos(t) + rng.standard_normal(n_points) * noise
    y = height
    z = t * np.sin(t) + rng.standard_normal(n_points) * noise

    pos = np.column_stack([x, y, z])
    return pos, t


def build_knn_graph(pos: np.ndarray, k: int = 8) -> dict:
    """Build symmetric k-NN adjacency from positions."""
    from scipy.spatial import cKDTree
    tree = cKDTree(pos)
    _, idx = tree.query(pos, k=k + 1)
    neighbors = {i: set(idx[i, 1:]) for i in range(len(pos))}
    # Symmetrize
    for i in range(len(pos)):
        for j in neighbors[i]:
            neighbors[j].add(i)
    return neighbors


# ===================================================================
# Clustering methods
# ===================================================================

def cluster_kmeans(pos: np.ndarray, n_clusters: int,
                   seed: int = 42) -> np.ndarray:
    """k-means on raw coordinates (current approach)."""
    from scipy.cluster.vq import kmeans2
    _, labels = kmeans2(pos, n_clusters, minit='points', seed=seed)
    return labels


def cluster_louvain(neighbors: dict, n_points: int,
                    seed: int = 42) -> np.ndarray:
    """Louvain community detection on the adjacency graph.

    Maximises modularity using only the adjacency matrix — no coordinates,
    no eigendecomposition.  Near-linear time O(N log N).

    The number of clusters is determined automatically by the algorithm.
    """
    import networkx as nx
    import community.community_louvain as community_louvain

    G = nx.Graph()
    G.add_nodes_from(range(n_points))
    for i, nbrs in neighbors.items():
        for j in nbrs:
            if j > i:
                G.add_edge(i, j)

    partition = community_louvain.best_partition(G, random_state=seed)
    labels = np.array([partition[i] for i in range(n_points)])
    return labels


def cluster_leiden(neighbors: dict, n_points: int,
                   seed: int = 42) -> np.ndarray:
    """Leiden community detection on the adjacency graph.

    Improvement over Louvain: guaranteed to produce well-connected
    communities (no disconnected clusters).  Same O(N log N) complexity.
    """
    import igraph as ig
    import leidenalg

    edges = []
    for i, nbrs in neighbors.items():
        for j in nbrs:
            if j > i:
                edges.append((i, j))

    G = ig.Graph(n=n_points, edges=edges, directed=False)
    partition = leidenalg.find_partition(
        G, leidenalg.ModularityVertexPartition, seed=seed)
    labels = np.array(partition.membership)
    return labels


def cluster_spectral(neighbors: dict, n_points: int, n_clusters: int,
                     seed: int = 42) -> np.ndarray:
    """Spectral clustering on graph Laplacian."""
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import eigsh
    from scipy.cluster.vq import kmeans2

    # Build adjacency matrix
    A = lil_matrix((n_points, n_points))
    for i, nbrs in neighbors.items():
        for j in nbrs:
            A[i, j] = 1.0
    A = A.tocsr()

    # Graph Laplacian: L = D - A
    degrees = np.array(A.sum(axis=1)).ravel()
    D = lil_matrix((n_points, n_points))
    for i in range(n_points):
        D[i, i] = degrees[i]
    L = D.tocsr() - A

    # Normalized Laplacian: D^{-1/2} L D^{-1/2}
    d_inv_sqrt = np.zeros(n_points)
    nz = degrees > 0
    d_inv_sqrt[nz] = 1.0 / np.sqrt(degrees[nz])
    D_inv_sqrt = lil_matrix((n_points, n_points))
    for i in range(n_points):
        D_inv_sqrt[i, i] = d_inv_sqrt[i]
    D_inv_sqrt = D_inv_sqrt.tocsr()
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt

    # Smallest k eigenvectors of L_norm
    eigenvalues, eigenvectors = eigsh(L_norm, k=n_clusters, which='SM')
    # Row-normalize
    norms = np.linalg.norm(eigenvectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    embedding = eigenvectors / norms

    # k-means in spectral space
    _, labels = kmeans2(embedding, n_clusters, minit='points', seed=seed)
    return labels


# ===================================================================
# Metrics
# ===================================================================

def edge_cut_ratio(neighbors: dict, labels: np.ndarray) -> float:
    """Fraction of edges that cross cluster boundaries."""
    total = 0
    cut = 0
    for i, nbrs in neighbors.items():
        for j in nbrs:
            if j > i:  # count each edge once
                total += 1
                if labels[i] != labels[j]:
                    cut += 1
    return cut / max(total, 1)


def cluster_size_stats(labels: np.ndarray, n_clusters: int) -> dict:
    """Cluster size statistics."""
    sizes = [np.sum(labels == c) for c in range(n_clusters)]
    sizes = [s for s in sizes if s > 0]
    return {
        "n_nonempty": len(sizes),
        "min": int(min(sizes)) if sizes else 0,
        "max": int(max(sizes)) if sizes else 0,
        "mean": float(np.mean(sizes)) if sizes else 0,
        "std": float(np.std(sizes)) if sizes else 0,
    }


def compute_quality(pos, gt, labels, n_clusters, neighbors):
    """Simulate coarse + refinement, return PSNR."""
    n = len(gt)
    # Coarse: cluster-mean
    coarse = np.zeros(n)
    for c in range(n_clusters):
        mask = labels == c
        if mask.any():
            coarse[mask] = gt[mask].mean()

    # Refine: one pass, refine clusters in rho order
    state = coarse.copy()
    rho_per_cluster = []
    for c in range(n_clusters):
        mask = labels == c
        pts = np.where(mask)[0]
        if len(pts) == 0:
            rho_per_cluster.append((c, 0.0))
        else:
            rho = float(np.mean((gt[pts] - state[pts]) ** 2))
            rho_per_cluster.append((c, rho))
    rho_per_cluster.sort(key=lambda x: -x[1])

    budget = max(1, n_clusters // 3)  # refine top 1/3
    for c, _ in rho_per_cluster[:budget]:
        pts = np.where(labels == c)[0]
        for p in pts:
            state[p] = gt[p]  # perfect refinement for test clarity
        # 1-hop halo smoothing
        halo = set()
        for p in pts:
            for nb in neighbors.get(p, set()):
                if labels[nb] != c:
                    halo.add(nb)
        for p in halo:
            state[p] = state[p] + (gt[p] - state[p]) * 0.3

    mse = float(np.mean((gt - state) ** 2))
    data_range = float(gt.max() - gt.min())
    psnr = 10 * np.log10(data_range ** 2 / max(mse, 1e-15))
    return float(psnr), mse


def compute_d_parent_stats(gt, labels, n_clusters, n_points):
    """Compute D_parent for synthetic deltas using GraphOps."""
    ops = GraphOps(labels, n_clusters, n_points)
    R_fn = ops.restrict
    Up_fn = lambda xc, tgt: ops.prolong(xc, tgt)

    rng = np.random.default_rng(0)
    d_values = []
    for _ in range(50):
        # Random high-frequency delta (per-node noise)
        delta = rng.standard_normal(n_points) * 0.1
        d = d_parent_lf_frac(delta, R_fn, Up_fn)
        d_values.append(d)

    return {
        "mean": float(np.mean(d_values)),
        "std": float(np.std(d_values)),
        "max": float(np.max(d_values)),
    }


# ===================================================================
# Main
# ===================================================================

def main():
    # Sweep configs: from benign to pathological
    CONFIGS = [
        {"label": "mild (500pt, noise=0.5)",
         "n_points": 500, "k": 8, "n_clusters": 15, "noise": 0.5},
        {"label": "medium (1000pt, noise=0.3)",
         "n_points": 1000, "k": 10, "n_clusters": 20, "noise": 0.3},
        {"label": "tight (2000pt, noise=0.1)",
         "n_points": 2000, "k": 12, "n_clusters": 25, "noise": 0.1},
        {"label": "pathological (3000pt, noise=0.05)",
         "n_points": 3000, "k": 15, "n_clusters": 30, "noise": 0.05},
    ]
    SEEDS = [0, 42, 99]

    print("=" * 75)
    print("Swiss Roll Stress Test: k-means vs Spectral Clustering")
    print("=" * 75)
    print(f"  Configs: {len(CONFIGS)}, Seeds: {SEEDS}")
    print()

    all_results = {}

    for cfg in CONFIGS:
        N_POINTS = cfg["n_points"]
        K = cfg["k"]
        N_CLUSTERS = cfg["n_clusters"]
        noise = cfg["noise"]
        label = cfg["label"]

        print(f"-- {label} --")
        results = {"kmeans": [], "spectral": [], "louvain": [], "leiden": []}

        for seed in SEEDS:
            pos, t = make_swiss_roll(N_POINTS, noise=noise, seed=seed)
            neighbors = build_knn_graph(pos, k=K)

            # GT: smooth function of manifold parameter t
            gt = (np.sin(t / t.max() * 2 * np.pi)
                  + 0.5 * np.cos(t / t.max() * 4 * np.pi))
            gt += np.random.default_rng(seed).standard_normal(N_POINTS) * 0.02

            for method_name, cluster_fn in [
                ("kmeans",
                 lambda: cluster_kmeans(pos, N_CLUSTERS, seed=seed)),
                ("spectral",
                 lambda: cluster_spectral(
                     neighbors, N_POINTS, N_CLUSTERS, seed=seed)),
                ("louvain",
                 lambda: cluster_louvain(
                     neighbors, N_POINTS, seed=seed)),
                ("leiden",
                 lambda: cluster_leiden(
                     neighbors, N_POINTS, seed=seed)),
            ]:
                labels = cluster_fn()
                n_actual = int(labels.max()) + 1
                ecr = edge_cut_ratio(neighbors, labels)
                sizes = cluster_size_stats(labels, n_actual)
                psnr, mse = compute_quality(
                    pos, gt, labels, n_actual, neighbors)
                dp = compute_d_parent_stats(
                    gt, labels, n_actual, N_POINTS)

                results[method_name].append({
                    "seed": seed,
                    "n_clusters_actual": n_actual,
                    "edge_cut_ratio": ecr,
                    "cluster_sizes": sizes,
                    "psnr": psnr,
                    "mse": mse,
                    "d_parent": dp,
                })

            print(".", end="", flush=True)
        print()

        # Summary for this config
        methods = ["kmeans", "spectral", "louvain", "leiden"]
        stats = {}
        for m in methods:
            stats[m] = {
                "ecr": np.mean([r["edge_cut_ratio"] for r in results[m]]),
                "psnr": np.mean([r["psnr"] for r in results[m]]),
                "dp": np.mean([r["d_parent"]["mean"] for r in results[m]]),
                "n_cl": np.mean([r["n_clusters_actual"] for r in results[m]]),
            }

        hdr = ["k-means", "spectral", "louvain", "leiden"]
        print(f"  {'':20s} " + " ".join(f"{h:>12s}" for h in hdr))
        for metric, key, fmt in [
            ("Edge Cut", "ecr", ".1%"),
            ("PSNR (dB)", "psnr", ".2f"),
            ("D_parent mean", "dp", ".4f"),
            ("Clusters", "n_cl", ".0f"),
        ]:
            vals = " ".join(f"{stats[m][key]:>12{fmt}}" for m in methods)
            print(f"  {metric:20s} {vals}")
        print()

        all_results[label] = results

    # ── Final verdict ─────────────────────────────────────────────
    print("=" * 95)
    print("SUMMARY ACROSS ALL CONFIGS")
    print("=" * 95)
    print(f"  {'Config':<30s} {'km ECR':>8s} {'sp ECR':>8s} "
          f"{'lv ECR':>8s} {'ld ECR':>8s}  "
          f"{'km/ld':>6s} {'ld PSNR':>8s} {'ld #cl':>7s}")
    print("  " + "-" * 88)

    for cfg in CONFIGS:
        label = cfg["label"]
        res = all_results[label]
        km_ecr = np.mean([r["edge_cut_ratio"] for r in res["kmeans"]])
        sp_ecr = np.mean([r["edge_cut_ratio"] for r in res["spectral"]])
        lv_ecr = np.mean([r["edge_cut_ratio"] for r in res["louvain"]])
        ld_ecr = np.mean([r["edge_cut_ratio"] for r in res["leiden"]])
        ld_psnr = np.mean([r["psnr"] for r in res["leiden"]])
        ld_ncl = np.mean([r["n_clusters_actual"] for r in res["leiden"]])
        ratio = km_ecr / max(ld_ecr, 1e-9)
        print(f"  {label:<30s} {km_ecr:>7.1%} {sp_ecr:>7.1%} "
              f"{lv_ecr:>7.1%} {ld_ecr:>7.1%}  "
              f"{ratio:>5.2f}x {ld_psnr:>7.2f} {ld_ncl:>7.0f}")

    print("=" * 95)

    # Save
    import json
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _safe(obj):
        if isinstance(obj, dict):
            return {k: _safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_safe(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(out_dir / "swiss_roll_test.json", "w") as f:
        json.dump(_safe(all_results), f, indent=2)
    print(f"\n  Saved: {out_dir / 'swiss_roll_test.json'}")


if __name__ == "__main__":
    main()
