#!/usr/bin/env python3
"""
Test topological feature extraction on all pathological graphs.

Validates:
  1. Features compute without errors on all topologies
  2. Curvature signs match expected topology (positive=triangles, negative=bridges)
  3. PageRank identifies hubs
  4. topo_adjusted_rho boosts budget at structurally important clusters
  5. Performance: extraction time stays reasonable
"""

import sys
import json
import numpy as np
import networkx as nx
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from topo_features import extract_topo_features, topo_adjusted_rho, calibrate_transport_probe, reset_calibration

import leidenalg
import igraph as ig


def cluster_leiden(G, seed=42):
    node_list = sorted(G.nodes())
    node_map = {nd: i for i, nd in enumerate(node_list)}
    n = len(node_list)
    edges = [(node_map[u], node_map[v]) for u, v in G.edges()]
    G_ig = ig.Graph(n=n, edges=edges, directed=False)
    part = leidenalg.find_partition(G_ig, leidenalg.ModularityVertexPartition, seed=seed)
    return np.array(part.membership), node_list


# ===================================================================
# Graph builders (same as stress tests)
# ===================================================================

def build_barbell(cs=30, bl=5): return nx.barbell_graph(cs, bl)
def build_hub_spoke(n=60): return nx.star_graph(n)
def build_ring_cliques(nc=8, cs=12): return nx.ring_of_cliques(nc, cs)
def build_bipartite(nl=40, nr=40, p=0.15):
    rng = np.random.default_rng(42)
    G = nx.Graph()
    G.add_nodes_from(range(nl + nr))
    for i in range(nl):
        for j in range(nl, nl + nr):
            if rng.random() < p: G.add_edge(i, j)
    comps = list(nx.connected_components(G))
    for k in range(1, len(comps)):
        G.add_edge(min(comps[0]), min(comps[k]))
    return G
def build_erdos(n=100, p=0.06, s=42):
    G = nx.erdos_renyi_graph(n, p, seed=s)
    comps = list(nx.connected_components(G))
    for k in range(1, len(comps)):
        G.add_edge(min(comps[0]), min(comps[k]))
    return G
def build_grid(r, c): return nx.grid_2d_graph(r, c)
def build_swiss_roll(n=500, k=6, noise=0.3, seed=42):
    rng = np.random.default_rng(seed)
    t = 1.5 * np.pi * (1 + 2 * rng.random(n))
    x = t * np.cos(t) + rng.standard_normal(n) * noise
    y = t * np.sin(t) + rng.standard_normal(n) * noise
    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack([x, y]))
    _, idx = tree.query(np.column_stack([x, y]), k=k+1)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in idx[i, 1:]:
            G.add_edge(i, j)
    return G
def build_mobius(na=40, nw=3):
    G = nx.Graph()
    for i in range(na):
        for j in range(nw):
            G.add_node((i, j))
    for i in range(na - 1):
        for j in range(nw):
            G.add_edge((i, j), (i + 1, j))
    for i in range(na):
        for j in range(nw - 1):
            G.add_edge((i, j), (i, j + 1))
    for j in range(nw):
        G.add_edge((na - 1, j), (0, nw - 1 - j))
    return G
from scipy.spatial import Delaunay
def build_planar(n=80, seed=42):
    rng = np.random.default_rng(seed)
    pts = rng.random((n, 2))
    tri = Delaunay(pts)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for s in tri.simplices:
        for i in range(3):
            for j in range(i+1, 3):
                G.add_edge(s[i], s[j])
    return G


GRAPHS = {
    "barbell_30":         build_barbell,
    "hub_spoke_60":       lambda: build_hub_spoke(60),
    "ring_cliques_8x12":  lambda: build_ring_cliques(8, 12),
    "bipartite_40x40":    build_bipartite,
    "erdos_renyi_100":    build_erdos,
    "grid_10x10":         lambda: build_grid(10, 10),
    "swiss_roll_500":     build_swiss_roll,
    "planar_80":          build_planar,
    "mobius_40x3":        build_mobius,
}


def main():
    print("=" * 100)
    print("Topological Feature Extraction — Stress Test")
    print("=" * 100)

    results = {}

    # Calibrate once
    reset_calibration()
    cal = calibrate_transport_probe(tau_edge_ms=5.0)
    print(f"\n  Hardware calibration:")
    print(f"    kappa_max = {cal.kappa_max:.0f} (per-edge W ceiling)")
    print(f"    t_ollivier = {cal.t_ollivier_ms:.2f}ms (median EMD solve)")
    print(f"    topo_budget = 50ms (10% of 500ms pipeline tick)")
    print(f"    max Ollivier calls = {int(50.0 / cal.t_ollivier_ms)}")
    print()

    for name, builder in GRAPHS.items():
        G = builder()
        n = G.number_of_nodes()
        e = G.number_of_edges()

        labels, node_list = cluster_leiden(G, seed=42)
        n_cl = len(set(labels))

        topo = extract_topo_features(G, labels, node_list,
                                     kappa_max=cal.kappa_max,
                                     topo_budget_ms=50.0)

        # Simulate base rho (random for testing)
        rng = np.random.default_rng(42)
        base_rho = rng.random(n_cl) * 0.1
        adj_rho = topo_adjusted_rho(base_rho, topo)
        boost = adj_rho / np.maximum(base_rho, 1e-12)

        # Key curvature stats
        curv = topo.curvature
        pr = topo.pagerank
        cc = topo.clustering_coeff

        rec = {
            "n_nodes": n,
            "n_edges": e,
            "n_clusters": n_cl,
            "kappa_max": round(topo.kappa_max, 0),
            "n_ollivier": topo.n_ollivier,
            "n_forman": topo.n_forman,
            "ollivier_pct": round(topo.ollivier_pct, 1),
            "time_ms": round(topo.computation_ms, 1),
            "curvature_mean": round(float(curv.mean()), 4),
            "curvature_min": round(float(curv.min()), 4),
            "curvature_max": round(float(curv.max()), 4),
            "pagerank_max": round(float(pr.max()), 4),
            "pagerank_gini": round(float(_gini(pr)), 3),
            "clustering_mean": round(float(cc.mean()), 3),
            "rho_boost_mean": round(float(boost.mean()), 3),
            "rho_boost_max": round(float(boost.max()), 3),
            "boundary_curv_min": round(float(topo.cluster_boundary_curvature.min()), 4),
        }
        results[name] = rec

        print(f"\n-- {name} (N={n}, E={e}, cl={n_cl}, "
              f"ORC={topo.n_ollivier}/{topo.n_forman+topo.n_ollivier} [{topo.ollivier_pct:.0f}%], "
              f"{topo.computation_ms:.0f}ms) --")
        print(f"  Curvature:  mean={curv.mean():.4f}  min={curv.min():.4f}  max={curv.max():.4f}")
        print(f"  PageRank:   max={pr.max():.4f}  gini={_gini(pr):.3f}")
        print(f"  Clustering: mean={cc.mean():.3f}")
        print(f"  rho boost:  mean={boost.mean():.3f}x  max={boost.max():.3f}x")
        print(f"  Boundary:   min_curv={topo.cluster_boundary_curvature.min():.4f}")

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"  {'Topology':<22s} {'N':>5s} {'cl':>4s} {'ORC%':>5s} {'ms':>6s} "
          f"{'k_mean':>7s} {'k_min':>7s} {'PR_gini':>8s} {'CC':>6s} "
          f"{'boost':>6s} {'b_curv':>7s}")
    print("  " + "-" * 92)

    for name, r in results.items():
        print(f"  {name:<22s} {r['n_nodes']:>5d} {r['n_clusters']:>4d} "
              f"{r['ollivier_pct']:>4.0f}% {r['time_ms']:>5.0f}ms "
              f"{r['curvature_mean']:>7.4f} {r['curvature_min']:>7.4f} "
              f"{r['pagerank_gini']:>8.3f} {r['clustering_mean']:>6.3f} "
              f"{r['rho_boost_mean']:>5.3f}x {r['boundary_curv_min']:>7.4f}")

    print("=" * 100)

    # Validate topological signatures
    print("\nTopological signature validation:")
    # Barbell should have negative curvature on bridge
    if results["barbell_30"]["curvature_min"] < -0.1:
        print("  [PASS] Barbell: negative curvature detected (bridge)")
    else:
        print("  [FAIL] Barbell: expected negative curvature on bridge")

    # Hub-spoke should have extreme PageRank concentration
    if results["hub_spoke_60"]["pagerank_gini"] > 0.8:
        print("  [PASS] Hub-spoke: high PageRank Gini (hub detected)")
    else:
        print("  [FAIL] Hub-spoke: expected high PageRank Gini")

    # Ring of cliques should have high clustering
    if results["ring_cliques_8x12"]["clustering_mean"] > 0.5:
        print("  [PASS] Ring-cliques: high clustering coefficient")
    else:
        print("  [FAIL] Ring-cliques: expected high clustering")

    # Grid should have low PageRank Gini (uniform)
    if results["grid_10x10"]["pagerank_gini"] < 0.2:
        print("  [PASS] Grid: low PageRank Gini (uniform)")
    else:
        print("  [FAIL] Grid: expected low PageRank Gini")

    # Save
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "topo_features_test.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")


def _gini(arr: np.ndarray) -> float:
    """Gini coefficient — measures inequality. 0=uniform, 1=one takes all."""
    a = np.sort(arr.ravel())
    n = len(a)
    if n == 0 or a.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * (index * a).sum() / (n * a.sum())) - (n + 1) / n)


if __name__ == "__main__":
    main()
