#!/usr/bin/env python3
"""
Three-zone topological classifier validation on full corpus.

All topologies from all previous tests + new variations + scaled versions.
Tests the k_mean + Gini(PageRank) classifier accuracy.
"""

import sys
import json
import numpy as np
import networkx as nx
from pathlib import Path
from scipy.spatial import Delaunay, cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from topo_features import (extract_topo_features, calibrate_transport_probe,
                           reset_calibration)

import leidenalg
import igraph as ig


# ===================================================================
# Helpers
# ===================================================================

def _connect(G):
    comps = list(nx.connected_components(G))
    for k in range(1, len(comps)):
        G.add_edge(min(comps[0]), min(comps[k]))
    return G

def cluster_leiden(G, seed=42):
    node_list = sorted(G.nodes())
    node_map = {nd: i for i, nd in enumerate(node_list)}
    n = len(node_list)
    edges = [(node_map[u], node_map[v]) for u, v in G.edges()]
    G_ig = ig.Graph(n=n, edges=edges, directed=False)
    part = leidenalg.find_partition(
        G_ig, leidenalg.ModularityVertexPartition, seed=seed)
    labels = np.array(part.membership)
    return labels, node_list, node_map

def ecr(G, labels, node_map):
    total = G.number_of_edges()
    cut = sum(1 for u, v in G.edges()
              if labels[node_map[u]] != labels[node_map[v]])
    return cut / max(total, 1)

def gini(arr):
    a = np.sort(arr.ravel())
    n = len(a)
    if n == 0 or a.sum() == 0: return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * a).sum() / (n * a.sum())) - (n + 1) / n)

def forman_std(G):
    """Compute std of Forman-Ricci curvature across all edges. O(E)."""
    vals = []
    for u, v in G.edges():
        d_u = G.degree(u)
        d_v = G.degree(v)
        nu = set(G.neighbors(u))
        nv = set(G.neighbors(v))
        n_tri = len(nu & nv)
        F = 4.0 - d_u - d_v + 3.0 * n_tri
        vals.append(F)
    if not vals:
        return 0.0
    return float(np.std(vals))


# ===================================================================
# Graph builders: full corpus
# ===================================================================

def make_corpus():
    """All topologies, multiple scales, parameter variations."""
    corpus = []

    # --- Original 9 from combined test ---
    corpus.append(("Barbell_30", nx.barbell_graph(30, 5)))
    corpus.append(("Hub-Spoke_60", nx.star_graph(60)))
    corpus.append(("Ring_8x12", nx.ring_of_cliques(8, 12)))

    rng = np.random.default_rng(42)
    G = nx.Graph(); G.add_nodes_from(range(80))
    for i in range(40):
        for j in range(40, 80):
            if rng.random() < 0.15: G.add_edge(i, j)
    corpus.append(("Bipartite_40x40", _connect(G)))

    corpus.append(("ER_100_p006", _connect(nx.erdos_renyi_graph(100, 0.06, seed=42))))
    corpus.append(("Grid_10x10", nx.grid_2d_graph(10, 10)))

    rng2 = np.random.default_rng(42); n_sr = 500
    t = 1.5 * np.pi * (1 + 2 * rng2.random(n_sr))
    x = t * np.cos(t) + rng2.standard_normal(n_sr) * 0.3
    y = t * np.sin(t) + rng2.standard_normal(n_sr) * 0.3
    tree = cKDTree(np.column_stack([x, y]))
    _, idx = tree.query(np.column_stack([x, y]), k=7)
    G_sr = nx.Graph(); G_sr.add_nodes_from(range(n_sr))
    for i in range(n_sr):
        for j in idx[i, 1:]: G_sr.add_edge(i, j)
    corpus.append(("Swiss_Roll_500", G_sr))

    rng3 = np.random.default_rng(42); pts = rng3.random((80, 2))
    tri = Delaunay(pts); G_pl = nx.Graph(); G_pl.add_nodes_from(range(80))
    for s in tri.simplices:
        for i in range(3):
            for j in range(i+1, 3): G_pl.add_edge(s[i], s[j])
    corpus.append(("Planar_80", G_pl))

    G_mob = nx.Graph(); na, nw = 40, 3
    for i in range(na):
        for j in range(nw): G_mob.add_node((i, j))
    for i in range(na-1):
        for j in range(nw): G_mob.add_edge((i, j), (i+1, j))
    for i in range(na):
        for j in range(nw-1): G_mob.add_edge((i, j), (i, j+1))
    for j in range(nw): G_mob.add_edge((na-1, j), (0, nw-1-j))
    corpus.append(("Mobius_40x3", G_mob))

    # --- Scale variations ---
    corpus.append(("Grid_5x5", nx.grid_2d_graph(5, 5)))
    corpus.append(("Grid_20x20", nx.grid_2d_graph(20, 20)))

    corpus.append(("Barbell_10", nx.barbell_graph(10, 3)))
    corpus.append(("Barbell_50", nx.barbell_graph(50, 8)))

    corpus.append(("Ring_4x8", nx.ring_of_cliques(4, 8)))
    corpus.append(("Ring_12x20", nx.ring_of_cliques(12, 20)))

    corpus.append(("Star_20", nx.star_graph(20)))
    corpus.append(("Star_120", nx.star_graph(120)))

    # --- ER variations ---
    corpus.append(("ER_100_p015", _connect(nx.erdos_renyi_graph(100, 0.15, seed=42))))
    corpus.append(("ER_200_p004", _connect(nx.erdos_renyi_graph(200, 0.04, seed=42))))
    corpus.append(("ER_50_p010", _connect(nx.erdos_renyi_graph(50, 0.10, seed=42))))

    # --- Bipartite variations ---
    rng4 = np.random.default_rng(42)
    G_bp2 = nx.Graph(); G_bp2.add_nodes_from(range(60))
    for i in range(30):
        for j in range(30, 60):
            if rng4.random() < 0.10: G_bp2.add_edge(i, j)
    corpus.append(("Bipartite_30x30_p01", _connect(G_bp2)))

    rng5 = np.random.default_rng(42)
    G_bp3 = nx.Graph(); G_bp3.add_nodes_from(range(100))
    for i in range(50):
        for j in range(50, 100):
            if rng5.random() < 0.25: G_bp3.add_edge(i, j)
    corpus.append(("Bipartite_50x50_p025", _connect(G_bp3)))

    # --- Special topologies ---
    corpus.append(("Petersen", nx.petersen_graph()))
    corpus.append(("Karate_Club", nx.karate_club_graph()))
    corpus.append(("Wheel_50", nx.wheel_graph(50)))
    corpus.append(("Ladder_40", nx.ladder_graph(40)))
    corpus.append(("Cycle_100", nx.cycle_graph(100)))
    corpus.append(("Complete_20", nx.complete_graph(20)))
    corpus.append(("Tree_bin_d7", nx.balanced_tree(2, 7)))
    corpus.append(("Barabasi_200", nx.barabasi_albert_graph(200, 3, seed=42)))
    corpus.append(("Watts_200", nx.watts_strogatz_graph(200, 6, 0.1, seed=42)))
    corpus.append(("Caveman_8x15", nx.connected_caveman_graph(8, 15)))

    # --- Planar variations ---
    for n_pts, label in [(40, "Planar_40"), (150, "Planar_150")]:
        rng_p = np.random.default_rng(42)
        pts_p = rng_p.random((n_pts, 2))
        tri_p = Delaunay(pts_p)
        G_p = nx.Graph(); G_p.add_nodes_from(range(n_pts))
        for s in tri_p.simplices:
            for i in range(3):
                for j in range(i+1, 3): G_p.add_edge(s[i], s[j])
        corpus.append((label, G_p))

    # --- Swiss roll variations ---
    for n_sr2, noise, label in [(200, 0.5, "Swiss_200_noisy"),
                                 (1000, 0.1, "Swiss_1000_tight")]:
        rng_s = np.random.default_rng(42)
        t2 = 1.5 * np.pi * (1 + 2 * rng_s.random(n_sr2))
        x2 = t2 * np.cos(t2) + rng_s.standard_normal(n_sr2) * noise
        y2 = t2 * np.sin(t2) + rng_s.standard_normal(n_sr2) * noise
        tree2 = cKDTree(np.column_stack([x2, y2]))
        _, idx2 = tree2.query(np.column_stack([x2, y2]), k=7)
        G_s2 = nx.Graph(); G_s2.add_nodes_from(range(n_sr2))
        for i in range(n_sr2):
            for j in idx2[i, 1:]: G_s2.add_edge(i, j)
        corpus.append((label, G_s2))

    return corpus


# ===================================================================
# Classifier
# ===================================================================

GINI_THRESHOLD = 0.12
SIGMA_F_THRESHOLD = 1.5
ETA_F_THRESHOLD = 0.70  # dimensionless: sigma_F / sqrt(2 * mean_degree)

def classify_v1(k_mean, gini_pr):
    """Original two-signal classifier."""
    if k_mean > 0:
        return "GREEN", "ECR < 5%"
    elif gini_pr < GINI_THRESHOLD:
        return "YELLOW", "ECR 10-25%"
    else:
        return "RED", "ECR>30%/refuse"

def classify_v2(k_mean, gini_pr, sigma_f):
    """Three-signal classifier with Forman variance (absolute threshold)."""
    if k_mean > 0:
        return "GREEN", "ECR < 5%"
    elif gini_pr < GINI_THRESHOLD and sigma_f <= SIGMA_F_THRESHOLD:
        return "YELLOW", "ECR 10-25%"
    else:
        return "RED", "ECR>30%/refuse"

def classify_v3(k_mean, gini_pr, eta_f):
    """Physics-based classifier: eta_F = sigma_F / sqrt(2*mean_degree)."""
    if k_mean > 0:
        return "GREEN", "ECR < 5%"
    elif gini_pr < GINI_THRESHOLD and eta_f <= ETA_F_THRESHOLD:
        return "YELLOW", "ECR 10-25%"
    else:
        return "RED", "ECR>30%/refuse"

def check_prediction(zone, actual_ecr, n_clusters, n_nodes):
    if zone == "GREEN":
        return actual_ecr < 0.05
    elif zone == "YELLOW":
        return 0.05 <= actual_ecr <= 0.30
    else:  # RED
        return actual_ecr > 0.30 or n_clusters <= 1


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 130)
    print("Three-Zone Topological Classifier -- Full Corpus Validation")
    print("=" * 130)

    reset_calibration()
    cal = calibrate_transport_probe()
    print(f"\n  Calibration: kappa_max={cal.kappa_max:.0f}, "
          f"t_ollivier={cal.t_ollivier_ms:.2f}ms\n")

    corpus = make_corpus()
    print(f"  Corpus size: {len(corpus)} graphs\n")

    results = []
    v1_correct = 0
    v2_correct = 0
    v3_correct = 0
    v1_misses = []
    v2_misses = []
    v3_misses = []

    print(f"  {'Graph':<24s} {'N':>5s} {'E':>6s} {'Cl':>3s} "
          f"{'ECR':>6s} {'k_mean':>7s} {'Gini':>6s} {'sF':>5s} {'etaF':>5s} "
          f"{'v1':>7s} {'v1?':>4s} {'v3':>7s} {'v3?':>4s}")
    print("  " + "-" * 120)

    for name, G in corpus:
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()

        labels, node_list, node_map = cluster_leiden(G)
        n_cl = len(set(labels))
        actual_ecr = ecr(G, labels, node_map)

        topo = extract_topo_features(G, labels, node_list,
                                     kappa_max=cal.kappa_max,
                                     topo_budget_ms=50.0)
        k_mean = float(topo.curvature.mean())

        pr_dict = nx.pagerank(G, alpha=0.85)
        pr_arr = np.array([pr_dict.get(nd, 0.0) for nd in node_list])
        gini_pr = gini(pr_arr)

        sig_f = forman_std(G)
        mean_deg = 2.0 * n_edges / max(n_nodes, 1)
        eta_f = sig_f / np.sqrt(2.0 * mean_deg) if mean_deg > 0 else 0.0

        z1, p1 = classify_v1(k_mean, gini_pr)
        z2, p2 = classify_v2(k_mean, gini_pr, sig_f)
        z3, p3 = classify_v3(k_mean, gini_pr, eta_f)
        ok1 = check_prediction(z1, actual_ecr, n_cl, n_nodes)
        ok2 = check_prediction(z2, actual_ecr, n_cl, n_nodes)
        ok3 = check_prediction(z3, actual_ecr, n_cl, n_nodes)
        v1_correct += ok1
        v2_correct += ok2
        v3_correct += ok3
        if not ok1:
            v1_misses.append((name, k_mean, gini_pr, sig_f, eta_f, actual_ecr, z1, n_cl))
        if not ok2:
            v2_misses.append((name, k_mean, gini_pr, sig_f, eta_f, actual_ecr, z2, n_cl))
        if not ok3:
            v3_misses.append((name, k_mean, gini_pr, sig_f, eta_f, actual_ecr, z3, n_cl))

        m1 = "OK" if ok1 else "MISS"
        m3 = "OK" if ok3 else "MISS"

        results.append({
            "name": name, "N": n_nodes, "E": n_edges,
            "Clusters": n_cl, "ECR": actual_ecr,
            "k_mean": k_mean, "Gini": gini_pr,
            "sigma_F": sig_f, "eta_F": eta_f,
            "zone_v1": z1, "zone_v2": z2, "zone_v3": z3,
            "match_v1": ok1, "match_v2": ok2, "match_v3": ok3,
        })

        # Highlight where v1 != v3
        diff = " <<" if z1 != z3 else ""
        print(f"  {name:<24s} {n_nodes:>5d} {n_edges:>6d} {n_cl:>3d} "
              f"{actual_ecr:>5.1%} {k_mean:>+7.3f} {gini_pr:>6.3f} {sig_f:>5.1f} {eta_f:>5.2f} "
              f"{z1:>7s} {m1:>4s} {z3:>7s} {m3:>4s}{diff}")

    # Summary
    total = len(corpus)
    print("\n" + "=" * 130)
    print(f"  v1 (k + Gini):               {v1_correct}/{total} ({v1_correct/total:.0%})")
    print(f"  v2 (k + Gini + sigma_F):     {v2_correct}/{total} ({v2_correct/total:.0%})")
    print(f"  v3 (k + Gini + eta_F=0.70):  {v3_correct}/{total} ({v3_correct/total:.0%})")

    if v1_misses:
        print(f"\n  v1 MISSES ({len(v1_misses)}):")
        for name, k, g, sf, ef, e, z, cl in v1_misses:
            print(f"    {name}: k={k:+.3f} gini={g:.3f} sF={sf:.1f} etaF={ef:.2f} ECR={e:.1%} zone={z} cl={cl}")

    if v3_misses:
        print(f"\n  v3 MISSES ({len(v3_misses)}):")
        for name, k, g, sf, ef, e, z, cl in v3_misses:
            print(f"    {name}: k={k:+.3f} gini={g:.3f} sF={sf:.1f} etaF={ef:.2f} ECR={e:.1%} zone={z} cl={cl}")

    # Divergences v1 vs v3
    diverged = [(r["name"], r["zone_v1"], r["zone_v3"], r["eta_F"], r["ECR"])
                for r in results if r["zone_v1"] != r["zone_v3"]]
    if diverged:
        print(f"\n  v1/v3 DIVERGENCES ({len(diverged)}):")
        for name, z1, z3, ef, e in diverged:
            print(f"    {name}: v1={z1} -> v3={z3} (etaF={ef:.2f}, ECR={e:.1%})")

    # Zone distribution for v3
    print("\n  v3 zone distribution:")
    for z in ["GREEN", "YELLOW", "RED"]:
        z_results = [r for r in results if r["zone_v3"] == z]
        z_ecr = [r["ECR"] for r in z_results]
        if z_ecr:
            print(f"    {z:>7s} ({len(z_ecr):>2d} graphs): "
                  f"ECR {min(z_ecr):.1%} - {max(z_ecr):.1%}, "
                  f"mean {np.mean(z_ecr):.1%}")

    print("=" * 130)

    # Save
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "zone_classifier_full.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
