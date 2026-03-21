#!/usr/bin/env python3
"""
Pre-runtime profiling benchmark.

Measures wall-clock cost of the entire pre-pipeline preparation:
  1. Hardware calibration (one-time)
  2. Leiden clustering
  3. extract_topo_features (Forman + hybrid Ollivier + PageRank + CC + v3 classifier)

All 35 graphs from the validation corpus.
"""

import sys
import time
import numpy as np
import networkx as nx
from pathlib import Path
from collections import defaultdict
from scipy.spatial import Delaunay, cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from topo_features import (extract_topo_features, calibrate_transport_probe,
                           reset_calibration)

import leidenalg
import igraph as ig


def cluster_leiden(G, seed=42):
    node_list = sorted(G.nodes())
    node_map = {nd: i for i, nd in enumerate(node_list)}
    n = len(node_list)
    edges = [(node_map[u], node_map[v]) for u, v in G.edges()]
    G_ig = ig.Graph(n=n, edges=edges, directed=False)
    part = leidenalg.find_partition(
        G_ig, leidenalg.ModularityVertexPartition, seed=seed)
    return np.array(part.membership), node_list


def _connect(G):
    comps = list(nx.connected_components(G))
    for k in range(1, len(comps)):
        G.add_edge(min(comps[0]), min(comps[k]))
    return G


def make_corpus():
    corpus = []

    # Original 9
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

    # Scale variations
    corpus.append(("Grid_5x5", nx.grid_2d_graph(5, 5)))
    corpus.append(("Grid_20x20", nx.grid_2d_graph(20, 20)))
    corpus.append(("Barbell_10", nx.barbell_graph(10, 3)))
    corpus.append(("Barbell_50", nx.barbell_graph(50, 8)))
    corpus.append(("Ring_4x8", nx.ring_of_cliques(4, 8)))
    corpus.append(("Ring_12x20", nx.ring_of_cliques(12, 20)))
    corpus.append(("Star_20", nx.star_graph(20)))
    corpus.append(("Star_120", nx.star_graph(120)))

    # ER variations
    corpus.append(("ER_100_p015", _connect(nx.erdos_renyi_graph(100, 0.15, seed=42))))
    corpus.append(("ER_200_p004", _connect(nx.erdos_renyi_graph(200, 0.04, seed=42))))
    corpus.append(("ER_50_p010", _connect(nx.erdos_renyi_graph(50, 0.10, seed=42))))

    # Bipartite variations
    rng4 = np.random.default_rng(42)
    G_bp2 = nx.Graph(); G_bp2.add_nodes_from(range(60))
    for i in range(30):
        for j in range(30, 60):
            if rng4.random() < 0.10: G_bp2.add_edge(i, j)
    corpus.append(("Bipartite_30x30", _connect(G_bp2)))

    rng5 = np.random.default_rng(42)
    G_bp3 = nx.Graph(); G_bp3.add_nodes_from(range(100))
    for i in range(50):
        for j in range(50, 100):
            if rng5.random() < 0.25: G_bp3.add_edge(i, j)
    corpus.append(("Bipartite_50x50", _connect(G_bp3)))

    # Special topologies
    corpus.append(("Karate_Club", nx.karate_club_graph()))
    corpus.append(("Wheel_50", nx.wheel_graph(50)))
    corpus.append(("Ladder_40", nx.ladder_graph(40)))
    corpus.append(("Cycle_100", nx.cycle_graph(100)))
    corpus.append(("Complete_20", nx.complete_graph(20)))
    corpus.append(("Tree_bin_d7", nx.balanced_tree(2, 7)))
    corpus.append(("Barabasi_200", nx.barabasi_albert_graph(200, 3, seed=42)))
    corpus.append(("Watts_200", nx.watts_strogatz_graph(200, 6, 0.1, seed=42)))
    corpus.append(("Caveman_8x15", nx.connected_caveman_graph(8, 15)))

    # Planar variations
    for n_pts, label in [(40, "Planar_40"), (150, "Planar_150")]:
        rng_p = np.random.default_rng(42); pts_p = rng_p.random((n_pts, 2))
        tri_p = Delaunay(pts_p); G_p = nx.Graph(); G_p.add_nodes_from(range(n_pts))
        for s in tri_p.simplices:
            for i in range(3):
                for j in range(i+1, 3): G_p.add_edge(s[i], s[j])
        corpus.append((label, G_p))

    # Swiss roll variations
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


def main():
    print("=" * 120)
    print("Pre-Runtime Profiling Benchmark: Calibration + Leiden + Topo Features + Zone Classifier")
    print("=" * 120)

    # Phase 0: Hardware calibration (one-time)
    reset_calibration()
    t_cal_start = time.perf_counter()
    cal = calibrate_transport_probe()
    t_cal = (time.perf_counter() - t_cal_start) * 1000

    print(f"\n  Phase 0 — Calibration (one-time): {t_cal:.0f}ms")
    print(f"    kappa_max={cal.kappa_max:.0f}, t_ollivier={cal.t_ollivier_ms:.2f}ms\n")

    corpus = make_corpus()

    print(f"  {'Graph':<22s} {'N':>5s} {'E':>6s} "
          f"{'Leiden':>8s} {'Topo':>8s} {'TOTAL':>8s} "
          f"{'zone':>7s} {'eta_F':>6s} {'Cl':>3s}")
    print("  " + "-" * 80)

    total_leiden = 0.0
    total_topo = 0.0
    zone_times = defaultdict(list)
    rows = []

    for name, G in corpus:
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()

        # Phase 1: Leiden
        t0 = time.perf_counter()
        labels, node_list = cluster_leiden(G)
        t_leiden = (time.perf_counter() - t0) * 1000

        # Phase 2: Topo features + zone classifier
        t1 = time.perf_counter()
        topo = extract_topo_features(G, labels, node_list,
                                     kappa_max=cal.kappa_max,
                                     topo_budget_ms=50.0)
        t_topo = (time.perf_counter() - t1) * 1000

        t_total = t_leiden + t_topo
        total_leiden += t_leiden
        total_topo += t_topo
        n_cl = len(set(labels))
        zone_times[topo.topo_zone].append(t_total)

        rows.append((name, n_nodes, n_edges, t_leiden, t_topo, t_total,
                      topo.topo_zone, topo.eta_F, n_cl))

        print(f"  {name:<22s} {n_nodes:>5d} {n_edges:>6d} "
              f"{t_leiden:>7.1f}ms {t_topo:>7.1f}ms {t_total:>7.1f}ms "
              f"{topo.topo_zone:>7s} {topo.eta_F:>6.2f} {n_cl:>3d}")

    # Totals
    n_graphs = len(corpus)
    total_all = total_leiden + total_topo
    print("  " + "-" * 80)
    print(f"  {'SUM (' + str(n_graphs) + ' graphs)':<22s} {'':>5s} {'':>6s} "
          f"{total_leiden:>7.0f}ms {total_topo:>7.0f}ms {total_all:>7.0f}ms")
    print(f"  {'MEAN per graph':<22s} {'':>5s} {'':>6s} "
          f"{total_leiden/n_graphs:>7.1f}ms {total_topo/n_graphs:>7.1f}ms "
          f"{total_all/n_graphs:>7.1f}ms")

    # Percentiles
    all_totals = [r[5] for r in rows]
    arr = np.array(all_totals)
    print(f"\n  Percentiles (Leiden + Topo combined):")
    print(f"    P50 = {np.percentile(arr, 50):>6.1f}ms")
    print(f"    P90 = {np.percentile(arr, 90):>6.1f}ms")
    print(f"    P99 = {np.percentile(arr, 99):>6.1f}ms")
    print(f"    MAX = {arr.max():>6.1f}ms")

    # By zone
    print(f"\n  By zone:")
    for z in ["GREEN", "YELLOW", "RED"]:
        if zone_times[z]:
            za = np.array(zone_times[z])
            print(f"    {z:>7s}: n={len(za):>2d}  "
                  f"mean={za.mean():>6.1f}ms  "
                  f"max={za.max():>6.1f}ms")

    # By scale
    print(f"\n  By scale:")
    for threshold, label in [(50, "N<=50"), (200, "50<N<=200"),
                              (500, "200<N<=500"), (10000, "N>500")]:
        bucket = [r for r in rows if r[1] <= threshold]
        rows = [r for r in rows if r[1] > threshold]  # consume
        if bucket:
            ba = np.array([r[5] for r in bucket])
            print(f"    {label:>12s}: n={len(bucket):>2d}  "
                  f"mean={ba.mean():>6.1f}ms  "
                  f"max={ba.max():>6.1f}ms")

    print(f"\n  Pipeline tick budget: 500ms")
    print(f"  Pre-runtime overhead vs tick: "
          f"{total_all/n_graphs/500*100:.1f}% mean, "
          f"{np.max(all_totals)/500*100:.1f}% worst case")

    print("=" * 120)


if __name__ == "__main__":
    main()
