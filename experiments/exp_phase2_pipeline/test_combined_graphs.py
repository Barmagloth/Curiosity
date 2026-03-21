#!/usr/bin/env python3
"""
Combined graph analysis: clustering + SC-enforce + topological features.

All 9 topologies, all metrics in one table.
"""

import sys
import json
import numpy as np
import networkx as nx
from pathlib import Path
from scipy.spatial import Delaunay, cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "exp14a_sc_enforce"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sc_baseline"))

from topo_features import (extract_topo_features, topo_adjusted_rho,
                           calibrate_transport_probe, reset_calibration)
from sc_enforce import SCEnforcer, StrictnessTracker

import leidenalg
import igraph as ig


# ===================================================================
# Graph builders
# ===================================================================

def build_barbell():
    return nx.barbell_graph(30, 5)

def build_hub_spoke():
    return nx.star_graph(60)

def build_ring_cliques():
    return nx.ring_of_cliques(8, 12)

def build_bipartite():
    rng = np.random.default_rng(42)
    G = nx.Graph()
    G.add_nodes_from(range(80))
    for i in range(40):
        for j in range(40, 80):
            if rng.random() < 0.15:
                G.add_edge(i, j)
    _connect(G)
    return G

def build_erdos_renyi():
    G = nx.erdos_renyi_graph(100, 0.06, seed=42)
    _connect(G)
    return G

def build_grid():
    return nx.grid_2d_graph(10, 10)

def build_swiss_roll():
    rng = np.random.default_rng(42)
    n = 500
    t = 1.5 * np.pi * (1 + 2 * rng.random(n))
    x = t * np.cos(t) + rng.standard_normal(n) * 0.3
    y = t * np.sin(t) + rng.standard_normal(n) * 0.3
    tree = cKDTree(np.column_stack([x, y]))
    _, idx = tree.query(np.column_stack([x, y]), k=7)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in idx[i, 1:]:
            G.add_edge(i, j)
    return G

def build_planar():
    rng = np.random.default_rng(42)
    pts = rng.random((80, 2))
    tri = Delaunay(pts)
    G = nx.Graph()
    G.add_nodes_from(range(80))
    for s in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                G.add_edge(s[i], s[j])
    return G

def build_mobius():
    G = nx.Graph()
    na, nw = 40, 3
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

def _connect(G):
    comps = list(nx.connected_components(G))
    for k in range(1, len(comps)):
        G.add_edge(min(comps[0]), min(comps[k]))


GRAPHS = [
    ("Barbell",        build_barbell),
    ("Hub-Spoke",      build_hub_spoke),
    ("Ring Cliques",   build_ring_cliques),
    ("Bipartite",      build_bipartite),
    ("Erdos-Renyi",    build_erdos_renyi),
    ("Grid 10x10",     build_grid),
    ("Swiss Roll",     build_swiss_roll),
    ("Planar",         build_planar),
    ("Mobius",         build_mobius),
]


# ===================================================================
# Leiden clustering
# ===================================================================

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


# ===================================================================
# Clustering metrics
# ===================================================================

def clustering_metrics(G, labels, node_list, node_map):
    n_cl = len(set(labels))
    total_e = G.number_of_edges()
    cut = sum(1 for u, v in G.edges()
              if labels[node_map[u]] != labels[node_map[v]])
    ecr = cut / max(total_e, 1)

    communities = []
    for c in range(n_cl):
        communities.append({node_list[i] for i in range(len(labels))
                            if labels[i] == c})
    mod = nx.algorithms.community.modularity(G, communities)

    return {"n_clusters": n_cl, "ecr": ecr, "modularity": mod}


# ===================================================================
# SC-enforce simulation
# ===================================================================

def sc_enforce_sim(G, labels, node_list, node_map, seed=42):
    rng = np.random.default_rng(seed)
    n = len(node_list)
    n_cl = len(set(labels))

    gt = np.array([0.5 * np.sin(i * 0.3) + rng.standard_normal() * 0.05
                   for i in range(n)])
    coarse = np.zeros(n)
    for c in range(n_cl):
        mask = labels == c
        if mask.any():
            coarse[mask] = gt[mask].mean()
    state = coarse.copy()

    def restrict(v):
        out = np.zeros(n_cl)
        for c in range(n_cl):
            m = labels == c
            if m.any(): out[c] = v[m].mean()
        return out

    def prolong(cv, shape):
        out = np.zeros(n)
        for c in range(n_cl):
            out[labels == c] = cv[c]
        return out

    tau_base = 0.078
    enforcer = SCEnforcer(
        tau_parent={("T3_graph", 1): tau_base},
        R_fn=restrict, Up_fn=prolong,
        space_type="T3_graph",
        damp_factor=0.5, max_damp_iterations=3,
        n_active=n_cl, beta=2.0)
    enforcer.strictness_tracker = StrictnessTracker()

    rho = np.array([float(np.mean((gt[labels == c] - state[labels == c])**2))
                     for c in range(n_cl)])
    budget = max(1, int(0.3 * n_cl))
    order = np.argsort(-rho)[:budget]

    n_pass = n_damp = n_reject = 0
    for unit in order:
        mask = labels == unit
        delta = np.zeros(n)
        delta[mask] = gt[mask] - state[mask]
        result = enforcer.check_and_enforce(delta, coarse, level=1,
                                            unit_id=str(unit))
        if result.action == "pass":
            state += delta; n_pass += 1
        elif result.action == "damped":
            state += result.enforced_delta; n_damp += 1
        else:
            n_reject += 1

    mse = float(np.mean((gt - state)**2))
    psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
    c_mse = float(np.mean((gt - coarse)**2))
    c_psnr = 10 * np.log10(1.0 / max(c_mse, 1e-12))

    return {
        "n_pass": n_pass, "n_damp": n_damp, "n_reject": n_reject,
        "psnr_gain": psnr - c_psnr,
    }


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 130)
    print("Combined Graph Analysis: Clustering + SC-Enforce + Topological Features")
    print("=" * 130)

    reset_calibration()
    cal = calibrate_transport_probe()
    print(f"\n  Calibration: kappa_max={cal.kappa_max:.0f}, "
          f"t_ollivier={cal.t_ollivier_ms:.2f}ms, "
          f"budget=50ms -> max {int(50/cal.t_ollivier_ms)} Ollivier calls\n")

    rows = []

    for name, builder in GRAPHS:
        G = builder()
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()

        labels, node_list, node_map = cluster_leiden(G)
        cl = clustering_metrics(G, labels, node_list, node_map)
        sc = sc_enforce_sim(G, labels, node_list, node_map)
        topo = extract_topo_features(G, labels, node_list,
                                     kappa_max=cal.kappa_max,
                                     topo_budget_ms=50.0)

        row = {
            "name": name,
            "N": n_nodes,
            "E": n_edges,
            "Clusters": cl["n_clusters"],
            "ECR": cl["ecr"],
            "Modularity": cl["modularity"],
            "Reject": sc["n_reject"],
            "PSNR": sc["psnr_gain"],
            "ORC": f"{topo.n_ollivier}/{topo.n_ollivier+topo.n_forman}",
            "ORC_pct": topo.ollivier_pct,
            "Time_ms": topo.computation_ms,
            "k_mean": float(topo.curvature.mean()),
            "k_min": float(topo.curvature.min()),
        }
        rows.append(row)

        print(f"  {name:<14s}  N={n_nodes:>4d}  E={n_edges:>5d}  "
              f"cl={cl['n_clusters']:>2d}  ECR={cl['ecr']:.1%}  "
              f"mod={cl['modularity']:.3f}  rej={sc['n_reject']}  "
              f"PSNR={sc['psnr_gain']:+.1f}dB  "
              f"ORC={topo.n_ollivier:>2d}/{topo.n_ollivier+topo.n_forman:<4d} "
              f"[{topo.ollivier_pct:>4.0f}%]  "
              f"{topo.computation_ms:>5.0f}ms  "
              f"k={topo.curvature.mean():>+.3f}  "
              f"k_min={topo.curvature.min():>+.3f}")

    # Summary table
    print("\n" + "=" * 130)
    hdr = (f"  {'Topology':<14s} {'N':>4s} {'E':>5s} {'Cl':>3s} "
           f"{'ECR':>6s} {'Mod':>5s} {'Rej':>3s} {'PSNR':>7s} "
           f"{'ORC':>10s} {'Time':>6s} {'k_mean':>7s} {'k_min':>7s}")
    print(hdr)
    print("  " + "-" * 120)

    for r in rows:
        print(f"  {r['name']:<14s} {r['N']:>4d} {r['E']:>5d} {r['Clusters']:>3d} "
              f"{r['ECR']:>5.1%} {r['Modularity']:>5.3f} {r['Reject']:>3d} "
              f"{r['PSNR']:>+6.1f}dB "
              f"{r['ORC']:>10s} "
              f"{r['Time_ms']:>5.0f}ms "
              f"{r['k_mean']:>+7.3f} {r['k_min']:>+7.3f}")

    print("=" * 130)

    # Save
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "combined_graphs_test.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2, default=float)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
