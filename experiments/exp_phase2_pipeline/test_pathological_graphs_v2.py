#!/usr/bin/env python3
"""
Stress test v2: structural graph topologies.

Topologies:
  1. Grid 10x10        -- uniform local connectivity, no natural clusters
  2. Grid 20x20        -- same but 4x larger
  3. Erdos-Renyi p=0.15 -- above percolation, weak clusters
  4. Planar (Delaunay) -- triangulation of random points, spatial topology
  5. Mobius strip      -- non-orientable surface as a graph

Uses same metrics + SC-enforce simulation as v1.
"""

import sys
import time
import json
import numpy as np
import networkx as nx
from pathlib import Path
from scipy.spatial import Delaunay

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "exp14a_sc_enforce"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sc_baseline"))

from sc_enforce import SCEnforcer, StrictnessTracker, WasteBudget


# ===================================================================
# Graph builders
# ===================================================================

def build_grid(rows: int, cols: int) -> nx.Graph:
    """2D grid graph with 4-connectivity."""
    return nx.grid_2d_graph(rows, cols)


def build_erdos_renyi_dense(n: int = 100, p: float = 0.15, seed: int = 42) -> nx.Graph:
    """Erdos-Renyi above percolation threshold."""
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    components = list(nx.connected_components(G))
    if len(components) > 1:
        for k in range(1, len(components)):
            a = min(components[0])
            b = min(components[k])
            G.add_edge(a, b)
    return G


def build_planar_delaunay(n_points: int = 80, seed: int = 42) -> nx.Graph:
    """Planar graph from Delaunay triangulation of random 2D points."""
    rng = np.random.default_rng(seed)
    points = rng.random((n_points, 2))
    tri = Delaunay(points)
    G = nx.Graph()
    G.add_nodes_from(range(n_points))
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                G.add_edge(simplex[i], simplex[j])
    return G


def build_mobius_strip(n_along: int = 40, n_across: int = 3) -> nx.Graph:
    """
    Mobius strip as a graph.

    Take a rectangular grid (n_along x n_across), then glue the two short edges
    with a twist: node (0, j) connects to node (n_along-1, n_across-1-j).

    This creates a non-orientable surface with a single edge boundary
    and no consistent notion of 'inside' vs 'outside'.
    """
    G = nx.Graph()
    # Create nodes
    for i in range(n_along):
        for j in range(n_across):
            G.add_node((i, j))

    # Horizontal edges (along the strip)
    for i in range(n_along - 1):
        for j in range(n_across):
            G.add_edge((i, j), (i + 1, j))

    # Vertical edges (across the strip width)
    for i in range(n_along):
        for j in range(n_across - 1):
            G.add_edge((i, j), (i, j + 1))

    # Twisted glue: connect last column to first with flip
    for j in range(n_across):
        G.add_edge((n_along - 1, j), (0, n_across - 1 - j))

    return G


TOPOLOGIES = {
    "grid_10x10":          lambda: build_grid(10, 10),
    "grid_20x20":          lambda: build_grid(20, 20),
    "erdos_renyi_p015":    lambda: build_erdos_renyi_dense(100, 0.15),
    "planar_delaunay_80":  lambda: build_planar_delaunay(80),
    "mobius_40x3":         lambda: build_mobius_strip(40, 3),
}


# ===================================================================
# Clustering (same as v1)
# ===================================================================

def cluster_leiden(G: nx.Graph, seed: int = 42):
    import igraph as ig
    import leidenalg
    node_list = sorted(G.nodes())
    node_map = {node: i for i, node in enumerate(node_list)}
    n = len(node_list)
    edges = [(node_map[u], node_map[v]) for u, v in G.edges()]
    G_ig = ig.Graph(n=n, edges=edges, directed=False)
    partition = leidenalg.find_partition(
        G_ig, leidenalg.ModularityVertexPartition, seed=seed)
    labels = np.array(partition.membership)
    return labels, node_list, node_map


def cluster_louvain_cc(G: nx.Graph, seed: int = 42):
    node_list = sorted(G.nodes())
    node_map = {node: i for i, node in enumerate(node_list)}
    n = len(node_list)
    communities = nx.algorithms.community.louvain_communities(G, seed=seed)

    labels = np.zeros(n, dtype=int)
    for cid, nodes in enumerate(communities):
        for nd in nodes:
            labels[node_map[nd]] = cid

    # CC post-fix
    next_id = len(communities)
    for cid, nodes in enumerate(communities):
        subg = G.subgraph(nodes)
        components = list(nx.connected_components(subg))
        if len(components) > 1:
            for comp in components[1:]:
                for nd in comp:
                    labels[node_map[nd]] = next_id
                next_id += 1

    unique = sorted(set(labels))
    remap = {old: new for new, old in enumerate(unique)}
    labels = np.array([remap[l] for l in labels])
    return labels, node_list, node_map


# ===================================================================
# Metrics
# ===================================================================

def compute_metrics(G: nx.Graph, labels: np.ndarray,
                    node_list, node_map) -> dict:
    n_clusters = len(set(labels))
    total_edges = G.number_of_edges()
    cut_edges = sum(1 for u, v in G.edges()
                    if labels[node_map[u]] != labels[node_map[v]])
    ecr = cut_edges / max(total_edges, 1)

    sizes = [int(np.sum(labels == c)) for c in range(n_clusters)]

    # Connected community check
    disconnected = 0
    for c in range(n_clusters):
        nodes_in_c = [node_list[i] for i in range(len(labels)) if labels[i] == c]
        if len(nodes_in_c) > 1:
            subg = G.subgraph(nodes_in_c)
            if nx.number_connected_components(subg) > 1:
                disconnected += 1

    communities_list = []
    for c in range(n_clusters):
        comm = {node_list[i] for i in range(len(labels)) if labels[i] == c}
        communities_list.append(comm)
    modularity = nx.algorithms.community.modularity(G, communities_list)

    return {
        "n_clusters": n_clusters,
        "edge_cut_ratio": ecr,
        "modularity": modularity,
        "min_size": min(sizes),
        "max_size": max(sizes),
        "mean_size": float(np.mean(sizes)),
        "size_ratio": max(sizes) / max(min(sizes), 1),
        "disconnected": disconnected,
    }


# ===================================================================
# SC-enforce simulation
# ===================================================================

def simulate_sc_enforce(G: nx.Graph, labels: np.ndarray,
                        node_list, node_map, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    n = len(node_list)
    n_clusters = len(set(labels))

    gt = np.zeros(n)
    for i, nd in enumerate(node_list):
        # Use node index for deterministic signal
        gt[i] = 0.5 * np.sin(i * 0.3) + rng.standard_normal() * 0.05

    coarse = np.zeros(n)
    for c in range(n_clusters):
        mask = labels == c
        if mask.any():
            coarse[mask] = gt[mask].mean()

    state = coarse.copy()

    def restrict(values):
        out = np.zeros(n_clusters)
        for c in range(n_clusters):
            mask = labels == c
            if mask.any():
                out[c] = values[mask].mean()
        return out

    def prolong(coarse_vals, target_shape):
        out = np.zeros(n)
        for c in range(n_clusters):
            mask = labels == c
            out[mask] = coarse_vals[c]
        return out

    tau_base = 0.078
    thresholds = {("T3_graph", 1): tau_base}

    enforcer = SCEnforcer(
        tau_parent=thresholds, R_fn=restrict, Up_fn=prolong,
        space_type="T3_graph",
        damp_factor=0.5, max_damp_iterations=3,
        n_active=n_clusters, beta=2.0)
    enforcer.strictness_tracker = StrictnessTracker()

    rho = np.array([float(np.mean((gt[labels == c] - state[labels == c])**2))
                     for c in range(n_clusters)])
    budget = max(1, int(0.3 * n_clusters))
    order = np.argsort(-rho)[:budget]

    n_pass = n_damp = n_reject = 0
    for unit in order:
        mask = labels == unit
        delta = np.zeros(n)
        delta[mask] = gt[mask] - state[mask]

        result = enforcer.check_and_enforce(delta, coarse, level=1,
                                            unit_id=str(unit))
        if result.action == "pass":
            state = state + delta
            n_pass += 1
        elif result.action == "damped":
            state = state + result.enforced_delta
            n_damp += 1
        else:
            n_reject += 1

    mse = float(np.mean((gt - state) ** 2))
    psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
    coarse_mse = float(np.mean((gt - coarse) ** 2))
    coarse_psnr = 10 * np.log10(1.0 / max(coarse_mse, 1e-12))

    effective_tau = tau_base * (1 + 2.0 / np.sqrt(max(n_clusters, 1)))

    return {
        "n_pass": n_pass,
        "n_damp": n_damp,
        "n_reject": n_reject,
        "reject_rate": n_reject / max(n_pass + n_damp + n_reject, 1),
        "budget": budget,
        "psnr_coarse": coarse_psnr,
        "psnr_final": psnr,
        "psnr_gain": psnr - coarse_psnr,
        "tau_base": tau_base,
        "tau_effective": effective_tau,
        "adaptive_mult": 1 + 2.0 / np.sqrt(max(n_clusters, 1)),
    }


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 95)
    print("Pathological Graph Topologies v2 -- Structural Stress Test")
    print("=" * 95)

    results_all = {}

    for topo_name, builder in TOPOLOGIES.items():
        G = builder()
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        density = 2 * n_edges / max(n_nodes * (n_nodes - 1), 1)

        print(f"\n-- {topo_name} (N={n_nodes}, E={n_edges}, density={density:.4f}) --")

        topo_results = {"n_nodes": n_nodes, "n_edges": n_edges, "density": density}

        for cluster_name, cluster_fn in [("leiden", cluster_leiden),
                                          ("louvain_cc", cluster_louvain_cc)]:
            labels, node_list, node_map = cluster_fn(G, seed=42)
            metrics = compute_metrics(G, labels, node_list, node_map)
            sc = simulate_sc_enforce(G, labels, node_list, node_map, seed=42)

            topo_results[cluster_name] = {**metrics, **sc}

            tag = "LD" if cluster_name == "leiden" else "LV"
            print(f"  [{tag}] clusters={metrics['n_clusters']:3d}  "
                  f"ECR={metrics['edge_cut_ratio']:.1%}  "
                  f"mod={metrics['modularity']:.3f}  "
                  f"sizes={metrics['min_size']}..{metrics['max_size']}  "
                  f"disconn={metrics['disconnected']}  |  "
                  f"pass={sc['n_pass']} damp={sc['n_damp']} rej={sc['n_reject']}  "
                  f"PSNR={sc['psnr_gain']:+.1f}dB  "
                  f"tau_eff={sc['tau_effective']:.3f} ({sc['adaptive_mult']:.2f}x)")

        results_all[topo_name] = topo_results

    # Summary
    print("\n" + "=" * 95)
    print("SUMMARY (Leiden)")
    print("=" * 95)
    print(f"  {'Topology':<22s} {'N':>5s} {'E':>6s} {'dens':>6s} "
          f"{'cl':>4s} {'ECR':>7s} {'mod':>6s} {'disc':>5s} "
          f"{'rej':>4s} {'PSNR':>8s} {'tau_eff':>8s}")
    print("  " + "-" * 88)

    for topo_name, r in results_all.items():
        ld = r["leiden"]
        print(f"  {topo_name:<22s} {r['n_nodes']:>5d} {r['n_edges']:>6d} "
              f"{r['density']:>6.4f} "
              f"{ld['n_clusters']:>4d} {ld['edge_cut_ratio']:>6.1%} "
              f"{ld['modularity']:>6.3f} {ld['disconnected']:>5d} "
              f"{ld['n_reject']:>4d} {ld['psnr_gain']:>+7.1f}dB "
              f"{ld['tau_effective']:>7.3f}")

    # Louvain+CC disconnected check
    print(f"\n  Louvain+CC disconnected communities:")
    for topo_name, r in results_all.items():
        lv = r["louvain_cc"]
        print(f"    {topo_name}: {lv['disconnected']}")

    print("=" * 95)

    # Save
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pathological_graphs_v2_test.json"
    with open(out_path, "w") as f:
        json.dump(results_all, f, indent=2, default=float)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
