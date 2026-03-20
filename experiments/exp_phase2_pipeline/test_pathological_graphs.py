#!/usr/bin/env python3
"""
Stress test: pathological graph topologies through the full pipeline.

Topologies:
  1. Barbell       -- two dense cliques connected by a thin bridge
  2. Hub-and-Spoke -- single hub with N spokes (star graph)
  3. Ring of Cliques -- cliques arranged in a ring, single edges between them
  4. Bipartite     -- two disjoint sets, edges only between sets
  5. Erdos-Renyi   -- random graph G(n, p)

For each topology we:
  - Build the adjacency structure
  - Run Leiden clustering (+ Louvain+CC for comparison)
  - Verify connected communities
  - Measure edge cut ratio, cluster balance, modularity
  - Run SC-enforce with adaptive tau to check reject rates
"""

import sys
import time
import json
import numpy as np
import networkx as nx
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Set, List, Tuple

# Local imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "exp14a_sc_enforce"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sc_baseline"))

from sc_enforce import SCEnforcer, load_thresholds, StrictnessTracker, WasteBudget


# ===================================================================
# Graph builders
# ===================================================================

def build_barbell(clique_size: int = 30, bridge_len: int = 5) -> nx.Graph:
    """Two dense cliques connected by a thin chain."""
    G = nx.barbell_graph(clique_size, bridge_len)
    return G


def build_hub_spoke(n_spokes: int = 60) -> nx.Graph:
    """Single hub node connected to all spokes (star)."""
    G = nx.star_graph(n_spokes)
    return G


def build_ring_of_cliques(n_cliques: int = 8, clique_size: int = 12) -> nx.Graph:
    """Cliques arranged in a ring, each connected to the next by one edge."""
    G = nx.ring_of_cliques(n_cliques, clique_size)
    return G


def build_bipartite(n_left: int = 40, n_right: int = 40, p: float = 0.15) -> nx.Graph:
    """Random bipartite graph."""
    rng = np.random.default_rng(42)
    G = nx.Graph()
    G.add_nodes_from(range(n_left + n_right))
    for i in range(n_left):
        for j in range(n_left, n_left + n_right):
            if rng.random() < p:
                G.add_edge(i, j)
    # Ensure connected
    components = list(nx.connected_components(G))
    if len(components) > 1:
        for k in range(1, len(components)):
            a = min(components[0])
            b = min(components[k])
            G.add_edge(a, b)
    return G


def build_erdos_renyi(n: int = 100, p: float = 0.06, seed: int = 42) -> nx.Graph:
    """Erdos-Renyi random graph G(n, p)."""
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    # Ensure connected
    components = list(nx.connected_components(G))
    if len(components) > 1:
        for k in range(1, len(components)):
            a = min(components[0])
            b = min(components[k])
            G.add_edge(a, b)
    return G


TOPOLOGIES = {
    "barbell_30":       lambda: build_barbell(30, 5),
    "hub_spoke_60":     lambda: build_hub_spoke(60),
    "ring_cliques_8x12": lambda: build_ring_of_cliques(8, 12),
    "bipartite_40x40":  lambda: build_bipartite(40, 40, 0.15),
    "erdos_renyi_100":  lambda: build_erdos_renyi(100, 0.06),
}


# ===================================================================
# Clustering
# ===================================================================

def cluster_leiden(G: nx.Graph, seed: int = 42):
    """Leiden community detection."""
    import igraph as ig
    import leidenalg
    edges = list(G.edges())
    n = G.number_of_nodes()
    node_map = {node: i for i, node in enumerate(sorted(G.nodes()))}
    mapped_edges = [(node_map[u], node_map[v]) for u, v in edges]
    G_ig = ig.Graph(n=n, edges=mapped_edges, directed=False)
    partition = leidenalg.find_partition(
        G_ig, leidenalg.ModularityVertexPartition, seed=seed)
    labels = np.zeros(n, dtype=int)
    for node_idx, cid in enumerate(partition.membership):
        labels[node_idx] = cid
    return labels, "leiden"


def cluster_louvain_cc(G: nx.Graph, seed: int = 42):
    """Louvain + connected_components post-fix."""
    node_list = sorted(G.nodes())
    node_map = {node: i for i, node in enumerate(node_list)}
    communities = nx.algorithms.community.louvain_communities(G, seed=seed, resolution=1.0)

    n = G.number_of_nodes()
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

    # Re-pack
    unique = sorted(set(labels))
    remap = {old: new for new, old in enumerate(unique)}
    labels = np.array([remap[l] for l in labels])
    return labels, "louvain+cc"


# ===================================================================
# Metrics
# ===================================================================

def compute_metrics(G: nx.Graph, labels: np.ndarray) -> dict:
    """Compute clustering quality metrics."""
    node_list = sorted(G.nodes())
    node_map = {node: i for i, node in enumerate(node_list)}
    n_clusters = len(set(labels))

    # Edge cut ratio
    total_edges = G.number_of_edges()
    cut_edges = 0
    for u, v in G.edges():
        if labels[node_map[u]] != labels[node_map[v]]:
            cut_edges += 1
    ecr = cut_edges / max(total_edges, 1)

    # Cluster sizes
    sizes = []
    for c in range(n_clusters):
        sizes.append(int(np.sum(labels == c)))

    # Connected community check
    disconnected = 0
    for c in range(n_clusters):
        nodes_in_c = [node_list[i] for i in range(len(labels)) if labels[i] == c]
        if len(nodes_in_c) > 1:
            subg = G.subgraph(nodes_in_c)
            n_cc = nx.number_connected_components(subg)
            if n_cc > 1:
                disconnected += 1

    # Modularity
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

def simulate_sc_enforce(G: nx.Graph, labels: np.ndarray, seed: int = 42) -> dict:
    """Simulate refinement + SC-enforce on graph clusters.

    Uses cluster-mean restrict / piecewise-constant prolong (same as pipeline).
    """
    rng = np.random.default_rng(seed)
    n = G.number_of_nodes()
    n_clusters = len(set(labels))

    # Ground truth: sin wave + noise
    node_list = sorted(G.nodes())
    gt = np.zeros(n)
    for i, nd in enumerate(node_list):
        gt[i] = 0.5 * np.sin(nd * 0.3) + rng.standard_normal() * 0.05

    # Coarse: cluster means
    coarse = np.zeros(n)
    for c in range(n_clusters):
        mask = labels == c
        if mask.any():
            coarse[mask] = gt[mask].mean()

    state = coarse.copy()

    # R/Up operators: cluster-mean / piecewise-constant
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

    # Build thresholds dict (use T3_graph baseline 0.078 from exp12a)
    tau_base = 0.078
    thresholds = {("T3_graph", 1): tau_base}

    enforcer = SCEnforcer(
        tau_parent=thresholds, R_fn=restrict, Up_fn=prolong,
        space_type="T3_graph",
        damp_factor=0.5, max_damp_iterations=3,
        n_active=n_clusters, beta=2.0)
    enforcer.strictness_tracker = StrictnessTracker()

    # Refine top-30% clusters by rho
    rho = np.array([float(np.mean((gt[labels == c] - state[labels == c])**2))
                     for c in range(n_clusters)])
    budget = max(1, int(0.3 * n_clusters))
    order = np.argsort(-rho)[:budget]

    n_pass = n_damp = n_reject = 0
    for unit in order:
        mask = labels == unit
        delta = np.zeros(n)
        delta[mask] = gt[mask] - state[mask]

        result = enforcer.check_and_enforce(delta, coarse, level=1, unit_id=str(unit))
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
    print("=" * 90)
    print("Pathological Graph Topologies -- Clustering + SC-Enforce Stress Test")
    print("=" * 90)

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
            labels, backend = cluster_fn(G, seed=42)
            metrics = compute_metrics(G, labels)
            sc = simulate_sc_enforce(G, labels, seed=42)

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

    # Summary table
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"  {'Topology':<22s} {'N':>5s} {'E':>6s} "
          f"{'LD cl':>5s} {'LD ECR':>7s} {'LD mod':>7s} {'LD rej':>6s} "
          f"{'LD PSNR':>8s} {'tau_eff':>8s}")
    print("  " + "-" * 82)

    for topo_name, r in results_all.items():
        ld = r["leiden"]
        print(f"  {topo_name:<22s} {r['n_nodes']:>5d} {r['n_edges']:>6d} "
              f"{ld['n_clusters']:>5d} {ld['edge_cut_ratio']:>6.1%} "
              f"{ld['modularity']:>7.3f} {ld['n_reject']:>6d} "
              f"{ld['psnr_gain']:>+7.1f}dB "
              f"{ld['tau_effective']:>7.3f}")

    print("=" * 90)

    # Save
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pathological_graphs_test.json"
    with open(out_path, "w") as f:
        json.dump(results_all, f, indent=2, default=float)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
