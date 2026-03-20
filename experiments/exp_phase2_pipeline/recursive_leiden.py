#!/usr/bin/env python3
"""
Recursive Leiden clustering: graph → hierarchical tree.

Takes a flat k-NN graph (irregular_graph / T3) and produces a multi-level
hierarchy of communities by recursively applying Leiden on the coarsened
community graph.  The result is a dendrogram that can be processed by the
tree_hierarchy branch of the pipeline.

Architecture:
  Level 0: original nodes
  Level 1: Leiden communities of the graph
  Level 2: Leiden communities of the community-graph
  ...until a level has ≤ 1 community or no inter-community edges.

Fallback: if igraph/leidenalg unavailable, uses Louvain+CC (same semantics).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Clustering backends
# ---------------------------------------------------------------------------

def _leiden_partition(edges: list, n_nodes: int, seed: int):
    """Leiden community detection.  Returns list of sets."""
    import igraph as ig
    import leidenalg
    G = ig.Graph(n=n_nodes, edges=edges, directed=False)
    partition = leidenalg.find_partition(
        G, leidenalg.ModularityVertexPartition, seed=seed)
    communities = [set() for _ in range(len(partition))]
    for node, cid in enumerate(partition.membership):
        communities[cid].add(node)
    return [c for c in communities if c]  # drop empties


def _louvain_cc_partition(edges: list, n_nodes: int, seed: int):
    """Louvain + connected_components post-fix.  Fallback."""
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    for u, v in edges:
        G.add_edge(u, v)
    raw = nx.algorithms.community.louvain_communities(G, seed=seed, resolution=1.0)
    # CC post-fix
    result = []
    for nodes in raw:
        subg = G.subgraph(nodes)
        components = list(nx.connected_components(subg))
        result.extend(components)
    return result


def _partition(edges: list, n_nodes: int, seed: int):
    """Try Leiden, fall back to Louvain+CC."""
    try:
        import igraph, leidenalg  # noqa: F401
        return _leiden_partition(edges, n_nodes, seed), "leiden"
    except ImportError:
        return _louvain_cc_partition(edges, n_nodes, seed), "louvain+cc"


# ---------------------------------------------------------------------------
# Hierarchy node
# ---------------------------------------------------------------------------

@dataclass
class HierNode:
    """One node in the community dendrogram."""
    id: int
    level: int
    children: List[int] = field(default_factory=list)   # child HierNode ids
    parent: Optional[int] = None
    leaf_nodes: Set[int] = field(default_factory=set)    # original graph nodes
    size: int = 0                                         # len(leaf_nodes)


# ---------------------------------------------------------------------------
# Recursive clustering
# ---------------------------------------------------------------------------

@dataclass
class CommunityTree:
    """Hierarchical community decomposition of a graph.

    Attributes:
        nodes: dict  id -> HierNode
        root: int    id of the root node (single super-community)
        depth: int   max depth of the tree
        n_leaves: int  number of leaf-level communities (= level 1 communities)
        n_original: int  number of original graph nodes
        backend: str  "leiden" or "louvain+cc"
    """
    nodes: Dict[int, HierNode] = field(default_factory=dict)
    root: int = 0
    depth: int = 0
    n_leaves: int = 0
    n_original: int = 0
    backend: str = ""

    # -- Tree navigation (TreeHierarchySpace-compatible interface) ----------

    def children(self, node_id: int) -> List[int]:
        return self.nodes[node_id].children

    def parent(self, node_id: int) -> Optional[int]:
        return self.nodes[node_id].parent

    def subtree_ids(self, node_id: int) -> List[int]:
        """All descendant node ids (inclusive)."""
        result = [node_id]
        queue = [node_id]
        while queue:
            curr = queue.pop()
            for c in self.nodes[curr].children:
                result.append(c)
                queue.append(c)
        return result

    def leaf_original_nodes(self, node_id: int) -> Set[int]:
        """Original graph nodes covered by this subtree."""
        return self.nodes[node_id].leaf_nodes

    def get_level(self, level: int) -> List[int]:
        """All node ids at a given level."""
        return [nid for nid, n in self.nodes.items() if n.level == level]

    def stats(self) -> dict:
        """Summary statistics."""
        level_sizes = {}
        for n in self.nodes.values():
            level_sizes.setdefault(n.level, []).append(n.size)
        return {
            "depth": self.depth,
            "n_tree_nodes": len(self.nodes),
            "n_original_nodes": self.n_original,
            "n_leaves": self.n_leaves,
            "backend": self.backend,
            "levels": {
                lv: {
                    "count": len(sizes),
                    "min_size": min(sizes),
                    "max_size": max(sizes),
                    "mean_size": np.mean(sizes),
                }
                for lv, sizes in sorted(level_sizes.items())
            },
        }


def build_community_tree(
    neighbors: Dict[int, Set[int]],
    n_points: int,
    seed: int = 42,
    max_depth: int = 10,
) -> CommunityTree:
    """Build a hierarchical community tree from a k-NN graph.

    Parameters
    ----------
    neighbors : dict  node_id -> set of neighbor node_ids
    n_points : int  number of original graph nodes
    seed : int  for reproducibility
    max_depth : int  safety limit on recursion depth

    Returns
    -------
    CommunityTree with full dendrogram
    """
    tree = CommunityTree(n_original=n_points)
    next_id = 0

    # --- Level 0: leaf communities from first Leiden pass ----------------
    edges_l0 = [(i, j) for i, nbrs in neighbors.items()
                 for j in nbrs if j > i]
    communities_l0, backend = _partition(edges_l0, n_points, seed)
    tree.backend = backend

    # Create leaf nodes (level 0 in tree = finest community level)
    leaf_ids = []
    community_of_node = np.zeros(n_points, dtype=int)  # original node -> leaf community id

    for comm in communities_l0:
        nid = next_id
        next_id += 1
        tree.nodes[nid] = HierNode(
            id=nid, level=0,
            leaf_nodes=set(comm), size=len(comm))
        leaf_ids.append(nid)
        for orig_node in comm:
            community_of_node[orig_node] = nid

    tree.n_leaves = len(leaf_ids)
    current_level_ids = leaf_ids

    # --- Recursive coarsening: levels 1, 2, ... -------------------------
    level = 1
    while len(current_level_ids) > 1 and level <= max_depth:
        # Build community graph: nodes = current_level_ids, edges = inter-community links
        # Map current_level_ids to 0..N-1 for the partition functions
        id_to_idx = {nid: idx for idx, nid in enumerate(current_level_ids)}
        n_meta = len(current_level_ids)

        meta_edges_set = set()
        for nid in current_level_ids:
            idx_a = id_to_idx[nid]
            # Gather all original-graph edges that cross from this community
            for orig_node in tree.nodes[nid].leaf_nodes:
                for nbr in neighbors.get(orig_node, set()):
                    nbr_comm = community_of_node[nbr]
                    if nbr_comm != nid:
                        # Find which current_level node owns nbr_comm
                        # nbr_comm is a leaf id; find its ancestor at current level
                        owner = nbr_comm
                        while tree.nodes[owner].parent is not None:
                            owner = tree.nodes[owner].parent
                        if owner in id_to_idx:
                            idx_b = id_to_idx[owner]
                            if idx_a != idx_b:
                                e = (min(idx_a, idx_b), max(idx_a, idx_b))
                                meta_edges_set.add(e)

        meta_edges = list(meta_edges_set)

        if not meta_edges:
            # No inter-community edges → everything merges into single root
            break

        # Partition the meta-graph
        meta_communities, _ = _partition(meta_edges, n_meta, seed + level)

        if len(meta_communities) >= n_meta:
            # No coarsening happened (each community = its own partition)
            break

        # Create new parent nodes at this level
        new_level_ids = []
        for meta_comm in meta_communities:
            nid = next_id
            next_id += 1
            child_ids = [current_level_ids[idx] for idx in meta_comm]
            all_leaves = set()
            for cid in child_ids:
                all_leaves |= tree.nodes[cid].leaf_nodes
                tree.nodes[cid].parent = nid
            tree.nodes[nid] = HierNode(
                id=nid, level=level,
                children=child_ids,
                leaf_nodes=all_leaves,
                size=len(all_leaves))
            new_level_ids.append(nid)

        current_level_ids = new_level_ids
        level += 1

    # --- Root node -------------------------------------------------------
    if len(current_level_ids) == 1:
        tree.root = current_level_ids[0]
    else:
        # Multiple nodes remain → create synthetic root
        root_id = next_id
        all_leaves = set()
        for nid in current_level_ids:
            tree.nodes[nid].parent = root_id
            all_leaves |= tree.nodes[nid].leaf_nodes
        tree.nodes[root_id] = HierNode(
            id=root_id, level=level,
            children=list(current_level_ids),
            leaf_nodes=all_leaves,
            size=len(all_leaves))
        tree.root = root_id

    tree.depth = tree.nodes[tree.root].level

    return tree


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from scipy.spatial import cKDTree
    import time

    print("=" * 70)
    print("Recursive Leiden: graph -> community tree")
    print("=" * 70)

    configs = [
        (200, 6, "small"),
        (500, 6, "medium"),
        (1000, 8, "large"),
        (3000, 10, "xlarge"),
    ]

    for n_pts, k, label in configs:
        rng = np.random.default_rng(42)
        pos = rng.random((n_pts, 2))
        tree_kd = cKDTree(pos)
        _, idx = tree_kd.query(pos, k=k + 1)
        neighbors = {i: set(idx[i, 1:]) for i in range(n_pts)}

        t0 = time.perf_counter()
        ctree = build_community_tree(neighbors, n_pts, seed=42)
        elapsed = time.perf_counter() - t0

        s = ctree.stats()
        print(f"\n-- {label} (N={n_pts}, k={k}) --  [{elapsed*1000:.1f}ms, {s['backend']}]")
        print(f"   Tree depth: {s['depth']}")
        print(f"   Tree nodes: {s['n_tree_nodes']}")
        print(f"   Leaf communities: {s['n_leaves']}")

        for lv, info in s["levels"].items():
            print(f"   Level {lv}: {info['count']} nodes, "
                  f"size {info['min_size']}..{info['max_size']} "
                  f"(mean {info['mean_size']:.1f})")

        # Verify: all original nodes covered, no overlaps at leaf level
        leaf_level = ctree.get_level(0)
        all_covered = set()
        for lid in leaf_level:
            orig = ctree.nodes[lid].leaf_nodes
            overlap = all_covered & orig
            assert not overlap, f"Overlap: {overlap}"
            all_covered |= orig
        assert all_covered == set(range(n_pts)), \
            f"Coverage: {len(all_covered)}/{n_pts}"
        print(f"   Coverage: {len(all_covered)}/{n_pts} OK  No overlaps OK")

        # Verify: tree connectivity (every non-root has parent)
        for nid, node in ctree.nodes.items():
            if nid != ctree.root:
                assert node.parent is not None, f"Node {nid} has no parent"
                assert nid in ctree.nodes[node.parent].children, \
                    f"Node {nid} not in parent's children"
        print(f"   Tree integrity: OK")

    print("\n" + "=" * 70)
    print("All checks passed.")
    print("=" * 70)
