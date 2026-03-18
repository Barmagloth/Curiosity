"""
Restriction (R) and Prolongation (Up) operators for scale-consistency verification.

R = gaussian blur + decimation (factor 2)
Up = bilinear upsampling back to original size

The pair (R, Up) is fixed for the entire verification cycle.
See concept_v1.6.md section 8.3 for design rationale.

Supports 4 space types from exp_seam_crossspace.py:
  T1: 2D scalar grid
  T2: 2D vector-valued grid (feature map)
  T3: Irregular graph (k-NN point cloud)
  T4: Tree hierarchy
"""

import numpy as np
from scipy.ndimage import gaussian_filter, zoom


# ═══════════════════════════════════════════════
# T1: 2D Scalar Grid
# ═══════════════════════════════════════════════

def restrict_scalar(x, sigma=1.0):
    """R for 2D scalar grid: gaussian blur + decimation by factor 2.

    Args:
        x: 2D array (H, W)
        sigma: gaussian blur sigma before decimation

    Returns:
        2D array (H//2, W//2)
    """
    blurred = gaussian_filter(x, sigma=sigma)
    return blurred[::2, ::2]


def prolong_scalar(x_coarse, target_shape):
    """Up for 2D scalar grid: bilinear upsampling to target_shape.

    Args:
        x_coarse: 2D array (H_c, W_c)
        target_shape: (H, W) — shape to upsample to

    Returns:
        2D array of target_shape
    """
    factors = (target_shape[0] / x_coarse.shape[0],
               target_shape[1] / x_coarse.shape[1])
    return zoom(x_coarse, factors, order=1)


# ═══════════════════════════════════════════════
# T2: 2D Vector Grid
# ═══════════════════════════════════════════════

def restrict_vector(x, sigma=1.0):
    """R for 2D vector grid: per-channel gaussian blur + decimation by factor 2.

    Args:
        x: 3D array (H, W, D)
        sigma: gaussian blur sigma before decimation

    Returns:
        3D array (H//2, W//2, D)
    """
    D = x.shape[2]
    out = np.empty((x.shape[0] // 2, x.shape[1] // 2, D))
    for d in range(D):
        blurred = gaussian_filter(x[:, :, d], sigma=sigma)
        out[:, :, d] = blurred[::2, ::2]
    return out


def prolong_vector(x_coarse, target_shape):
    """Up for 2D vector grid: per-channel bilinear upsampling.

    Args:
        x_coarse: 3D array (H_c, W_c, D)
        target_shape: (H, W, D) — shape to upsample to

    Returns:
        3D array of target_shape
    """
    D = x_coarse.shape[2]
    H, W = target_shape[0], target_shape[1]
    out = np.empty((H, W, D))
    factors = (H / x_coarse.shape[0], W / x_coarse.shape[1])
    for d in range(D):
        out[:, :, d] = zoom(x_coarse[:, :, d], factors, order=1)
    return out


# ═══════════════════════════════════════════════
# T3: Irregular Graph
# ═══════════════════════════════════════════════

def restrict_graph(values, labels, n_clusters):
    """R for irregular graph: cluster-mean pooling.

    Each cluster's mean value becomes the coarse representation.
    This is the natural analogue of gaussian-blur+decimation on graphs:
    spatial averaging within each cluster, then one value per cluster.

    Args:
        values: 1D array (n_points,) — per-node values
        labels: 1D array (n_points,) — cluster assignment per node
        n_clusters: int — number of clusters

    Returns:
        1D array (n_clusters,) — one value per cluster
    """
    coarse = np.zeros(n_clusters)
    for c in range(n_clusters):
        mask = labels == c
        if mask.any():
            coarse[c] = values[mask].mean()
    return coarse


def prolong_graph(coarse_values, labels, n_points):
    """Up for irregular graph: scatter cluster means back to nodes.

    Each node gets the value of its cluster. This is the natural
    bilinear-upsampling analogue for graphs (piecewise constant prolongation).

    Args:
        coarse_values: 1D array (n_clusters,) — one value per cluster
        labels: 1D array (n_points,) — cluster assignment per node
        n_points: int

    Returns:
        1D array (n_points,) — prolonged values
    """
    out = np.zeros(n_points)
    for i in range(n_points):
        out[i] = coarse_values[labels[i]]
    return out


# ═══════════════════════════════════════════════
# T4: Tree Hierarchy
# ═══════════════════════════════════════════════

def restrict_tree(values, n_nodes, coarse_depth):
    """R for tree: average subtree values up to coarse_depth ancestors.

    For a binary tree, restriction means collapsing fine-level nodes
    into their coarse-depth ancestor by averaging.

    Args:
        values: 1D array (n_nodes,)
        n_nodes: int — total node count
        coarse_depth: int — depth at which coarse representation lives

    Returns:
        1D array (n_coarse_nodes,) where n_coarse_nodes = 2^coarse_depth - 1 + 2^coarse_depth
        Actually returns values only for nodes at depth < coarse_depth,
        aggregated from their subtrees.
    """
    n_coarse = 2 ** coarse_depth
    coarse = np.zeros(n_coarse)
    for c in range(n_coarse):
        node_idx = (2 ** coarse_depth - 1) + c
        subtree = _subtree_nodes(node_idx, n_nodes)
        if subtree:
            coarse[c] = np.mean([values[j] for j in subtree])
        else:
            coarse[c] = 0.0
    return coarse


def prolong_tree(coarse_values, n_nodes, coarse_depth):
    """Up for tree: broadcast coarse ancestor value to all subtree descendants.

    Args:
        coarse_values: 1D array (n_coarse_nodes,) — values at coarse level
        n_nodes: int — total node count
        coarse_depth: int — depth at which coarse_values live

    Returns:
        1D array (n_nodes,) — prolonged values
    """
    out = np.zeros(n_nodes)
    n_coarse = len(coarse_values)
    for c in range(n_coarse):
        node_idx = (2 ** coarse_depth - 1) + c
        subtree = _subtree_nodes(node_idx, n_nodes)
        for j in subtree:
            out[j] = coarse_values[c]
    # Nodes above coarse_depth keep their original mapping (identity for ancestors)
    for i in range(min(2 ** coarse_depth - 1, n_nodes)):
        d = int(np.log2(i + 1))
        if d < coarse_depth:
            # Map ancestor to mean of its coarse-level descendants
            desc_coarse_indices = []
            _find_coarse_descendants(i, coarse_depth, n_nodes, desc_coarse_indices)
            if desc_coarse_indices:
                mapped = [coarse_values[idx - (2 ** coarse_depth - 1)]
                          for idx in desc_coarse_indices
                          if 0 <= idx - (2 ** coarse_depth - 1) < n_coarse]
                if mapped:
                    out[i] = np.mean(mapped)
    return out


def _subtree_nodes(root, n_nodes):
    """All nodes in subtree rooted at root (inclusive)."""
    nodes = []
    q = [root]
    while q:
        curr = q.pop()
        if curr >= n_nodes:
            continue
        nodes.append(curr)
        left = 2 * curr + 1
        right = 2 * curr + 2
        if left < n_nodes:
            q.append(left)
        if right < n_nodes:
            q.append(right)
    return nodes


def _find_coarse_descendants(node, coarse_depth, n_nodes, result):
    """Find all descendants of node at exactly coarse_depth."""
    d = int(np.log2(node + 1))
    if d == coarse_depth:
        if node < n_nodes:
            result.append(node)
        return
    left = 2 * node + 1
    right = 2 * node + 2
    if left < n_nodes:
        _find_coarse_descendants(left, coarse_depth, n_nodes, result)
    if right < n_nodes:
        _find_coarse_descendants(right, coarse_depth, n_nodes, result)
