#!/usr/bin/env python3
"""
Curiosity -- Exp10k: Cost Surface C(I, M, p) for Layout Selection

Motivation:
  Previous experiments (exp10g-10j) showed that layout performance depends on
  space characteristics. This experiment maps the full cost surface C(I, M, p)
  where:
    I = topological isotropy (degree entropy H(D))
    M = metric gap (Kendall tau between BFS distance and linear address)
    p = occupancy (fraction of active nodes)

  For each (I, M, p) point, we benchmark three layouts:
    A_bitset:   full tensor + bitmask
    D_direct:   packed array + tile_map (gather/scatter)
    D_blocked:  block-structured storage with spatial partitioning

  The argmin surface = layout switching boundary.
  If the boundary is smooth -> Layout Selection Invariant confirmed.
  If jagged -> the invariant is just a classification, not a law.

Sweep grid:
  I_values = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]   (8 points)
  M_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]               (6 points)
  p_values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]         (7 points)
  Total: 8 x 6 x 7 = 336 grid points x 10 seeds = 3360 trials

Chunked execution:
  python exp10k_cost_surface.py --chunk 0 --n_chunks 10 --output results/chunk_0.json
  python exp10k_cost_surface.py --merge --output results/merged.json

Outputs (in results/ subdirectory):
  exp10k_summary.json        -- all data + verdicts
  exp10k_report.md           -- human-readable report
  exp10k_cost_surface.png    -- cost heatmaps sliced at different p values
  exp10k_winners.png         -- winner layout map sliced at different p values
"""

import argparse
import itertools
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp_sparse
import scipy.sparse.csgraph as csgraph
# kendalltau removed — M now computed via spectral gap λ₂

import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

# =====================================================================
# Configuration
# =====================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

N = 1024            # number of nodes per synthetic space
FEAT_DIM = 16       # feature dimension (16 x float32 = 64 bytes per node)
SEED_BASE = 42
N_SEEDS = 10
N_WARMUP = 5
N_REPEAT = 20
BLOCK_SIZE = 8      # for D_blocked

RESULTS_DIR = Path(__file__).parent / "results"

I_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
M_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
P_VALUES = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]

LAYOUT_NAMES = ["A_bitset", "D_direct", "D_blocked"]

# =====================================================================
# Helpers
# =====================================================================

def _sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)


def _timed_runs(func, n_warmup=N_WARMUP, n_repeat=N_REPEAT):
    """Return array of wall-clock seconds for n_repeat runs after warmup."""
    for _ in range(n_warmup):
        func()
        _sync()
    times = []
    for _ in range(n_repeat):
        _sync()
        t0 = time.perf_counter()
        func()
        _sync()
        dt = time.perf_counter() - t0
        times.append(dt)
    return np.array(times)


def _measure_memory(build_fn, compute_fn):
    """Build layout, measure resident; run compute, measure peak.

    Returns dict with resident_bytes, peak_vram_bytes.
    """
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(DEVICE)

    state = build_fn()
    _sync()

    if DEVICE.type == "cuda":
        resident = torch.cuda.memory_allocated(DEVICE)
        torch.cuda.reset_peak_memory_stats(DEVICE)
        compute_fn(state)
        _sync()
        peak = torch.cuda.max_memory_allocated(DEVICE)
    else:
        resident = 0
        compute_fn(state)
        peak = 0

    return {"resident_bytes": int(resident), "peak_vram_bytes": int(peak)}


# =====================================================================
# 1. Space Generator
# =====================================================================

def _make_grid_graph(side):
    """Create a 2D grid graph as a CSR adjacency matrix. N = side*side."""
    n = side * side
    rows, cols = [], []
    for r in range(side):
        for c in range(side):
            node = r * side + c
            if c + 1 < side:
                rows.extend([node, node + 1])
                cols.extend([node + 1, node])
            if r + 1 < side:
                rows.extend([node, node + side])
                cols.extend([node + side, node])
    data = np.ones(len(rows), dtype=np.float32)
    adj = sp_sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    return adj


def _rewire_edges(adj, rewire_prob, rng):
    """Rewire edges of a CSR adjacency matrix with given probability.

    For each edge (u, v), with probability rewire_prob, reconnect v to a
    random node w (avoiding self-loops and duplicates).
    """
    n = adj.shape[0]
    coo = sp_sparse.triu(adj).tocoo()
    rows = coo.row.copy()
    cols = coo.col.copy()

    for i in range(len(rows)):
        if rng.random() < rewire_prob:
            u = rows[i]
            # Pick a random target that is not u
            w = rng.integers(0, n)
            while w == u:
                w = rng.integers(0, n)
            cols[i] = w

    # Rebuild symmetric adjacency
    all_rows = np.concatenate([rows, cols])
    all_cols = np.concatenate([cols, rows])
    data = np.ones(len(all_rows), dtype=np.float32)
    adj_new = sp_sparse.csr_matrix((data, (all_rows, all_cols)), shape=(n, n))
    # Remove self-loops and duplicates
    adj_new.setdiag(0)
    adj_new.eliminate_zeros()
    adj_new.data[:] = 1.0
    return adj_new


def _ba_graph(n, m, rng):
    """Barabasi-Albert preferential attachment graph.

    m = number of edges to attach from each new node.
    Returns CSR adjacency.
    """
    if HAS_NX:
        G = nx.barabasi_albert_graph(n, m, seed=int(rng.integers(0, 2**31)))
        adj = nx.to_scipy_sparse_array(G, format="csr")
        return adj.astype(np.float32)

    # Fallback: manual BA
    # Start with m+1 fully connected nodes
    targets = list(range(m + 1))
    edges = []
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            edges.append((i, j))

    degree = np.zeros(n, dtype=np.float64)
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1

    for new_node in range(m + 1, n):
        # Preferential attachment: probability proportional to degree
        total_deg = degree[:new_node].sum()
        if total_deg == 0:
            probs = np.ones(new_node) / new_node
        else:
            probs = degree[:new_node] / total_deg
        chosen = rng.choice(new_node, size=min(m, new_node), replace=False, p=probs)
        for t in chosen:
            edges.append((new_node, t))
            degree[new_node] += 1
            degree[t] += 1

    rows = np.array([e[0] for e in edges] + [e[1] for e in edges], dtype=np.int32)
    cols = np.array([e[1] for e in edges] + [e[0] for e in edges], dtype=np.int32)
    data = np.ones(len(rows), dtype=np.float32)
    adj = sp_sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    adj.setdiag(0)
    adj.eliminate_zeros()
    adj.data[:] = 1.0
    return adj


def _er_graph(n, p_edge, rng):
    """Erdos-Renyi random graph. Returns CSR adjacency."""
    if HAS_NX:
        G = nx.erdos_renyi_graph(n, p_edge, seed=int(rng.integers(0, 2**31)))
        adj = nx.to_scipy_sparse_array(G, format="csr")
        return adj.astype(np.float32)

    # Fallback: manual
    rows, cols = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p_edge:
                rows.extend([i, j])
                cols.extend([j, i])
    if len(rows) == 0:
        # Ensure at least a connected pair
        rows, cols = [0, 1], [1, 0]
    data = np.ones(len(rows), dtype=np.float32)
    adj = sp_sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    adj.setdiag(0)
    adj.eliminate_zeros()
    adj.data[:] = 1.0
    return adj


def _clustered_hubs_graph(N, n_clusters, hub_m, rng):
    """Graph with local hubs: high I (degree entropy) + high M (locality).

    Creates n_clusters isolated sub-graphs, each with BA-like preferential
    attachment internally. Hubs are powerful but local (feudal lords).
    Clusters connected by sparse bridges.

    Returns CSR adjacency with nodes ordered cluster-by-cluster (natural locality).
    """
    cluster_size = N // n_clusters
    rows, cols = [], []
    offset = 0

    for c in range(n_clusters):
        sz = cluster_size if c < n_clusters - 1 else N - offset
        # BA graph within cluster
        m = min(hub_m, sz - 1)
        if m < 1:
            m = 1
        # Manual BA within cluster
        targets = list(range(min(m + 1, sz)))
        edges_local = []
        for i in range(len(targets)):
            for j in range(i + 1, len(targets)):
                edges_local.append((i, j))
        degree = np.zeros(sz, dtype=np.float64)
        for u, v in edges_local:
            degree[u] += 1
            degree[v] += 1
        for new_node in range(len(targets), sz):
            total_deg = degree[:new_node].sum()
            if total_deg == 0:
                probs = np.ones(new_node) / new_node
            else:
                probs = degree[:new_node] / total_deg
            chosen = rng.choice(new_node, size=min(m, new_node), replace=False, p=probs)
            for t in chosen:
                edges_local.append((new_node, t))
                degree[new_node] += 1
                degree[t] += 1

        # Shift to global indices
        for u, v in edges_local:
            rows.extend([u + offset, v + offset])
            cols.extend([v + offset, u + offset])

        offset += sz

    # Add sparse bridges between adjacent clusters (1-2 edges each)
    offset = 0
    for c in range(n_clusters - 1):
        sz = cluster_size
        next_offset = offset + sz
        # Bridge: last node of cluster c -> first node of cluster c+1
        bridge_src = offset + rng.integers(0, sz)
        next_sz = cluster_size if c + 1 < n_clusters - 1 else N - next_offset
        bridge_dst = next_offset + rng.integers(0, next_sz)
        rows.extend([bridge_src, bridge_dst])
        cols.extend([bridge_dst, bridge_src])
        offset = next_offset

    data = np.ones(len(rows), dtype=np.float32)
    adj = sp_sparse.csr_matrix((data, (rows, cols)), shape=(N, N))
    adj.setdiag(0)
    adj.eliminate_zeros()
    if adj.nnz > 0:
        adj.data[:] = 1.0
    return adj


def generate_space(I_target, M_target, p_target, N=1024, seed=42):
    """Generate a space with approximately the target (I, M, p) values.

    Four quadrants of (I, M):
      I low  + M high = regular grid (uniform degree, spatial locality)
      I low  + M low  = Erdos-Renyi (uniform degree, no locality)
      I high + M low  = Barabasi-Albert (hub degree, no locality)
      I high + M high = clustered hubs (hub degree, local hubs)

    M is controlled through TOPOLOGY, not through ordering perturbation.
    RCM ordering is applied to all graphs to maximize whatever locality exists.

    Returns: adj, node_features, active_mask, actual_I, actual_M, actual_p
    """
    rng = np.random.default_rng(seed)
    side = int(math.isqrt(N))

    # -----------------------------------------------------------------
    # Step 1: Select topology based on (I_target, M_target) quadrant
    # -----------------------------------------------------------------
    I_high = I_target >= 1.5
    M_high = M_target >= 0.5

    if not I_high and M_high:
        # Quadrant: low I, high M -> regular grid (+ light rewiring for I)
        assert side * side == N, f"N={N} must be perfect square for grid"
        adj = _make_grid_graph(side)
        if I_target > 0.3:
            rewire_prob = min(0.15, I_target / 6.0)
            adj = _rewire_edges(adj, rewire_prob, rng)

    elif not I_high and not M_high:
        # Quadrant: low I, low M -> Erdos-Renyi (uniform degree, no locality)
        mean_deg = 4.0 + I_target * 2.0
        p_edge = mean_deg / N
        adj = _er_graph(N, p_edge, rng)

    elif I_high and not M_high:
        # Quadrant: high I, low M -> Barabasi-Albert (global hubs)
        m = max(2, int(1 + (I_target - 1.5) * 2))
        adj = _ba_graph(N, m, rng)

    else:
        # Quadrant: high I, high M -> clustered local hubs (feudal lords)
        n_clusters = max(4, int(8 * M_target))
        hub_m = max(2, int(1 + (I_target - 1.5) * 2))
        adj = _clustered_hubs_graph(N, n_clusters, hub_m, rng)

    # -----------------------------------------------------------------
    # Step 2: Apply RCM ordering to maximize locality (for ALL topologies)
    # -----------------------------------------------------------------
    try:
        rcm_order = csgraph.reverse_cuthill_mckee(adj)
    except Exception:
        rcm_order = np.arange(N)

    # Permute adjacency to RCM order
    inv_order = np.empty(N, dtype=np.int64)
    inv_order[rcm_order] = np.arange(N)
    perm_mat = sp_sparse.csr_matrix(
        (np.ones(N, dtype=np.float32), (np.arange(N), rcm_order)),
        shape=(N, N),
    )
    adj = perm_mat @ adj @ perm_mat.T
    adj.setdiag(0)
    adj.eliminate_zeros()
    if adj.nnz > 0:
        adj.data[:] = 1.0

    # -----------------------------------------------------------------
    # Step 3: Generate active mask targeting p (clustered BFS activation)
    # -----------------------------------------------------------------
    k = max(1, int(N * p_target))
    bfs_seed = rng.integers(0, N)
    visited = np.zeros(N, dtype=bool)
    queue = [bfs_seed]
    visited[bfs_seed] = True
    bfs_order = [bfs_seed]
    head = 0
    while head < len(queue) and len(bfs_order) < k:
        node = queue[head]
        head += 1
        start, end = adj.indptr[node], adj.indptr[node + 1]
        neighbors = adj.indices[start:end]
        rng.shuffle(neighbors)
        for nb in neighbors:
            if not visited[nb]:
                visited[nb] = True
                queue.append(nb)
                bfs_order.append(nb)
                if len(bfs_order) >= k:
                    break

    if len(bfs_order) < k:
        remaining = np.where(~visited)[0]
        need = k - len(bfs_order)
        if len(remaining) >= need:
            extra = rng.choice(remaining, size=need, replace=False)
        else:
            extra = remaining
        bfs_order.extend(extra.tolist())

    active_mask = np.zeros(N, dtype=bool)
    active_mask[bfs_order[:k]] = True

    # -----------------------------------------------------------------
    # Step 4: Generate node features
    # -----------------------------------------------------------------
    torch.manual_seed(seed)
    node_features = torch.randn(N, FEAT_DIM, dtype=DTYPE, device=DEVICE)

    # -----------------------------------------------------------------
    # Step 5: Compute actual metrics
    # -----------------------------------------------------------------
    actual_I = compute_I(adj)
    actual_M = compute_M(adj)
    actual_p = compute_p(active_mask)

    return adj, node_features, active_mask, actual_I, actual_M, actual_p


# =====================================================================
# 2. Metrics Computation
# =====================================================================

def compute_I(adj):
    """Degree entropy H(D) = -sum P(k) log P(k)).

    Parameters
    ----------
    adj : scipy.sparse.csr_matrix

    Returns
    -------
    float : degree entropy in nats.
    """
    degrees = np.diff(adj.indptr)
    if len(degrees) == 0:
        return 0.0
    counts = Counter(degrees)
    total = sum(counts.values())
    entropy = 0.0
    for k, cnt in counts.items():
        if cnt > 0 and k > 0:
            pk = cnt / total
            entropy -= pk * math.log(pk)
    return entropy


def compute_M(adj):
    """Algebraic connectivity λ₂ of the graph Laplacian.

    λ₂ = second smallest eigenvalue of L = D - A (Fiedler value).
    Measures how strongly the graph resists being cut and laid out linearly.

    - λ₂ → 0: graph folds neatly into 1D memory (grid, tree)
    - λ₂ large: expander graph, 1D embedding destroys all locality

    We return normalized_lambda2 = λ₂ / d_max to make it comparable
    across graphs with different mean degree.

    For layout selection, LOWER λ₂ = BETTER locality = HIGHER M.
    So we return M = 1 / (1 + normalized_lambda2) to keep M ∈ (0, 1]
    with M=1 meaning perfect locality.
    """
    from scipy.sparse.linalg import eigsh

    n = adj.shape[0]
    if n < 3 or adj.nnz == 0:
        return 0.5

    # Build Laplacian L = D - A
    degrees = np.diff(adj.indptr).astype(np.float64)
    d_max = degrees.max()
    if d_max == 0:
        return 0.5

    L = sp_sparse.diags(degrees) - adj.astype(np.float64)

    try:
        # k=2: we need λ₁ (=0 for connected) and λ₂
        eigenvalues = eigsh(L, k=2, which='SM', return_eigenvectors=False)
        lambda2 = float(max(eigenvalues))  # second smallest
        if lambda2 < 1e-10:
            lambda2 = 0.0
    except Exception:
        return 0.5

    # Normalize and invert: low λ₂ = high M
    normalized = lambda2 / d_max
    M = 1.0 / (1.0 + normalized * 10.0)  # scale factor for sensitivity
    return float(M)


def compute_p(active_mask):
    """Occupancy = fraction of active nodes."""
    return float(active_mask.sum()) / len(active_mask)


# =====================================================================
# 3. Layout Benchmark
# =====================================================================

def _compute_operator(features, adj, active_ids):
    """Operator = matmul: for each active node, compute mean(neighbor features)
    then matmul with a weight matrix.

    This is the compute kernel that all layouts must implement equivalently.
    """
    # Weight matrix (fixed, seeded)
    W = torch.ones(FEAT_DIM, FEAT_DIM, dtype=DTYPE, device=DEVICE) / FEAT_DIM
    return W  # Returned so layouts can reuse it


# --- A_bitset layout ---

def build_A_bitset(adj, node_features, active_mask):
    """A_bitset: full tensor + bitmask.

    Returns state dict for compute.
    """
    mask_t = torch.from_numpy(active_mask).to(DEVICE)
    indices = torch.from_numpy(adj.indices.astype(np.int64)).to(DEVICE)
    indptr = torch.from_numpy(adj.indptr.astype(np.int64)).to(DEVICE)
    W = torch.ones(FEAT_DIM, FEAT_DIM, dtype=DTYPE, device=DEVICE) / FEAT_DIM
    return {
        "data": node_features,
        "mask": mask_t,
        "indices": indices,
        "indptr": indptr,
        "W": W,
    }


def compute_A_bitset(state):
    """A_bitset compute: element-wise on full tensor, mask result."""
    data = state["data"]
    mask = state["mask"]
    indices = state["indices"]
    indptr = state["indptr"]
    W = state["W"]

    active_ids = torch.where(mask)[0]
    if len(active_ids) == 0:
        return data

    result = data.clone()
    for idx in active_ids:
        i = idx.item()
        start, end = indptr[i].item(), indptr[i + 1].item()
        if start == end:
            continue
        nb_ids = indices[start:end]
        nb_feats = data[nb_ids]  # [deg, FEAT_DIM]
        agg = nb_feats.mean(dim=0)  # [FEAT_DIM]
        result[i] = agg @ W

    return result


# --- D_direct layout ---

def build_D_direct(adj, node_features, active_mask):
    """D_direct: packed array + tile_map.

    packed_data[k, FEAT_DIM] for k active nodes.
    tile_map[N] -> slot index (-1 = inactive).
    """
    active_ids = np.where(active_mask)[0].astype(np.int32)
    k = len(active_ids)

    tile_map = torch.full((N,), -1, dtype=torch.int32, device=DEVICE)
    if k > 0:
        tile_map[torch.from_numpy(active_ids).to(DEVICE).long()] = \
            torch.arange(k, dtype=torch.int32, device=DEVICE)

    packed_data = node_features[active_ids] if k > 0 else \
        torch.zeros(1, FEAT_DIM, dtype=DTYPE, device=DEVICE)

    # Precompute neighbor info for active nodes
    nb_slots_list = []
    nb_counts = []
    for aid in active_ids:
        start, end = adj.indptr[aid], adj.indptr[aid + 1]
        nbs = adj.indices[start:end]
        # Map through tile_map: only keep neighbors that are active
        nb_slots = []
        for nb in nbs:
            slot = tile_map[nb].item()
            if slot >= 0:
                nb_slots.append(slot)
        nb_slots_list.append(nb_slots)
        nb_counts.append(len(nb_slots))

    # Flatten into padded tensor for vectorized gather
    max_deg = max(nb_counts) if nb_counts else 1
    max_deg = max(max_deg, 1)
    nb_indices = torch.zeros(k, max_deg, dtype=torch.int64, device=DEVICE)
    nb_mask = torch.zeros(k, max_deg, dtype=torch.bool, device=DEVICE)
    for i, slots in enumerate(nb_slots_list):
        for j, s in enumerate(slots):
            nb_indices[i, j] = s
            nb_mask[i, j] = True

    W = torch.ones(FEAT_DIM, FEAT_DIM, dtype=DTYPE, device=DEVICE) / FEAT_DIM

    return {
        "packed_data": packed_data,
        "tile_map": tile_map,
        "nb_indices": nb_indices,
        "nb_mask": nb_mask,
        "W": W,
        "k": k,
    }


def compute_D_direct(state):
    """D_direct compute: gather neighbors via tile_map, op on packed, scatter."""
    packed = state["packed_data"]
    nb_idx = state["nb_indices"]
    nb_mask = state["nb_mask"]
    W = state["W"]
    k = state["k"]

    if k == 0:
        return packed

    # Gather neighbor features: [k, max_deg, FEAT_DIM]
    nb_feats = packed[nb_idx]
    # Zero out inactive neighbors
    nb_feats = nb_feats * nb_mask.unsqueeze(-1).float()

    # Mean aggregation (count only valid neighbors)
    valid_counts = nb_mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
    agg = nb_feats.sum(dim=1) / valid_counts  # [k, FEAT_DIM]

    # Matmul
    result = agg @ W  # [k, FEAT_DIM]
    return result


# --- D_blocked layout ---

def _spatial_partition(adj, active_mask, block_size):
    """Partition active nodes into fixed-size blocks using BFS ordering.

    Returns:
        block_ids: np.ndarray[N] with block assignment (-1 = inactive)
        n_blocks: int
    """
    active_ids = np.where(active_mask)[0]
    k = len(active_ids)
    if k == 0:
        return np.full(adj.shape[0], -1, dtype=np.int32), 0

    # BFS from first active node for spatial ordering
    n = adj.shape[0]
    visited = np.zeros(n, dtype=bool)
    ordered = []
    start = active_ids[0]
    queue = [start]
    visited[start] = True

    head = 0
    while head < len(queue):
        node = queue[head]
        head += 1
        if active_mask[node]:
            ordered.append(node)
        s, e = adj.indptr[node], adj.indptr[node + 1]
        for nb in adj.indices[s:e]:
            if not visited[nb]:
                visited[nb] = True
                queue.append(nb)

    # Add any unreached active nodes
    for aid in active_ids:
        if aid not in set(ordered):
            ordered.append(aid)

    # Assign to blocks sequentially
    n_blocks = math.ceil(len(ordered) / block_size)
    block_ids = np.full(n, -1, dtype=np.int32)
    for i, node in enumerate(ordered):
        block_ids[node] = i // block_size

    return block_ids, n_blocks


def build_D_blocked(adj, node_features, active_mask):
    """D_blocked: block-structured storage with spatial partitioning.

    packed_blocks[n_blocks, block_size, FEAT_DIM]
    block_map[n_blocks_total] -> slot (-1 = empty)
    node_to_block[N] -> (block_id, local_offset)
    """
    block_ids, n_blocks = _spatial_partition(adj, active_mask, BLOCK_SIZE)
    if n_blocks == 0:
        W = torch.ones(FEAT_DIM, FEAT_DIM, dtype=DTYPE, device=DEVICE) / FEAT_DIM
        return {
            "packed_blocks": torch.zeros(1, BLOCK_SIZE, FEAT_DIM,
                                         dtype=DTYPE, device=DEVICE),
            "node_block_id": torch.zeros(N, dtype=torch.int32, device=DEVICE),
            "node_local_offset": torch.zeros(N, dtype=torch.int32, device=DEVICE),
            "nb_block_ids": torch.zeros(1, 1, dtype=torch.int64, device=DEVICE),
            "nb_offsets": torch.zeros(1, 1, dtype=torch.int64, device=DEVICE),
            "nb_mask": torch.zeros(1, 1, dtype=torch.bool, device=DEVICE),
            "W": W,
            "n_blocks": 0,
            "k": 0,
        }

    active_ids = np.where(active_mask)[0]
    k = len(active_ids)

    # Build block contents
    # For each block, track which nodes belong and their local offset
    block_contents = [[] for _ in range(n_blocks)]
    node_local_offset = np.zeros(N, dtype=np.int32)
    for node in active_ids:
        bid = block_ids[node]
        local_off = len(block_contents[bid])
        block_contents[bid].append(node)
        node_local_offset[node] = local_off

    # Pack blocks
    packed_blocks = torch.zeros(n_blocks, BLOCK_SIZE, FEAT_DIM,
                                dtype=DTYPE, device=DEVICE)
    for bid, members in enumerate(block_contents):
        for local_off, node in enumerate(members):
            packed_blocks[bid, local_off] = node_features[node]

    node_block_id_t = torch.from_numpy(block_ids).to(dtype=torch.int32,
                                                      device=DEVICE)
    node_local_off_t = torch.from_numpy(node_local_offset).to(
        dtype=torch.int32, device=DEVICE)

    # Precompute neighbor info for active nodes (block-structured)
    nb_bids_list = []
    nb_offs_list = []
    nb_counts = []
    for aid in active_ids:
        start, end = adj.indptr[aid], adj.indptr[aid + 1]
        nbs = adj.indices[start:end]
        bids_local = []
        offs_local = []
        for nb in nbs:
            bid = block_ids[nb]
            if bid >= 0:
                bids_local.append(bid)
                offs_local.append(node_local_offset[nb])
        nb_bids_list.append(bids_local)
        nb_offs_list.append(offs_local)
        nb_counts.append(len(bids_local))

    max_deg = max(nb_counts) if nb_counts else 1
    max_deg = max(max_deg, 1)
    nb_bids = torch.zeros(k, max_deg, dtype=torch.int64, device=DEVICE)
    nb_offs = torch.zeros(k, max_deg, dtype=torch.int64, device=DEVICE)
    nb_mask = torch.zeros(k, max_deg, dtype=torch.bool, device=DEVICE)
    for i in range(k):
        for j in range(nb_counts[i]):
            nb_bids[i, j] = nb_bids_list[i][j]
            nb_offs[i, j] = nb_offs_list[i][j]
            nb_mask[i, j] = True

    W = torch.ones(FEAT_DIM, FEAT_DIM, dtype=DTYPE, device=DEVICE) / FEAT_DIM

    return {
        "packed_blocks": packed_blocks,
        "node_block_id": node_block_id_t,
        "node_local_offset": node_local_off_t,
        "nb_block_ids": nb_bids,
        "nb_offsets": nb_offs,
        "nb_mask": nb_mask,
        "W": W,
        "n_blocks": n_blocks,
        "k": k,
    }


def compute_D_blocked(state):
    """D_blocked compute: block-structured gather/scatter."""
    packed = state["packed_blocks"]
    nb_bids = state["nb_block_ids"]
    nb_offs = state["nb_offsets"]
    nb_mask = state["nb_mask"]
    W = state["W"]
    k = state["k"]

    if k == 0:
        return packed

    # Gather neighbor features through block addressing:
    # packed[nb_bids, nb_offs] -> [k, max_deg, FEAT_DIM]
    nb_feats = packed[nb_bids, nb_offs]  # [k, max_deg, FEAT_DIM]
    nb_feats = nb_feats * nb_mask.unsqueeze(-1).float()

    valid_counts = nb_mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
    agg = nb_feats.sum(dim=1) / valid_counts  # [k, FEAT_DIM]

    result = agg @ W  # [k, FEAT_DIM]
    return result


# =====================================================================
# 4. Single Grid Point Benchmark
# =====================================================================

def benchmark_point(I_target, M_target, p_target, seed):
    """Benchmark all three layouts at a single (I, M, p, seed) point.

    Returns dict with costs per layout and actual metrics.
    """
    adj, features, mask, actual_I, actual_M, actual_p = \
        generate_space(I_target, M_target, p_target, N=N, seed=seed)

    results = {
        "I_target": I_target,
        "M_target": M_target,
        "p_target": p_target,
        "seed": seed,
        "actual_I": round(actual_I, 4),
        "actual_M": round(actual_M, 4),
        "actual_p": round(actual_p, 4),
        "layouts": {},
    }

    layout_builders = [
        ("A_bitset", build_A_bitset, compute_A_bitset),
        ("D_direct", build_D_direct, compute_D_direct),
        ("D_blocked", build_D_blocked, compute_D_blocked),
    ]

    for name, build_fn, compute_fn in layout_builders:
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(DEVICE)

        # Build
        state = build_fn(adj, features, mask)
        _sync()

        if DEVICE.type == "cuda":
            resident = torch.cuda.memory_allocated(DEVICE)
            torch.cuda.reset_peak_memory_stats(DEVICE)
        else:
            resident = 0

        # Timed runs
        times = _timed_runs(lambda: compute_fn(state))
        wall_us = float(np.median(times)) * 1e6

        if DEVICE.type == "cuda":
            peak = torch.cuda.max_memory_allocated(DEVICE)
        else:
            peak = 0

        results["layouts"][name] = {
            "resident_bytes": int(resident),
            "wall_clock_us": round(wall_us, 2),
            "peak_vram_bytes": int(peak),
        }

        # Cleanup
        del state
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # Cleanup space
    del adj, features, mask

    return results


# =====================================================================
# 5. Sweep Grid + Chunked Execution
# =====================================================================

def build_grid():
    """Build the full (I, M, p) sweep grid as a list of (I, M, p) tuples."""
    return list(itertools.product(I_VALUES, M_VALUES, P_VALUES))


def run_chunk(chunk_idx, n_chunks, output_path):
    """Run a chunk of the sweep grid.

    Splits the grid into n_chunks pieces, runs chunk_idx, saves results
    incrementally to output_path.
    """
    grid = build_grid()
    total = len(grid)
    chunk_size = math.ceil(total / n_chunks)
    start = chunk_idx * chunk_size
    end = min(start + chunk_size, total)
    my_points = grid[start:end]

    print(f"Chunk {chunk_idx}/{n_chunks}: grid points [{start}, {end}) "
          f"= {len(my_points)} points x {N_SEEDS} seeds")

    results = []
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    for pt_idx, (I_t, M_t, p_t) in enumerate(my_points):
        for seed_idx in range(N_SEEDS):
            seed = SEED_BASE + seed_idx
            try:
                rec = benchmark_point(I_t, M_t, p_t, seed)
                results.append(rec)
            except Exception as e:
                print(f"  ERROR at I={I_t} M={M_t} p={p_t} seed={seed}: {e}")
                results.append({
                    "I_target": I_t, "M_target": M_t, "p_target": p_t,
                    "seed": seed, "error": str(e),
                })

            # Progress
            done = pt_idx * N_SEEDS + seed_idx + 1
            total_trials = len(my_points) * N_SEEDS
            if done % 10 == 0 or done == total_trials:
                print(f"  [{done}/{total_trials}] I={I_t} M={M_t} p={p_t} "
                      f"seed={seed}")

        # Save incrementally after each grid point (all seeds)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=1)

    print(f"Chunk {chunk_idx} done: {len(results)} trials -> {output_file}")
    return results


# =====================================================================
# 6. Analysis (--merge mode)
# =====================================================================

def merge_and_analyze(output_path):
    """Merge all chunk files and produce analysis outputs."""
    results_dir = RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    # Find all chunk files
    chunk_files = sorted(results_dir.glob("chunk_*.json"))
    if not chunk_files:
        print("ERROR: No chunk files found in", results_dir)
        return

    # Load and merge
    all_results = []
    for cf in chunk_files:
        with open(cf) as f:
            data = json.load(f)
        all_results.extend(data)
        print(f"  Loaded {len(data)} trials from {cf.name}")

    print(f"Total trials: {len(all_results)}")

    # Filter out errors
    valid = [r for r in all_results if "error" not in r]
    errors = [r for r in all_results if "error" in r]
    if errors:
        print(f"  {len(errors)} trials had errors (skipped)")

    # -----------------------------------------------------------------
    # Aggregate: for each (I, M, p) point, median cost across seeds
    # -----------------------------------------------------------------
    grid_data = {}  # (I, M, p) -> {layout: median_cost}

    for r in valid:
        key = (r["I_target"], r["M_target"], r["p_target"])
        if key not in grid_data:
            grid_data[key] = {name: [] for name in LAYOUT_NAMES}

        for name in LAYOUT_NAMES:
            if name in r["layouts"]:
                cost = r["layouts"][name]["wall_clock_us"]
                grid_data[key][name].append(cost)

    # Compute medians and find winners
    grid_summary = {}
    for key, costs_by_layout in grid_data.items():
        entry = {}
        best_cost = float("inf")
        best_layout = None
        for name in LAYOUT_NAMES:
            vals = costs_by_layout[name]
            if vals:
                med = float(np.median(vals))
                entry[name] = med
                if med < best_cost:
                    best_cost = med
                    best_layout = name
            else:
                entry[name] = float("inf")
        entry["winner"] = best_layout
        entry["winner_cost"] = best_cost
        grid_summary[str(key)] = entry

    # -----------------------------------------------------------------
    # Smoothness analysis
    # -----------------------------------------------------------------
    grid = build_grid()
    grid_set = set(grid)

    # Build a map from (I, M, p) -> winner
    winner_map = {}
    for key, entry in grid_summary.items():
        # Parse key string back to tuple
        tup = eval(key)
        winner_map[tup] = entry["winner"]

    # Count adjacent pairs with same winner
    same_winner = 0
    total_pairs = 0

    for I_t, M_t, p_t in grid:
        if (I_t, M_t, p_t) not in winner_map:
            continue
        w = winner_map[(I_t, M_t, p_t)]

        # Check neighbors along each axis
        i_idx = I_VALUES.index(I_t) if I_t in I_VALUES else -1
        m_idx = M_VALUES.index(M_t) if M_t in M_VALUES else -1
        p_idx = P_VALUES.index(p_t) if p_t in P_VALUES else -1

        neighbors = []
        if i_idx + 1 < len(I_VALUES):
            neighbors.append((I_VALUES[i_idx + 1], M_t, p_t))
        if m_idx + 1 < len(M_VALUES):
            neighbors.append((I_t, M_VALUES[m_idx + 1], p_t))
        if p_idx + 1 < len(P_VALUES):
            neighbors.append((I_t, M_t, P_VALUES[p_idx + 1]))

        for nb in neighbors:
            if nb in winner_map:
                total_pairs += 1
                if winner_map[nb] == w:
                    same_winner += 1

    smoothness_score = same_winner / total_pairs if total_pairs > 0 else 0.0
    print(f"\nBoundary smoothness score: {smoothness_score:.4f} "
          f"({same_winner}/{total_pairs} adjacent pairs agree)")

    if smoothness_score > 0.85:
        verdict = "SMOOTH -- Layout Selection Invariant confirmed as a law."
    elif smoothness_score > 0.70:
        verdict = ("MODERATE -- Boundaries exist but are somewhat noisy. "
                    "The invariant is a useful heuristic but not a strict law.")
    else:
        verdict = ("JAGGED -- The cost surface is noisy. Layout selection "
                    "is a classification problem, not a deterministic law.")

    # -----------------------------------------------------------------
    # Decision boundaries: find (I, M, p) where layout switches
    # -----------------------------------------------------------------
    boundary_points = []
    for I_t, M_t, p_t in grid:
        if (I_t, M_t, p_t) not in winner_map:
            continue
        w = winner_map[(I_t, M_t, p_t)]

        i_idx = I_VALUES.index(I_t) if I_t in I_VALUES else -1

        neighbors = []
        if i_idx + 1 < len(I_VALUES):
            neighbors.append((I_VALUES[i_idx + 1], M_t, p_t))
        m_idx = M_VALUES.index(M_t) if M_t in M_VALUES else -1
        if m_idx + 1 < len(M_VALUES):
            neighbors.append((I_t, M_VALUES[m_idx + 1], p_t))
        p_idx = P_VALUES.index(p_t) if p_t in P_VALUES else -1
        if p_idx + 1 < len(P_VALUES):
            neighbors.append((I_t, M_t, P_VALUES[p_idx + 1]))

        for nb in neighbors:
            if nb in winner_map and winner_map[nb] != w:
                boundary_points.append({
                    "from": (I_t, M_t, p_t),
                    "to": nb,
                    "from_winner": w,
                    "to_winner": winner_map[nb],
                })

    # -----------------------------------------------------------------
    # Count overall wins
    # -----------------------------------------------------------------
    win_counts = Counter(winner_map.values())
    print(f"\nLayout win counts: {dict(win_counts)}")
    print(f"Boundary transitions: {len(boundary_points)}")
    print(f"Verdict: {verdict}")

    # -----------------------------------------------------------------
    # Save summary JSON
    # -----------------------------------------------------------------
    summary = {
        "experiment": "exp10k_cost_surface",
        "device": str(DEVICE),
        "N": N,
        "feat_dim": FEAT_DIM,
        "n_seeds": N_SEEDS,
        "grid_shape": {
            "I_values": I_VALUES,
            "M_values": M_VALUES,
            "p_values": P_VALUES,
        },
        "total_trials": len(all_results),
        "valid_trials": len(valid),
        "error_trials": len(errors),
        "grid_summary": grid_summary,
        "smoothness_score": round(smoothness_score, 4),
        "same_winner_pairs": same_winner,
        "total_adjacent_pairs": total_pairs,
        "verdict": verdict,
        "win_counts": dict(win_counts),
        "n_boundary_transitions": len(boundary_points),
        "boundary_points": [
            {
                "from": list(bp["from"]),
                "to": list(bp["to"]),
                "from_winner": bp["from_winner"],
                "to_winner": bp["to_winner"],
            }
            for bp in boundary_points
        ],
    }

    summary_path = results_dir / "exp10k_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {summary_path}")

    # -----------------------------------------------------------------
    # Generate plots
    # -----------------------------------------------------------------
    _plot_cost_surface(grid_summary, results_dir)
    _plot_winners(winner_map, results_dir)

    # -----------------------------------------------------------------
    # Generate report
    # -----------------------------------------------------------------
    _write_report(summary, boundary_points, results_dir)


def _plot_cost_surface(grid_summary, results_dir):
    """Plot cost heatmaps sliced at different p values."""
    n_p = len(P_VALUES)
    n_cols = min(4, n_p)
    n_rows = math.ceil(n_p / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
                             squeeze=False)

    for layout_idx, layout_name in enumerate(LAYOUT_NAMES):
        fig_l, axes_l = plt.subplots(n_rows, n_cols,
                                     figsize=(5 * n_cols, 4 * n_rows),
                                     squeeze=False)

        for p_idx, p_val in enumerate(P_VALUES):
            row = p_idx // n_cols
            col = p_idx % n_cols
            ax = axes_l[row, col]

            # Build heatmap: I x M
            heatmap = np.full((len(I_VALUES), len(M_VALUES)), np.nan)
            for i, I_val in enumerate(I_VALUES):
                for j, M_val in enumerate(M_VALUES):
                    key = str((I_val, M_val, p_val))
                    if key in grid_summary:
                        heatmap[i, j] = grid_summary[key].get(layout_name,
                                                               float("nan"))

            im = ax.imshow(heatmap, aspect="auto", origin="lower",
                           cmap="viridis")
            ax.set_xticks(range(len(M_VALUES)))
            ax.set_xticklabels([f"{v:.1f}" for v in M_VALUES], fontsize=7)
            ax.set_yticks(range(len(I_VALUES)))
            ax.set_yticklabels([f"{v:.1f}" for v in I_VALUES], fontsize=7)
            ax.set_xlabel("M (metric gap)")
            ax.set_ylabel("I (degree entropy)")
            ax.set_title(f"p={p_val}")
            fig_l.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Hide unused subplots
        for idx in range(n_p, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes_l[row, col].set_visible(False)

        fig_l.suptitle(f"exp10k: Cost surface -- {layout_name} (wall_clock_us)",
                       fontsize=13)
        fig_l.tight_layout()
        fig_l.savefig(results_dir / f"exp10k_cost_{layout_name}.png", dpi=150)
        plt.close(fig_l)
        print(f"  Wrote exp10k_cost_{layout_name}.png")

    # Combined cost surface: for each p-slice, show min cost across layouts
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    for p_idx, p_val in enumerate(P_VALUES):
        row = p_idx // n_cols
        col = p_idx % n_cols
        ax = axes[row, col]

        heatmap = np.full((len(I_VALUES), len(M_VALUES)), np.nan)
        for i, I_val in enumerate(I_VALUES):
            for j, M_val in enumerate(M_VALUES):
                key = str((I_val, M_val, p_val))
                if key in grid_summary:
                    costs = [grid_summary[key].get(ln, float("inf"))
                             for ln in LAYOUT_NAMES]
                    heatmap[i, j] = min(costs)

        im = ax.imshow(heatmap, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xticks(range(len(M_VALUES)))
        ax.set_xticklabels([f"{v:.1f}" for v in M_VALUES], fontsize=7)
        ax.set_yticks(range(len(I_VALUES)))
        ax.set_yticklabels([f"{v:.1f}" for v in I_VALUES], fontsize=7)
        ax.set_xlabel("M (metric gap)")
        ax.set_ylabel("I (degree entropy)")
        ax.set_title(f"p={p_val}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for idx in range(n_p, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    fig.suptitle("exp10k: Minimum cost C(I,M,p) across layouts (wall_clock_us)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(results_dir / "exp10k_cost_surface.png", dpi=150)
    plt.close(fig)
    print(f"  Wrote exp10k_cost_surface.png")


def _plot_winners(winner_map, results_dir):
    """Plot winner layout map sliced at different p values."""
    layout_to_int = {name: i for i, name in enumerate(LAYOUT_NAMES)}
    cmap = ListedColormap(["#2196F3", "#FF9800", "#4CAF50"])  # blue, orange, green

    n_p = len(P_VALUES)
    n_cols = min(4, n_p)
    n_rows = math.ceil(n_p / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    for p_idx, p_val in enumerate(P_VALUES):
        row = p_idx // n_cols
        col = p_idx % n_cols
        ax = axes[row, col]

        heatmap = np.full((len(I_VALUES), len(M_VALUES)), np.nan)
        for i, I_val in enumerate(I_VALUES):
            for j, M_val in enumerate(M_VALUES):
                key = (I_val, M_val, p_val)
                if key in winner_map and winner_map[key] is not None:
                    heatmap[i, j] = layout_to_int[winner_map[key]]

        im = ax.imshow(heatmap, aspect="auto", origin="lower",
                       cmap=cmap, vmin=0, vmax=len(LAYOUT_NAMES) - 1)
        ax.set_xticks(range(len(M_VALUES)))
        ax.set_xticklabels([f"{v:.1f}" for v in M_VALUES], fontsize=7)
        ax.set_yticks(range(len(I_VALUES)))
        ax.set_yticklabels([f"{v:.1f}" for v in I_VALUES], fontsize=7)
        ax.set_xlabel("M (metric gap)")
        ax.set_ylabel("I (degree entropy)")
        ax.set_title(f"p={p_val}")

    for idx in range(n_p, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2196F3", label="A_bitset"),
        Patch(facecolor="#FF9800", label="D_direct"),
        Patch(facecolor="#4CAF50", label="D_blocked"),
    ]
    fig.legend(handles=legend_elements, loc="lower right", fontsize=10,
               ncol=3, framealpha=0.9)

    fig.suptitle("exp10k: Winner layout argmin C(I,M,p)", fontsize=13)
    fig.tight_layout()
    fig.savefig(results_dir / "exp10k_winners.png", dpi=150)
    plt.close(fig)
    print(f"  Wrote exp10k_winners.png")


def _write_report(summary, boundary_points, results_dir):
    """Write human-readable report."""
    lines = [
        "# exp10k Cost Surface Report",
        "",
        f"Device: {summary['device']}",
        f"N = {summary['N']}, feat_dim = {summary['feat_dim']}, "
        f"seeds = {summary['n_seeds']}",
        f"Total trials: {summary['valid_trials']} valid, "
        f"{summary['error_trials']} errors",
        "",
        "## Layout Win Counts",
        "",
    ]

    for name, count in sorted(summary["win_counts"].items()):
        total = sum(summary["win_counts"].values())
        lines.append(f"- {name}: {count}/{total} "
                      f"({count/total*100:.1f}%)")

    lines += [
        "",
        "## Smoothness Analysis",
        "",
        f"Boundary smoothness score: **{summary['smoothness_score']:.4f}**",
        f"Adjacent pairs agreeing: {summary['same_winner_pairs']}"
        f"/{summary['total_adjacent_pairs']}",
        f"Boundary transitions: {summary['n_boundary_transitions']}",
        "",
        f"**{summary['verdict']}**",
        "",
        "## Decision Boundaries",
        "",
        "Transitions between layouts at adjacent grid points:",
        "",
        "| From (I, M, p) | To (I, M, p) | From Layout | To Layout |",
        "|-----------------|--------------|-------------|-----------|",
    ]

    for bp in boundary_points[:50]:  # Limit to 50 for readability
        lines.append(
            f"| ({bp['from'][0]}, {bp['from'][1]}, {bp['from'][2]}) "
            f"| ({bp['to'][0]}, {bp['to'][1]}, {bp['to'][2]}) "
            f"| {bp['from_winner']} | {bp['to_winner']} |"
        )

    if len(boundary_points) > 50:
        lines.append(f"| ... | ... | ... | ... |")
        lines.append(f"| ({len(boundary_points)} total transitions) | | | |")

    lines += [
        "",
        "## Interpretation",
        "",
        "If smoothness > 0.85: the layout switching boundary is a smooth "
        "surface in (I, M, p) space. The Layout Selection Invariant holds as "
        "a deterministic law: given (I, M, p), the optimal layout is "
        "predictable.",
        "",
        "If smoothness 0.70-0.85: boundaries exist but are noisy. The "
        "invariant is a useful heuristic with some stochastic overlap zones.",
        "",
        "If smoothness < 0.70: the cost surface is chaotic. Layout selection "
        "is a classification problem requiring per-instance benchmarking, "
        "not a closed-form law.",
        "",
    ]

    report_path = results_dir / "exp10k_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote {report_path}")


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="exp10k: Cost surface C(I, M, p) for layout selection")
    parser.add_argument("--chunk", type=int, default=None,
                        help="Chunk index (0-based)")
    parser.add_argument("--n_chunks", type=int, default=10,
                        help="Total number of chunks")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for chunk results or merged output")
    parser.add_argument("--merge", action="store_true",
                        help="Merge all chunk files and produce analysis")
    args = parser.parse_args()

    if args.merge:
        output = args.output or str(RESULTS_DIR / "exp10k_summary.json")
        merge_and_analyze(output)
    elif args.chunk is not None:
        if args.output is None:
            args.output = str(RESULTS_DIR / f"chunk_{args.chunk}.json")
        run_chunk(args.chunk, args.n_chunks, args.output)
    else:
        # Run all chunks sequentially (single-machine mode)
        print(f"Running full sweep: {len(build_grid())} grid points x "
              f"{N_SEEDS} seeds = {len(build_grid()) * N_SEEDS} trials")
        print(f"Device: {DEVICE}")
        print()

        all_results = []
        grid = build_grid()
        total = len(grid) * N_SEEDS
        done = 0

        for I_t, M_t, p_t in grid:
            for seed_idx in range(N_SEEDS):
                seed = SEED_BASE + seed_idx
                try:
                    rec = benchmark_point(I_t, M_t, p_t, seed)
                    all_results.append(rec)
                except Exception as e:
                    print(f"  ERROR at I={I_t} M={M_t} p={p_t} "
                          f"seed={seed}: {e}")
                    all_results.append({
                        "I_target": I_t, "M_target": M_t, "p_target": p_t,
                        "seed": seed, "error": str(e),
                    })

                done += 1
                if done % 50 == 0 or done == total:
                    print(f"  [{done}/{total}] I={I_t} M={M_t} p={p_t}")

        # Save as single chunk then merge
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        chunk_path = RESULTS_DIR / "chunk_full.json"
        with open(chunk_path, "w") as f:
            json.dump(all_results, f, indent=1)
        print(f"\nSaved {len(all_results)} trials -> {chunk_path}")

        # Run analysis
        print("\n" + "=" * 72)
        print("Analysis")
        print("=" * 72)
        merge_and_analyze(str(RESULTS_DIR / "exp10k_summary.json"))

        print("\n" + "=" * 72)
        print("exp10k COMPLETE")
        print("=" * 72)


if __name__ == "__main__":
    main()
