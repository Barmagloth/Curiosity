#!/usr/bin/env python3
"""
Curiosity -- Exp10i: Block-Based Addressing for Irregular Graphs

Motivation:
  D_direct (packed tiles + tile_map) won for grids (exp10g/10h). For irregular
  graphs there is no natural tile grid. This experiment tests whether block-based
  addressing -- partitioning graph nodes into fixed-size blocks, with a block_map
  for slot lookup -- works as a foundation for graph-structured sparse storage.

  This is NOT a full sparse layout yet. It tests whether the blocking step is
  viable: can we partition nodes so that most edges stay within blocks
  (low cross_block_ratio) without too much padding waste?

Approach:
  1. Generate synthetic graphs (random_geometric, barabasi_albert, grid_graph)
  2. Partition nodes into fixed-size blocks using 3 strategies:
     - random_partition (baseline, worst case)
     - spatial_partition (Morton order for spatial graphs, BFS for non-spatial)
     - greedy_partition (BFS/DFS growth from random seeds)
  3. Compare graph_baseline (full arrays + CSR) vs D_graph_blocks (packed blocks
     + block_map + cross_block edges)

Candidates:
  graph_baseline:
    data[N, feat_dim] + active_mask[N] + CSR adjacency (indices, indptr)
    Compute: for each active node, mean of active neighbor features

  D_graph_blocks:
    packed_data[n_blocks, block_size, feat_dim] (padded)
    block_map[n_blocks_total] -> slot (int32, -1 = inactive)
    block_membership[N] -> (block_id, local_offset)
    cross_block_edges for inter-block neighbor access
    Compute: same aggregation through block addressing

Metrics:
  resident_bytes, peak_vram_bytes, wall_clock_us, build_cost_us
  cross_block_ratio: fraction of edges crossing block boundaries
  padding_waste: fraction of packed_data that is padding

Dual kill criteria:
  Contour A (architectural): D resident < baseline AND time overhead < 20%
                              AND cross_block_ratio < 0.5
  Contour B (operational):   D peak < baseline AND time overhead < 20%
  WARNING if padding_waste > 50%

Outputs (in results/ subdirectory):
  exp10i_summary.json           -- all data + verdicts
  exp10i_time_comparison.png    -- wall-clock comparison
  exp10i_memory_comparison.png  -- resident + peak VRAM comparison
  exp10i_blocking_quality.png   -- cross_block_ratio by partitioning method
  exp10i_report.md              -- human-readable report
"""

import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =====================================================================
# Configuration
# =====================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
SEED_BASE = 42
N_SEEDS = 10
N_WARMUP = 5
N_REPEAT = 20
KILL_THRESH = 0.20  # 20% overhead tolerance
CROSS_BLOCK_THRESH = 0.50
PADDING_WASTE_THRESH = 0.50
FEATURE_DIM = 8

RESULTS_DIR = Path(__file__).parent / "results"

N_NODES_LIST = [256, 1024, 4096, 16384]
SPARSITIES = [0.05, 0.1, 0.3, 0.5]
GRAPH_TYPES = ["random_geometric", "barabasi_albert", "grid_graph"]
BLOCK_SIZES = [8, 16, 32, 64]
PARTITION_METHODS = ["random_partition", "spatial_partition", "greedy_partition"]

CAND_COLORS = {
    "graph_baseline":    "#1f77b4",
    "D_graph_blocks":    "#2ca02c",
}

PARTITION_COLORS = {
    "random_partition":  "#d62728",
    "spatial_partition": "#ff7f0e",
    "greedy_partition":  "#9467bd",
}

# =====================================================================
# Helpers
# =====================================================================

def _sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)


def _timed_runs(func, n_warmup: int = N_WARMUP,
                n_repeat: int = N_REPEAT) -> np.ndarray:
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


def _build_time(build_fn) -> float:
    """Measure build cost in microseconds (median of 5 runs)."""
    times = []
    for _ in range(5):
        torch.cuda.empty_cache()
        _sync()
        t0 = time.perf_counter()
        build_fn()
        _sync()
        dt = time.perf_counter() - t0
        times.append(dt)
    return float(np.median(times)) * 1e6


# =====================================================================
# Synthetic Graph Generation (pure numpy, no networkx)
# =====================================================================

def generate_graph(graph_type: str, N: int, rng: np.random.Generator
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a graph in CSR format.

    Returns:
        positions: [N, 2] float64 node positions (None for barabasi_albert)
        adj_indices: flat int32 neighbor list
        adj_indptr: int32 offsets per node [N+1]
    """
    if graph_type == "random_geometric":
        return _gen_random_geometric(N, rng)
    elif graph_type == "barabasi_albert":
        return _gen_barabasi_albert(N, rng)
    elif graph_type == "grid_graph":
        return _gen_grid_graph(N, rng)
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")


def _gen_random_geometric(N: int, rng: np.random.Generator
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random geometric graph: N nodes in [0,1]^2, edge if dist < r.

    r is chosen to give ~6 neighbors per node on average:
        pi * r^2 * N ~ 6  =>  r = sqrt(6 / (pi * N))
    """
    positions = rng.random((N, 2)).astype(np.float64)
    r = math.sqrt(6.0 / (math.pi * N))

    # Build adjacency lists
    neighbors: List[List[int]] = [[] for _ in range(N)]

    # For moderate N, brute force is fine. For large N, use cell grid.
    if N <= 4096:
        # Brute force pairwise distances
        for i in range(N):
            dx = positions[i, 0] - positions[:, 0]
            dy = positions[i, 1] - positions[:, 1]
            dist2 = dx * dx + dy * dy
            mask = dist2 < r * r
            mask[i] = False  # no self-loops
            nbrs = np.where(mask)[0]
            neighbors[i] = nbrs.tolist()
    else:
        # Cell-based spatial hashing for efficiency
        cell_size = r
        n_cells = max(1, int(1.0 / cell_size) + 1)
        cells: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for i in range(N):
            cx = min(int(positions[i, 0] / cell_size), n_cells - 1)
            cy = min(int(positions[i, 1] / cell_size), n_cells - 1)
            cells[(cx, cy)].append(i)

        r2 = r * r
        for i in range(N):
            cx = min(int(positions[i, 0] / cell_size), n_cells - 1)
            cy = min(int(positions[i, 1] / cell_size), n_cells - 1)
            px, py = positions[i, 0], positions[i, 1]
            for dcx in range(-1, 2):
                for dcy in range(-1, 2):
                    key = (cx + dcx, cy + dcy)
                    if key not in cells:
                        continue
                    for j in cells[key]:
                        if j <= i:
                            continue
                        dx = px - positions[j, 0]
                        dy = py - positions[j, 1]
                        if dx * dx + dy * dy < r2:
                            neighbors[i].append(j)
                            neighbors[j].append(i)

    return positions, *_lists_to_csr(neighbors, N)


def _gen_barabasi_albert(N: int, rng: np.random.Generator
                         ) -> Tuple[None, np.ndarray, np.ndarray]:
    """Barabasi-Albert preferential attachment. m=3 edges per new node."""
    m = 3
    if N <= m:
        # Trivial: complete graph
        neighbors: List[List[int]] = [[] for _ in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                neighbors[i].append(j)
                neighbors[j].append(i)
        return None, *_lists_to_csr(neighbors, N)

    neighbors = [[] for _ in range(N)]
    degree = np.zeros(N, dtype=np.int64)

    # Start with complete graph on m+1 nodes
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            neighbors[i].append(j)
            neighbors[j].append(i)
            degree[i] += 1
            degree[j] += 1

    # Preferential attachment for remaining nodes
    for new_node in range(m + 1, N):
        # Probability proportional to degree
        total_deg = degree[:new_node].sum()
        if total_deg == 0:
            probs = np.ones(new_node, dtype=np.float64) / new_node
        else:
            probs = degree[:new_node].astype(np.float64) / total_deg

        # Sample m distinct targets
        targets = set()
        attempts = 0
        while len(targets) < m and attempts < m * 10:
            t = rng.choice(new_node, p=probs)
            targets.add(t)
            attempts += 1

        for t in targets:
            neighbors[new_node].append(t)
            neighbors[t].append(new_node)
            degree[new_node] += 1
            degree[t] += 1

    return None, *_lists_to_csr(neighbors, N)


def _gen_grid_graph(N: int, rng: np.random.Generator
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Regular grid graph: sqrt(N) x sqrt(N) with 4-neighbor connectivity."""
    side = int(math.sqrt(N))
    actual_N = side * side  # may differ slightly from N

    positions = np.zeros((actual_N, 2), dtype=np.float64)
    neighbors: List[List[int]] = [[] for _ in range(actual_N)]

    for r in range(side):
        for c in range(side):
            node = r * side + c
            positions[node] = [c / max(side - 1, 1), r / max(side - 1, 1)]
            # 4-neighbors
            if r > 0:
                neighbors[node].append((r - 1) * side + c)
            if r < side - 1:
                neighbors[node].append((r + 1) * side + c)
            if c > 0:
                neighbors[node].append(r * side + (c - 1))
            if c < side - 1:
                neighbors[node].append(r * side + (c + 1))

    return positions, *_lists_to_csr(neighbors, actual_N)


def _lists_to_csr(neighbors: List[List[int]], N: int
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Convert adjacency lists to CSR format (indices, indptr)."""
    indptr = np.zeros(N + 1, dtype=np.int32)
    all_indices = []
    for i in range(N):
        nbrs = sorted(set(neighbors[i]))  # deduplicate + sort
        all_indices.extend(nbrs)
        indptr[i + 1] = indptr[i] + len(nbrs)
    indices = np.array(all_indices, dtype=np.int32) if all_indices else np.empty(0, dtype=np.int32)
    return indices, indptr


# =====================================================================
# Block Partitioning Strategies
# =====================================================================

def partition_nodes(method: str, N: int, block_size: int,
                    positions: np.ndarray,
                    adj_indices: np.ndarray, adj_indptr: np.ndarray,
                    graph_type: str,
                    rng: np.random.Generator) -> np.ndarray:
    """Assign each node to a block_id.

    Returns:
        assignment: int32 [N], block_id for each node.
                    block_ids are contiguous from 0.
    """
    if method == "random_partition":
        return _partition_random(N, block_size, rng)
    elif method == "spatial_partition":
        return _partition_spatial(N, block_size, positions, adj_indices,
                                  adj_indptr, graph_type, rng)
    elif method == "greedy_partition":
        return _partition_greedy(N, block_size, adj_indices, adj_indptr, rng)
    else:
        raise ValueError(f"Unknown partition method: {method}")


def _partition_random(N: int, block_size: int,
                      rng: np.random.Generator) -> np.ndarray:
    """Random assignment of nodes to blocks."""
    n_blocks = int(math.ceil(N / block_size))
    assignment = np.zeros(N, dtype=np.int32)
    perm = rng.permutation(N)
    for i, node in enumerate(perm):
        assignment[node] = i // block_size
    return assignment


def _partition_spatial(N: int, block_size: int,
                       positions: np.ndarray,
                       adj_indices: np.ndarray, adj_indptr: np.ndarray,
                       graph_type: str,
                       rng: np.random.Generator) -> np.ndarray:
    """Spatial partition using Morton order for spatial graphs.

    For barabasi_albert (no positions), falls back to BFS-based partitioning.
    """
    if positions is None:
        # No spatial info -> BFS fallback
        return _partition_bfs(N, block_size, adj_indices, adj_indptr, rng)

    # Morton (Z-order) curve: interleave bits of quantized x, y
    # Quantize to [0, 2^16) range
    x = positions[:, 0]
    y = positions[:, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_range = max(x_max - x_min, 1e-12)
    y_range = max(y_max - y_min, 1e-12)

    qx = np.clip(((x - x_min) / x_range * 65535).astype(np.int64), 0, 65535)
    qy = np.clip(((y - y_min) / y_range * 65535).astype(np.int64), 0, 65535)

    # Interleave bits for Morton code
    def _spread_bits(v):
        """Spread 16-bit value into even bit positions of 32-bit result."""
        v = v & 0xFFFF
        v = (v | (v << 8)) & 0x00FF00FF
        v = (v | (v << 4)) & 0x0F0F0F0F
        v = (v | (v << 2)) & 0x33333333
        v = (v | (v << 1)) & 0x55555555
        return v

    morton = _spread_bits(qx) | (_spread_bits(qy) << 1)

    # Sort nodes by Morton code, then chunk into blocks
    order = np.argsort(morton)
    assignment = np.zeros(N, dtype=np.int32)
    for i, node in enumerate(order):
        assignment[node] = i // block_size

    return assignment


def _partition_bfs(N: int, block_size: int,
                   adj_indices: np.ndarray, adj_indptr: np.ndarray,
                   rng: np.random.Generator) -> np.ndarray:
    """BFS-based partitioning: start from random node, grow BFS, chunk."""
    visited = np.zeros(N, dtype=bool)
    order = []

    # BFS from random start
    start = int(rng.integers(0, N))
    queue = [start]
    visited[start] = True

    while len(order) < N:
        if not queue:
            # Find unvisited node (disconnected component)
            unvisited = np.where(~visited)[0]
            if len(unvisited) == 0:
                break
            start = int(rng.choice(unvisited))
            queue = [start]
            visited[start] = True

        node = queue.pop(0)
        order.append(node)

        # Add unvisited neighbors
        start_idx = adj_indptr[node]
        end_idx = adj_indptr[node + 1]
        for idx in range(start_idx, end_idx):
            nbr = adj_indices[idx]
            if not visited[nbr]:
                visited[nbr] = True
                queue.append(nbr)

    assignment = np.zeros(N, dtype=np.int32)
    for i, node in enumerate(order):
        assignment[node] = i // block_size

    return assignment


def _partition_greedy(N: int, block_size: int,
                      adj_indices: np.ndarray, adj_indptr: np.ndarray,
                      rng: np.random.Generator) -> np.ndarray:
    """Greedy BFS growth from random seeds.

    Start a new block from an unassigned node. Grow it via BFS, prioritizing
    nodes with the most already-assigned neighbors in the current block.
    Stop when block reaches block_size.
    """
    assignment = np.full(N, -1, dtype=np.int32)
    block_id = 0
    unassigned = set(range(N))

    while unassigned:
        # Pick a random seed from unassigned nodes
        seed = int(rng.choice(list(unassigned)))
        block_nodes = []
        frontier = [seed]

        while len(block_nodes) < block_size and frontier:
            # Pick frontier node with most neighbors already in this block
            best_node = frontier[0]
            best_score = -1
            for fn in frontier:
                score = 0
                s = adj_indptr[fn]
                e = adj_indptr[fn + 1]
                for idx in range(s, e):
                    nbr = adj_indices[idx]
                    if assignment[nbr] == block_id:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_node = fn

            frontier.remove(best_node)

            if best_node not in unassigned:
                continue

            assignment[best_node] = block_id
            block_nodes.append(best_node)
            unassigned.discard(best_node)

            # Add unassigned neighbors to frontier
            s = adj_indptr[best_node]
            e = adj_indptr[best_node + 1]
            for idx in range(s, e):
                nbr = adj_indices[idx]
                if nbr in unassigned and nbr not in frontier:
                    frontier.append(nbr)

        # If frontier exhausted but block not full, that's ok (last block)
        block_id += 1

    return assignment


# =====================================================================
# Blocking Quality Metrics
# =====================================================================

def compute_cross_block_ratio(assignment: np.ndarray,
                              adj_indices: np.ndarray,
                              adj_indptr: np.ndarray,
                              N: int) -> float:
    """Fraction of edges that cross block boundaries."""
    total_edges = 0
    cross_edges = 0
    for node in range(N):
        s = adj_indptr[node]
        e = adj_indptr[node + 1]
        for idx in range(s, e):
            nbr = adj_indices[idx]
            total_edges += 1
            if assignment[node] != assignment[nbr]:
                cross_edges += 1
    if total_edges == 0:
        return 0.0
    return cross_edges / total_edges


def compute_padding_waste(assignment: np.ndarray, active_mask: np.ndarray,
                          block_size: int) -> float:
    """Fraction of packed block slots that are padding (not active nodes)."""
    active_nodes = np.where(active_mask)[0]
    if len(active_nodes) == 0:
        return 1.0

    # Count active nodes per block (only blocks that contain active nodes)
    active_assignment = assignment[active_nodes]
    unique_blocks = np.unique(active_assignment)
    n_active_blocks = len(unique_blocks)

    total_slots = n_active_blocks * block_size
    used_slots = len(active_nodes)

    if total_slots == 0:
        return 1.0
    return 1.0 - used_slots / total_slots


# =====================================================================
# Layout: graph_baseline
# =====================================================================

def build_graph_baseline(N: int, feat_dim: int, active_mask: np.ndarray,
                         adj_indices: np.ndarray, adj_indptr: np.ndarray,
                         seed: int):
    """Baseline: full arrays + CSR adjacency on device.

    Returns dict with data, active_mask_t, adj_indices_t, adj_indptr_t.
    """
    torch.manual_seed(seed)
    data = torch.randn(N, feat_dim, dtype=DTYPE, device=DEVICE)
    active_mask_t = torch.from_numpy(active_mask).to(DEVICE)
    adj_indices_t = torch.from_numpy(adj_indices).to(torch.int64).to(DEVICE)
    adj_indptr_t = torch.from_numpy(adj_indptr).to(torch.int64).to(DEVICE)

    return {
        "data": data,
        "active_mask": active_mask_t,
        "adj_indices": adj_indices_t,
        "adj_indptr": adj_indptr_t,
    }


def compute_graph_baseline(bl: Dict) -> torch.Tensor:
    """For each active node, compute mean of active neighbor features.

    Returns output tensor [N, feat_dim].
    """
    data = bl["data"]
    active = bl["active_mask"]
    indices = bl["adj_indices"]
    indptr = bl["adj_indptr"]
    N, feat_dim = data.shape

    output = data.clone()
    active_nodes = torch.where(active)[0]

    for i_idx in range(len(active_nodes)):
        node = active_nodes[i_idx].item()
        s = indptr[node].item()
        e = indptr[node + 1].item()
        if s == e:
            continue
        nbr_ids = indices[s:e]
        nbr_active = active[nbr_ids]
        active_nbrs = nbr_ids[nbr_active]
        if len(active_nbrs) == 0:
            continue
        output[node] = data[active_nbrs].mean(dim=0)

    return output


# =====================================================================
# Layout: D_graph_blocks
# =====================================================================

def build_graph_blocks(N: int, feat_dim: int, active_mask: np.ndarray,
                       adj_indices: np.ndarray, adj_indptr: np.ndarray,
                       assignment: np.ndarray, block_size: int,
                       seed: int):
    """Block-based layout.

    Returns dict with:
        packed_data: [n_active_blocks, block_size, feat_dim]
        block_map: [n_blocks_total] int32, -1 = inactive block
        node_to_block: [N] int32, block_id per node
        node_to_local: [N] int32, local offset within block
        cross_block_edges: (src_block_slot, src_local, dst_block_slot, dst_local)
                           tensors for inter-block neighbor access
        intra_block_edges: similar for intra-block
        original_data: reference data for validation
    """
    torch.manual_seed(seed)
    original_data = torch.randn(N, feat_dim, dtype=DTYPE, device=DEVICE)

    # n_blocks_total must cover all block IDs produced by the partitioner.
    # greedy_partition can create more blocks than ceil(N/block_size) when
    # blocks don't fill completely (e.g. disconnected components), so we
    # derive the count from the actual assignment array.
    n_blocks_total = int(assignment.max()) + 1 if N > 0 else 0

    # Determine which blocks contain active nodes
    active_nodes = np.where(active_mask)[0]
    active_blocks_set = set(assignment[active_nodes].tolist())

    # Build block_map: block_id -> slot
    block_map_np = np.full(n_blocks_total, -1, dtype=np.int32)
    sorted_active_blocks = sorted(active_blocks_set)
    for slot, bid in enumerate(sorted_active_blocks):
        block_map_np[bid] = slot
    n_active_blocks = len(sorted_active_blocks)

    # Compute local offset within each block
    # For each block, nodes are ordered by their first appearance
    block_fill = np.zeros(n_blocks_total, dtype=np.int32)
    node_to_local = np.zeros(N, dtype=np.int32)
    for node in range(N):
        bid = assignment[node]
        node_to_local[node] = block_fill[bid]
        block_fill[bid] += 1

    # Pack data into blocks
    n_ab = max(n_active_blocks, 1)
    packed_data = torch.zeros(n_ab, block_size, feat_dim,
                              dtype=DTYPE, device=DEVICE)
    for node in active_nodes:
        bid = assignment[node]
        slot = block_map_np[bid]
        if slot >= 0:
            local = node_to_local[node]
            if local < block_size:
                packed_data[slot, local] = original_data[node]

    # Build cross-block and intra-block edge lists for compute
    src_blocks = []
    src_locals = []
    dst_blocks = []
    dst_locals = []
    intra_src_blocks = []
    intra_src_locals = []
    intra_dst_locals = []

    for node in active_nodes:
        bid_src = assignment[node]
        slot_src = block_map_np[bid_src]
        local_src = node_to_local[node]
        if slot_src < 0 or local_src >= block_size:
            continue

        s = adj_indptr[node]
        e = adj_indptr[node + 1]
        for idx in range(s, e):
            nbr = adj_indices[idx]
            if not active_mask[nbr]:
                continue
            bid_dst = assignment[nbr]
            slot_dst = block_map_np[bid_dst]
            local_dst = node_to_local[nbr]
            if slot_dst < 0 or local_dst >= block_size:
                continue

            if bid_src == bid_dst:
                intra_src_blocks.append(slot_src)
                intra_src_locals.append(local_src)
                intra_dst_locals.append(local_dst)
            else:
                src_blocks.append(slot_src)
                src_locals.append(local_src)
                dst_blocks.append(slot_dst)
                dst_locals.append(local_dst)

    block_map_t = torch.from_numpy(block_map_np).to(DEVICE)
    node_to_block_t = torch.from_numpy(assignment).to(torch.int32).to(DEVICE)
    node_to_local_t = torch.from_numpy(node_to_local).to(torch.int32).to(DEVICE)

    # Cross-block edge tensors
    if src_blocks:
        cross_src_block = torch.tensor(src_blocks, dtype=torch.int64, device=DEVICE)
        cross_src_local = torch.tensor(src_locals, dtype=torch.int64, device=DEVICE)
        cross_dst_block = torch.tensor(dst_blocks, dtype=torch.int64, device=DEVICE)
        cross_dst_local = torch.tensor(dst_locals, dtype=torch.int64, device=DEVICE)
    else:
        cross_src_block = torch.empty(0, dtype=torch.int64, device=DEVICE)
        cross_src_local = torch.empty(0, dtype=torch.int64, device=DEVICE)
        cross_dst_block = torch.empty(0, dtype=torch.int64, device=DEVICE)
        cross_dst_local = torch.empty(0, dtype=torch.int64, device=DEVICE)

    # Intra-block edge tensors
    if intra_src_blocks:
        intra_src_block_t = torch.tensor(intra_src_blocks, dtype=torch.int64, device=DEVICE)
        intra_src_local_t = torch.tensor(intra_src_locals, dtype=torch.int64, device=DEVICE)
        intra_dst_local_t = torch.tensor(intra_dst_locals, dtype=torch.int64, device=DEVICE)
    else:
        intra_src_block_t = torch.empty(0, dtype=torch.int64, device=DEVICE)
        intra_src_local_t = torch.empty(0, dtype=torch.int64, device=DEVICE)
        intra_dst_local_t = torch.empty(0, dtype=torch.int64, device=DEVICE)

    return {
        "packed_data": packed_data,
        "block_map": block_map_t,
        "node_to_block": node_to_block_t,
        "node_to_local": node_to_local_t,
        "n_active_blocks": n_active_blocks,
        "cross_edges": (cross_src_block, cross_src_local,
                        cross_dst_block, cross_dst_local),
        "intra_edges": (intra_src_block_t, intra_src_local_t,
                        intra_dst_local_t),
        "original_data": original_data,
        "active_mask": torch.from_numpy(active_mask).to(DEVICE),
    }


def compute_graph_blocks(bl: Dict) -> torch.Tensor:
    """Neighbor aggregation through block addressing.

    For each active node (via edge lists), accumulate neighbor features
    and divide by count. Both intra-block and cross-block edges.

    Returns: packed_data-shaped output (modified in place conceptually).
    """
    packed = bl["packed_data"]
    n_ab = bl["n_active_blocks"]
    if n_ab == 0:
        return packed

    # Output accumulator: [n_blocks, block_size, feat_dim]
    output_sum = torch.zeros_like(packed)
    output_count = torch.zeros(packed.shape[0], packed.shape[1],
                               dtype=DTYPE, device=DEVICE)

    # Intra-block edges
    intra_src_block, intra_src_local, intra_dst_local = bl["intra_edges"]
    if len(intra_src_block) > 0:
        # Gather destination features
        dst_feats = packed[intra_src_block, intra_dst_local]  # [E_intra, feat_dim]
        # Scatter-add to source positions
        output_sum.index_put_(
            (intra_src_block, intra_src_local),
            dst_feats, accumulate=True)
        ones = torch.ones(len(intra_src_block), dtype=DTYPE, device=DEVICE)
        output_count.index_put_(
            (intra_src_block, intra_src_local),
            ones, accumulate=True)

    # Cross-block edges
    cross_src_block, cross_src_local, cross_dst_block, cross_dst_local = bl["cross_edges"]
    if len(cross_src_block) > 0:
        dst_feats = packed[cross_dst_block, cross_dst_local]  # [E_cross, feat_dim]
        output_sum.index_put_(
            (cross_src_block, cross_src_local),
            dst_feats, accumulate=True)
        ones = torch.ones(len(cross_src_block), dtype=DTYPE, device=DEVICE)
        output_count.index_put_(
            (cross_src_block, cross_src_local),
            ones, accumulate=True)

    # Compute mean where count > 0
    mask = output_count > 0
    result = packed.clone()
    if mask.any():
        safe_count = output_count.clamp(min=1.0).unsqueeze(-1)
        mean_feats = output_sum / safe_count
        # Only update where we have neighbors
        mask_3d = mask.unsqueeze(-1).expand_as(result)
        result[mask_3d] = mean_feats[mask_3d]

    return result


# =====================================================================
# Active mask generation
# =====================================================================

def make_active_mask(N: int, sparsity: float,
                     rng: np.random.Generator) -> np.ndarray:
    """Generate 1D boolean mask. sparsity = fraction active."""
    n_active = max(1, int(round(N * sparsity)))
    idx = rng.choice(N, size=n_active, replace=False)
    mask = np.zeros(N, dtype=bool)
    mask[idx] = True
    return mask


# #####################################################################
# BENCHMARK LOOP
# #####################################################################

print("=" * 72)
print("Exp10i: Block-Based Addressing for Irregular Graphs")
print(f"Device: {DEVICE}")
print("=" * 72)

all_results: List[Dict[str, Any]] = []

total_configs = (len(N_NODES_LIST) * len(SPARSITIES) * len(GRAPH_TYPES)
                 * len(BLOCK_SIZES) * len(PARTITION_METHODS))
config_count = 0

for N_nodes in N_NODES_LIST:
    for graph_type in GRAPH_TYPES:
        # Adjust actual N for grid_graph
        if graph_type == "grid_graph":
            side = int(math.sqrt(N_nodes))
            actual_N = side * side
        else:
            actual_N = N_nodes

        print(f"\n--- {graph_type} N={actual_N} ---")

        for sparsity in SPARSITIES:
            for block_size in BLOCK_SIZES:
                for partition_method in PARTITION_METHODS:
                    seed_records_base = []
                    seed_records_d = []
                    seed_cross_ratios = []
                    seed_padding_wastes = []

                    for seed_idx in range(N_SEEDS):
                        seed = SEED_BASE + seed_idx
                        rng = np.random.default_rng(seed)
                        torch.manual_seed(seed)

                        # Generate graph
                        positions, adj_indices, adj_indptr = generate_graph(
                            graph_type, actual_N, rng)

                        # Actual N may differ for grid
                        N = len(adj_indptr) - 1

                        # Generate active mask
                        active_mask = make_active_mask(N, sparsity, rng)

                        # Partition nodes
                        assignment = partition_nodes(
                            partition_method, N, block_size,
                            positions, adj_indices, adj_indptr,
                            graph_type, rng)

                        # Quality metrics (numpy, before GPU work)
                        cbr = compute_cross_block_ratio(
                            assignment, adj_indices, adj_indptr, N)
                        pw = compute_padding_waste(
                            assignment, active_mask, block_size)
                        seed_cross_ratios.append(cbr)
                        seed_padding_wastes.append(pw)

                        # ---- graph_baseline ----
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats(DEVICE)

                        bl_base = build_graph_baseline(
                            N, FEATURE_DIM, active_mask,
                            adj_indices, adj_indptr, seed)
                        _sync()

                        resident_base = (torch.cuda.memory_allocated(DEVICE)
                                         if DEVICE.type == "cuda" else 0)

                        torch.cuda.reset_peak_memory_stats(DEVICE)
                        times_base = _timed_runs(
                            lambda: compute_graph_baseline(bl_base))
                        peak_base = (torch.cuda.max_memory_allocated(DEVICE)
                                     if DEVICE.type == "cuda" else 0)

                        wall_base_us = float(np.median(times_base)) * 1e6

                        build_base_us = _build_time(
                            lambda: build_graph_baseline(
                                N, FEATURE_DIM, active_mask,
                                adj_indices, adj_indptr, seed))

                        seed_records_base.append({
                            "resident_bytes": int(resident_base),
                            "peak_vram_bytes": int(peak_base),
                            "wall_clock_us": wall_base_us,
                            "build_cost_us": build_base_us,
                        })

                        del bl_base
                        torch.cuda.empty_cache()

                        # ---- D_graph_blocks ----
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats(DEVICE)

                        bl_d = build_graph_blocks(
                            N, FEATURE_DIM, active_mask,
                            adj_indices, adj_indptr,
                            assignment, block_size, seed)
                        _sync()

                        resident_d = (torch.cuda.memory_allocated(DEVICE)
                                      if DEVICE.type == "cuda" else 0)

                        torch.cuda.reset_peak_memory_stats(DEVICE)
                        times_d = _timed_runs(
                            lambda: compute_graph_blocks(bl_d))
                        peak_d = (torch.cuda.max_memory_allocated(DEVICE)
                                  if DEVICE.type == "cuda" else 0)

                        wall_d_us = float(np.median(times_d)) * 1e6

                        build_d_us = _build_time(
                            lambda: build_graph_blocks(
                                N, FEATURE_DIM, active_mask,
                                adj_indices, adj_indptr,
                                assignment, block_size, seed))

                        seed_records_d.append({
                            "resident_bytes": int(resident_d),
                            "peak_vram_bytes": int(peak_d),
                            "wall_clock_us": wall_d_us,
                            "build_cost_us": build_d_us,
                        })

                        del bl_d
                        torch.cuda.empty_cache()

                    # Aggregate across seeds (median)
                    def _median_dict(records):
                        out = {}
                        for key in records[0]:
                            vals = [r[key] for r in records]
                            out[key] = float(np.median(vals))
                        return out

                    base_agg = _median_dict(seed_records_base)
                    d_agg = _median_dict(seed_records_d)
                    median_cbr = float(np.median(seed_cross_ratios))
                    median_pw = float(np.median(seed_padding_wastes))

                    # Kill criteria
                    time_overhead = ((d_agg["wall_clock_us"] - base_agg["wall_clock_us"])
                                     / max(base_agg["wall_clock_us"], 1e-9))

                    contour_a = (d_agg["resident_bytes"] < base_agg["resident_bytes"]
                                 and time_overhead < KILL_THRESH
                                 and median_cbr < CROSS_BLOCK_THRESH)
                    contour_b = (d_agg["peak_vram_bytes"] < base_agg["peak_vram_bytes"]
                                 and time_overhead < KILL_THRESH)

                    padding_warning = median_pw > PADDING_WASTE_THRESH

                    rec = {
                        "graph_type": graph_type,
                        "N_nodes": N,
                        "sparsity": sparsity,
                        "block_size": block_size,
                        "partition_method": partition_method,
                        "graph_baseline": base_agg,
                        "D_graph_blocks": d_agg,
                        "time_overhead_frac": round(time_overhead, 4),
                        "cross_block_ratio": round(median_cbr, 4),
                        "padding_waste": round(median_pw, 4),
                        "contour_A": "PASS" if contour_a else "FAIL",
                        "contour_B": "PASS" if contour_b else "FAIL",
                        "padding_warning": padding_warning,
                    }
                    all_results.append(rec)

                    config_count += 1
                    ca_tag = "PASS" if contour_a else "FAIL"
                    cb_tag = "PASS" if contour_b else "FAIL"
                    pw_tag = " !!PADDING" if padding_warning else ""
                    print(f"  [{config_count}/{total_configs}] "
                          f"sp={sparsity} bs={block_size} "
                          f"part={partition_method} | "
                          f"A={ca_tag} B={cb_tag} "
                          f"time_oh={time_overhead:+.1%} "
                          f"cbr={median_cbr:.2f} pw={median_pw:.2f}"
                          f"{pw_tag}")


# #####################################################################
# OUTPUT: Summary JSON
# #####################################################################

print("\n" + "=" * 72)
print("Writing outputs...")
print("=" * 72)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

pass_a = sum(1 for r in all_results if r["contour_A"] == "PASS")
pass_b = sum(1 for r in all_results if r["contour_B"] == "PASS")
n_warnings = sum(1 for r in all_results if r["padding_warning"])

summary = {
    "experiment": "exp10i_graph_blocks",
    "device": str(DEVICE),
    "dtype": str(DTYPE),
    "n_seeds": N_SEEDS,
    "feature_dim": FEATURE_DIM,
    "kill_threshold": KILL_THRESH,
    "cross_block_threshold": CROSS_BLOCK_THRESH,
    "padding_waste_threshold": PADDING_WASTE_THRESH,
    "config": {
        "N_nodes_list": N_NODES_LIST,
        "sparsities": SPARSITIES,
        "graph_types": GRAPH_TYPES,
        "block_sizes": BLOCK_SIZES,
        "partition_methods": PARTITION_METHODS,
    },
    "results": all_results,
    "contour_A_pass": pass_a,
    "contour_A_total": len(all_results),
    "contour_B_pass": pass_b,
    "contour_B_total": len(all_results),
    "padding_warnings": n_warnings,
}

with open(RESULTS_DIR / "exp10i_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"  Wrote {RESULTS_DIR / 'exp10i_summary.json'}")


# #####################################################################
# OUTPUT: Plots
# #####################################################################

# =====================================================================
# Plot 1: Time comparison -- wall clock ratio by sparsity, per graph type
# =====================================================================

fig, axes = plt.subplots(1, len(GRAPH_TYPES), figsize=(6 * len(GRAPH_TYPES), 5),
                         squeeze=False)

for gi, gtype in enumerate(GRAPH_TYPES):
    ax = axes[0, gi]
    for pm in PARTITION_METHODS:
        subset = [r for r in all_results
                  if r["graph_type"] == gtype and r["partition_method"] == pm]
        if not subset:
            continue
        sps = sorted(set(r["sparsity"] for r in subset))
        ratios = []
        for sp in sps:
            sp_recs = [r for r in subset if r["sparsity"] == sp]
            ratio = np.mean([r["D_graph_blocks"]["wall_clock_us"]
                             / max(r["graph_baseline"]["wall_clock_us"], 1e-9)
                             for r in sp_recs])
            ratios.append(ratio)
        ax.plot(sps, ratios, "o-", label=pm.replace("_partition", ""),
                color=PARTITION_COLORS.get(pm, None))
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(1.0 + KILL_THRESH, color="red", linestyle=":", alpha=0.5)
    ax.set_xlabel("Sparsity (fraction active)")
    ax.set_ylabel("Time ratio (D / baseline)")
    ax.set_title(f"{gtype}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle("exp10i: Wall Clock Ratio by Graph Type & Partition Method",
             fontsize=13)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "exp10i_time_comparison.png", dpi=150)
plt.close(fig)
print(f"  Wrote {RESULTS_DIR / 'exp10i_time_comparison.png'}")


# =====================================================================
# Plot 2: Memory comparison -- resident and peak ratios
# =====================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: resident ratio by sparsity, colored by graph type
ax = axes[0]
for gtype in GRAPH_TYPES:
    # Use best partition method (greedy) for memory plot
    subset = [r for r in all_results
              if r["graph_type"] == gtype and r["partition_method"] == "greedy_partition"]
    if not subset:
        continue
    sps = sorted(set(r["sparsity"] for r in subset))
    ratios = []
    for sp in sps:
        sp_recs = [r for r in subset if r["sparsity"] == sp]
        ratio = np.mean([r["D_graph_blocks"]["resident_bytes"]
                         / max(r["graph_baseline"]["resident_bytes"], 1)
                         for r in sp_recs])
        ratios.append(ratio)
    ax.plot(sps, ratios, "o-", label=gtype)
ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("Sparsity (fraction active)")
ax.set_ylabel("Resident ratio (D / baseline)")
ax.set_title("Resident Memory (greedy partition)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Right: peak ratio
ax = axes[1]
for gtype in GRAPH_TYPES:
    subset = [r for r in all_results
              if r["graph_type"] == gtype and r["partition_method"] == "greedy_partition"]
    if not subset:
        continue
    sps = sorted(set(r["sparsity"] for r in subset))
    ratios = []
    for sp in sps:
        sp_recs = [r for r in subset if r["sparsity"] == sp]
        ratio = np.mean([r["D_graph_blocks"]["peak_vram_bytes"]
                         / max(r["graph_baseline"]["peak_vram_bytes"], 1)
                         for r in sp_recs])
        ratios.append(ratio)
    ax.plot(sps, ratios, "o-", label=gtype)
ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("Sparsity (fraction active)")
ax.set_ylabel("Peak VRAM ratio (D / baseline)")
ax.set_title("Peak VRAM (greedy partition)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

fig.suptitle("exp10i: Memory Ratios (D_graph_blocks / graph_baseline)", fontsize=13)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "exp10i_memory_comparison.png", dpi=150)
plt.close(fig)
print(f"  Wrote {RESULTS_DIR / 'exp10i_memory_comparison.png'}")


# =====================================================================
# Plot 3: Blocking quality -- cross_block_ratio by partition method
# =====================================================================

fig, axes = plt.subplots(1, len(GRAPH_TYPES), figsize=(6 * len(GRAPH_TYPES), 5),
                         squeeze=False)

for gi, gtype in enumerate(GRAPH_TYPES):
    ax = axes[0, gi]
    for pm in PARTITION_METHODS:
        subset = [r for r in all_results
                  if r["graph_type"] == gtype and r["partition_method"] == pm]
        if not subset:
            continue
        block_sizes_seen = sorted(set(r["block_size"] for r in subset))
        cbrs = []
        for bs in block_sizes_seen:
            bs_recs = [r for r in subset if r["block_size"] == bs]
            cbr = np.mean([r["cross_block_ratio"] for r in bs_recs])
            cbrs.append(cbr)
        ax.plot(block_sizes_seen, cbrs, "o-",
                label=pm.replace("_partition", ""),
                color=PARTITION_COLORS.get(pm, None))
    ax.axhline(CROSS_BLOCK_THRESH, color="red", linestyle=":", alpha=0.5,
               label=f"threshold={CROSS_BLOCK_THRESH}")
    ax.set_xlabel("Block size")
    ax.set_ylabel("Cross-block ratio")
    ax.set_title(f"{gtype}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

fig.suptitle("exp10i: Blocking Quality (cross_block_ratio by method)",
             fontsize=13)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "exp10i_blocking_quality.png", dpi=150)
plt.close(fig)
print(f"  Wrote {RESULTS_DIR / 'exp10i_blocking_quality.png'}")


# #####################################################################
# OUTPUT: Report
# #####################################################################

report_lines = [
    "# exp10i Graph Blocks Report",
    "",
    "## Summary",
    "",
    f"Contour A (architectural + cross_block < {CROSS_BLOCK_THRESH}): "
    f"{pass_a}/{len(all_results)} PASS",
    f"Contour B (operational): {pass_b}/{len(all_results)} PASS",
    f"Padding warnings (waste > {PADDING_WASTE_THRESH:.0%}): "
    f"{n_warnings}/{len(all_results)}",
    "",
    "## Results by graph type",
    "",
]

for gtype in GRAPH_TYPES:
    gt_recs = [r for r in all_results if r["graph_type"] == gtype]
    gt_a = sum(1 for r in gt_recs if r["contour_A"] == "PASS")
    gt_b = sum(1 for r in gt_recs if r["contour_B"] == "PASS")
    gt_w = sum(1 for r in gt_recs if r["padding_warning"])

    report_lines += [
        f"### {gtype}",
        "",
        f"Contour A: {gt_a}/{len(gt_recs)} PASS | "
        f"Contour B: {gt_b}/{len(gt_recs)} PASS | "
        f"Padding warnings: {gt_w}",
        "",
        "| N | sp | bs | partition | A | B | time_oh | cbr | pw | res_ratio | peak_ratio |",
        "|---|----|----|----------|---|---|---------|-----|-----|-----------|------------|",
    ]

    for r in gt_recs:
        g = r["graph_baseline"]
        d = r["D_graph_blocks"]
        res_ratio = d["resident_bytes"] / max(g["resident_bytes"], 1)
        peak_ratio = d["peak_vram_bytes"] / max(g["peak_vram_bytes"], 1)
        pw_flag = " !!" if r["padding_warning"] else ""
        report_lines.append(
            f"| {r['N_nodes']} | {r['sparsity']} | {r['block_size']} "
            f"| {r['partition_method'].replace('_partition', '')} "
            f"| {r['contour_A']} | {r['contour_B']} "
            f"| {r['time_overhead_frac']:+.1%} "
            f"| {r['cross_block_ratio']:.2f} "
            f"| {r['padding_waste']:.2f}{pw_flag} "
            f"| {res_ratio:.2f} | {peak_ratio:.2f} |"
        )

    report_lines.append("")

# Best partition method analysis
report_lines += [
    "## Partition method comparison (cross_block_ratio, lower is better)",
    "",
]

for pm in PARTITION_METHODS:
    pm_recs = [r for r in all_results if r["partition_method"] == pm]
    mean_cbr = np.mean([r["cross_block_ratio"] for r in pm_recs])
    mean_pw = np.mean([r["padding_waste"] for r in pm_recs])
    pm_a = sum(1 for r in pm_recs if r["contour_A"] == "PASS")
    report_lines.append(
        f"- **{pm}**: mean cbr={mean_cbr:.3f}, mean pw={mean_pw:.3f}, "
        f"Contour A: {pm_a}/{len(pm_recs)}")

report_lines += [
    "",
    "## Verdict",
    "",
]

if pass_a == len(all_results) and pass_b == len(all_results):
    report_lines.append(
        "**OVERALL: PASS** -- Block-based addressing works for irregular graphs.")
elif pass_a > len(all_results) * 0.5:
    report_lines.append(
        "**OVERALL: PARTIAL** -- Block-based addressing works for some configs. "
        "Check partition method and block size selection.")
else:
    report_lines.append(
        "**OVERALL: FAIL** -- Block-based addressing does not reliably work for "
        "irregular graphs. Graph structure resists blocking.")

report_text = "\n".join(report_lines) + "\n"
with open(RESULTS_DIR / "exp10i_report.md", "w") as f:
    f.write(report_text)
print(f"  Wrote {RESULTS_DIR / 'exp10i_report.md'}")

print("\n" + "=" * 72)
print("exp10i COMPLETE")
print("=" * 72)
print(f"\nContour A: {pass_a}/{len(all_results)} PASS")
print(f"Contour B: {pass_b}/{len(all_results)} PASS")
print(f"Padding warnings: {n_warnings}/{len(all_results)}")
