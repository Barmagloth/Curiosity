#!/usr/bin/env python3
"""
S2 -- Halo Cross-Space Validation

Validates that cosine feathering (halo, overlap >= 3 elements) reduces seam
artifacts across ALL 4 space types, not only 2D pixel grids.

Key insight: seam artifacts appear at boundaries between REFINED and UNREFINED
tiles. We refine ~30% of tiles (highest-error), then compare:
  - Hard insert (w=0): sharp discontinuity at refined/unrefined boundary
  - Halo insert (w=3): cosine-feathered blending smooths the boundary

Space types:
  T1: 2D scalar grid 64x64, tile=8
  T2: 2D vector grid 64x64, dim=32, tile=8
  T3: Irregular graph 500 pts, k=8, 10 clusters
  T4: Tree hierarchy depth=8 (255 nodes)

Protocol:
  - For each space type x 20 seeds:
      1. Generate GT + coarse
      2. Select ~30% highest-error tiles for refinement
      3. Refine selected tiles WITHOUT halo -> measure SeamScore at boundary
      4. Refine selected tiles WITH halo (w=3) -> measure SeamScore at boundary
      5. Record ratio = SeamScore_no_halo / SeamScore_halo
  - Paired Wilcoxon signed-rank test per space type
  - Holm-Bonferroni correction for 4 comparisons
  - Kill criterion: ratio >= 1.5 on ALL 4 space types
"""

import numpy as np
import json
import sys
from pathlib import Path
from scipy import stats
from scipy.spatial import cKDTree
from scipy.cluster.vq import kmeans2

EPS = 1e-10
N_SEEDS = 20
HALO_WIDTH = 3
BUDGET = 0.30  # fraction of tiles to refine

OUT_DIR = Path(__file__).parent / "results"


# =====================================================================
# Utilities
# =====================================================================

def robust_jump(diffs):
    if len(diffs) == 0:
        return 0.0
    return float(np.median(diffs))


def cosine_ramp(n):
    if n <= 0:
        return np.array([])
    return 0.5 * (1 - np.cos(np.pi * np.arange(n) / n))


# =====================================================================
# T1: 2D Scalar Grid
# =====================================================================

def t1_run(seed):
    N, tile = 64, 8
    NT = N // tile
    rng = np.random.RandomState(seed)
    x = np.linspace(0, 1, N, endpoint=False)
    xx, yy = np.meshgrid(x, x)
    gt = (0.8 * (xx > 0.35)
          + 0.5 * np.sin(2 * np.pi * 4 * xx) * np.cos(2 * np.pi * 3 * yy)
          + rng.randn(N, N) * 0.02)

    # Coarse
    coarse = np.zeros_like(gt)
    for ti in range(NT):
        for tj in range(NT):
            s, c = slice(ti * tile, (ti + 1) * tile), slice(tj * tile, (tj + 1) * tile)
            coarse[s, c] = gt[s, c].mean()

    delta = gt - coarse

    # Select top-BUDGET tiles by error
    energies = []
    for ti in range(NT):
        for tj in range(NT):
            s, c = slice(ti * tile, (ti + 1) * tile), slice(tj * tile, (tj + 1) * tile)
            energies.append(((ti, tj), np.mean(delta[s, c] ** 2)))
    energies.sort(key=lambda x: -x[1])
    k = max(1, int(BUDGET * NT * NT))
    active = set(t for t, _ in energies[:k])

    # Hard insert: replace active tiles with GT, leave rest as coarse
    hard = coarse.copy()
    for (ti, tj) in active:
        s, c = slice(ti * tile, (ti + 1) * tile), slice(tj * tile, (tj + 1) * tile)
        hard[s, c] = gt[s, c]

    # Halo insert: cosine-feathered blending
    d_acc = np.zeros((N, N), dtype=np.float64)
    w_acc = np.zeros((N, N), dtype=np.float64)
    for (ti, tj) in active:
        w = HALO_WIDTH
        er0 = max(0, ti * tile - w)
        er1 = min(N, (ti + 1) * tile + w)
        ec0 = max(0, tj * tile - w)
        ec1 = min(N, (tj + 1) * tile + w)
        ph, pw = er1 - er0, ec1 - ec0
        wy = np.ones(ph)
        wx = np.ones(pw)
        top = ti * tile - er0
        if top > 0:
            wy[:top] = cosine_ramp(top)
        bot = er1 - (ti + 1) * tile
        if bot > 0:
            wy[-bot:] = cosine_ramp(bot)[::-1]
        left = tj * tile - ec0
        if left > 0:
            wx[:left] = cosine_ramp(left)
        right = ec1 - (tj + 1) * tile
        if right > 0:
            wx[-right:] = cosine_ramp(right)[::-1]
        W2d = np.outer(wy, wx)
        d_acc[er0:er1, ec0:ec1] += W2d * delta[er0:er1, ec0:ec1]
        w_acc[er0:er1, ec0:ec1] += W2d
    halo_out = coarse.copy()
    valid = w_acc > 1e-12
    halo_out[valid] = coarse[valid] + d_acc[valid] / w_acc[valid]

    # SeamScore at active/inactive boundary
    def seam_at_boundary(state):
        bp_diffs = []
        ip_diffs = []
        for (ti, tj) in active:
            r0, c0 = ti * tile, tj * tile
            r1, c1 = r0 + tile, c0 + tile
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = ti + di, tj + dj
                if 0 <= ni < NT and 0 <= nj < NT and (ni, nj) not in active:
                    # Boundary between active (ti,tj) and inactive (ni,nj)
                    if di == -1:
                        for cc in range(c0, c1):
                            bp_diffs.append(abs(state[r0, cc] - state[r0 - 1, cc]))
                            if r0 + 1 < N:
                                ip_diffs.append(abs(state[r0 + 1, cc] - state[r0, cc]))
                    elif di == 1:
                        for cc in range(c0, c1):
                            bp_diffs.append(abs(state[r1 - 1, cc] - state[r1, cc]))
                            if r1 - 2 >= r0:
                                ip_diffs.append(abs(state[r1 - 2, cc] - state[r1 - 1, cc]))
                    elif dj == -1:
                        for rr in range(r0, r1):
                            bp_diffs.append(abs(state[rr, c0] - state[rr, c0 - 1]))
                            if c0 + 1 < N:
                                ip_diffs.append(abs(state[rr, c0 + 1] - state[rr, c0]))
                    elif dj == 1:
                        for rr in range(r0, r1):
                            bp_diffs.append(abs(state[rr, c1 - 1] - state[rr, c1]))
                            if c1 - 2 >= c0:
                                ip_diffs.append(abs(state[rr, c1 - 2] - state[rr, c1 - 1]))
        jo = robust_jump(bp_diffs)
        ji = robust_jump(ip_diffs) if ip_diffs else 0.0
        return jo / (ji + EPS)

    ss_hard = seam_at_boundary(hard)
    ss_halo = seam_at_boundary(halo_out)
    return ss_hard, ss_halo


# =====================================================================
# T2: 2D Vector Grid (dim=32)
# =====================================================================

def t2_run(seed):
    N, tile, D = 64, 8, 32
    NT = N // tile
    rng = np.random.RandomState(seed)
    x = np.linspace(0, 1, N, endpoint=False)
    xx, yy = np.meshgrid(x, x)
    gt = np.zeros((N, N, D))
    for d in range(D):
        freq = 2 + d * 0.5
        phase = rng.uniform(0, 2 * np.pi)
        amp = 0.5 / (1 + d * 0.1)
        gt[:, :, d] = amp * np.sin(2 * np.pi * freq * xx + phase) * np.cos(2 * np.pi * (freq * 0.7) * yy)
        if d % 3 == 0:
            gt[:, :, d] += 0.3 * (xx > rng.uniform(0.2, 0.8)).astype(float)
    gt += rng.randn(N, N, D) * 0.02

    coarse = np.zeros_like(gt)
    for ti in range(NT):
        for tj in range(NT):
            s, c = slice(ti * tile, (ti + 1) * tile), slice(tj * tile, (tj + 1) * tile)
            coarse[s, c, :] = gt[s, c, :].mean(axis=(0, 1), keepdims=True)

    delta = gt - coarse

    # Select active tiles
    energies = []
    for ti in range(NT):
        for tj in range(NT):
            s, c = slice(ti * tile, (ti + 1) * tile), slice(tj * tile, (tj + 1) * tile)
            energies.append(((ti, tj), np.mean(delta[s, c, :] ** 2)))
    energies.sort(key=lambda x: -x[1])
    k = max(1, int(BUDGET * NT * NT))
    active = set(t for t, _ in energies[:k])

    # Hard insert
    hard = coarse.copy()
    for (ti, tj) in active:
        s, c = slice(ti * tile, (ti + 1) * tile), slice(tj * tile, (tj + 1) * tile)
        hard[s, c, :] = gt[s, c, :]

    # Halo insert
    d_acc = np.zeros((N, N, D), dtype=np.float64)
    w_acc = np.zeros((N, N, 1), dtype=np.float64)
    for (ti, tj) in active:
        w = HALO_WIDTH
        er0 = max(0, ti * tile - w)
        er1 = min(N, (ti + 1) * tile + w)
        ec0 = max(0, tj * tile - w)
        ec1 = min(N, (tj + 1) * tile + w)
        ph, pw = er1 - er0, ec1 - ec0
        wy = np.ones(ph)
        wx = np.ones(pw)
        top = ti * tile - er0
        if top > 0:
            wy[:top] = cosine_ramp(top)
        bot = er1 - (ti + 1) * tile
        if bot > 0:
            wy[-bot:] = cosine_ramp(bot)[::-1]
        left = tj * tile - ec0
        if left > 0:
            wx[:left] = cosine_ramp(left)
        right = ec1 - (tj + 1) * tile
        if right > 0:
            wx[-right:] = cosine_ramp(right)[::-1]
        W2d = np.outer(wy, wx)[:, :, np.newaxis]
        d_acc[er0:er1, ec0:ec1, :] += W2d * delta[er0:er1, ec0:ec1, :]
        w_acc[er0:er1, ec0:ec1, :] += W2d
    halo_out = coarse.copy()
    valid = w_acc > 1e-12
    valid3 = np.broadcast_to(valid, d_acc.shape)
    halo_out[valid3] = coarse[valid3] + d_acc[valid3] / np.broadcast_to(w_acc, d_acc.shape)[valid3]

    def seam_at_boundary(state):
        bp_diffs = []
        ip_diffs = []
        for (ti, tj) in active:
            r0, c0 = ti * tile, tj * tile
            r1, c1 = r0 + tile, c0 + tile
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = ti + di, tj + dj
                if 0 <= ni < NT and 0 <= nj < NT and (ni, nj) not in active:
                    if di == -1:
                        for cc in range(c0, c1):
                            bp_diffs.append(np.linalg.norm(state[r0, cc, :] - state[r0 - 1, cc, :]))
                            if r0 + 1 < N:
                                ip_diffs.append(np.linalg.norm(state[r0 + 1, cc, :] - state[r0, cc, :]))
                    elif di == 1:
                        for cc in range(c0, c1):
                            bp_diffs.append(np.linalg.norm(state[r1 - 1, cc, :] - state[r1, cc, :]))
                            if r1 - 2 >= r0:
                                ip_diffs.append(np.linalg.norm(state[r1 - 2, cc, :] - state[r1 - 1, cc, :]))
                    elif dj == -1:
                        for rr in range(r0, r1):
                            bp_diffs.append(np.linalg.norm(state[rr, c0, :] - state[rr, c0 - 1, :]))
                            if c0 + 1 < N:
                                ip_diffs.append(np.linalg.norm(state[rr, c0 + 1, :] - state[rr, c0, :]))
                    elif dj == 1:
                        for rr in range(r0, r1):
                            bp_diffs.append(np.linalg.norm(state[rr, c1 - 1, :] - state[rr, c1, :]))
                            if c1 - 2 >= c0:
                                ip_diffs.append(np.linalg.norm(state[rr, c1 - 2, :] - state[rr, c1 - 1, :]))
        jo = robust_jump(bp_diffs)
        ji = robust_jump(ip_diffs) if ip_diffs else 0.0
        return jo / (ji + EPS)

    ss_hard = seam_at_boundary(hard)
    ss_halo = seam_at_boundary(halo_out)
    return ss_hard, ss_halo


# =====================================================================
# T3: Irregular Graph (k-NN point cloud)
# =====================================================================

def t3_run(seed):
    n_points, k, n_clusters = 500, 8, 10
    rng = np.random.RandomState(seed)
    pos = rng.rand(n_points, 2)
    tree = cKDTree(pos)
    _, idx = tree.query(pos, k=k + 1)
    neighbors = {i: set(idx[i, 1:]) for i in range(n_points)}

    gt = (0.5 * np.sin(4 * np.pi * pos[:, 0]) * np.cos(3 * np.pi * pos[:, 1])
          + 0.7 * (pos[:, 0] > 0.4).astype(float)
          + rng.randn(n_points) * 0.03)

    _, labels = kmeans2(pos, n_clusters, minit='points', seed=seed)
    coarse = np.zeros(n_points)
    for c in range(n_clusters):
        mask = labels == c
        coarse[mask] = gt[mask].mean()

    delta = gt - coarse

    # Select active clusters (~30%)
    cluster_errors = []
    for c in range(n_clusters):
        pts = np.where(labels == c)[0]
        cluster_errors.append((c, np.mean(delta[pts] ** 2)))
    cluster_errors.sort(key=lambda x: -x[1])
    n_active = max(1, int(BUDGET * n_clusters))
    active_clusters = set(c for c, _ in cluster_errors[:n_active])

    active_pts = set()
    for c in active_clusters:
        active_pts |= set(np.where(labels == c)[0])

    # Hard insert: copy GT into active clusters only
    hard = coarse.copy()
    for p in active_pts:
        hard[p] = gt[p]

    # Halo insert: cosine feathering over hops
    hops = HALO_WIDTH
    d_acc = np.zeros(n_points, dtype=np.float64)
    w_acc = np.zeros(n_points, dtype=np.float64)
    for cid in active_clusters:
        cluster = set(np.where(labels == cid)[0])
        for p in cluster:
            d_acc[p] += delta[p]
            w_acc[p] += 1.0
        visited = set(cluster)
        frontier = set(cluster)
        for hop in range(1, hops + 1):
            nf = set()
            for p in frontier:
                for nb in neighbors[p]:
                    if nb not in visited:
                        visited.add(nb)
                        nf.add(nb)
            fade = 0.5 * (1 + np.cos(np.pi * hop / (hops + 1)))
            for p in nf:
                d_acc[p] += fade * delta[p]
                w_acc[p] += fade
            frontier = nf
    halo_out = coarse.copy()
    valid = w_acc > 1e-12
    halo_out[valid] = coarse[valid] + d_acc[valid] / w_acc[valid]

    # SeamScore at active/inactive cluster boundary
    def seam_at_boundary(state):
        bp_diffs = []
        ip_diffs = []
        for p in active_pts:
            for nb in neighbors[p]:
                if nb not in active_pts:
                    bp_diffs.append(abs(state[p] - state[nb]))
            for nb in neighbors[p]:
                if nb in active_pts:
                    ip_diffs.append(abs(state[p] - state[nb]))
        jo = robust_jump(bp_diffs)
        ji = robust_jump(ip_diffs) if ip_diffs else 0.0
        return jo / (ji + EPS)

    ss_hard = seam_at_boundary(hard)
    ss_halo = seam_at_boundary(halo_out)
    return ss_hard, ss_halo


# =====================================================================
# T4: Tree Hierarchy
# =====================================================================

def _t4_children(i, n):
    c = []
    if 2 * i + 1 < n:
        c.append(2 * i + 1)
    if 2 * i + 2 < n:
        c.append(2 * i + 2)
    return c


def _t4_parent(i):
    return (i - 1) // 2 if i > 0 else None


def _t4_subtree(i, n):
    nodes = [i]
    q = [i]
    while q:
        curr = q.pop()
        for c in _t4_children(curr, n):
            nodes.append(c)
            q.append(c)
    return nodes


def _t4_neighbors(i, n):
    nb = set()
    p = _t4_parent(i)
    if p is not None:
        nb.add(p)
        for c in _t4_children(p, n):
            if c != i:
                nb.add(c)
    for c in _t4_children(i, n):
        nb.add(c)
    return nb


def t4_run(seed):
    depth = 8
    n = 2 ** depth - 1
    rng = np.random.RandomState(seed)

    gt = np.zeros(n)
    for i in range(n):
        d = int(np.log2(i + 1))
        gt[i] = 0.5 * np.sin(i * 0.3) / (1 + d * 0.2) + rng.randn() * 0.05
        if d >= 3:
            gt[i] += 0.3 * ((i % 7) > 3)

    coarse_depth = 3
    coarse = np.zeros(n)
    for i in range(n):
        d = int(np.log2(i + 1))
        if d < coarse_depth:
            coarse[i] = gt[i]
        else:
            ancestor = i
            while int(np.log2(ancestor + 1)) >= coarse_depth:
                ancestor = (ancestor - 1) // 2
            subtree = _t4_subtree(ancestor, n)
            coarse[i] = np.mean([gt[j] for j in subtree])

    delta = gt - coarse

    # Subtrees at depth_level=3
    subtree_roots = [i for i in range(n) if int(np.log2(i + 1)) == 3]

    # Select active subtrees (~30%)
    subtree_errors = []
    for root in subtree_roots:
        pts = _t4_subtree(root, n)
        subtree_errors.append((root, np.mean([delta[p] ** 2 for p in pts])))
    subtree_errors.sort(key=lambda x: -x[1])
    n_active = max(1, int(BUDGET * len(subtree_roots)))
    active_roots = set(r for r, _ in subtree_errors[:n_active])

    active_pts = set()
    for root in active_roots:
        active_pts |= set(_t4_subtree(root, n))

    # Hard insert
    hard = coarse.copy()
    for p in active_pts:
        hard[p] = gt[p]

    # Halo insert
    hops = HALO_WIDTH
    d_acc = np.zeros(n, dtype=np.float64)
    w_acc = np.zeros(n, dtype=np.float64)
    for root in active_roots:
        core = set(_t4_subtree(root, n))
        for p in core:
            d_acc[p] += delta[p]
            w_acc[p] += 1.0
        visited = set(core)
        frontier = set(core)
        for hop in range(1, hops + 1):
            nf = set()
            for p in frontier:
                for nb in _t4_neighbors(p, n):
                    if nb not in visited:
                        visited.add(nb)
                        nf.add(nb)
            fade = 0.5 * (1 + np.cos(np.pi * hop / (hops + 1)))
            for p in nf:
                d_acc[p] += fade * delta[p]
                w_acc[p] += fade
            frontier = nf
    halo_out = coarse.copy()
    valid = w_acc > 1e-12
    halo_out[valid] = coarse[valid] + d_acc[valid] / w_acc[valid]

    # SeamScore
    def seam_at_boundary(state):
        bp_diffs = []
        ip_diffs = []
        for p in active_pts:
            for nb in _t4_neighbors(p, n):
                if nb not in active_pts:
                    bp_diffs.append(abs(state[p] - state[nb]))
            for nb in _t4_neighbors(p, n):
                if nb in active_pts:
                    ip_diffs.append(abs(state[p] - state[nb]))
        jo = robust_jump(bp_diffs)
        ji = robust_jump(ip_diffs) if ip_diffs else 0.0
        return jo / (ji + EPS)

    ss_hard = seam_at_boundary(hard)
    ss_halo = seam_at_boundary(halo_out)
    return ss_hard, ss_halo


# =====================================================================
# Holm-Bonferroni correction
# =====================================================================

def holm_bonferroni(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.zeros(m)
    cummax = 0.0
    for rank, idx in enumerate(order):
        adj = pvals[idx] * (m - rank)
        adj = max(adj, cummax)
        adj = min(adj, 1.0)
        adjusted[idx] = adj
        cummax = adj
    return adjusted


# =====================================================================
# Main
# =====================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    spaces = [
        ("T1_scalar_grid_64x64", t1_run),
        ("T2_vector_grid_64x64_d32", t2_run),
        ("T3_irregular_graph_500_k8", t3_run),
        ("T4_tree_depth8_255nodes", t4_run),
    ]

    seeds = list(range(100, 100 + N_SEEDS))

    print("=" * 70)
    print("S2 -- Halo Cross-Space Validation")
    print(f"  Halo width: {HALO_WIDTH} elements/hops")
    print(f"  Budget: {BUDGET:.0%} of tiles refined")
    print(f"  Seeds: {N_SEEDS}")
    print(f"  Spaces: {len(spaces)}")
    print("=" * 70)

    all_results = {}
    raw_pvals = []

    for space_name, run_fn in spaces:
        print(f"\n[{space_name}]")
        ss_hard_list = []
        ss_halo_list = []

        for si, seed in enumerate(seeds):
            ss_hard, ss_halo = run_fn(seed)
            ss_hard_list.append(ss_hard)
            ss_halo_list.append(ss_halo)
            if (si + 1) % 5 == 0:
                print(f"  seed {si+1}/{N_SEEDS}: "
                      f"ss_hard={ss_hard:.4f}, ss_halo={ss_halo:.4f}, "
                      f"ratio={ss_hard / (ss_halo + EPS):.2f}")

        ss_hard_arr = np.array(ss_hard_list)
        ss_halo_arr = np.array(ss_halo_list)
        ratios = ss_hard_arr / (ss_halo_arr + EPS)

        # Paired Wilcoxon: H0 = no difference; H1 = hard > halo
        try:
            stat_w, p_val = stats.wilcoxon(ss_hard_arr, ss_halo_arr, alternative='greater')
        except ValueError:
            stat_w, p_val = 0.0, 1.0

        median_ratio = float(np.median(ratios))
        mean_ratio = float(np.mean(ratios))
        min_ratio = float(np.min(ratios))

        all_results[space_name] = {
            "ss_hard": ss_hard_list,
            "ss_halo": ss_halo_list,
            "ratios": ratios.tolist(),
            "median_ratio": median_ratio,
            "mean_ratio": mean_ratio,
            "min_ratio": min_ratio,
            "wilcoxon_stat": float(stat_w),
            "p_value_raw": float(p_val),
        }
        raw_pvals.append(float(p_val))

        print(f"  SeamScore hard:  median={np.median(ss_hard_arr):.4f}, "
              f"mean={np.mean(ss_hard_arr):.4f}")
        print(f"  SeamScore halo:  median={np.median(ss_halo_arr):.4f}, "
              f"mean={np.mean(ss_halo_arr):.4f}")
        print(f"  Ratio (hard/halo): median={median_ratio:.3f}, "
              f"mean={mean_ratio:.3f}, min={min_ratio:.3f}")
        print(f"  Wilcoxon: W={stat_w:.1f}, p={p_val:.2e}")

    # Holm-Bonferroni
    adjusted = holm_bonferroni(np.array(raw_pvals))
    for i, (sn, _) in enumerate(spaces):
        all_results[sn]["p_value_adjusted"] = float(adjusted[i])

    # Results table
    print("\n" + "=" * 70)
    print("HOLM-BONFERRONI CORRECTED RESULTS")
    print("=" * 70)
    print(f"{'Space':40s} {'Med.Ratio':>10s} {'p_raw':>12s} {'p_adj':>12s} {'Sig?':>6s} {'>=1.5x?':>8s}")
    print("-" * 90)

    all_pass = True
    all_sig = True
    for i, (sn, _) in enumerate(spaces):
        r = all_results[sn]
        sig = "YES" if adjusted[i] < 0.05 else "NO"
        passes = "YES" if r["median_ratio"] >= 1.5 else "NO"
        if r["median_ratio"] < 1.5:
            all_pass = False
        if adjusted[i] >= 0.05:
            all_sig = False
        print(f"  {sn:38s} {r['median_ratio']:10.3f} {r['p_value_raw']:12.2e} "
              f"{adjusted[i]:12.2e} {sig:>6s} {passes:>8s}")

    print("\n" + "=" * 70)
    if all_pass and all_sig:
        verdict = "PASS"
        print("VERDICT: PASS -- Halo reduces seam artifacts by >= 1.5x on ALL 4 space types")
        print("         with statistical significance (Holm-Bonferroni p < 0.05)")
    elif all_pass:
        verdict = "PASS (marginal)"
        print("VERDICT: PASS (marginal) -- Ratio >= 1.5x on all spaces,")
        print("         but not all statistically significant after correction")
    else:
        verdict = "FAIL"
        failing = [sn for sn, _ in spaces if all_results[sn]["median_ratio"] < 1.5]
        print(f"VERDICT: FAIL -- Ratio < 1.5x on: {', '.join(failing)}")
    print("=" * 70)

    all_results["_meta"] = {
        "n_seeds": N_SEEDS,
        "halo_width": HALO_WIDTH,
        "budget": BUDGET,
        "alpha": 0.05,
        "correction": "Holm-Bonferroni",
        "kill_criterion": "median_ratio >= 1.5 on all 4 spaces",
        "verdict": verdict,
    }

    out_path = OUT_DIR / "halo_crossspace.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, '__float__') else str(x))
    print(f"\nSaved: {out_path}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(f"S2: Halo Cross-Space Validation (w={HALO_WIDTH}, budget={BUDGET:.0%}, {N_SEEDS} seeds)",
                     fontsize=14, fontweight="bold")

        for ax, (sn, _) in zip(axes, spaces):
            r = all_results[sn]
            ratios = np.array(r["ratios"])
            ax.hist(ratios, bins=12, color="#2ca02c", alpha=0.7, edgecolor="black")
            ax.axvline(1.5, color="red", ls="--", lw=2, label="kill=1.5x")
            ax.axvline(r["median_ratio"], color="blue", ls="-", lw=2,
                       label=f"median={r['median_ratio']:.2f}")
            ax.set_xlabel("Seam reduction ratio (hard / halo)")
            ax.set_ylabel("Count")
            short = sn.split("_", 1)[0]
            ax.set_title(f"{short}: p_adj={r['p_value_adjusted']:.2e}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = OUT_DIR / "halo_crossspace.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fig_path}")
    except Exception as e:
        print(f"Plot skipped: {e}")

    return verdict


if __name__ == "__main__":
    verdict = main()
    sys.exit(0 if "PASS" in verdict else 1)
