#!/usr/bin/env python3
"""
Curiosity -- Scale-Consistency Baseline: Cross-Space Validation (SC-0..SC-3)

Runs SC-0 through SC-3 on ALL 4 space types:
  T1: Scalar grid 64x64
  T2: Vector grid 64x64 dim=32
  T3: Irregular graph 500 pts k=8
  T4: Tree hierarchy depth=8

Uses best D_parent config: R=gauss_s3.0 + lf_frac normalization +
d_abs_vs_signed auxiliary + two_stage combiner.

Uses FIXED coarse_shift generators from baselines_v2.py (coherent, block, gradient).

Key questions:
  1. Does D_parent work equally well across all 4 space types?
  2. Does the Halo applicability rule affect SC results on trees?
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from scipy.spatial import cKDTree
from scipy.cluster.vq import kmeans2

sys.path.insert(0, str(Path(__file__).parent))

from operators import (
    restrict_scalar, prolong_scalar,
    restrict_vector, prolong_vector,
    restrict_graph, prolong_graph,
    restrict_tree, prolong_tree,
)
from operators_v2 import make_restrict_gaussian, prolong_gaussian
from baselines_v2 import (
    positive_oracle, positive_scaled, positive_noisy,
    negative_lf_drift, negative_random_lf, negative_semant_wrong,
    negative_coarse_shift_coherent,
    negative_coarse_shift_block,
    negative_coarse_shift_gradient,
)


# ===================================================
# R/Up operators per space type
# ===================================================

def make_grid_ops(sigma=3.0):
    """R/Up for scalar grid using gauss sigma=3.0."""
    R_fn = make_restrict_gaussian(sigma)
    Up_fn = lambda xc, tgt: prolong_gaussian(xc, tgt)
    return R_fn, Up_fn


def make_vector_ops(sigma=3.0):
    """R/Up for vector grid: per-channel gauss sigma + decimation."""
    def R_fn(x):
        D = x.shape[2]
        out = np.empty((x.shape[0] // 2, x.shape[1] // 2, D))
        for d in range(D):
            blurred = gaussian_filter(x[:, :, d], sigma=sigma)
            out[:, :, d] = blurred[::2, ::2]
        return out

    def Up_fn(xc, tgt):
        D = xc.shape[2]
        H, W = tgt[0], tgt[1]
        out = np.empty((H, W, D))
        factors = (H / xc.shape[0], W / xc.shape[1])
        for d in range(D):
            out[:, :, d] = zoom(xc[:, :, d], factors, order=1)
        return out

    return R_fn, Up_fn


class GraphOps:
    """R/Up for irregular graph."""
    def __init__(self, labels, n_clusters, n_points):
        self.labels = labels
        self.n_clusters = n_clusters
        self.n_points = n_points

    def restrict(self, values):
        return restrict_graph(values, self.labels, self.n_clusters)

    def prolong(self, coarse_values, target_shape):
        return prolong_graph(coarse_values, self.labels, self.n_points)


class TreeOps:
    """R/Up for tree hierarchy."""
    def __init__(self, n_nodes, coarse_depth):
        self.n_nodes = n_nodes
        self.coarse_depth = coarse_depth

    def restrict(self, values):
        return restrict_tree(values, self.n_nodes, self.coarse_depth)

    def prolong(self, coarse_values, target_shape):
        return prolong_tree(coarse_values, self.n_nodes, self.coarse_depth)


# ===================================================
# D_parent metrics (best config)
# ===================================================

def d_parent_lf_frac(delta, coarse, R_fn, Up_fn):
    """||Up(R(delta))|| / ||delta|| -- LF fraction."""
    R_delta = R_fn(delta)
    Up_R_delta = Up_fn(R_delta, delta.shape)
    numer = np.linalg.norm(Up_R_delta.ravel())
    denom = np.linalg.norm(delta.ravel()) + 1e-12
    return numer / denom


def det_abs_vs_signed(delta, coarse, R_fn):
    """||R(|delta|)|| / ||R(delta)|| -- sign cancellation detector."""
    n_abs = np.linalg.norm(R_fn(np.abs(delta)).ravel())
    n_d = np.linalg.norm(R_fn(delta).ravel())
    if n_d < 1e-12:
        return 1.0
    return n_abs / n_d


def combine_two_stage(d_parent_val, d_shift_val):
    """Two-stage: D_parent + 0.3 * D_shift (both z-scored before calling)."""
    if d_shift_val is None:
        return d_parent_val
    return d_parent_val + 0.3 * d_shift_val


# ===================================================
# Space data generators
# ===================================================

def make_scalar_grid(N=64, seed=42):
    """T1: Scalar grid NxN with tile-mean coarse."""
    rng = np.random.RandomState(seed)
    x = np.linspace(0, 1, N, endpoint=False)
    xx, yy = np.meshgrid(x, x)
    gt = (0.3 * np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy)
          + 0.8 * (xx > 0.45).astype(float)
          + 0.2 * np.sin(2 * np.pi * 12 * xx) * np.cos(2 * np.pi * 10 * yy)
          + rng.randn(N, N) * 0.01)
    T = 8; NT = N // T
    coarse = np.zeros_like(gt)
    for ti in range(NT):
        for tj in range(NT):
            s = slice(ti * T, (ti + 1) * T)
            cs = slice(tj * T, (tj + 1) * T)
            coarse[s, cs] = gt[s, cs].mean()
    return gt, coarse, {'type': 'scalar_grid', 'N': N, 'T': T}


def make_vector_grid(N=64, D=32, seed=42):
    """T2: Vector grid NxN dim=D."""
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
    T = 8; NT = N // T
    coarse = np.zeros_like(gt)
    for ti in range(NT):
        for tj in range(NT):
            s = slice(ti * T, (ti + 1) * T)
            cs = slice(tj * T, (tj + 1) * T)
            coarse[s, cs, :] = gt[s, cs, :].mean(axis=(0, 1), keepdims=True)
    return gt, coarse, {'type': 'vector_grid', 'N': N, 'D': D, 'T': T}


def make_graph(n_points=500, k=8, n_clusters=25, seed=42):
    """T3: Irregular graph with k-NN edges."""
    rng = np.random.RandomState(seed)
    positions = rng.rand(n_points, 2)
    tree = cKDTree(positions)
    _, idx = tree.query(positions, k=k + 1)
    neighbors = {i: set(idx[i, 1:]) for i in range(n_points)}

    gt = (0.5 * np.sin(4 * np.pi * positions[:, 0]) * np.cos(3 * np.pi * positions[:, 1])
          + 0.7 * (positions[:, 0] > 0.4).astype(float)
          + rng.randn(n_points) * 0.03)

    _, labels = kmeans2(positions, n_clusters, minit='points', seed=seed)
    coarse = np.zeros(n_points)
    for c in range(n_clusters):
        mask = labels == c
        if mask.any():
            coarse[mask] = gt[mask].mean()

    info = {
        'type': 'graph',
        'n_points': n_points, 'k': k, 'n_clusters': n_clusters,
        'positions': positions, 'neighbors': neighbors, 'labels': labels,
    }
    return gt, coarse, info


def make_tree(depth=8, seed=42):
    """T4: Binary tree with depth levels."""
    rng = np.random.RandomState(seed)
    n_nodes = 2 ** depth - 1
    coarse_depth = max(3, depth // 2)

    gt = np.zeros(n_nodes)
    for i in range(n_nodes):
        d = int(np.log2(i + 1))
        gt[i] = 0.5 * np.sin(i * 0.3) / (1 + d * 0.2) + rng.randn() * 0.05
        if d >= 3:
            gt[i] += 0.3 * ((i % 7) > 3)

    coarse = np.zeros(n_nodes)
    for i in range(n_nodes):
        d = int(np.log2(i + 1))
        if d < coarse_depth:
            coarse[i] = gt[i]
        else:
            ancestor = i
            while int(np.log2(ancestor + 1)) >= coarse_depth and ancestor > 0:
                ancestor = (ancestor - 1) // 2
            subtree = _subtree_nodes(ancestor, n_nodes)
            coarse[i] = np.mean([gt[j] for j in subtree])

    info = {
        'type': 'tree', 'depth': depth,
        'n_nodes': n_nodes, 'coarse_depth': coarse_depth,
    }
    return gt, coarse, info


def _subtree_nodes(root, n_nodes):
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


# ===================================================
# SC-0: Idempotence per space type
# ===================================================

def sc0_idempotence(verbose=True):
    results = {}
    IDEMPOTENCE_TOL = 0.30

    if verbose:
        print("=" * 70)
        print("SC-0: Idempotence Check (all 4 space types)")
        print("=" * 70)

    # T1
    gt1, coarse1, info1 = make_scalar_grid(64, seed=42)
    R1, Up1 = make_grid_ops(3.0)
    Rc = R1(coarse1)
    UpRc = Up1(Rc, coarse1.shape)
    RUpRc = R1(UpRc)
    err = np.linalg.norm(Rc - RUpRc) / (np.linalg.norm(Rc) + 1e-12)
    results['T1_scalar'] = float(err)

    # T2
    gt2, coarse2, info2 = make_vector_grid(64, 32, seed=42)
    R2, Up2 = make_vector_ops(3.0)
    Rc2 = R2(coarse2)
    UpRc2 = Up2(Rc2, coarse2.shape)
    RUpRc2 = R2(UpRc2)
    err2 = np.linalg.norm(Rc2.ravel() - RUpRc2.ravel()) / (np.linalg.norm(Rc2.ravel()) + 1e-12)
    results['T2_vector'] = float(err2)

    # T3
    gt3, coarse3, info3 = make_graph(500, 8, 25, seed=42)
    gops = GraphOps(info3['labels'], info3['n_clusters'], info3['n_points'])
    Rc3 = gops.restrict(coarse3)
    UpRc3 = gops.prolong(Rc3, coarse3.shape)
    RUpRc3 = gops.restrict(UpRc3)
    err3 = np.linalg.norm(Rc3 - RUpRc3) / (np.linalg.norm(Rc3) + 1e-12)
    results['T3_graph'] = float(err3)

    # T4
    gt4, coarse4, info4 = make_tree(8, seed=42)
    tops = TreeOps(info4['n_nodes'], info4['coarse_depth'])
    Rc4 = tops.restrict(coarse4)
    UpRc4 = tops.prolong(Rc4, coarse4.shape)
    RUpRc4 = tops.restrict(UpRc4)
    err4 = np.linalg.norm(Rc4 - RUpRc4) / (np.linalg.norm(Rc4) + 1e-12)
    results['T4_tree'] = float(err4)

    all_pass = True
    if verbose:
        for name, err in results.items():
            status = "OK" if err < IDEMPOTENCE_TOL else "KILL"
            if err >= IDEMPOTENCE_TOL:
                all_pass = False
            print(f"  {name}: ||R - R(Up(R))|| / ||R|| = {err:.2e}  [{status}]")
        if all_pass:
            print("  All idempotence checks passed.\n")

    return results, all_pass


# ===================================================
# SC-1 + SC-2: Generate cases and compute metrics per space type
# ===================================================

def generate_and_compute(space_name, gt, coarse, R_fn, Up_fn, space_info,
                         n_seeds=5, base_seed=42):
    """Generate pos/neg baselines and compute D_parent (lf_frac + abs_vs_signed)."""
    cases = []

    for seed_offset in range(n_seeds):
        seed = base_seed + seed_offset

        # Positive baselines
        for pos_type, delta in [
            ('oracle', positive_oracle(gt, coarse)),
            ('scaled', positive_scaled(gt, coarse, 0.5, seed)),
            ('noisy', positive_noisy(gt, coarse, 0.005, seed)),
        ]:
            dp = d_parent_lf_frac(delta, coarse, R_fn, Up_fn)
            ds = det_abs_vs_signed(delta, coarse, R_fn)
            cases.append({
                'space': space_name,
                'case_type': 'pos',
                'pos_type': pos_type,
                'neg_type': None,
                'seed': seed,
                'd_parent_lf_frac': float(dp),
                'd_abs_vs_signed': float(ds),
            })

        # Negative baselines
        neg_generators = [
            ('lf_drift', lambda s: negative_lf_drift(gt, coarse, 0.3, 0.5, s)),
            ('random_lf', lambda s: negative_random_lf(coarse, sigma=8.0, amplitude=0.5, seed=s)),
            ('semant_wrong', lambda s: negative_semant_wrong(coarse, 1.0)),
        ]

        # coarse_shift variants (the FIXED ones)
        for cs_variant, cs_fn in [
            ('coarse_shift_coherent', negative_coarse_shift_coherent),
            ('coarse_shift_block', negative_coarse_shift_block),
            ('coarse_shift_gradient', negative_coarse_shift_gradient),
        ]:
            neg_generators.append(
                (cs_variant, lambda s, _fn=cs_fn: _fn(gt, coarse, 0.2, s, space_info))
            )

        for neg_type, gen_fn in neg_generators:
            delta = gen_fn(seed)
            dp = d_parent_lf_frac(delta, coarse, R_fn, Up_fn)
            ds = det_abs_vs_signed(delta, coarse, R_fn)
            cases.append({
                'space': space_name,
                'case_type': 'neg',
                'pos_type': None,
                'neg_type': neg_type,
                'seed': seed,
                'd_parent_lf_frac': float(dp),
                'd_abs_vs_signed': float(ds),
            })

    return cases


# ===================================================
# SC-3: Separability analysis
# ===================================================

def roc_auc(pos_scores, neg_scores):
    from sklearn.metrics import roc_auc_score
    if len(pos_scores) < 2 or len(neg_scores) < 2:
        return float('nan')
    y_true = np.array([0] * len(pos_scores) + [1] * len(neg_scores))
    y_score = np.array(list(pos_scores) + list(neg_scores))
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float('nan')


def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return float('nan')
    m1, m2 = np.mean(g1), np.mean(g2)
    s1, s2 = np.std(g1, ddof=1), np.std(g2, ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    if pooled < 1e-12:
        return float('nan')
    return float(abs(m1 - m2) / pooled)


def zscore_combine(cases):
    """Z-score normalize d_parent_lf_frac and d_abs_vs_signed, then combine."""
    dp_vals = np.array([c['d_parent_lf_frac'] for c in cases])
    ds_vals = np.array([c['d_abs_vs_signed'] for c in cases])

    dp_mu, dp_s = dp_vals.mean(), dp_vals.std()
    ds_mu, ds_s = ds_vals.mean(), ds_vals.std()
    if dp_s < 1e-12: dp_s = 1.0
    if ds_s < 1e-12: ds_s = 1.0

    for i, c in enumerate(cases):
        dp_z = (dp_vals[i] - dp_mu) / dp_s
        ds_z = (ds_vals[i] - ds_mu) / ds_s
        c['score'] = float(combine_two_stage(dp_z, ds_z))

    return cases


def analyze_space(cases, space_name, verbose=True):
    """Full SC-3 analysis for one space type.

    Primary metric: D_parent_lf_frac (works universally with fixed generators).
    Secondary: combined lf_frac + abs_vs_signed (reported for comparison).
    The abs_vs_signed detector was designed for the OLD broken sign-flip generator
    and can invert on vector data with coherent shifts, so lf_frac alone is primary.
    """
    cases = zscore_combine(cases)

    # Primary: lf_frac only
    pos_dp = [c['d_parent_lf_frac'] for c in cases if c['case_type'] == 'pos']
    neg_dp = [c['d_parent_lf_frac'] for c in cases if c['case_type'] == 'neg']

    result = {
        'space': space_name,
        'n_pos': len(pos_dp),
        'n_neg': len(neg_dp),
        'global_auc': roc_auc(pos_dp, neg_dp),
        'cohens_d': cohens_d(pos_dp, neg_dp),
        'pos_mean': float(np.mean(pos_dp)),
        'neg_mean': float(np.mean(neg_dp)),
        'pos_std': float(np.std(pos_dp)),
        'neg_std': float(np.std(neg_dp)),
    }

    # Per negative type (using lf_frac)
    neg_types = sorted(set(c['neg_type'] for c in cases if c['neg_type']))
    result['by_neg_type'] = {}
    for nt in neg_types:
        nv = [c['d_parent_lf_frac'] for c in cases if c['case_type'] == 'neg' and c['neg_type'] == nt]
        auc = roc_auc(pos_dp, nv)
        d = cohens_d(pos_dp, nv)
        result['by_neg_type'][nt] = {'auc': auc, 'cohens_d': d, 'n': len(nv),
                                      'mean': float(np.mean(nv))}

    # Secondary: combined score (for reference)
    pos_combined = [c['score'] for c in cases if c['case_type'] == 'pos']
    neg_combined = [c['score'] for c in cases if c['case_type'] == 'neg']
    result['combined_auc'] = roc_auc(pos_combined, neg_combined)
    result['combined_cohens_d'] = cohens_d(pos_combined, neg_combined)

    # Per-type combined (for comparison)
    result['combined_by_neg_type'] = {}
    for nt in neg_types:
        nv = [c['score'] for c in cases if c['case_type'] == 'neg' and c['neg_type'] == nt]
        result['combined_by_neg_type'][nt] = {
            'auc': roc_auc(pos_combined, nv),
            'cohens_d': cohens_d(pos_combined, nv),
        }

    if verbose:
        print(f"\n  [{space_name}] lf_frac AUC={result['global_auc']:.4f}  "
              f"Cohen's d={result['cohens_d']:.4f}  "
              f"(combined AUC={result['combined_auc']:.4f})")
        print(f"    pos: n={result['n_pos']} mean={result['pos_mean']:.4f} std={result['pos_std']:.4f}")
        print(f"    neg: n={result['n_neg']} mean={result['neg_mean']:.4f} std={result['neg_std']:.4f}")
        for nt in neg_types:
            info = result['by_neg_type'][nt]
            comb = result['combined_by_neg_type'][nt]
            print(f"    vs {nt:30s}: lf_frac AUC={info['auc']:.4f}  "
                  f"combined AUC={comb['auc']:.4f}  d={info['cohens_d']:.4f}  n={info['n']}")

    return result


# ===================================================
# Main
# ===================================================

def main():
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Curiosity -- SC Baseline Cross-Space Validation")
    print("Config: R=gauss_s3.0 + lf_frac + abs_vs_signed + two_stage")
    print("Coarse-shift variants: coherent, block, gradient (FIXED)")
    print("=" * 70)

    # SC-0
    idem_results, idem_pass = sc0_idempotence(verbose=True)
    if not idem_pass:
        print("\nAborting: idempotence check failed.")
        sys.exit(1)

    all_results = {'sc0': idem_results, 'spaces': {}}

    # --- T1: Scalar grid ---
    print("\n" + "=" * 70)
    print("T1: Scalar Grid 64x64")
    print("=" * 70)
    gt1, coarse1, info1 = make_scalar_grid(64, seed=42)
    R1, Up1 = make_grid_ops(3.0)
    cases1 = generate_and_compute('T1_scalar', gt1, coarse1, R1, Up1, info1, n_seeds=5)
    res1 = analyze_space(cases1, 'T1_scalar')
    all_results['spaces']['T1'] = res1

    # --- T2: Vector grid ---
    print("\n" + "=" * 70)
    print("T2: Vector Grid 64x64 dim=32")
    print("=" * 70)
    gt2, coarse2, info2 = make_vector_grid(64, 32, seed=42)
    R2, Up2 = make_vector_ops(3.0)
    cases2 = generate_and_compute('T2_vector', gt2, coarse2, R2, Up2, info2, n_seeds=5)
    res2 = analyze_space(cases2, 'T2_vector')
    all_results['spaces']['T2'] = res2

    # --- T3: Irregular graph ---
    print("\n" + "=" * 70)
    print("T3: Irregular Graph 500 pts k=8")
    print("=" * 70)
    gt3, coarse3, info3 = make_graph(500, 8, 25, seed=42)
    gops = GraphOps(info3['labels'], info3['n_clusters'], info3['n_points'])
    R3 = gops.restrict
    Up3 = gops.prolong
    cases3 = generate_and_compute('T3_graph', gt3, coarse3, R3, Up3, info3, n_seeds=5)
    res3 = analyze_space(cases3, 'T3_graph')
    all_results['spaces']['T3'] = res3

    # --- T4: Tree hierarchy ---
    print("\n" + "=" * 70)
    print("T4: Tree Hierarchy depth=8")
    print("=" * 70)
    gt4, coarse4, info4 = make_tree(8, seed=42)
    tops = TreeOps(info4['n_nodes'], info4['coarse_depth'])
    R4 = tops.restrict
    Up4 = tops.prolong
    cases4 = generate_and_compute('T4_tree', gt4, coarse4, R4, Up4, info4, n_seeds=5)
    res4 = analyze_space(cases4, 'T4_tree')
    all_results['spaces']['T4'] = res4

    # ===================================================
    # Cross-space summary
    # ===================================================
    print("\n" + "=" * 70)
    print("CROSS-SPACE SUMMARY")
    print("=" * 70)

    print(f"\n  {'Space':15s} {'lf_frac AUC':>12s} {'Cohen d':>10s} {'comb AUC':>10s} "
          f"{'cs_coh':>8s} {'cs_blk':>8s} {'cs_grad':>8s} {'lf_drift':>8s} {'rand_lf':>8s} {'sem_wr':>8s}")
    print(f"  {'-'*15} {'-'*12} {'-'*10} {'-'*10} "
          f"{'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for tkey in ['T1', 'T2', 'T3', 'T4']:
        r = all_results['spaces'][tkey]
        nt = r['by_neg_type']
        cs_coh = nt.get('coarse_shift_coherent', {}).get('auc', float('nan'))
        cs_blk = nt.get('coarse_shift_block', {}).get('auc', float('nan'))
        cs_grad = nt.get('coarse_shift_gradient', {}).get('auc', float('nan'))
        lf = nt.get('lf_drift', {}).get('auc', float('nan'))
        rlf = nt.get('random_lf', {}).get('auc', float('nan'))
        sw = nt.get('semant_wrong', {}).get('auc', float('nan'))
        print(f"  {r['space']:15s} {r['global_auc']:12.4f} {r['cohens_d']:10.4f} "
              f"{r['combined_auc']:10.4f} "
              f"{cs_coh:8.4f} {cs_blk:8.4f} {cs_grad:8.4f} {lf:8.4f} {rlf:8.4f} {sw:8.4f}")

    # Kill criteria check per space
    print(f"\n  Kill Criteria (Global AUC >= 0.75, Cohen's d >= 0.5):")
    for tkey in ['T1', 'T2', 'T3', 'T4']:
        r = all_results['spaces'][tkey]
        auc_ok = r['global_auc'] >= 0.75
        d_ok = r['cohens_d'] >= 0.5
        status = "PASS" if (auc_ok and d_ok) else "FAIL"
        print(f"    {r['space']:15s}: AUC={r['global_auc']:.4f} {'OK' if auc_ok else 'FAIL'}  "
              f"d={r['cohens_d']:.4f} {'OK' if d_ok else 'FAIL'}  [{status}]")

    # Halo applicability analysis
    print(f"\n  Halo Applicability Impact:")
    print(f"    T1-T3 support Halo (boundary parallelism >= 3)")
    print(f"    T4 (tree) does NOT support Halo (single parent edge, context leakage)")
    t4 = all_results['spaces']['T4']
    t1 = all_results['spaces']['T1']
    print(f"    T1 (grid) global AUC = {t1['global_auc']:.4f}")
    print(f"    T4 (tree) global AUC = {t4['global_auc']:.4f}")
    diff = t4['global_auc'] - t1['global_auc']
    print(f"    Difference: {diff:+.4f}")
    if abs(diff) < 0.05:
        print(f"    -> Halo non-applicability does NOT significantly affect D_parent SC results")
    else:
        direction = "better" if diff > 0 else "worse"
        print(f"    -> Tree is {direction} by {abs(diff):.4f} -- may need space-specific tuning")

    # Space-specific tuning question
    print(f"\n  Space-Specific Tuning Assessment:")
    aucs = [all_results['spaces'][t]['global_auc'] for t in ['T1', 'T2', 'T3', 'T4']]
    spread = max(aucs) - min(aucs)
    mean_auc = np.mean(aucs)
    print(f"    AUC range: [{min(aucs):.4f}, {max(aucs):.4f}]  spread={spread:.4f}  mean={mean_auc:.4f}")
    if spread < 0.10:
        print(f"    -> D_parent generalizes well across all 4 space types (spread < 0.10)")
        print(f"    -> No space-specific tuning needed")
    elif spread < 0.20:
        print(f"    -> Moderate variation across spaces (0.10 <= spread < 0.20)")
        print(f"    -> Consider space-specific R sigma or normalization")
    else:
        print(f"    -> Large variation across spaces (spread >= 0.20)")
        print(f"    -> Space-specific tuning is recommended")

    # Save results
    def jd(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return str(obj)

    out_file = out_dir / "crossspace_sc_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2, default=jd)
    print(f"\n[Saved] {out_file}")

    # Generate report
    _write_report(all_results, out_dir)

    print("[Done]")
    return all_results


def _write_report(results, out_dir):
    """Write markdown report."""
    lines = []
    lines.append("# Cross-Space SC-Baseline Report")
    lines.append("")
    lines.append("**Date:** 2026-03-18")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append("- R operator: Gaussian blur sigma=3.0 + decimation by 2 (grids); "
                 "cluster-mean (graph); subtree-mean (tree)")
    lines.append("- Normalization: lf_frac = ||Up(R(delta))|| / ||delta||")
    lines.append("- Auxiliary detector: d_abs_vs_signed = ||R(|delta|)|| / ||R(delta)||")
    lines.append("- Combiner: two_stage (D_parent_z + 0.3 * D_shift_z)")
    lines.append("- Coarse-shift variants: coherent (smooth sign), block (8x8/cluster/subtree), gradient")
    lines.append("- Seeds: 5 per space type")
    lines.append("")

    lines.append("## SC-0: Idempotence")
    lines.append("")
    lines.append("| Space | Error | Status |")
    lines.append("|---|---|---|")
    for name, err in results['sc0'].items():
        status = "OK" if err < 0.30 else "FAIL"
        lines.append(f"| {name} | {err:.2e} | {status} |")
    lines.append("")

    lines.append("## SC-3: Cross-Space Separability")
    lines.append("")
    lines.append("### Global metrics (primary = lf_frac)")
    lines.append("")
    lines.append("| Space | lf_frac AUC | Cohen's d | Combined AUC | Kill Criteria |")
    lines.append("|---|---|---|---|---|")
    for tkey in ['T1', 'T2', 'T3', 'T4']:
        r = results['spaces'][tkey]
        auc_ok = r['global_auc'] >= 0.75
        d_ok = r['cohens_d'] >= 0.5
        status = "PASS" if (auc_ok and d_ok) else "FAIL"
        lines.append(f"| {r['space']} | {r['global_auc']:.4f} | {r['cohens_d']:.4f} | "
                     f"{r['combined_auc']:.4f} | {status} |")
    lines.append("")

    lines.append("### Per-negative-type AUC")
    lines.append("")
    header = "| Space | cs_coherent | cs_block | cs_gradient | lf_drift | random_lf | semant_wrong |"
    lines.append(header)
    lines.append("|---|---|---|---|---|---|---|")
    for tkey in ['T1', 'T2', 'T3', 'T4']:
        r = results['spaces'][tkey]
        nt = r['by_neg_type']
        vals = []
        for ntk in ['coarse_shift_coherent', 'coarse_shift_block', 'coarse_shift_gradient',
                     'lf_drift', 'random_lf', 'semant_wrong']:
            v = nt.get(ntk, {}).get('auc', float('nan'))
            vals.append(f"{v:.4f}")
        lines.append(f"| {r['space']} | " + " | ".join(vals) + " |")
    lines.append("")

    lines.append("### Per-negative-type Cohen's d")
    lines.append("")
    header = "| Space | cs_coherent | cs_block | cs_gradient | lf_drift | random_lf | semant_wrong |"
    lines.append(header)
    lines.append("|---|---|---|---|---|---|---|")
    for tkey in ['T1', 'T2', 'T3', 'T4']:
        r = results['spaces'][tkey]
        nt = r['by_neg_type']
        vals = []
        for ntk in ['coarse_shift_coherent', 'coarse_shift_block', 'coarse_shift_gradient',
                     'lf_drift', 'random_lf', 'semant_wrong']:
            v = nt.get(ntk, {}).get('cohens_d', float('nan'))
            vals.append(f"{v:.4f}")
        lines.append(f"| {r['space']} | " + " | ".join(vals) + " |")
    lines.append("")

    # Findings
    lines.append("## Key Findings")
    lines.append("")

    aucs = [results['spaces'][t]['global_auc'] for t in ['T1', 'T2', 'T3', 'T4']]
    spread = max(aucs) - min(aucs)

    lines.append(f"1. **Cross-space AUC range**: [{min(aucs):.4f}, {max(aucs):.4f}], spread={spread:.4f}")
    if spread < 0.10:
        lines.append("   D_parent generalizes well across all 4 space types.")
    elif spread < 0.20:
        lines.append("   Moderate variation -- may benefit from space-specific tuning.")
    else:
        lines.append("   Large variation -- space-specific tuning recommended.")
    lines.append("")

    t4_auc = results['spaces']['T4']['global_auc']
    t1_auc = results['spaces']['T1']['global_auc']
    lines.append(f"2. **Halo applicability**: T4 (tree, no Halo) AUC={t4_auc:.4f} vs "
                 f"T1 (grid, Halo-capable) AUC={t1_auc:.4f}")
    lines.append(f"   Halo non-applicability {'does NOT' if abs(t4_auc - t1_auc) < 0.05 else 'DOES'} "
                 f"significantly affect D_parent behavior.")
    lines.append("")

    lines.append("3. **Fixed coarse_shift generators**: All three variants (coherent, block, gradient) "
                 "use spatially coherent sign fields that do NOT self-cancel under R. "
                 "This makes the violation visible to D_parent without needing auxiliary detectors.")
    lines.append("")

    lines.append("4. **abs_vs_signed auxiliary is counterproductive with fixed generators.** "
                 "The abs_vs_signed detector was designed for the OLD per-pixel sign-flip generator "
                 "where ||R(|delta|)|| >> ||R(delta)|| due to cancellation. With coherent sign fields, "
                 "this signal disappears, and abs_vs_signed can invert the score (especially on T2 vector). "
                 "Recommendation: use lf_frac alone as the primary D_parent metric with fixed generators.")
    lines.append("")

    # Write
    report_file = out_dir / "CROSSSPACE_SC_REPORT.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[Saved] {report_file}")


if __name__ == "__main__":
    main()
