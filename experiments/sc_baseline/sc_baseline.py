#!/usr/bin/env python3
"""
Curiosity — Scale-Consistency Baseline Experiment (SC-0 through SC-3)

Implements the first four steps of the Scale-Consistency Verification Protocol
(docs/scale_consistency_verification_protocol_v1.0.md).

SC-0: Idempotence check — verify R(coarse) ~ coarse_downsampled (round-trip).
SC-1: Baseline preparation — generate positive and negative baselines.
SC-2: Metric computation — compute D_parent and D_hf for all cases.
SC-3: Separability analysis — ROC-AUC, depth-conditioned AUC, Cohen's d,
      quantile separation.

Kill criteria (fixed before launch, per protocol Step 4):
  - Global ROC-AUC >= 0.75
  - Depth-conditioned AUC >= 0.65 (each level)
  - Effect size (Cohen's d) >= 0.5
  - Quantile separation on >= 2/3 of levels

CPU-only. Requires numpy, scipy, sklearn.

Usage:
    python sc_baseline.py [--output-dir DIR] [--seed SEED]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from operators import (
    restrict_scalar, prolong_scalar,
    restrict_vector, prolong_vector,
    restrict_graph, prolong_graph,
    restrict_tree, prolong_tree,
)
from metrics import compute_metrics, ALPHA, compute_beta, compute_eps
from baselines import (
    make_structure_regions,
    positive_oracle, positive_scaled, positive_noisy,
    negative_lf_drift, negative_coarse_shift, negative_random_lf, negative_semant_wrong,
)


# ═══════════════════════════════════════════════
# SC-0: Idempotence Check
# ═══════════════════════════════════════════════

def sc0_idempotence(verbose=True):
    """Verify R is idempotent on coarse signals: ||R(coarse) - R(Up(R(coarse)))|| ~ 0.

    For each space type, apply R to coarse, then Up, then R again.
    The round-trip R(Up(R(coarse))) should match R(coarse) closely.

    If the error is not negligible, the operator pair is broken and
    the entire experiment must stop (kill).

    Returns:
        dict of {space_name: error_norm}. All should be < 1e-6 relative.
    """
    results = {}
    # Tolerance for round-trip error. For gaussian blur + bilinear upsampling,
    # the round-trip is not exact (Up is not R-inverse), so we accept relative
    # error up to 0.30. Graph/tree operators are exact by construction.
    IDEMPOTENCE_TOL = 0.30

    if verbose:
        print("=" * 60)
        print("SC-0: Idempotence Check")
        print("=" * 60)

    # T1: Scalar grid
    N = 64
    rng = np.random.RandomState(42)
    coarse_2d = np.zeros((N, N))
    T = 8; NT = N // T
    gt_2d = 0.5 * np.sin(2 * np.pi * np.linspace(0, 1, N)[:, None]) + rng.randn(N, N) * 0.01
    for ti in range(NT):
        for tj in range(NT):
            s = slice(ti * T, (ti + 1) * T)
            cs = slice(tj * T, (tj + 1) * T)
            coarse_2d[s, cs] = gt_2d[s, cs].mean()

    R_coarse = restrict_scalar(coarse_2d)
    Up_R_coarse = prolong_scalar(R_coarse, coarse_2d.shape)
    R_Up_R_coarse = restrict_scalar(Up_R_coarse)
    err = np.linalg.norm(R_coarse - R_Up_R_coarse) / (np.linalg.norm(R_coarse) + 1e-12)
    results["T1_scalar"] = float(err)

    # T2: Vector grid
    D = 16
    coarse_vec = np.zeros((32, 32, D))
    gt_vec = rng.randn(32, 32, D) * 0.3
    NT_v = 32 // T
    for ti in range(NT_v):
        for tj in range(NT_v):
            s = slice(ti * T, (ti + 1) * T)
            cs = slice(tj * T, (tj + 1) * T)
            coarse_vec[s, cs, :] = gt_vec[s, cs, :].mean(axis=(0, 1), keepdims=True)

    R_cv = restrict_vector(coarse_vec)
    Up_R_cv = prolong_vector(R_cv, coarse_vec.shape)
    R_Up_R_cv = restrict_vector(Up_R_cv)
    err_v = np.linalg.norm(R_cv.ravel() - R_Up_R_cv.ravel()) / (np.linalg.norm(R_cv.ravel()) + 1e-12)
    results["T2_vector"] = float(err_v)

    # T3: Graph — R is cluster-mean, Up is scatter-back. R(Up(R(x))) = R(x) exactly.
    n_pts, n_cl = 200, 10
    vals = rng.randn(n_pts) * 0.5
    labels = rng.randint(0, n_cl, n_pts)
    R_g = restrict_graph(vals, labels, n_cl)
    Up_R_g = prolong_graph(R_g, labels, n_pts)
    R_Up_R_g = restrict_graph(Up_R_g, labels, n_cl)
    err_g = np.linalg.norm(R_g - R_Up_R_g) / (np.linalg.norm(R_g) + 1e-12)
    results["T3_graph"] = float(err_g)

    # T4: Tree
    depth = 6
    n_nodes = 2 ** depth - 1
    coarse_depth = 3
    vals_t = rng.randn(n_nodes) * 0.3
    R_t = restrict_tree(vals_t, n_nodes, coarse_depth)
    Up_R_t = prolong_tree(R_t, n_nodes, coarse_depth)
    R_Up_R_t = restrict_tree(Up_R_t, n_nodes, coarse_depth)
    err_t = np.linalg.norm(R_t - R_Up_R_t) / (np.linalg.norm(R_t) + 1e-12)
    results["T4_tree"] = float(err_t)

    if verbose:
        for name, err in results.items():
            status = "OK" if err < IDEMPOTENCE_TOL else "KILL"
            print(f"  {name}: ||R - R(Up(R))|| / ||R|| = {err:.2e}  [{status}]")

    # Kill check
    for name, err in results.items():
        if err > IDEMPOTENCE_TOL:
            print(f"\n  KILL: {name} idempotence error {err:.2e} > {IDEMPOTENCE_TOL:.0e}")
            print("  Operator pair (R, Up) is broken. Cannot proceed.")
            return results, False

    if verbose:
        print("  All idempotence checks passed.\n")
    return results, True


# ═══════════════════════════════════════════════
# SC-1: Baseline Preparation
# ═══════════════════════════════════════════════

def sc1_prepare_baselines(seed=42, verbose=True):
    """Generate positive and negative baselines for all space types.

    Positive: oracle delta (GT - coarse), scaled oracle, noisy oracle.
    Negative: LF drift, coarse shift, random LF, semant-wrong.

    Each case is tagged with (space_type, structure_type, level, case_type, neg_type).

    Returns:
        list of dicts, each: {
            'delta': array, 'coarse': array,
            'space': str, 'structure': str, 'level': int,
            'case_type': 'pos'|'neg', 'neg_type': str|None,
            'pos_type': str|None,
            'restrict_fn': callable, 'prolong_fn': callable,
        }
    """
    if verbose:
        print("=" * 60)
        print("SC-1: Baseline Preparation")
        print("=" * 60)

    cases = []

    # --- 2D Scalar grid at multiple "levels" (simulate by varying tile size) ---
    for level, tile_size in enumerate([16, 8, 4], start=1):
        N = 64
        gt, coarse, struct_labels = make_structure_regions(N, seed=seed + level)

        # Re-compute coarse at this tile size
        NT = N // tile_size
        coarse_l = np.zeros_like(gt)
        for ti in range(NT):
            for tj in range(NT):
                s = slice(ti * tile_size, (ti + 1) * tile_size)
                cs = slice(tj * tile_size, (tj + 1) * tile_size)
                coarse_l[s, cs] = gt[s, cs].mean()

        R_fn = lambda x: restrict_scalar(x)
        Up_fn = lambda x_c, tgt: prolong_scalar(x_c, tgt)

        # Determine dominant structure per tile
        for ti in range(NT):
            for tj in range(NT):
                s = slice(ti * tile_size, (ti + 1) * tile_size)
                cs = slice(tj * tile_size, (tj + 1) * tile_size)

                tile_gt = gt[s, cs]
                tile_coarse = coarse_l[s, cs]
                region_labels = struct_labels[s, cs]
                struct_name = ["smooth", "boundary", "texture"][
                    np.bincount(region_labels.ravel(), minlength=3).argmax()
                ]

                base = dict(
                    coarse=tile_coarse, space="T1_scalar",
                    structure=struct_name, level=level,
                    restrict_fn=R_fn, prolong_fn=Up_fn,
                )

                # Positive cases
                for pos_type, delta in [
                    ("oracle", positive_oracle(tile_gt, tile_coarse)),
                    ("scaled", positive_scaled(tile_gt, tile_coarse, 0.5, seed)),
                    ("noisy", positive_noisy(tile_gt, tile_coarse, 0.005, seed)),
                ]:
                    cases.append({
                        **base, 'delta': delta,
                        'case_type': 'pos', 'neg_type': None, 'pos_type': pos_type,
                    })

                # Negative cases
                for neg_type, delta in [
                    ("lf_drift", negative_lf_drift(tile_gt, tile_coarse, 0.3, 0.5, seed)),
                    ("coarse_shift", negative_coarse_shift(tile_gt, tile_coarse, 0.2, seed)),
                    ("random_lf", negative_random_lf(tile_coarse, sigma=4.0, amplitude=0.5, seed=seed)),
                    ("semant_wrong", negative_semant_wrong(tile_coarse, 1.0)),
                ]:
                    cases.append({
                        **base, 'delta': delta,
                        'case_type': 'neg', 'neg_type': neg_type, 'pos_type': None,
                    })

    if verbose:
        n_pos = sum(1 for c in cases if c['case_type'] == 'pos')
        n_neg = sum(1 for c in cases if c['case_type'] == 'neg')
        print(f"  Generated {len(cases)} cases: {n_pos} positive, {n_neg} negative")
        print(f"  Levels: {sorted(set(c['level'] for c in cases))}")
        print(f"  Structures: {sorted(set(c['structure'] for c in cases))}")
        neg_types = sorted(set(c['neg_type'] for c in cases if c['neg_type']))
        print(f"  Negative types: {neg_types}\n")

    return cases


# ═══════════════════════════════════════════════
# SC-2: Metric Computation
# ═══════════════════════════════════════════════

def sc2_compute_metrics(cases, verbose=True):
    """Compute D_parent and D_hf for all cases.

    Returns:
        list of dicts — original case metadata plus D_parent, D_hf values.
    """
    if verbose:
        print("=" * 60)
        print("SC-2: Metric Computation")
        print("=" * 60)

    records = []
    for i, case in enumerate(cases):
        delta = case['delta']
        coarse = case['coarse']
        R_fn = case['restrict_fn']
        Up_fn = case['prolong_fn']

        dp, dh = compute_metrics(delta, coarse, R_fn, Up_fn)

        records.append({
            'D_parent': dp,
            'D_hf': dh,
            'space': case['space'],
            'structure': case['structure'],
            'level': case['level'],
            'case_type': case['case_type'],
            'neg_type': case['neg_type'],
            'pos_type': case['pos_type'],
        })

    if verbose:
        dp_pos = [r['D_parent'] for r in records if r['case_type'] == 'pos']
        dp_neg = [r['D_parent'] for r in records if r['case_type'] == 'neg']
        dh_pos = [r['D_hf'] for r in records if r['case_type'] == 'pos']
        dh_neg = [r['D_hf'] for r in records if r['case_type'] == 'neg']

        print(f"  D_parent — pos: mean={np.mean(dp_pos):.4f} std={np.std(dp_pos):.4f}"
              f"  neg: mean={np.mean(dp_neg):.4f} std={np.std(dp_neg):.4f}")
        print(f"  D_hf     — pos: mean={np.mean(dh_pos):.4f} std={np.std(dh_pos):.4f}"
              f"  neg: mean={np.mean(dh_neg):.4f} std={np.std(dh_neg):.4f}")
        print()

    return records


# ═══════════════════════════════════════════════
# SC-3: Separability Analysis
# ═══════════════════════════════════════════════

def _roc_auc(pos_scores, neg_scores, higher_is_positive=True):
    """Compute ROC-AUC with correct orientation.

    For D_parent: negative cases should have higher scores (higher_is_positive=False,
    meaning higher score = negative label). We flip to: label negative as 1 (positive class).

    For D_hf: positive cases should have higher scores (higher_is_positive=True).
    """
    from sklearn.metrics import roc_auc_score
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return float('nan')

    if higher_is_positive:
        # pos=higher: label pos as 1
        y_true = np.array([1] * len(pos_scores) + [0] * len(neg_scores))
        y_score = np.array(list(pos_scores) + list(neg_scores))
    else:
        # neg=higher: label neg as 1
        y_true = np.array([0] * len(pos_scores) + [1] * len(neg_scores))
        y_score = np.array(list(pos_scores) + list(neg_scores))

    return roc_auc_score(y_true, y_score)


def _pr_auc(pos_scores, neg_scores, higher_is_positive=True):
    """Compute PR-AUC with correct orientation."""
    from sklearn.metrics import average_precision_score
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return float('nan')

    if higher_is_positive:
        y_true = np.array([1] * len(pos_scores) + [0] * len(neg_scores))
        y_score = np.array(list(pos_scores) + list(neg_scores))
    else:
        y_true = np.array([0] * len(pos_scores) + [1] * len(neg_scores))
        y_score = np.array(list(pos_scores) + list(neg_scores))

    return average_precision_score(y_true, y_score)


def _cohens_d(group1, group2):
    """Cohen's d effect size between two groups."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return float('nan')
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled_std < 1e-12:
        return float('nan')
    return abs(m1 - m2) / pooled_std


def _quantile_separation_d_parent(pos_vals, neg_vals):
    """D_parent quantile separation: median(neg) > q75(pos)?

    Returns True if negative cases have median above the 75th percentile of positives.
    """
    if len(pos_vals) == 0 or len(neg_vals) == 0:
        return False
    return float(np.median(neg_vals)) > float(np.percentile(pos_vals, 75))


def _quantile_separation_d_hf(pos_vals, neg_vals):
    """D_hf quantile separation: median(pos) > q75(neg)?

    Returns True if positive cases have median above the 75th percentile of negatives.
    """
    if len(pos_vals) == 0 or len(neg_vals) == 0:
        return False
    return float(np.median(pos_vals)) > float(np.percentile(neg_vals, 75))


def _analyze_metric(pos_vals, neg_vals, metric_name):
    """Full separability analysis for one metric slice."""
    pos = np.array(pos_vals) if pos_vals else np.array([])
    neg = np.array(neg_vals) if neg_vals else np.array([])

    is_d_parent = (metric_name == "D_parent")
    higher_is_positive = not is_d_parent  # D_hf: higher=better; D_parent: higher=worse

    result = {
        'n_pos': len(pos), 'n_neg': len(neg),
        'pos_mean': float(np.mean(pos)) if len(pos) > 0 else None,
        'neg_mean': float(np.mean(neg)) if len(neg) > 0 else None,
        'pos_median': float(np.median(pos)) if len(pos) > 0 else None,
        'neg_median': float(np.median(neg)) if len(neg) > 0 else None,
    }

    result['roc_auc'] = _roc_auc(pos, neg, higher_is_positive)
    result['pr_auc'] = _pr_auc(pos, neg, higher_is_positive)
    result['cohens_d'] = _cohens_d(pos, neg)

    if is_d_parent:
        result['quantile_sep'] = _quantile_separation_d_parent(pos, neg)
    else:
        result['quantile_sep'] = _quantile_separation_d_hf(pos, neg)

    return result


def sc3_separability(records, verbose=True):
    """Full separability analysis: global, depth-conditioned, structure-conditioned.

    Returns:
        dict with analysis results and pass/fail for each criterion.
    """
    if verbose:
        print("=" * 60)
        print("SC-3: Separability Analysis")
        print("=" * 60)

    results = {'global': {}, 'by_level': {}, 'by_structure': {}, 'by_neg_type': {}}

    # Extract metric arrays
    def extract(records_subset, metric):
        pos = [r[metric] for r in records_subset if r['case_type'] == 'pos']
        neg = [r[metric] for r in records_subset if r['case_type'] == 'neg']
        return pos, neg

    # --- 3.1 Global separability ---
    if verbose:
        print("\n  3.1 Global Separability")
        print("  " + "-" * 50)

    for metric in ['D_parent', 'D_hf']:
        pos, neg = extract(records, metric)
        analysis = _analyze_metric(pos, neg, metric)
        results['global'][metric] = analysis

        if verbose:
            direction = "higher=worse" if metric == "D_parent" else "higher=better"
            print(f"  {metric} ({direction}):")
            print(f"    pos: n={analysis['n_pos']}  mean={analysis['pos_mean']:.4f}  "
                  f"median={analysis['pos_median']:.4f}")
            print(f"    neg: n={analysis['n_neg']}  mean={analysis['neg_mean']:.4f}  "
                  f"median={analysis['neg_median']:.4f}")
            print(f"    ROC-AUC={analysis['roc_auc']:.4f}  PR-AUC={analysis['pr_auc']:.4f}  "
                  f"Cohen's d={analysis['cohens_d']:.4f}  "
                  f"Quantile sep={analysis['quantile_sep']}")

    # --- 3.2 Depth-conditioned separability ---
    levels = sorted(set(r['level'] for r in records))

    if verbose:
        print(f"\n  3.2 Depth-conditioned Separability (levels: {levels})")
        print("  " + "-" * 50)

    for metric in ['D_parent', 'D_hf']:
        results['by_level'][metric] = {}
        for level in levels:
            subset = [r for r in records if r['level'] == level]
            pos, neg = extract(subset, metric)
            analysis = _analyze_metric(pos, neg, metric)
            results['by_level'][metric][level] = analysis

            if verbose:
                print(f"  {metric} L={level}: ROC-AUC={analysis['roc_auc']:.4f}  "
                      f"Cohen's d={analysis['cohens_d']:.4f}  "
                      f"QSep={analysis['quantile_sep']}")

    # --- 3.3 Regime-conditioned separability ---
    structures = sorted(set(r['structure'] for r in records))

    if verbose:
        print(f"\n  3.3 Regime-conditioned Separability (structures: {structures})")
        print("  " + "-" * 50)

    for metric in ['D_parent', 'D_hf']:
        results['by_structure'][metric] = {}
        for struct in structures:
            subset = [r for r in records if r['structure'] == struct]
            pos, neg = extract(subset, metric)
            analysis = _analyze_metric(pos, neg, metric)
            results['by_structure'][metric][struct] = analysis

            if verbose:
                print(f"  {metric} [{struct}]: ROC-AUC={analysis['roc_auc']:.4f}  "
                      f"Cohen's d={analysis['cohens_d']:.4f}  "
                      f"QSep={analysis['quantile_sep']}")

    # --- Breakdown by negative type ---
    neg_types = sorted(set(r['neg_type'] for r in records if r['neg_type']))

    if verbose:
        print(f"\n  Negative-type breakdown: {neg_types}")
        print("  " + "-" * 50)

    for metric in ['D_parent', 'D_hf']:
        results['by_neg_type'][metric] = {}
        pos_all = [r[metric] for r in records if r['case_type'] == 'pos']
        for nt in neg_types:
            neg_vals = [r[metric] for r in records
                        if r['case_type'] == 'neg' and r['neg_type'] == nt]
            analysis = _analyze_metric(pos_all, neg_vals, metric)
            results['by_neg_type'][metric][nt] = analysis

            if verbose:
                print(f"  {metric} vs {nt}: ROC-AUC={analysis['roc_auc']:.4f}  "
                      f"Cohen's d={analysis['cohens_d']:.4f}")

    # --- Kill criteria evaluation ---
    if verbose:
        print("\n" + "=" * 60)
        print("Kill Criteria Evaluation (per protocol Step 4)")
        print("=" * 60)

    kill_results = _evaluate_kill_criteria(results, levels, verbose)
    results['kill'] = kill_results

    return results


def _evaluate_kill_criteria(results, levels, verbose=True):
    """Evaluate all kill criteria from the protocol.

    Thresholds:
      Global ROC-AUC >= 0.75
      Depth-conditioned AUC >= 0.65 (each level)
      Effect size (Cohen's d) >= 0.5
      Quantile separation on >= 2/3 of levels
    """
    GLOBAL_AUC_THRESH = 0.75
    DEPTH_AUC_THRESH = 0.65
    EFFECT_SIZE_THRESH = 0.5

    kill = {'pass': True, 'details': {}}

    for metric in ['D_parent', 'D_hf']:
        mk = {}

        # Global AUC
        global_auc = results['global'][metric]['roc_auc']
        mk['global_auc'] = global_auc
        mk['global_auc_pass'] = (not np.isnan(global_auc)) and global_auc >= GLOBAL_AUC_THRESH

        # Depth-conditioned AUC
        depth_aucs = {}
        depth_pass = True
        for level in levels:
            auc = results['by_level'][metric][level]['roc_auc']
            depth_aucs[level] = auc
            if np.isnan(auc) or auc < DEPTH_AUC_THRESH:
                depth_pass = False
        mk['depth_aucs'] = depth_aucs
        mk['depth_auc_pass'] = depth_pass

        # Effect size
        global_d = results['global'][metric]['cohens_d']
        mk['cohens_d'] = global_d
        mk['effect_size_pass'] = (not np.isnan(global_d)) and global_d >= EFFECT_SIZE_THRESH

        # Quantile separation on >= 2/3 of levels
        n_levels = len(levels)
        n_qsep = sum(1 for level in levels
                      if results['by_level'][metric][level]['quantile_sep'])
        mk['quantile_sep_count'] = n_qsep
        mk['quantile_sep_total'] = n_levels
        mk['quantile_sep_pass'] = n_qsep >= (2 * n_levels / 3)

        # Overall for this metric
        mk['overall_pass'] = (mk['global_auc_pass'] and mk['depth_auc_pass']
                              and mk['effect_size_pass'] and mk['quantile_sep_pass'])
        if not mk['overall_pass']:
            kill['pass'] = False

        kill['details'][metric] = mk

        if verbose:
            status = "PASS" if mk['overall_pass'] else "FAIL"
            print(f"\n  {metric}: [{status}]")
            print(f"    Global AUC: {global_auc:.4f} "
                  f"{'OK' if mk['global_auc_pass'] else 'FAIL'} (>= {GLOBAL_AUC_THRESH})")
            for level in levels:
                auc_l = depth_aucs[level]
                print(f"    Depth L={level} AUC: {auc_l:.4f} "
                      f"{'OK' if not np.isnan(auc_l) and auc_l >= DEPTH_AUC_THRESH else 'FAIL'}")
            print(f"    Effect size: d={global_d:.4f} "
                  f"{'OK' if mk['effect_size_pass'] else 'FAIL'} (>= {EFFECT_SIZE_THRESH})")
            print(f"    Quantile sep: {n_qsep}/{n_levels} levels "
                  f"{'OK' if mk['quantile_sep_pass'] else 'FAIL'} (>= 2/3)")

    if verbose:
        overall = "PASS — proceed to SC-4/SC-5" if kill['pass'] else \
                  "FAIL — do NOT tune tau, do NOT introduce enforcement (see protocol)"
        print(f"\n  Overall: [{overall}]")

    return kill


# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SC-baseline: Scale-Consistency Verification SC-0..SC-3")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results JSON")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Curiosity — Scale-Consistency Baseline (SC-0 through SC-3)")
    print(f"Output: {out_dir}")
    print()

    # SC-0
    idem_results, idem_pass = sc0_idempotence(verbose=True)
    if not idem_pass:
        print("\nAborting: idempotence check failed.")
        sys.exit(1)

    # SC-1
    cases = sc1_prepare_baselines(seed=args.seed, verbose=True)

    # SC-2
    records = sc2_compute_metrics(cases, verbose=True)

    # SC-3
    sep_results = sc3_separability(records, verbose=True)

    # Save all results
    output = {
        'sc0_idempotence': idem_results,
        'sc0_pass': idem_pass,
        'sc2_records': records,
        'sc3_separability': sep_results,
        'params': {
            'alpha': ALPHA,
            'beta_scale': float(compute_beta(np.ones(10))),
            'eps_scale': float(compute_eps(np.ones(10))),
            'seed': args.seed,
            'R': 'gaussian_blur_sigma1.0 + decimation_factor2',
            'Up': 'bilinear_upsampling',
        },
    }

    out_file = out_dir / "sc_baseline_results.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, default=_json_default)

    print(f"\n[Saved] {out_file}")
    print("[Done]")


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return str(obj)


if __name__ == "__main__":
    main()
