#!/usr/bin/env python3
"""
Diagnostic: decompose D_parent into its signal components and analyze
where positive/negative overlap happens.

Outputs:
  - Per-subtype distribution stats (mean, std, median, q25, q75)
  - Numerator (||R(delta)||) and denominator (alpha*||coarse||+beta) separately
  - Identifies which positive cases score high and which negative cases score low
"""

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from sc_baseline import sc0_idempotence, sc1_prepare_baselines
from metrics import compute_beta, ALPHA


def decompose_d_parent(cases):
    records = []
    for case in cases:
        delta = case['delta']
        coarse = case['coarse']
        R_fn = case['restrict_fn']

        R_delta = R_fn(delta)
        beta = compute_beta(coarse)

        numer = np.linalg.norm(R_delta.ravel())
        denom = ALPHA * np.linalg.norm(coarse.ravel()) + beta
        d_parent = numer / denom

        norm_delta = np.linalg.norm(delta.ravel())
        norm_coarse = np.linalg.norm(coarse.ravel())

        label = case['pos_type'] if case['case_type'] == 'pos' else case['neg_type']

        records.append({
            'case_type': case['case_type'],
            'subtype': label,
            'level': case['level'],
            'structure': case['structure'],
            'd_parent': d_parent,
            'numer': numer,
            'denom': denom,
            'norm_delta': norm_delta,
            'norm_coarse': norm_coarse,
            'beta': beta,
            'numer_over_norm_delta': numer / (norm_delta + 1e-12),
            'norm_delta_over_norm_coarse': norm_delta / (norm_coarse + 1e-12),
        })
    return records


def print_stats(vals, label):
    a = np.array(vals)
    print(f"  {label:25s}  n={len(a):4d}  "
          f"mean={a.mean():.5f}  std={a.std():.5f}  "
          f"med={np.median(a):.5f}  q25={np.percentile(a, 25):.5f}  q75={np.percentile(a, 75):.5f}  "
          f"min={a.min():.5f}  max={a.max():.5f}")


def main():
    print("=" * 80)
    print("D_parent Diagnostic: Signal Component Decomposition")
    print("=" * 80)

    _, idem_pass = sc0_idempotence(verbose=False)
    if not idem_pass:
        print("Idempotence check failed, aborting.")
        sys.exit(1)

    cases = sc1_prepare_baselines(seed=42, verbose=False)
    records = decompose_d_parent(cases)

    # Group by subtype
    by_subtype = defaultdict(list)
    for r in records:
        key = f"{r['case_type']}/{r['subtype']}"
        by_subtype[key].append(r)

    # 1. D_parent distributions by subtype
    print("\n--- D_parent by subtype ---")
    for key in sorted(by_subtype.keys()):
        vals = [r['d_parent'] for r in by_subtype[key]]
        print_stats(vals, key)

    # 2. Numerator distributions
    print("\n--- Numerator ||R(delta)|| by subtype ---")
    for key in sorted(by_subtype.keys()):
        vals = [r['numer'] for r in by_subtype[key]]
        print_stats(vals, key)

    # 3. Denominator distributions
    print("\n--- Denominator (alpha*||coarse||+beta) by subtype ---")
    for key in sorted(by_subtype.keys()):
        vals = [r['denom'] for r in by_subtype[key]]
        print_stats(vals, key)

    # 4. Ratio: what fraction of delta energy survives restriction?
    print("\n--- ||R(delta)|| / ||delta|| by subtype (restriction survival ratio) ---")
    for key in sorted(by_subtype.keys()):
        vals = [r['numer_over_norm_delta'] for r in by_subtype[key]]
        print_stats(vals, key)

    # 5. ||delta|| / ||coarse|| ratio
    print("\n--- ||delta|| / ||coarse|| by subtype ---")
    for key in sorted(by_subtype.keys()):
        vals = [r['norm_delta_over_norm_coarse'] for r in by_subtype[key]]
        print_stats(vals, key)

    # 6. By level
    print("\n--- D_parent by level ---")
    for level in sorted(set(r['level'] for r in records)):
        for ct in ['pos', 'neg']:
            vals = [r['d_parent'] for r in records if r['level'] == level and r['case_type'] == ct]
            if vals:
                print_stats(vals, f"L={level} {ct}")

    # 7. Overlap analysis: positive cases with high D_parent, negative with low
    pos_vals = np.array([r['d_parent'] for r in records if r['case_type'] == 'pos'])
    neg_vals = np.array([r['d_parent'] for r in records if r['case_type'] == 'neg'])
    threshold = np.percentile(pos_vals, 75)

    print(f"\n--- Overlap Analysis ---")
    print(f"  Positive q75 = {threshold:.5f}")
    print(f"  Negative cases BELOW pos q75 (misclassified as positive):")
    for key in sorted(by_subtype.keys()):
        if not key.startswith('neg'):
            continue
        vals = np.array([r['d_parent'] for r in by_subtype[key]])
        n_below = np.sum(vals < threshold)
        print(f"    {key:25s}: {n_below}/{len(vals)} ({100*n_below/len(vals):.1f}%) below threshold")

    neg_q25 = np.percentile(neg_vals, 25)
    print(f"\n  Negative q25 = {neg_q25:.5f}")
    print(f"  Positive cases ABOVE neg q25 (misclassified as negative):")
    for key in sorted(by_subtype.keys()):
        if not key.startswith('pos'):
            continue
        vals = np.array([r['d_parent'] for r in by_subtype[key]])
        n_above = np.sum(vals > neg_q25)
        print(f"    {key:25s}: {n_above}/{len(vals)} ({100*n_above/len(vals):.1f}%) above threshold")

    # 8. Key insight: denominator variance vs numerator variance
    print("\n--- Coefficient of Variation (std/mean) ---")
    for ct in ['pos', 'neg']:
        subset = [r for r in records if r['case_type'] == ct]
        numers = np.array([r['numer'] for r in subset])
        denoms = np.array([r['denom'] for r in subset])
        dps = np.array([r['d_parent'] for r in subset])
        print(f"  {ct}: numer CV={numers.std()/numers.mean():.3f}  "
              f"denom CV={denoms.std()/denoms.mean():.3f}  "
              f"d_parent CV={dps.std()/dps.mean():.3f}")

    # 9. Correlation: denom vs d_parent
    all_denoms = np.array([r['denom'] for r in records])
    all_dp = np.array([r['d_parent'] for r in records])
    all_numers = np.array([r['numer'] for r in records])
    print(f"\n--- Correlations ---")
    print(f"  corr(denom, d_parent) = {np.corrcoef(all_denoms, all_dp)[0,1]:.4f}")
    print(f"  corr(numer, d_parent) = {np.corrcoef(all_numers, all_dp)[0,1]:.4f}")
    print(f"  corr(denom, numer)    = {np.corrcoef(all_denoms, all_numers)[0,1]:.4f}")

    # 10. The core problem: when coarse is large, denominator is large,
    # compressing both positive AND negative D_parent values
    print(f"\n--- Denominator quartiles and D_parent separation within each ---")
    denom_q = np.percentile(all_denoms, [0, 25, 50, 75, 100])
    for i in range(4):
        lo, hi = denom_q[i], denom_q[i+1]
        mask = (all_denoms >= lo) & (all_denoms < hi + 1e-12)
        pos_in = [r['d_parent'] for r in records
                  if r['case_type'] == 'pos' and lo <= r['denom'] < hi + 1e-12]
        neg_in = [r['d_parent'] for r in records
                  if r['case_type'] == 'neg' and lo <= r['denom'] < hi + 1e-12]
        if pos_in and neg_in:
            from metrics import ALPHA
            from sklearn.metrics import roc_auc_score
            y_true = np.array([0]*len(pos_in) + [1]*len(neg_in))
            y_score = np.array(pos_in + neg_in)
            auc = roc_auc_score(y_true, y_score)
            print(f"  Denom Q{i+1} [{lo:.3f}, {hi:.3f}]: "
                  f"n_pos={len(pos_in)} n_neg={len(neg_in)}  "
                  f"AUC={auc:.4f}  "
                  f"pos_mean={np.mean(pos_in):.4f} neg_mean={np.mean(neg_in):.4f}")

    print("\n[Done]")


if __name__ == "__main__":
    main()
