#!/usr/bin/env python3
"""
SC-baseline R-variant experiment: tests alternative R operators
and compares D_parent separability against the gaussian+bilinear baseline.

Runs SC-0 through SC-3 for each R variant:
  - baseline: gaussian blur (sigma=1.0) + decimation / bilinear upsample
  - lanczos:  Lanczos-3 downsample / cubic upsample
  - area:     Area averaging (box filter) / bilinear upsample
  - wavelet:  Haar wavelet (block mean) / nearest-neighbor (piecewise constant) upsample

Reports: AUC, Cohen's d, per-negative-type breakdown for D_parent.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from operators import restrict_scalar, prolong_scalar
from operators_v2 import (
    restrict_lanczos, prolong_lanczos,
    restrict_area, prolong_area,
    restrict_wavelet, prolong_wavelet,
    make_restrict_gaussian, prolong_gaussian,
    prolong_wavelet_bilinear,
)
from metrics import compute_metrics, ALPHA, compute_beta, compute_eps
from baselines import (
    make_structure_regions,
    positive_oracle, positive_scaled, positive_noisy,
    negative_lf_drift, negative_coarse_shift, negative_random_lf, negative_semant_wrong,
)


_gauss_s2 = make_restrict_gaussian(2.0)
_gauss_s3 = make_restrict_gaussian(3.0)
_gauss_s05 = make_restrict_gaussian(0.5)

ALL_VARIANTS = {
    'baseline': (
        lambda x: restrict_scalar(x),
        lambda x_c, tgt: prolong_scalar(x_c, tgt),
    ),
    'lanczos': (
        lambda x: restrict_lanczos(x),
        lambda x_c, tgt: prolong_lanczos(x_c, tgt),
    ),
    'area': (
        lambda x: restrict_area(x),
        lambda x_c, tgt: prolong_area(x_c, tgt),
    ),
    'wavelet': (
        lambda x: restrict_wavelet(x),
        lambda x_c, tgt: prolong_wavelet(x_c, tgt),
    ),
    'gauss_s0.5': (
        lambda x: _gauss_s05(x),
        lambda x_c, tgt: prolong_gaussian(x_c, tgt),
    ),
    'gauss_s2.0': (
        lambda x: _gauss_s2(x),
        lambda x_c, tgt: prolong_gaussian(x_c, tgt),
    ),
    'gauss_s3.0': (
        lambda x: _gauss_s3(x),
        lambda x_c, tgt: prolong_gaussian(x_c, tgt),
    ),
    'wavelet_bl': (
        lambda x: restrict_wavelet(x),
        lambda x_c, tgt: prolong_wavelet_bilinear(x_c, tgt),
    ),
}


def sc0_idempotence(R_fn, Up_fn, name, verbose=True):
    IDEMPOTENCE_TOL = 0.30
    N = 64
    rng = np.random.RandomState(42)
    T = 8; NT = N // T
    gt = 0.5 * np.sin(2 * np.pi * np.linspace(0, 1, N)[:, None]) + rng.randn(N, N) * 0.01
    coarse = np.zeros((N, N))
    for ti in range(NT):
        for tj in range(NT):
            s = slice(ti * T, (ti + 1) * T)
            cs = slice(tj * T, (tj + 1) * T)
            coarse[s, cs] = gt[s, cs].mean()

    R_c = R_fn(coarse)
    Up_R_c = Up_fn(R_c, coarse.shape)
    R_Up_R_c = R_fn(Up_R_c)
    err = np.linalg.norm(R_c - R_Up_R_c) / (np.linalg.norm(R_c) + 1e-12)

    passed = err < IDEMPOTENCE_TOL
    if verbose:
        status = "OK" if passed else "KILL"
        print(f"  SC-0 [{name}]: idempotence err = {err:.2e}  [{status}]")
    return err, passed


def generate_cases(R_fn, Up_fn, seed=42):
    cases = []
    for level, tile_size in enumerate([16, 8, 4], start=1):
        N = 64
        gt, coarse_base, struct_labels = make_structure_regions(N, seed=seed + level)

        NT = N // tile_size
        coarse_l = np.zeros_like(gt)
        for ti in range(NT):
            for tj in range(NT):
                s = slice(ti * tile_size, (ti + 1) * tile_size)
                cs = slice(tj * tile_size, (tj + 1) * tile_size)
                coarse_l[s, cs] = gt[s, cs].mean()

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

                for pos_type, delta in [
                    ("oracle", positive_oracle(tile_gt, tile_coarse)),
                    ("scaled", positive_scaled(tile_gt, tile_coarse, 0.5, seed)),
                    ("noisy", positive_noisy(tile_gt, tile_coarse, 0.005, seed)),
                ]:
                    cases.append({
                        **base, 'delta': delta,
                        'case_type': 'pos', 'neg_type': None, 'pos_type': pos_type,
                    })

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

    return cases


def compute_all_metrics(cases):
    records = []
    for case in cases:
        dp, dh = compute_metrics(
            case['delta'], case['coarse'],
            case['restrict_fn'], case['prolong_fn'],
        )
        records.append({
            'D_parent': dp, 'D_hf': dh,
            'space': case['space'], 'structure': case['structure'],
            'level': case['level'], 'case_type': case['case_type'],
            'neg_type': case['neg_type'], 'pos_type': case['pos_type'],
        })
    return records


def roc_auc(pos_scores, neg_scores, higher_is_positive=True):
    from sklearn.metrics import roc_auc_score
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return float('nan')
    if higher_is_positive:
        y_true = np.array([1] * len(pos_scores) + [0] * len(neg_scores))
    else:
        y_true = np.array([0] * len(pos_scores) + [1] * len(neg_scores))
    y_score = np.array(list(pos_scores) + list(neg_scores))
    return roc_auc_score(y_true, y_score)


def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return float('nan')
    m1, m2 = np.mean(g1), np.mean(g2)
    s1, s2 = np.std(g1, ddof=1), np.std(g2, ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled < 1e-12:
        return float('nan')
    return abs(m1 - m2) / pooled


def analyze_variant(records):
    pos_dp = [r['D_parent'] for r in records if r['case_type'] == 'pos']
    neg_dp = [r['D_parent'] for r in records if r['case_type'] == 'neg']
    pos_dh = [r['D_hf'] for r in records if r['case_type'] == 'pos']
    neg_dh = [r['D_hf'] for r in records if r['case_type'] == 'neg']

    result = {
        'D_parent': {
            'global_auc': roc_auc(pos_dp, neg_dp, higher_is_positive=False),
            'cohens_d': cohens_d(pos_dp, neg_dp),
            'pos_mean': float(np.mean(pos_dp)),
            'neg_mean': float(np.mean(neg_dp)),
            'pos_median': float(np.median(pos_dp)),
            'neg_median': float(np.median(neg_dp)),
        },
        'D_hf': {
            'global_auc': roc_auc(pos_dh, neg_dh, higher_is_positive=True),
            'cohens_d': cohens_d(pos_dh, neg_dh),
            'pos_mean': float(np.mean(pos_dh)),
            'neg_mean': float(np.mean(neg_dh)),
        },
    }

    # Per-level for D_parent
    levels = sorted(set(r['level'] for r in records))
    result['D_parent']['by_level'] = {}
    for level in levels:
        sub = [r for r in records if r['level'] == level]
        p = [r['D_parent'] for r in sub if r['case_type'] == 'pos']
        n = [r['D_parent'] for r in sub if r['case_type'] == 'neg']
        result['D_parent']['by_level'][level] = {
            'auc': roc_auc(p, n, higher_is_positive=False),
            'cohens_d': cohens_d(p, n),
        }

    # Per negative type for D_parent
    neg_types = sorted(set(r['neg_type'] for r in records if r['neg_type']))
    result['D_parent']['by_neg_type'] = {}
    for nt in neg_types:
        neg_vals = [r['D_parent'] for r in records
                    if r['case_type'] == 'neg' and r['neg_type'] == nt]
        result['D_parent']['by_neg_type'][nt] = {
            'auc': roc_auc(pos_dp, neg_vals, higher_is_positive=False),
            'cohens_d': cohens_d(pos_dp, neg_vals),
        }

    # Quantile separation count
    n_qsep = 0
    for level in levels:
        sub = [r for r in records if r['level'] == level]
        p = np.array([r['D_parent'] for r in sub if r['case_type'] == 'pos'])
        n = np.array([r['D_parent'] for r in sub if r['case_type'] == 'neg'])
        if len(p) > 0 and len(n) > 0:
            if float(np.median(n)) > float(np.percentile(p, 75)):
                n_qsep += 1
    result['D_parent']['quantile_sep_count'] = n_qsep
    result['D_parent']['quantile_sep_total'] = len(levels)

    # Kill criteria pass/fail for D_parent
    dp = result['D_parent']
    dp['global_auc_pass'] = dp['global_auc'] >= 0.75
    dp['cohens_d_pass'] = dp['cohens_d'] >= 0.5
    depth_pass = all(
        dp['by_level'][l]['auc'] >= 0.65 for l in levels
    )
    dp['depth_auc_pass'] = depth_pass
    dp['quantile_sep_pass'] = n_qsep >= (2 * len(levels) / 3)
    dp['overall_pass'] = (
        dp['global_auc_pass'] and dp['cohens_d_pass']
        and dp['depth_auc_pass'] and dp['quantile_sep_pass']
    )

    return result


def main():
    seed = 42
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SC-baseline R-variant comparison")
    print("=" * 70)

    all_results = {}

    for name, (R_fn, Up_fn) in ALL_VARIANTS.items():
        print(f"\n{'-' * 70}")
        print(f"Variant: {name}")
        print(f"{'-' * 70}")

        # SC-0
        err, passed = sc0_idempotence(R_fn, Up_fn, name)
        if not passed:
            print(f"  KILL: idempotence failed for {name}, skipping.")
            all_results[name] = {'sc0_pass': False, 'sc0_err': err}
            continue

        # SC-1 + SC-2
        cases = generate_cases(R_fn, Up_fn, seed=seed)
        records = compute_all_metrics(cases)

        # SC-3
        analysis = analyze_variant(records)
        analysis['sc0_pass'] = True
        analysis['sc0_err'] = err
        all_results[name] = analysis

        dp = analysis['D_parent']
        dh = analysis['D_hf']
        status = "PASS" if dp['overall_pass'] else "FAIL"

        print(f"  D_parent: AUC={dp['global_auc']:.4f}  d={dp['cohens_d']:.4f}  [{status}]")
        print(f"    pos mean={dp['pos_mean']:.4f}  neg mean={dp['neg_mean']:.4f}")
        print(f"    pos median={dp['pos_median']:.4f}  neg median={dp['neg_median']:.4f}")

        for level, ldata in sorted(dp['by_level'].items()):
            print(f"    L={level}: AUC={ldata['auc']:.4f}  d={ldata['cohens_d']:.4f}")

        print(f"    Quantile sep: {dp['quantile_sep_count']}/{dp['quantile_sep_total']}")

        print(f"    Per negative type:")
        for nt, ntdata in sorted(dp['by_neg_type'].items()):
            print(f"      {nt}: AUC={ntdata['auc']:.4f}  d={ntdata['cohens_d']:.4f}")

        print(f"  D_hf: AUC={dh['global_auc']:.4f}  d={dh['cohens_d']:.4f}")

    # Summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY - D_parent kill criteria")
    print(f"{'=' * 70}")
    print(f"{'Variant':<12} {'AUC':>7} {'d':>7} {'Depth':>6} {'QSep':>6} {'Result':>8}")
    print(f"{'-' * 12} {'-' * 7} {'-' * 7} {'-' * 6} {'-' * 6} {'-' * 8}")

    best_name = None
    best_auc = -1.0

    for name, res in all_results.items():
        if not res.get('sc0_pass', False):
            print(f"{name:<12} {'N/A':>7} {'N/A':>7} {'N/A':>6} {'N/A':>6} {'KILL':>8}")
            continue
        dp = res['D_parent']
        auc_s = f"{dp['global_auc']:.3f}"
        d_s = f"{dp['cohens_d']:.3f}"
        depth_s = "OK" if dp['depth_auc_pass'] else "FAIL"
        qsep_s = f"{dp['quantile_sep_count']}/{dp['quantile_sep_total']}"
        result_s = "PASS" if dp['overall_pass'] else "FAIL"
        print(f"{name:<12} {auc_s:>7} {d_s:>7} {depth_s:>6} {qsep_s:>6} {result_s:>8}")

        if dp['global_auc'] > best_auc:
            best_auc = dp['global_auc']
            best_name = name

    print(f"\nBest variant by D_parent AUC: {best_name} (AUC={best_auc:.4f})")

    # Save JSON
    out_file = out_dir / "r_variants_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2, default=_json_default)
    print(f"\n[Saved] {out_file}")

    return all_results, best_name


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
