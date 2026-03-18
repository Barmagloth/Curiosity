#!/usr/bin/env python3
"""
D_parent combination sweep: find optimal combination of R operator,
normalization, and coarse_shift detector.

Tests all combinations of:
  - R operator: [sigma=1.0, sigma=3.0]
  - Normalization: [baseline, survival, lf_frac]
  - Coarse_shift detector: [none, d_abs_vs_signed, d_local_var, d_gradient]
  - Combination: [max, weighted_sum, two_stage]

Also tests robustness across coarse_shift generation variants.

Ranks all combinations by composite score:
  0.4 * global_AUC + 0.3 * min_per_type_AUC + 0.3 * cohens_d_norm
"""

import sys
import json
from pathlib import Path
from itertools import product

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).parent))

from operators import restrict_scalar, prolong_scalar
from operators_v2 import make_restrict_gaussian, prolong_gaussian
from baselines import (
    make_structure_regions,
    positive_oracle, positive_scaled, positive_noisy,
    negative_lf_drift, negative_coarse_shift, negative_random_lf, negative_semant_wrong,
)
from metrics_v2 import d_parent_survival, d_parent_baseline, d_parent_lf_frac


def negative_coarse_shift_coherent(gt, coarse, shift_frac=0.2, seed=42):
    delta = gt - coarse
    return delta + shift_frac * coarse


def negative_coarse_shift_smooth(gt, coarse, shift_frac=0.2, seed=42):
    rng = np.random.RandomState(seed)
    delta = gt - coarse
    sign_field = gaussian_filter(rng.randn(*coarse.shape), sigma=3.0)
    sign_field = np.sign(sign_field)
    return delta + shift_frac * coarse * sign_field


def negative_coarse_shift_scaled(gt, coarse, shift_frac, seed=42):
    return negative_coarse_shift(gt, coarse, shift_frac, seed)


# ---- Coarse-shift detectors (from analysis) ----

def det_abs_vs_signed(delta, coarse, R_fn):
    n_abs = np.linalg.norm(R_fn(np.abs(delta)))
    n_d = np.linalg.norm(R_fn(delta))
    if n_d < 1e-12:
        return 1.0
    return n_abs / n_d


def det_local_var(delta, coarse, R_fn):
    R_dsq = R_fn(delta ** 2)
    R_d = R_fn(delta)
    local_var = np.mean(R_dsq - R_d ** 2)
    return local_var / (np.var(delta) + 1e-12)


def det_gradient(delta, coarse, R_fn):
    R_ref = R_fn(coarse + delta)
    R_c = R_fn(coarse)
    gy_r = np.diff(R_ref, axis=0, prepend=R_ref[:1, :])
    gx_r = np.diff(R_ref, axis=1, prepend=R_ref[:, :1])
    gy_c = np.diff(R_c, axis=0, prepend=R_c[:1, :])
    gx_c = np.diff(R_c, axis=1, prepend=R_c[:, :1])
    diff = np.sqrt((gy_r - gy_c) ** 2 + (gx_r - gx_c) ** 2).mean()
    base = np.sqrt(gy_c ** 2 + gx_c ** 2).mean() + 1e-12
    return diff / base


CS_DETECTORS = {
    'none': None,
    'd_abs_vs_signed': det_abs_vs_signed,
    'd_local_var': det_local_var,
    'd_gradient': det_gradient,
}


# ---- Normalization variants ----

def norm_baseline(delta, coarse, R_fn, Up_fn):
    return d_parent_baseline(delta, coarse, R_fn)


def norm_survival(delta, coarse, R_fn, Up_fn):
    return d_parent_survival(delta, coarse, R_fn)


def norm_lf_frac(delta, coarse, R_fn, Up_fn):
    return d_parent_lf_frac(delta, coarse, R_fn, Up_fn)


NORMS = {
    'baseline': norm_baseline,
    'survival': norm_survival,
    'lf_frac': norm_lf_frac,
}


# ---- Combination strategies ----

def combine_max(d_parent_val, d_shift_val):
    if d_shift_val is None:
        return d_parent_val
    return max(d_parent_val, d_shift_val)


def combine_weighted(d_parent_val, d_shift_val, w_parent=0.6, w_shift=0.4):
    if d_shift_val is None:
        return d_parent_val
    return w_parent * d_parent_val + w_shift * d_shift_val


def combine_two_stage(d_parent_val, d_shift_val, threshold=0.5):
    """Use d_parent primarily; add d_shift bonus when d_parent is ambiguous."""
    if d_shift_val is None:
        return d_parent_val
    return d_parent_val + 0.3 * d_shift_val


COMBINERS = {
    'max': combine_max,
    'weighted': combine_weighted,
    'two_stage': combine_two_stage,
}


# ---- Case generation ----

def generate_cases(R_fn, Up_fn, seed=42, cs_mode='original'):
    """Generate cases with a specific coarse_shift mode."""
    cases = []
    for level, tile_size in enumerate([16, 8, 4], start=1):
        N = 64
        gt, _, struct_labels = make_structure_regions(N, seed=seed + level)
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
                tile_c = coarse_l[s, cs]
                struct_name = ["smooth", "boundary", "texture"][
                    np.bincount(struct_labels[s, cs].ravel(), minlength=3).argmax()
                ]
                base = dict(coarse=tile_c, level=level, structure=struct_name)

                for pos_type, delta in [
                    ("oracle", positive_oracle(tile_gt, tile_c)),
                    ("scaled", positive_scaled(tile_gt, tile_c, 0.5, seed)),
                    ("noisy", positive_noisy(tile_gt, tile_c, 0.005, seed)),
                ]:
                    cases.append({**base, 'delta': delta, 'case_type': 'pos',
                                  'neg_type': None, 'pos_type': pos_type})

                if cs_mode == 'original':
                    cs_delta = negative_coarse_shift(tile_gt, tile_c, 0.2, seed)
                elif cs_mode == 'coherent':
                    cs_delta = negative_coarse_shift_coherent(tile_gt, tile_c, 0.2, seed)
                elif cs_mode == 'smooth':
                    cs_delta = negative_coarse_shift_smooth(tile_gt, tile_c, 0.2, seed)
                elif cs_mode == 'half':
                    cs_delta = negative_coarse_shift_scaled(tile_gt, tile_c, 0.1, seed)
                elif cs_mode == 'double':
                    cs_delta = negative_coarse_shift_scaled(tile_gt, tile_c, 0.4, seed)
                else:
                    cs_delta = negative_coarse_shift(tile_gt, tile_c, 0.2, seed)

                for neg_type, delta in [
                    ("lf_drift", negative_lf_drift(tile_gt, tile_c, 0.3, 0.5, seed)),
                    ("coarse_shift", cs_delta),
                    ("random_lf", negative_random_lf(tile_c, sigma=4.0, amplitude=0.5, seed=seed)),
                    ("semant_wrong", negative_semant_wrong(tile_c, 1.0)),
                ]:
                    cases.append({**base, 'delta': delta, 'case_type': 'neg',
                                  'neg_type': neg_type, 'pos_type': None})
    return cases


def roc_auc(pos_scores, neg_scores):
    from sklearn.metrics import roc_auc_score
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return float('nan')
    y_true = np.array([0] * len(pos_scores) + [1] * len(neg_scores))
    y_score = np.array(list(pos_scores) + list(neg_scores))
    try:
        return roc_auc_score(y_true, y_score)
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
    return abs(m1 - m2) / pooled


def normalize_scores(raw_scores, records, key):
    """Z-score normalize detector values across records for fair combination."""
    vals = np.array(raw_scores)
    mu, sigma = vals.mean(), vals.std()
    if sigma < 1e-12:
        return [0.0] * len(vals)
    return ((vals - mu) / sigma).tolist()


def evaluate_combo(cases, R_fn, Up_fn, norm_fn, cs_det_fn, combiner_fn):
    """Compute combined metric for all cases."""
    records = []
    d_parent_raw = []
    d_shift_raw = []

    for case in cases:
        d_p = norm_fn(case['delta'], case['coarse'], R_fn, Up_fn)
        d_s = cs_det_fn(case['delta'], case['coarse'], R_fn) if cs_det_fn else None
        d_parent_raw.append(d_p)
        d_shift_raw.append(d_s)
        records.append({
            'case_type': case['case_type'],
            'neg_type': case['neg_type'],
            'level': case['level'],
        })

    # Normalize both scores for fair combination
    d_p_norm = normalize_scores(d_parent_raw, records, 'dp')
    if cs_det_fn is not None:
        d_s_norm = normalize_scores(d_shift_raw, records, 'ds')
    else:
        d_s_norm = [None] * len(records)

    for i, rec in enumerate(records):
        rec['score'] = combiner_fn(d_p_norm[i], d_s_norm[i])

    return records


def full_analysis(records):
    """Compute global AUC, per-type AUC, per-level AUC, Cohen's d."""
    pos = [r['score'] for r in records if r['case_type'] == 'pos']
    neg = [r['score'] for r in records if r['case_type'] == 'neg']

    result = {
        'global_auc': roc_auc(pos, neg),
        'cohens_d': cohens_d(pos, neg),
    }

    neg_types = sorted(set(r['neg_type'] for r in records if r['neg_type']))
    result['by_neg_type'] = {}
    for nt in neg_types:
        nv = [r['score'] for r in records if r['case_type'] == 'neg' and r['neg_type'] == nt]
        result['by_neg_type'][nt] = roc_auc(pos, nv)

    levels = sorted(set(r['level'] for r in records))
    result['by_level'] = {}
    for lv in levels:
        lp = [r['score'] for r in records if r['level'] == lv and r['case_type'] == 'pos']
        ln = [r['score'] for r in records if r['level'] == lv and r['case_type'] == 'neg']
        result['by_level'][lv] = roc_auc(lp, ln)

    return result


def composite_score(analysis):
    """0.4*global_AUC + 0.3*min_per_type_AUC + 0.3*cohen_d_norm."""
    g_auc = analysis['global_auc']
    if np.isnan(g_auc):
        return 0.0
    per_type = list(analysis['by_neg_type'].values())
    min_type = min(per_type) if per_type else 0.0
    if np.isnan(min_type):
        min_type = 0.0
    d = analysis['cohens_d']
    if np.isnan(d):
        d = 0.0
    d_norm = min(d / 2.0, 1.0)  # normalize d to [0,1] range (d=2 -> 1.0)
    return 0.4 * g_auc + 0.3 * min_type + 0.3 * d_norm


def main():
    seed = 42
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("D_parent Combination Sweep")
    print("=" * 80)

    R_VARIANTS = {
        's1.0': (lambda x: restrict_scalar(x), lambda xc, tgt: prolong_scalar(xc, tgt)),
        's3.0': (make_restrict_gaussian(3.0), lambda xc, tgt: prolong_gaussian(xc, tgt)),
    }

    # Run sweep on original coarse_shift first
    print("\nPhase 1: Sweep on original coarse_shift")
    print("-" * 70)

    all_combos = []
    for r_name, (R_fn, Up_fn) in R_VARIANTS.items():
        cases = generate_cases(R_fn, Up_fn, seed=seed, cs_mode='original')
        for norm_name, norm_fn in NORMS.items():
            for cs_name, cs_fn in CS_DETECTORS.items():
                for comb_name, comb_fn in COMBINERS.items():
                    if cs_name == 'none' and comb_name != 'max':
                        continue  # skip redundant combiner variants when no detector
                    records = evaluate_combo(cases, R_fn, Up_fn, norm_fn, cs_fn, comb_fn)
                    analysis = full_analysis(records)
                    score = composite_score(analysis)

                    combo = {
                        'R': r_name, 'norm': norm_name,
                        'cs_det': cs_name, 'combiner': comb_name,
                        'score': score,
                        'analysis': analysis,
                    }
                    all_combos.append(combo)

    # Sort by composite score
    all_combos.sort(key=lambda x: x['score'], reverse=True)

    # Print top 20
    print(f"\n  Top 20 combinations:")
    print(f"  {'#':>3} {'R':>5} {'Norm':>10} {'CS_det':>16} {'Comb':>10} "
          f"{'Score':>7} {'AUC':>7} {'d':>7} {'cs':>7} {'lf':>7} {'rlf':>7} {'sw':>7}")
    print(f"  {'-' * 3} {'-' * 5} {'-' * 10} {'-' * 16} {'-' * 10} "
          f"{'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7}")

    for i, c in enumerate(all_combos[:20]):
        a = c['analysis']
        nt = a['by_neg_type']
        print(f"  {i + 1:3d} {c['R']:>5} {c['norm']:>10} {c['cs_det']:>16} {c['combiner']:>10} "
              f"{c['score']:7.4f} {a['global_auc']:7.4f} {a['cohens_d']:7.4f} "
              f"{nt.get('coarse_shift', float('nan')):7.4f} "
              f"{nt.get('lf_drift', float('nan')):7.4f} "
              f"{nt.get('random_lf', float('nan')):7.4f} "
              f"{nt.get('semant_wrong', float('nan')):7.4f}")

    # Best combo
    best = all_combos[0]
    print(f"\n  Best combination: R={best['R']}, norm={best['norm']}, "
          f"cs_det={best['cs_det']}, combiner={best['combiner']}")
    print(f"  Composite score: {best['score']:.4f}")
    ba = best['analysis']
    print(f"  Global AUC: {ba['global_auc']:.4f}, Cohen's d: {ba['cohens_d']:.4f}")
    for nt, auc in sorted(ba['by_neg_type'].items()):
        print(f"    vs {nt}: AUC={auc:.4f}")
    for lv, auc in sorted(ba['by_level'].items()):
        print(f"    L={lv}: AUC={auc:.4f}")

    # Phase 2: Robustness across shift variants
    print(f"\n\nPhase 2: Robustness across coarse_shift variants")
    print("-" * 70)

    # Take top 5 combos and test on shift variants
    top5 = all_combos[:5]
    cs_modes = ['original', 'coherent', 'smooth', 'half', 'double']

    print(f"\n  {'Config':>50} " + " ".join(f"{m:>10}" for m in cs_modes) + f" {'mean':>8}")
    print(f"  {'-' * 50} " + " ".join("-" * 10 for _ in cs_modes) + f" {'-' * 8}")

    robustness_results = []
    for c in top5:
        R_fn, Up_fn = R_VARIANTS[c['R']]
        norm_fn = NORMS[c['norm']]
        cs_fn = CS_DETECTORS[c['cs_det']]
        comb_fn = COMBINERS[c['combiner']]

        label = f"R={c['R']} {c['norm']} {c['cs_det']} {c['combiner']}"
        aucs = []
        for mode in cs_modes:
            cases = generate_cases(R_fn, Up_fn, seed=seed, cs_mode=mode)
            records = evaluate_combo(cases, R_fn, Up_fn, norm_fn, cs_fn, comb_fn)
            analysis = full_analysis(records)
            aucs.append(analysis['by_neg_type'].get('coarse_shift', float('nan')))

        mean_auc = np.nanmean(aucs)
        print(f"  {label:>50} " + " ".join(f"{a:10.4f}" for a in aucs) + f" {mean_auc:8.4f}")
        robustness_results.append({'config': label, 'aucs': aucs, 'mean': float(mean_auc)})

    # Phase 3: Also test pure norm variants (no combination) for reference
    print(f"\n\nPhase 3: Reference -- pure norm variants (no cs detector)")
    print("-" * 70)

    ref_combos = []
    for r_name, (R_fn, Up_fn) in R_VARIANTS.items():
        cases = generate_cases(R_fn, Up_fn, seed=seed, cs_mode='original')
        for norm_name, norm_fn in NORMS.items():
            records = evaluate_combo(cases, R_fn, Up_fn, norm_fn, None, combine_max)
            analysis = full_analysis(records)
            score = composite_score(analysis)
            ref_combos.append({
                'R': r_name, 'norm': norm_name,
                'score': score, 'analysis': analysis,
            })

    ref_combos.sort(key=lambda x: x['score'], reverse=True)
    print(f"\n  {'R':>5} {'Norm':>10} {'Score':>7} {'AUC':>7} {'d':>7} "
          f"{'cs':>7} {'lf':>7} {'rlf':>7} {'sw':>7}")
    print(f"  {'-' * 5} {'-' * 10} {'-' * 7} {'-' * 7} {'-' * 7} "
          f"{'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7}")
    for c in ref_combos:
        a = c['analysis']
        nt = a['by_neg_type']
        print(f"  {c['R']:>5} {c['norm']:>10} {c['score']:7.4f} {a['global_auc']:7.4f} "
              f"{a['cohens_d']:7.4f} "
              f"{nt.get('coarse_shift', float('nan')):7.4f} "
              f"{nt.get('lf_drift', float('nan')):7.4f} "
              f"{nt.get('random_lf', float('nan')):7.4f} "
              f"{nt.get('semant_wrong', float('nan')):7.4f}")

    # Save full results
    save_data = {
        'all_combos': [{k: v for k, v in c.items() if k != 'analysis'} | {'analysis': c['analysis']}
                       for c in all_combos[:30]],
        'robustness': robustness_results,
        'reference': [{k: v for k, v in c.items()} for c in ref_combos],
        'best': {
            'R': best['R'], 'norm': best['norm'],
            'cs_det': best['cs_det'], 'combiner': best['combiner'],
            'score': best['score'], 'analysis': best['analysis'],
        },
    }

    def jd(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return str(obj)

    out_file = out_dir / "combo_sweep_results.json"
    with open(out_file, "w") as f:
        json.dump(save_data, f, indent=2, default=jd)
    print(f"\n[Saved] {out_file}")

    return all_combos, robustness_results


if __name__ == "__main__":
    main()
