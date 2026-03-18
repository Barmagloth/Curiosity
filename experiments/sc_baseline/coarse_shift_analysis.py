#!/usr/bin/env python3
"""
Non-norm-based detectors for coarse_shift negative type.

Root cause analysis:
  coarse_shift adds `shift_frac * coarse * sign_map` to the oracle delta.
  The random per-pixel sign_map creates HF discontinuities that get
  attenuated by R (gaussian blur + decimation). Within a constant-valued
  tile, the coarse is spatially constant, so:
    - The shift component is +/- 0.2*c per pixel (random sign)
    - R averages over a patch, and random +/- signs cancel: R(shift) ~ 0
    - ||R(delta_shift)|| ~ ||R(delta_oracle)|| (shift is invisible to R)
    - ||delta_shift|| > ||delta_oracle|| (shift adds energy to denominator)
    - survival ratio for shift is LOWER than oracle (inverted AUC!)

  This cancellation is fundamental to any norm-based measure of R(delta).
  The shift is a per-pixel semantic violation that averages out at the
  coarse scale, making it invisible to restriction-based norms.

Detectors tested (8 original + 3 additional):
  Structural: d_correlation, d_cosine
  Spectral: d_spectral, d_phase
  Distribution: d_histogram, d_quantile
  Gradient/SSIM: d_gradient, d_ssim
  New: d_abs_vs_signed, d_local_var, d_energy_ratio

Also tests coherent (no sign-flip) and smooth coarse_shift variants to
confirm the diagnosis.
"""

import sys
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import wasserstein_distance

sys.path.insert(0, str(Path(__file__).parent))

from operators import restrict_scalar, prolong_scalar
from operators_v2 import make_restrict_gaussian, prolong_gaussian
from baselines import (
    make_structure_regions,
    positive_oracle, positive_scaled, positive_noisy,
    negative_lf_drift, negative_coarse_shift, negative_random_lf, negative_semant_wrong,
)
from metrics_v2 import d_parent_survival, d_parent_baseline


def negative_coarse_shift_coherent(gt, coarse, shift_frac=0.2, seed=42):
    """Coherent coarse_shift: no sign flip, uniform positive shift."""
    delta = gt - coarse
    return delta + shift_frac * coarse


def negative_coarse_shift_smooth(gt, coarse, shift_frac=0.2, seed=42):
    """Smooth coarse_shift: spatially coherent sign field (large patches)."""
    rng = np.random.RandomState(seed)
    delta = gt - coarse
    sign_field = gaussian_filter(rng.randn(*coarse.shape), sigma=3.0)
    sign_field = np.sign(sign_field)
    return delta + shift_frac * coarse * sign_field


def negative_coarse_shift_scaled(gt, coarse, shift_frac, seed=42):
    """Original coarse_shift with variable magnitude."""
    return negative_coarse_shift(gt, coarse, shift_frac, seed)


# ---- Detectors ----

def d_correlation(delta, coarse, R_fn):
    """1 - Pearson correlation between R(refined) and R(coarse)."""
    R_ref = R_fn(coarse + delta).ravel()
    R_c = R_fn(coarse).ravel()
    if R_ref.std() < 1e-12 or R_c.std() < 1e-12:
        return 0.0
    return 1.0 - np.corrcoef(R_ref, R_c)[0, 1]


def d_cosine(delta, coarse, R_fn):
    """1 - cosine similarity between R(refined) and R(coarse)."""
    R_ref = R_fn(coarse + delta).ravel()
    R_c = R_fn(coarse).ravel()
    n_ref = np.linalg.norm(R_ref)
    n_c = np.linalg.norm(R_c)
    if n_ref < 1e-12 or n_c < 1e-12:
        return 0.0
    return 1.0 - np.dot(R_ref, R_c) / (n_ref * n_c)


def d_spectral(delta, coarse, R_fn):
    """Spectral divergence: LF fraction difference between R(delta) and delta."""
    R_delta = R_fn(delta)
    psd_d = np.abs(np.fft.fft2(delta)) ** 2
    psd_R = np.abs(np.fft.fft2(R_delta)) ** 2
    h_d, w_d = delta.shape
    h_r, w_r = R_delta.shape
    lf_d = psd_d[:h_d // 4, :w_d // 4].sum() / (psd_d.sum() + 1e-12)
    lf_R = psd_R[:h_r // 4, :w_r // 4].sum() / (psd_R.sum() + 1e-12)
    return abs(lf_R - lf_d)


def d_phase(delta, coarse, R_fn):
    """Phase misalignment between R(refined) and R(coarse)."""
    R_ref = R_fn(coarse + delta)
    R_c = R_fn(coarse)
    fft_ref = np.fft.fft2(R_ref)
    fft_c = np.fft.fft2(R_c)
    mag = np.abs(fft_c) + 1e-12
    phase_diff = np.angle(fft_ref) - np.angle(fft_c)
    phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi
    return (np.abs(phase_diff) * mag).sum() / mag.sum()


def d_histogram(delta, coarse, R_fn):
    """Wasserstein distance between R(refined) and R(coarse) distributions."""
    R_ref = R_fn(coarse + delta).ravel()
    R_c = R_fn(coarse).ravel()
    c_range = R_c.max() - R_c.min()
    if c_range < 1e-12:
        return 0.0
    return wasserstein_distance(R_ref, R_c) / c_range


def d_quantile(delta, coarse, R_fn, n_quantiles=20):
    """Max quantile deviation between R(refined) and R(coarse)."""
    R_ref = R_fn(coarse + delta).ravel()
    R_c = R_fn(coarse).ravel()
    c_range = R_c.max() - R_c.min()
    if c_range < 1e-12:
        return 0.0
    qs = np.linspace(0, 100, n_quantiles + 2)[1:-1]
    return np.max(np.abs(np.percentile(R_ref, qs) - np.percentile(R_c, qs))) / c_range


def d_gradient(delta, coarse, R_fn):
    """Gradient field difference between R(refined) and R(coarse)."""
    R_ref = R_fn(coarse + delta)
    R_c = R_fn(coarse)
    def _grad(x):
        return np.diff(x, axis=0, prepend=x[:1, :]), np.diff(x, axis=1, prepend=x[:, :1])
    gy_r, gx_r = _grad(R_ref)
    gy_c, gx_c = _grad(R_c)
    diff = np.sqrt((gy_r - gy_c) ** 2 + (gx_r - gx_c) ** 2).mean()
    base = np.sqrt(gy_c ** 2 + gx_c ** 2).mean() + 1e-12
    return diff / base


def d_ssim(delta, coarse, R_fn, win_size=3):
    """1 - SSIM between R(refined) and R(coarse)."""
    R_ref = R_fn(coarse + delta)
    R_c = R_fn(coarse)
    C1 = (0.01 * (R_c.max() - R_c.min() + 1e-12)) ** 2
    C2 = (0.03 * (R_c.max() - R_c.min() + 1e-12)) ** 2
    s = win_size / 2
    mu_r = gaussian_filter(R_ref, sigma=s)
    mu_c = gaussian_filter(R_c, sigma=s)
    sig_r2 = gaussian_filter(R_ref ** 2, sigma=s) - mu_r ** 2
    sig_c2 = gaussian_filter(R_c ** 2, sigma=s) - mu_c ** 2
    sig_rc = gaussian_filter(R_ref * R_c, sigma=s) - mu_r * mu_c
    ssim_map = ((2 * mu_r * mu_c + C1) * (2 * sig_rc + C2)) / \
               ((mu_r ** 2 + mu_c ** 2 + C1) * (sig_r2 + sig_c2 + C2))
    return 1.0 - ssim_map.mean()


def d_abs_vs_signed(delta, coarse, R_fn):
    """Ratio ||R(|delta|)|| / ||R(delta)|| -- measures sign cancellation in R.

    For sign-consistent signals, |delta| ~ delta (up to sign), so ratio ~ 1.
    For sign-flipped signals, R averages away the cancelling components,
    so ||R(delta)|| << ||R(|delta|)||, ratio >> 1.
    """
    n_abs = np.linalg.norm(R_fn(np.abs(delta)))
    n_d = np.linalg.norm(R_fn(delta))
    if n_d < 1e-12:
        return 1.0
    return n_abs / n_d


def d_local_var(delta, coarse, R_fn):
    """Local variance ratio: mean(R(delta^2) - R(delta)^2) / var(delta).

    Measures how much within-patch variance delta has relative to global.
    Sign-flipped shift creates high local variance within R patches.
    """
    R_dsq = R_fn(delta ** 2)
    R_d = R_fn(delta)
    local_var = np.mean(R_dsq - R_d ** 2)
    return local_var / (np.var(delta) + 1e-12)


def d_energy_ratio(delta, coarse, R_fn):
    """Absolute relative energy change: |  ||R(refined)||/||R(coarse)|| - 1  |."""
    R_ref = R_fn(coarse + delta)
    R_c = R_fn(coarse)
    n_c = np.linalg.norm(R_c)
    if n_c < 1e-12:
        return 0.0
    return abs(np.linalg.norm(R_ref) / n_c - 1.0)


ALL_DETECTORS = {
    'd_correlation': d_correlation,
    'd_cosine': d_cosine,
    'd_spectral': d_spectral,
    'd_phase': d_phase,
    'd_histogram': d_histogram,
    'd_quantile': d_quantile,
    'd_gradient': d_gradient,
    'd_ssim': d_ssim,
    'd_abs_vs_signed': d_abs_vs_signed,
    'd_local_var': d_local_var,
    'd_energy_ratio': d_energy_ratio,
}


def generate_cases(R_fn, seed=42, shift_variants=False):
    """Generate test cases. If shift_variants, also generate coherent/smooth/scaled shifts."""
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

                neg_list = [
                    ("lf_drift", negative_lf_drift(tile_gt, tile_c, 0.3, 0.5, seed)),
                    ("coarse_shift", negative_coarse_shift(tile_gt, tile_c, 0.2, seed)),
                    ("random_lf", negative_random_lf(tile_c, sigma=4.0, amplitude=0.5, seed=seed)),
                    ("semant_wrong", negative_semant_wrong(tile_c, 1.0)),
                ]
                if shift_variants:
                    neg_list += [
                        ("cs_coherent", negative_coarse_shift_coherent(tile_gt, tile_c, 0.2, seed)),
                        ("cs_smooth", negative_coarse_shift_smooth(tile_gt, tile_c, 0.2, seed)),
                        ("cs_half", negative_coarse_shift_scaled(tile_gt, tile_c, 0.1, seed)),
                        ("cs_double", negative_coarse_shift_scaled(tile_gt, tile_c, 0.4, seed)),
                    ]
                for neg_type, delta in neg_list:
                    cases.append({**base, 'delta': delta, 'case_type': 'neg',
                                  'neg_type': neg_type, 'pos_type': None})
    return cases


def roc_auc(pos_scores, neg_scores):
    from sklearn.metrics import roc_auc_score
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return float('nan')
    y_true = np.array([0] * len(pos_scores) + [1] * len(neg_scores))
    y_score = np.array(list(pos_scores) + list(neg_scores))
    return roc_auc_score(y_true, y_score)


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


def analyze(records, key):
    pos = [r[key] for r in records if r['case_type'] == 'pos']
    neg = [r[key] for r in records if r['case_type'] == 'neg']
    result = {
        'global_auc': roc_auc(pos, neg),
        'cohens_d': cohens_d(pos, neg),
        'pos_mean': float(np.mean(pos)),
        'neg_mean': float(np.mean(neg)),
    }
    neg_types = sorted(set(r['neg_type'] for r in records if r['neg_type']))
    result['by_neg_type'] = {}
    for nt in neg_types:
        nv = [r[key] for r in records if r['case_type'] == 'neg' and r['neg_type'] == nt]
        result['by_neg_type'][nt] = {
            'auc': roc_auc(pos, nv), 'cohens_d': cohens_d(pos, nv),
            'mean': float(np.mean(nv)),
        }
    levels = sorted(set(r['level'] for r in records))
    result['by_level'] = {}
    for lv in levels:
        lp = [r[key] for r in records if r['level'] == lv and r['case_type'] == 'pos']
        ln = [r[key] for r in records if r['level'] == lv and r['case_type'] == 'neg']
        result['by_level'][lv] = {'auc': roc_auc(lp, ln), 'cohens_d': cohens_d(lp, ln)}
    return result


def main():
    seed = 42
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Coarse-Shift Analysis: Non-Norm-Based Detectors")
    print("=" * 80)

    r_variants = {
        'gauss_s1.0': lambda x: restrict_scalar(x),
        'gauss_s3.0': make_restrict_gaussian(3.0),
    }

    all_results = {}

    for r_name, R_fn in r_variants.items():
        print(f"\n{'=' * 70}")
        print(f"R operator: {r_name}")
        print(f"{'=' * 70}")

        cases = generate_cases(R_fn, seed=seed, shift_variants=False)
        n_pos = sum(1 for c in cases if c['case_type'] == 'pos')
        n_neg = sum(1 for c in cases if c['case_type'] == 'neg')
        print(f"  Cases: {len(cases)} ({n_pos} pos, {n_neg} neg)")

        records = []
        for case in cases:
            rec = {k: case[k] for k in ['case_type', 'neg_type', 'pos_type', 'level', 'structure']}
            for name, fn in ALL_DETECTORS.items():
                rec[name] = fn(case['delta'], case['coarse'], R_fn)
            rec['d_survival'] = d_parent_survival(case['delta'], case['coarse'], R_fn)
            rec['d_baseline'] = d_parent_baseline(case['delta'], case['coarse'], R_fn)
            records.append(rec)

        r_results = {}
        all_keys = list(ALL_DETECTORS.keys()) + ['d_survival', 'd_baseline']
        for key in all_keys:
            r_results[key] = analyze(records, key)

        all_results[r_name] = r_results

        print(f"\n  {'Detector':<18} {'AUC':>7} {'d':>7} {'cs':>7} {'lf_d':>7} {'rlf':>7} {'sw':>7}")
        print(f"  {'-' * 18} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7}")
        for key in all_keys:
            a = r_results[key]
            nt = a['by_neg_type']
            vals = [nt.get(t, {}).get('auc', float('nan'))
                    for t in ['coarse_shift', 'lf_drift', 'random_lf', 'semant_wrong']]
            print(f"  {key:<18} {a['global_auc']:7.4f} {a['cohens_d']:7.4f} "
                  + " ".join(f"{v:7.4f}" for v in vals))

    # Test shift variants with best R (sigma=3.0)
    print(f"\n{'=' * 70}")
    print("Shift Variant Comparison (R=gauss_s3.0, survival metric)")
    print(f"{'=' * 70}")

    R_fn = make_restrict_gaussian(3.0)
    cases_ext = generate_cases(R_fn, seed=seed, shift_variants=True)
    records_ext = []
    for case in cases_ext:
        rec = {k: case[k] for k in ['case_type', 'neg_type', 'pos_type', 'level', 'structure']}
        rec['d_survival'] = d_parent_survival(case['delta'], case['coarse'], R_fn)
        for name, fn in ALL_DETECTORS.items():
            rec[name] = fn(case['delta'], case['coarse'], R_fn)
        records_ext.append(rec)

    pos_surv = [r['d_survival'] for r in records_ext if r['case_type'] == 'pos']
    shift_types = ['coarse_shift', 'cs_coherent', 'cs_smooth', 'cs_half', 'cs_double']
    print(f"\n  Survival ratio AUC by shift variant:")
    for st in shift_types:
        neg_surv = [r['d_survival'] for r in records_ext
                    if r['case_type'] == 'neg' and r['neg_type'] == st]
        if neg_surv:
            auc = roc_auc(pos_surv, neg_surv)
            d = cohens_d(pos_surv, neg_surv)
            print(f"    {st:16s}: AUC={auc:.4f}  d={d:.4f}  mean={np.mean(neg_surv):.4f}")

    # Best detectors on shift variants
    print(f"\n  Best detectors on shift variants (AUC):")
    print(f"  {'Detector':<18} " + " ".join(f"{st:>16s}" for st in shift_types))
    print(f"  {'-' * 18} " + " ".join("-" * 16 for _ in shift_types))
    for key in list(ALL_DETECTORS.keys()) + ['d_survival']:
        row = f"  {key:<18} "
        for st in shift_types:
            pv = [r[key] for r in records_ext if r['case_type'] == 'pos']
            nv = [r[key] for r in records_ext if r['case_type'] == 'neg' and r['neg_type'] == st]
            if nv:
                row += f"{roc_auc(pv, nv):16.4f} "
            else:
                row += f"{'N/A':>16s} "
        print(row)

    # Save
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

    out_file = out_dir / "coarse_shift_detectors.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2, default=jd)
    print(f"\n[Saved] {out_file}")

    return all_results


if __name__ == "__main__":
    main()
