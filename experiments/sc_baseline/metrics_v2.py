"""
Alternative D_parent normalizations to fix the separability failure.

Baseline problem (from diagnostic):
  D_parent = ||R(delta)|| / (alpha * ||coarse|| + beta)
  - Denominator is identical for pos and neg cases on the same tile
  - Large denominator variance (CV=1.24) compresses dynamic range
  - coarse_shift and random_lf become indistinguishable from positives

Seven alternative normalizations:
  1. D_parent_log: log-scale compression
  2. D_parent_relative: normalize by ||R(refined)|| instead of ||coarse||
  3. D_parent_zscore: z-score within each depth level
  4. D_parent_rank: rank-based (distribution-free)
  5. D_parent_survival: ||R(delta)||/||delta|| (restriction survival ratio)
  6. D_parent_lf_frac: ||Up(R(delta))||/||delta|| (LF fraction = 1-D_hf)
  7. D_parent_combined: survival * (||delta||/||coarse||) (survival weighted by relative energy)
"""

import numpy as np
from collections import defaultdict

from metrics import ALPHA, compute_beta


def d_parent_baseline(delta, coarse, restrict_fn, alpha=ALPHA, beta=None):
    if beta is None:
        beta = compute_beta(coarse)
    R_delta = restrict_fn(delta)
    numer = np.linalg.norm(R_delta.ravel())
    denom = alpha * np.linalg.norm(coarse.ravel()) + beta
    return numer / denom


def d_parent_log(delta, coarse, restrict_fn, alpha=ALPHA, beta=None):
    """Log-scale normalization: log(1 + ||R(delta)|| / ||coarse||).

    Compresses large values, expands small differences near zero.
    The log transform stabilizes the heavy-tailed distribution
    caused by small-||coarse|| tiles blowing up the ratio.
    """
    if beta is None:
        beta = compute_beta(coarse)
    R_delta = restrict_fn(delta)
    numer = np.linalg.norm(R_delta.ravel())
    denom = alpha * np.linalg.norm(coarse.ravel()) + beta
    return np.log1p(numer / denom)


def d_parent_relative(delta, coarse, restrict_fn, prolong_fn, alpha=ALPHA):
    """Relative to refined energy: ||R(delta)|| / ||R(coarse + delta)||.

    Instead of normalizing by ||coarse|| (which is the same for all deltas
    on the same tile), normalizes by the total energy of the refined signal.
    This makes the denominator depend on the actual delta, not just the tile.

    The key insight: a good delta should have R(delta) small relative to
    R(refined), because refinement should not change the coarse picture.
    A bad delta (LF contamination) will have R(delta) large relative to
    R(refined) because the refinement IS the coarse picture change.
    """
    R_delta = restrict_fn(delta)
    refined = coarse + delta
    R_refined = restrict_fn(refined)

    numer = np.linalg.norm(R_delta.ravel())
    denom = np.linalg.norm(R_refined.ravel()) + 1e-8
    return numer / denom


def d_parent_survival(delta, coarse, restrict_fn):
    """Restriction survival ratio: ||R(delta)|| / ||delta||.

    Measures what fraction of delta energy survives restriction.
    Pure HF signals have low survival (~0.12-0.20); pure DC has 0.5.
    Does not depend on ||coarse|| at all -- removes the shared-denominator problem.
    """
    R_delta = restrict_fn(delta)
    numer = np.linalg.norm(R_delta.ravel())
    denom = np.linalg.norm(delta.ravel()) + 1e-12
    return numer / denom


def d_parent_lf_frac(delta, coarse, restrict_fn, prolong_fn):
    """LF fraction of delta: ||Up(R(delta))|| / ||delta||.

    This is essentially 1 - D_hf (the complement of HF purity).
    Measures how much of delta is low-frequency content that would
    project onto the coarse level.
    """
    R_delta = restrict_fn(delta)
    Up_R_delta = prolong_fn(R_delta, delta.shape)
    numer = np.linalg.norm(Up_R_delta.ravel())
    denom = np.linalg.norm(delta.ravel()) + 1e-12
    return numer / denom


def d_parent_combined(delta, coarse, restrict_fn):
    """Combined: survival ratio * relative energy.

    survival = ||R(delta)|| / ||delta||   (how LF is the delta?)
    rel_energy = ||delta|| / ||coarse||   (how big is delta vs coarse?)

    Product = ||R(delta)|| / ||coarse|| = baseline numerator / ||coarse||.

    But the key difference from baseline: this formulation separates the
    two independent failure modes (LF contamination vs energy magnitude)
    and combines them multiplicatively rather than as a single ratio.

    Actually this simplifies to the baseline without beta. The insight is
    that we should use a DIFFERENT combination. Let's use:

    ||R(delta)||^2 / (||R(delta)|| * ||delta|| + eps)
    = ||R(delta)|| / ||delta|| = survival ratio

    So instead let's try: survival * log(1 + ||delta||/||coarse||)
    This amplifies survival ratio when delta is large relative to coarse.
    """
    R_delta = restrict_fn(delta)
    norm_R_delta = np.linalg.norm(R_delta.ravel())
    norm_delta = np.linalg.norm(delta.ravel()) + 1e-12
    norm_coarse = np.linalg.norm(coarse.ravel()) + 1e-12

    survival = norm_R_delta / norm_delta
    rel_energy = norm_delta / norm_coarse
    return survival * np.log1p(rel_energy)


def d_parent_zscore_raw(delta, coarse, restrict_fn, alpha=ALPHA, beta=None):
    """Compute raw D_parent value for later z-score normalization.

    Returns the raw value; z-scoring is done in a batch post-processing step
    per depth level (see zscore_normalize).
    """
    return d_parent_baseline(delta, coarse, restrict_fn, alpha, beta)


def zscore_normalize(records, raw_key='d_parent_zscore_raw', out_key='d_parent_zscore'):
    """Z-score normalize raw D_parent values per depth level.

    For each level, compute mean and std of the raw metric across ALL cases
    (both positive and negative), then transform: (x - mu) / sigma.

    This removes depth-dependent bias where different tile sizes produce
    systematically different D_parent ranges.
    """
    by_level = defaultdict(list)
    for i, r in enumerate(records):
        by_level[r['level']].append((i, r[raw_key]))

    for level, items in by_level.items():
        vals = np.array([v for _, v in items])
        mu = vals.mean()
        sigma = vals.std()
        if sigma < 1e-12:
            sigma = 1.0
        for idx, raw_val in items:
            records[idx][out_key] = (raw_val - mu) / sigma

    return records


def rank_normalize(records, raw_key='d_parent_baseline', out_key='d_parent_rank'):
    """Replace raw values with rank percentiles (0 to 1).

    Maximally robust to distributional shape. Converts to uniform
    distribution, preserving only ordinal information.
    """
    vals = np.array([r[raw_key] for r in records])
    n = len(vals)
    order = vals.argsort()
    ranks = np.empty(n)
    ranks[order] = np.arange(n) / (n - 1) if n > 1 else np.array([0.5])

    for i, r in enumerate(records):
        r[out_key] = float(ranks[i])

    return records
