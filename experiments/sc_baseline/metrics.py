"""
D_parent and D_hf metric implementations for scale-consistency verification.

D_parent measures how much of delta "leaks" into the parent (coarser) scale.
  D_parent = ||R(delta)|| / (alpha * ||coarse|| + beta)
  Higher D_parent = worse (delta is smuggling coarse-level meaning).

D_hf measures what fraction of delta energy is high-frequency (detail).
  D_hf = ||delta_HF|| / (||delta|| + eps)
  where delta_HF = delta - Up(R(delta))
  Higher D_hf = better (delta is mostly detail, not LF drift).

See concept_v1.6.md section 8.4 for interpretation table.
See scale_consistency_verification_protocol_v1.0.md Step 2 for formulas.
"""

import numpy as np


ALPHA = 1.0
BETA_SCALE = 1e-4
EPS_SCALE = 1e-4


def compute_beta(coarse):
    """Compute beta stabilizer: 1e-4 * mean(||coarse||).

    Protocol parameter: beta = small constant, e.g. 1e-4 * mean(||coarse||).
    """
    return BETA_SCALE * np.mean(np.abs(coarse))


def compute_eps(delta):
    """Compute eps stabilizer: 1e-4 * mean(||delta||).

    Analogous to beta but for D_hf denominator.
    """
    return EPS_SCALE * np.mean(np.abs(delta))


def d_parent(delta, coarse, restrict_fn, alpha=ALPHA, beta=None):
    """Compute D_parent: LF leakage of delta into parent scale.

    D_parent = ||R(delta)|| / (alpha * ||coarse|| + beta)

    Args:
        delta: array — refinement correction (any shape)
        coarse: array — parent coarse representation (any shape)
        restrict_fn: callable — R operator, takes array and returns restricted array
        alpha: float — normalization scale (default 1.0)
        beta: float or None — stabilizer (auto-computed if None)

    Returns:
        float — D_parent value. Lower is better.
    """
    if beta is None:
        beta = compute_beta(coarse)

    R_delta = restrict_fn(delta)
    norm_R_delta = np.linalg.norm(R_delta.ravel())
    norm_coarse = np.linalg.norm(coarse.ravel())
    return norm_R_delta / (alpha * norm_coarse + beta)


def d_hf(delta, restrict_fn, prolong_fn, eps=None):
    """Compute D_hf: high-frequency purity of delta.

    delta_HF = delta - Up(R(delta))
    D_hf = ||delta_HF|| / (||delta|| + eps)

    Args:
        delta: array — refinement correction
        restrict_fn: callable — R operator
        prolong_fn: callable(restricted, target_shape) — Up operator
        eps: float or None — stabilizer (auto-computed if None)

    Returns:
        float — D_hf value. Higher is better.
    """
    if eps is None:
        eps = compute_eps(delta)

    R_delta = restrict_fn(delta)
    P_LF_delta = prolong_fn(R_delta, delta.shape)
    delta_HF = delta - P_LF_delta

    norm_delta_HF = np.linalg.norm(delta_HF.ravel())
    norm_delta = np.linalg.norm(delta.ravel())
    return norm_delta_HF / (norm_delta + eps)


def compute_metrics(delta, coarse, restrict_fn, prolong_fn,
                    alpha=ALPHA, beta=None, eps=None):
    """Compute both D_parent and D_hf in one pass (shared R(delta)).

    Args:
        delta: array — refinement correction
        coarse: array — parent coarse representation
        restrict_fn: callable — R operator
        prolong_fn: callable(restricted, target_shape) — Up operator
        alpha: float
        beta: float or None
        eps: float or None

    Returns:
        (D_parent, D_hf) tuple of floats
    """
    if beta is None:
        beta = compute_beta(coarse)
    if eps is None:
        eps = compute_eps(delta)

    R_delta = restrict_fn(delta)
    P_LF_delta = prolong_fn(R_delta, delta.shape)
    delta_HF = delta - P_LF_delta

    norm_R_delta = np.linalg.norm(R_delta.ravel())
    norm_coarse = np.linalg.norm(coarse.ravel())
    norm_delta_HF = np.linalg.norm(delta_HF.ravel())
    norm_delta = np.linalg.norm(delta.ravel())

    dp = norm_R_delta / (alpha * norm_coarse + beta)
    dh = norm_delta_HF / (norm_delta + eps)
    return dp, dh
