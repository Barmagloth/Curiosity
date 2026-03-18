"""
Positive and negative baseline generators for scale-consistency verification.

Positive baselines: scale-consistency guaranteed by construction.
  - Strong positive: delta = GT - coarse (near-oracle)

Negative baselines: scale-consistency intentionally violated.
  - lf_drift: low-frequency sinusoid added to correct delta
  - coarse_shift: delta shifts coarse-mean by 10-30%
  - random_lf: random low-frequency noise unrelated to GT
  - semant_wrong: delta flips sign of coarse in region (extreme)

See scale_consistency_verification_protocol_v1.0.md Steps 1.1 and 1.2.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


# ═══════════════════════════════════════════════
# Structure types for 2D grids
# ═══════════════════════════════════════════════

def make_structure_regions(N, seed=42):
    """Generate labeled structure-type map for an NxN grid.

    Returns:
        gt: (N, N) ground truth signal
        coarse: (N, N) tile-mean approximation (tile=8)
        labels: (N, N) int array — 0=smooth, 1=boundary, 2=texture
    """
    rng = np.random.RandomState(seed)
    T = 8
    NT = N // T
    x = np.linspace(0, 1, N, endpoint=False)
    xx, yy = np.meshgrid(x, x)

    # GT with three regimes
    gt = np.zeros((N, N))
    labels = np.zeros((N, N), dtype=int)

    # Smooth region: low-frequency
    smooth = 0.3 * np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy)

    # Boundary: sharp edge
    edge = 0.8 * (xx > 0.45).astype(float)

    # Texture: high-frequency
    texture = 0.2 * np.sin(2 * np.pi * 12 * xx) * np.cos(2 * np.pi * 10 * yy)

    # Assign regions (vertical thirds)
    third = N // 3
    gt[:, :third] = smooth[:, :third]
    labels[:, :third] = 0  # smooth

    gt[:, third:2*third] = edge[:, third:2*third] + smooth[:, third:2*third] * 0.3
    labels[:, third:2*third] = 1  # boundary

    gt[:, 2*third:] = texture[:, 2*third:] + smooth[:, 2*third:] * 0.5
    labels[:, 2*third:] = 2  # texture

    gt += rng.randn(N, N) * 0.01

    # Coarse: tile-mean
    coarse = np.zeros_like(gt)
    for ti in range(NT):
        for tj in range(NT):
            s = slice(ti * T, (ti + 1) * T)
            cs = slice(tj * T, (tj + 1) * T)
            coarse[s, cs] = gt[s, cs].mean()

    return gt, coarse, labels


# ═══════════════════════════════════════════════
# Positive baselines
# ═══════════════════════════════════════════════

def positive_oracle(gt, coarse):
    """Strong positive: delta = GT - coarse.

    Scale-consistency is guaranteed by construction because delta
    contains only the true detail that coarse doesn't capture.

    Returns:
        delta: array same shape as gt
    """
    return gt - coarse


def positive_scaled(gt, coarse, scale=0.5, seed=42):
    """Scaled positive: delta = scale * (GT - coarse).

    Simulates partial refinement. Still scale-consistent.
    """
    return scale * (gt - coarse)


def positive_noisy(gt, coarse, noise_std=0.01, seed=42):
    """Noisy positive: delta = (GT - coarse) + small HF noise.

    Small noise doesn't violate scale-consistency if it's high-frequency.
    """
    rng = np.random.RandomState(seed)
    delta = gt - coarse
    noise = rng.randn(*delta.shape) * noise_std
    return delta + noise


# ═══════════════════════════════════════════════
# Negative baselines
# ═══════════════════════════════════════════════

def negative_lf_drift(gt, coarse, amplitude=0.3, freq_scale=0.5, seed=42):
    """LF drift: correct delta + low-frequency sinusoid (scale > tile_size).

    The added sinusoid has spatial scale larger than a tile, so it will
    project strongly onto the coarse level via R, violating scale-consistency.

    Args:
        amplitude: strength of the LF sinusoid relative to delta energy
        freq_scale: spatial frequency multiplier (lower = larger scale drift)
    """
    rng = np.random.RandomState(seed)
    delta = gt - coarse
    shape = delta.shape

    if delta.ndim == 2:
        N = shape[0]
        x = np.linspace(0, 1, N, endpoint=False)
        xx, yy = np.meshgrid(x, x)
        phase = rng.uniform(0, 2 * np.pi)
        lf = amplitude * np.sin(2 * np.pi * freq_scale * xx + phase)
    elif delta.ndim == 3:
        N = shape[0]
        x = np.linspace(0, 1, N, endpoint=False)
        xx, yy = np.meshgrid(x, x)
        phase = rng.uniform(0, 2 * np.pi)
        lf = amplitude * np.sin(2 * np.pi * freq_scale * xx + phase)
        lf = lf[:, :, np.newaxis]
    elif delta.ndim == 1:
        n = shape[0]
        t = np.linspace(0, 1, n, endpoint=False)
        phase = rng.uniform(0, 2 * np.pi)
        lf = amplitude * np.sin(2 * np.pi * freq_scale * t + phase)
    else:
        raise ValueError(f"Unsupported delta ndim={delta.ndim}")

    return delta + lf


def negative_coarse_shift(gt, coarse, shift_frac=0.2, seed=42):
    """Coarse shift: delta intentionally shifts coarse-mean by shift_frac (10-30%).

    The delta contains a DC offset proportional to the local coarse value,
    directly contradicting the coarse level.
    """
    rng = np.random.RandomState(seed)
    delta = gt - coarse
    shift = shift_frac * coarse
    # Add random sign per region for variety
    if delta.ndim >= 2:
        sign_map = (2 * (rng.rand(*coarse.shape[:2]) > 0.5).astype(float) - 1)
        if delta.ndim == 3:
            sign_map = sign_map[:, :, np.newaxis]
    else:
        sign_map = 2 * (rng.rand(*coarse.shape) > 0.5).astype(float) - 1
    return delta + shift * sign_map


def negative_random_lf(coarse, sigma=8.0, amplitude=0.5, seed=42):
    """Random LF delta: low-frequency noise unrelated to GT.

    The delta is purely random low-pass noise, not connected to any
    ground truth structure. Strong violation of scale-consistency.
    """
    rng = np.random.RandomState(seed)
    if coarse.ndim == 2:
        raw = rng.randn(*coarse.shape) * amplitude
        return gaussian_filter(raw, sigma=sigma)
    elif coarse.ndim == 3:
        result = np.empty_like(coarse)
        for d in range(coarse.shape[2]):
            raw = rng.randn(coarse.shape[0], coarse.shape[1]) * amplitude
            result[:, :, d] = gaussian_filter(raw, sigma=sigma)
        return result
    elif coarse.ndim == 1:
        raw = rng.randn(*coarse.shape) * amplitude
        # 1D low-pass: simple moving average
        kernel_size = max(3, int(sigma))
        kernel = np.ones(kernel_size) / kernel_size
        return np.convolve(raw, kernel, mode='same')
    else:
        raise ValueError(f"Unsupported coarse ndim={coarse.ndim}")


def negative_semant_wrong(coarse, scale=1.0):
    """Semant-wrong: delta flips the sign of coarse in the region.

    Extreme case: delta = -2 * coarse, so refined = coarse + delta = -coarse.
    This is the most severe violation of scale-consistency.
    """
    return -2.0 * scale * coarse
