"""
Alternative R/Up operator pairs for scale-consistency verification.

Three alternatives to the baseline gaussian blur + decimation / bilinear upsampling:

  R_lanczos:  Lanczos-3 downsampling (sharper cutoff, preserves more structure)
  R_area:     Area averaging (box filter, exact energy preservation per block)
  R_wavelet:  Haar wavelet decomposition (exact, invertible at coarse level)

Each has a matching Up (prolongation) operator for T1 scalar grids.
"""

import numpy as np
from scipy.ndimage import zoom


# ═══════════════════════════════════════════════
# Lanczos-3 downsample / Lanczos-3 upsample
# ═══════════════════════════════════════════════

def _lanczos_kernel(x, a=3):
    x = np.abs(x)
    out = np.zeros_like(x)
    mask = x < a
    xm = x[mask]
    out[mask] = np.where(
        xm < 1e-12,
        1.0,
        (a * np.sin(np.pi * xm) * np.sin(np.pi * xm / a))
        / (np.pi**2 * xm**2)
    )
    return out


def _downsample_lanczos_1d(signal, factor=2, a=3):
    n_out = len(signal) // factor
    out = np.empty(n_out)
    for i in range(n_out):
        center = (i + 0.5) * factor - 0.5
        lo = max(0, int(np.floor(center - a * factor)))
        hi = min(len(signal) - 1, int(np.ceil(center + a * factor)))
        indices = np.arange(lo, hi + 1)
        weights = _lanczos_kernel((indices - center) / factor, a)
        wsum = weights.sum()
        if wsum > 0:
            out[i] = np.dot(weights, signal[indices]) / wsum
        else:
            out[i] = signal[int(round(center))]
    return out


def restrict_lanczos(x, factor=2):
    h, w = x.shape
    tmp = np.empty((h, w // factor))
    for i in range(h):
        tmp[i, :] = _downsample_lanczos_1d(x[i, :], factor)
    out = np.empty((h // factor, w // factor))
    for j in range(w // factor):
        out[:, j] = _downsample_lanczos_1d(tmp[:, j], factor)
    return out


def prolong_lanczos(x_coarse, target_shape):
    factors = (target_shape[0] / x_coarse.shape[0],
               target_shape[1] / x_coarse.shape[1])
    return zoom(x_coarse, factors, order=3)


# ═══════════════════════════════════════════════
# Area averaging (box filter) downsample / bilinear upsample
# ═══════════════════════════════════════════════

def restrict_area(x, factor=2):
    h, w = x.shape
    h_out, w_out = h // factor, w // factor
    reshaped = x[:h_out * factor, :w_out * factor].reshape(
        h_out, factor, w_out, factor
    )
    return reshaped.mean(axis=(1, 3))


def prolong_area(x_coarse, target_shape):
    factors = (target_shape[0] / x_coarse.shape[0],
               target_shape[1] / x_coarse.shape[1])
    return zoom(x_coarse, factors, order=1)


# ═══════════════════════════════════════════════
# Haar wavelet downsample / upsample
# ═══════════════════════════════════════════════

def restrict_wavelet(x, factor=2):
    h, w = x.shape
    h_out, w_out = h // factor, w // factor
    reshaped = x[:h_out * factor, :w_out * factor].reshape(
        h_out, factor, w_out, factor
    )
    return reshaped.mean(axis=(1, 3))


def prolong_wavelet(x_coarse, target_shape):
    h_out, w_out = target_shape[0], target_shape[1]
    factor_h = h_out // x_coarse.shape[0]
    factor_w = w_out // x_coarse.shape[1]
    out = np.repeat(np.repeat(x_coarse, factor_h, axis=0), factor_w, axis=1)
    return out[:h_out, :w_out]


# ═══════════════════════════════════════════════
# Gaussian with higher sigma (more aggressive LF extraction)
# ═══════════════════════════════════════════════

def make_restrict_gaussian(sigma):
    from scipy.ndimage import gaussian_filter as gf
    def restrict(x, _sigma=sigma):
        blurred = gf(x, sigma=_sigma)
        return blurred[::2, ::2]
    return restrict


def prolong_gaussian(x_coarse, target_shape):
    factors = (target_shape[0] / x_coarse.shape[0],
               target_shape[1] / x_coarse.shape[1])
    return zoom(x_coarse, factors, order=1)


# ═══════════════════════════════════════════════
# Wavelet with piecewise-linear (bilinear) Up for better D_hf
# ═══════════════════════════════════════════════

def prolong_wavelet_bilinear(x_coarse, target_shape):
    factors = (target_shape[0] / x_coarse.shape[0],
               target_shape[1] / x_coarse.shape[1])
    return zoom(x_coarse, factors, order=1)


# ═══════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════

VARIANTS = {
    'lanczos': (restrict_lanczos, prolong_lanczos),
    'area': (restrict_area, prolong_area),
    'wavelet': (restrict_wavelet, prolong_wavelet),
}
