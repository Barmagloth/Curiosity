"""
Fixed baseline generators for all 4 space types.

Task A fix: replace per-pixel random sign flip in coarse_shift with
spatially coherent sign fields that do NOT self-cancel under restriction R.

Three coarse_shift variants:
  - coherent:  smooth sign field (low-freq Gaussian noise -> sign)
  - block:     sign constant within blocks (8x8 for grids, cluster/subtree for graph/tree)
  - gradient:  sign follows smooth linear gradient across the domain

All variants work for:
  T1: scalar grid (2D)
  T2: vector grid (2D, per-channel)
  T3: irregular graph (1D node values, graph topology)
  T4: tree hierarchy (1D node values, tree topology)

Also includes positive and negative baselines adapted for all space types.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


# ===================================================
# Smooth sign field generators (for grids)
# ===================================================

def _smooth_sign_field_2d(shape, sigma=6.0, seed=42):
    """Low-freq Gaussian noise -> sign. Produces spatially coherent +/- regions."""
    rng = np.random.RandomState(seed)
    raw = rng.randn(*shape[:2])
    smoothed = gaussian_filter(raw, sigma=sigma)
    return np.sign(smoothed)


def _block_sign_field_2d(shape, block_size=8, seed=42):
    """Sign constant within blocks of given size."""
    rng = np.random.RandomState(seed)
    H, W = shape[0], shape[1]
    nH = max(1, (H + block_size - 1) // block_size)
    nW = max(1, (W + block_size - 1) // block_size)
    block_signs = 2 * (rng.rand(nH, nW) > 0.5).astype(float) - 1
    field = np.zeros((H, W))
    for bi in range(nH):
        for bj in range(nW):
            r0 = bi * block_size
            r1 = min(r0 + block_size, H)
            c0 = bj * block_size
            c1 = min(c0 + block_size, W)
            field[r0:r1, c0:c1] = block_signs[bi, bj]
    return field


def _gradient_sign_field_2d(shape, seed=42):
    """Sign follows a smooth linear gradient, producing one +/- boundary."""
    rng = np.random.RandomState(seed)
    H, W = shape[0], shape[1]
    angle = rng.uniform(0, 2 * np.pi)
    offset = rng.uniform(-0.3, 0.3)
    x = np.linspace(-1, 1, W)
    y = np.linspace(-1, 1, H)
    xx, yy = np.meshgrid(x, y)
    plane = np.cos(angle) * xx + np.sin(angle) * yy + offset
    return np.sign(plane)


# ===================================================
# Sign field generators for graphs (topology-aware)
# ===================================================

def _smooth_sign_field_graph(n_points, neighbors, sigma_hops=3, seed=42):
    """Smooth sign field on graph via iterative neighbor averaging.

    Start with random values, diffuse for sigma_hops iterations, then take sign.
    Neighboring nodes end up with the same sign -> spatially coherent.
    """
    rng = np.random.RandomState(seed)
    field = rng.randn(n_points)
    for _ in range(sigma_hops):
        new_field = np.zeros(n_points)
        for i in range(n_points):
            nbrs = list(neighbors.get(i, []))
            vals = [field[i]] + [field[j] for j in nbrs]
            new_field[i] = np.mean(vals)
        field = new_field
    return np.sign(field)


def _block_sign_field_graph(n_points, labels, seed=42):
    """Sign constant within each cluster."""
    rng = np.random.RandomState(seed)
    n_clusters = int(labels.max()) + 1
    cluster_signs = 2 * (rng.rand(n_clusters) > 0.5).astype(float) - 1
    field = np.zeros(n_points)
    for i in range(n_points):
        field[i] = cluster_signs[labels[i]]
    return field


def _gradient_sign_field_graph(n_points, positions, seed=42):
    """Sign follows a linear gradient in the embedding space."""
    rng = np.random.RandomState(seed)
    angle = rng.uniform(0, 2 * np.pi)
    offset = rng.uniform(-0.3, 0.3)
    proj = np.cos(angle) * positions[:, 0] + np.sin(angle) * positions[:, 1] + offset
    return np.sign(proj)


# ===================================================
# Sign field generators for trees (topology-aware)
# ===================================================

def _smooth_sign_field_tree(n_nodes, sigma_hops=3, seed=42):
    """Smooth sign field on tree via parent/child averaging."""
    rng = np.random.RandomState(seed)
    field = rng.randn(n_nodes)
    for _ in range(sigma_hops):
        new_field = field.copy()
        for i in range(n_nodes):
            vals = [field[i]]
            parent = (i - 1) // 2 if i > 0 else -1
            if parent >= 0:
                vals.append(field[parent])
            left = 2 * i + 1
            right = 2 * i + 2
            if left < n_nodes:
                vals.append(field[left])
            if right < n_nodes:
                vals.append(field[right])
            new_field[i] = np.mean(vals)
        field = new_field
    return np.sign(field)


def _block_sign_field_tree(n_nodes, coarse_depth=3, seed=42):
    """Sign constant within each subtree at coarse_depth."""
    rng = np.random.RandomState(seed)
    n_coarse = 2 ** coarse_depth
    subtree_signs = 2 * (rng.rand(n_coarse) > 0.5).astype(float) - 1
    field = np.zeros(n_nodes)
    for i in range(n_nodes):
        ancestor = i
        while int(np.log2(ancestor + 1)) >= coarse_depth and ancestor > 0:
            ancestor = (ancestor - 1) // 2
        d = int(np.log2(ancestor + 1))
        if d == coarse_depth:
            idx = ancestor - (2 ** coarse_depth - 1)
            if 0 <= idx < n_coarse:
                field[i] = subtree_signs[idx]
            else:
                field[i] = 1.0
        elif d < coarse_depth:
            field[i] = 1.0
        else:
            field[i] = 1.0
    return field


def _gradient_sign_field_tree(n_nodes, seed=42):
    """Sign follows node index gradient (left subtree vs right subtree)."""
    rng = np.random.RandomState(seed)
    offset = rng.uniform(-0.2, 0.2)
    field = np.zeros(n_nodes)
    for i in range(n_nodes):
        normalized = (i / max(n_nodes - 1, 1)) * 2 - 1 + offset
        field[i] = np.sign(normalized) if abs(normalized) > 0.01 else 1.0
    return field


# ===================================================
# Positive baselines (all space types)
# ===================================================

def positive_oracle(gt, coarse):
    return gt - coarse


def positive_scaled(gt, coarse, scale=0.5, seed=42):
    return scale * (gt - coarse)


def positive_noisy(gt, coarse, noise_std=0.01, seed=42):
    rng = np.random.RandomState(seed)
    delta = gt - coarse
    noise = rng.randn(*delta.shape) * noise_std
    return delta + noise


# ===================================================
# Negative baselines (all space types)
# ===================================================

def negative_lf_drift(gt, coarse, amplitude=0.3, freq_scale=0.5, seed=42):
    """LF drift: correct delta + low-frequency sinusoid."""
    rng = np.random.RandomState(seed)
    delta = gt - coarse

    if delta.ndim == 2:
        N = delta.shape[0]
        x = np.linspace(0, 1, N, endpoint=False)
        xx, yy = np.meshgrid(x, x)
        phase = rng.uniform(0, 2 * np.pi)
        lf = amplitude * np.sin(2 * np.pi * freq_scale * xx + phase)
    elif delta.ndim == 3:
        N = delta.shape[0]
        x = np.linspace(0, 1, N, endpoint=False)
        xx, yy = np.meshgrid(x, x)
        phase = rng.uniform(0, 2 * np.pi)
        lf = amplitude * np.sin(2 * np.pi * freq_scale * xx + phase)
        lf = lf[:, :, np.newaxis]
    elif delta.ndim == 1:
        n = delta.shape[0]
        t = np.linspace(0, 1, n, endpoint=False)
        phase = rng.uniform(0, 2 * np.pi)
        lf = amplitude * np.sin(2 * np.pi * freq_scale * t + phase)
    else:
        raise ValueError(f"Unsupported delta ndim={delta.ndim}")

    return delta + lf


def negative_random_lf(coarse, sigma=8.0, amplitude=0.5, seed=42):
    """Random LF delta: low-frequency noise unrelated to GT."""
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
        kernel_size = max(3, int(sigma))
        kernel = np.ones(kernel_size) / kernel_size
        return np.convolve(raw, kernel, mode='same')
    else:
        raise ValueError(f"Unsupported coarse ndim={coarse.ndim}")


def negative_semant_wrong(coarse, scale=1.0):
    """Semant-wrong: delta flips the sign of coarse (extreme violation)."""
    return -2.0 * scale * coarse


# ===================================================
# FIXED coarse_shift: coherent variant
# ===================================================

def negative_coarse_shift_coherent(gt, coarse, shift_frac=0.2, seed=42,
                                   space_info=None):
    """Coarse shift with smooth (low-freq noise) sign field.

    For grids: 2D Gaussian smoothed sign field (sigma=6).
    For graphs: diffused sign field on graph topology.
    For trees: diffused sign field on tree topology.
    """
    delta = gt - coarse
    sp = space_info or {}
    space_type = sp.get('type', _infer_type(delta))

    if space_type in ('scalar_grid', 'vector_grid'):
        sign = _smooth_sign_field_2d(coarse.shape, sigma=6.0, seed=seed)
        if coarse.ndim == 3:
            sign = sign[:, :, np.newaxis]
    elif space_type == 'graph':
        neighbors = sp.get('neighbors', {})
        sign = _smooth_sign_field_graph(len(coarse), neighbors, sigma_hops=3, seed=seed)
    elif space_type == 'tree':
        sign = _smooth_sign_field_tree(len(coarse), sigma_hops=3, seed=seed)
    else:
        sign = _smooth_sign_field_2d(coarse.shape, sigma=6.0, seed=seed)
        if coarse.ndim == 3:
            sign = sign[:, :, np.newaxis]

    return delta + shift_frac * coarse * sign


# ===================================================
# FIXED coarse_shift: block variant
# ===================================================

def negative_coarse_shift_block(gt, coarse, shift_frac=0.2, seed=42,
                                space_info=None):
    """Coarse shift with block-constant sign field.

    For grids: sign constant within 8x8 blocks.
    For graphs: sign constant within each cluster.
    For trees: sign constant within each subtree at coarse_depth.
    """
    delta = gt - coarse
    sp = space_info or {}
    space_type = sp.get('type', _infer_type(delta))

    if space_type in ('scalar_grid', 'vector_grid'):
        sign = _block_sign_field_2d(coarse.shape, block_size=8, seed=seed)
        if coarse.ndim == 3:
            sign = sign[:, :, np.newaxis]
    elif space_type == 'graph':
        labels = sp.get('labels', np.zeros(len(coarse), dtype=int))
        sign = _block_sign_field_graph(len(coarse), labels, seed=seed)
    elif space_type == 'tree':
        coarse_depth = sp.get('coarse_depth', 3)
        sign = _block_sign_field_tree(len(coarse), coarse_depth=coarse_depth, seed=seed)
    else:
        sign = _block_sign_field_2d(coarse.shape, block_size=8, seed=seed)
        if coarse.ndim == 3:
            sign = sign[:, :, np.newaxis]

    return delta + shift_frac * coarse * sign


# ===================================================
# FIXED coarse_shift: gradient variant
# ===================================================

def negative_coarse_shift_gradient(gt, coarse, shift_frac=0.2, seed=42,
                                   space_info=None):
    """Coarse shift with smooth gradient sign field.

    For grids: sign follows a linear plane (one diagonal boundary).
    For graphs: sign follows a linear gradient in embedding space.
    For trees: sign follows node index gradient (left vs right).
    """
    delta = gt - coarse
    sp = space_info or {}
    space_type = sp.get('type', _infer_type(delta))

    if space_type in ('scalar_grid', 'vector_grid'):
        sign = _gradient_sign_field_2d(coarse.shape, seed=seed)
        if coarse.ndim == 3:
            sign = sign[:, :, np.newaxis]
    elif space_type == 'graph':
        positions = sp.get('positions', np.zeros((len(coarse), 2)))
        sign = _gradient_sign_field_graph(len(coarse), positions, seed=seed)
    elif space_type == 'tree':
        sign = _gradient_sign_field_tree(len(coarse), seed=seed)
    else:
        sign = _gradient_sign_field_2d(coarse.shape, seed=seed)
        if coarse.ndim == 3:
            sign = sign[:, :, np.newaxis]

    return delta + shift_frac * coarse * sign


def _infer_type(arr):
    if arr.ndim == 3:
        return 'vector_grid'
    if arr.ndim == 2:
        return 'scalar_grid'
    return 'scalar_grid'


# ===================================================
# Convenience: all shift variants
# ===================================================

COARSE_SHIFT_VARIANTS = {
    'coherent': negative_coarse_shift_coherent,
    'block': negative_coarse_shift_block,
    'gradient': negative_coarse_shift_gradient,
}
