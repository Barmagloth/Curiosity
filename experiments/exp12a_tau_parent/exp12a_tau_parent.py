#!/usr/bin/env python3
"""
Curiosity -- exp12a_tau_parent: Data-driven thresholds tau_parent[L] per depth level.

Determines optimal thresholds for Scale-Consistency enforcement at each tree depth
using D_parent (lf_frac variant) from the SC-baseline cross-space validation.

Question: What are the optimal data-driven thresholds tau_parent[L] for each tree depth?
Kill criteria: accuracy drops >15 percentage points on held-out space = problem.

Roadmap level: SC-5
Depends on: SC-baseline (SC-0..SC-4) PASSED.

Methods for finding thresholds:
  - youden_j:        ROC optimal point (max sensitivity + specificity - 1)
  - f1_optimal:      threshold maximizing F1 score
  - sensitivity_at_90: threshold achieving >=90% sensitivity

Cross-validation:
  Leave-one-space-out (4 folds). Train on 3 spaces, validate on held-out.
  10 training seeds + 10 validation seeds per fold.
  Kill criterion: accuracy drops >15pp on held-out vs training.

CPU-only. Requires numpy, scipy, sklearn, matplotlib.

Usage:
    python exp12a_tau_parent.py [--output-dir DIR] [--n-train-seeds N]
                                [--n-val-seeds N] [--base-seed SEED]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from scipy.spatial import cKDTree
from scipy.cluster.vq import kmeans2

# ---------------------------------------------------------------------------
# Imports from sc_baseline (add parent to path)
# ---------------------------------------------------------------------------
SC_BASELINE_DIR = Path(__file__).resolve().parent.parent / "sc_baseline"
sys.path.insert(0, str(SC_BASELINE_DIR))

from operators import (
    restrict_graph, prolong_graph,
    restrict_tree, prolong_tree,
)
from operators_v2 import make_restrict_gaussian, prolong_gaussian
from baselines_v2 import (
    positive_oracle, positive_scaled, positive_noisy,
    negative_lf_drift, negative_random_lf, negative_semant_wrong,
    negative_coarse_shift_coherent,
    negative_coarse_shift_block,
    negative_coarse_shift_gradient,
)


# ===================================================================
# R/Up operators per space type (from sc_baseline_crossspace)
# ===================================================================

def make_grid_ops(sigma=3.0):
    """R/Up for scalar grid using gauss sigma=3.0."""
    R_fn = make_restrict_gaussian(sigma)
    Up_fn = lambda xc, tgt: prolong_gaussian(xc, tgt)
    return R_fn, Up_fn


def make_vector_ops(sigma=3.0):
    """R/Up for vector grid: per-channel gauss sigma + decimation."""
    def R_fn(x):
        D = x.shape[2]
        out = np.empty((x.shape[0] // 2, x.shape[1] // 2, D))
        for d in range(D):
            blurred = gaussian_filter(x[:, :, d], sigma=sigma)
            out[:, :, d] = blurred[::2, ::2]
        return out

    def Up_fn(xc, tgt):
        D = xc.shape[2]
        H, W = tgt[0], tgt[1]
        out = np.empty((H, W, D))
        factors = (H / xc.shape[0], W / xc.shape[1])
        for d in range(D):
            out[:, :, d] = zoom(xc[:, :, d], factors, order=1)
        return out

    return R_fn, Up_fn


class GraphOps:
    """R/Up for irregular graph."""
    def __init__(self, labels, n_clusters, n_points):
        self.labels = labels
        self.n_clusters = n_clusters
        self.n_points = n_points

    def restrict(self, values):
        return restrict_graph(values, self.labels, self.n_clusters)

    def prolong(self, coarse_values, target_shape):
        return prolong_graph(coarse_values, self.labels, self.n_points)


class TreeOps:
    """R/Up for tree hierarchy."""
    def __init__(self, n_nodes, coarse_depth):
        self.n_nodes = n_nodes
        self.coarse_depth = coarse_depth

    def restrict(self, values):
        return restrict_tree(values, self.n_nodes, self.coarse_depth)

    def prolong(self, coarse_values, target_shape):
        return prolong_tree(coarse_values, self.n_nodes, self.coarse_depth)


# ===================================================================
# D_parent metric (lf_frac variant, primary from cross-space results)
# ===================================================================

def d_parent_lf_frac(delta, coarse, R_fn, Up_fn):
    """||Up(R(delta))|| / ||delta|| -- LF fraction."""
    R_delta = R_fn(delta)
    Up_R_delta = Up_fn(R_delta, delta.shape)
    numer = np.linalg.norm(Up_R_delta.ravel())
    denom = np.linalg.norm(delta.ravel()) + 1e-12
    return numer / denom


# ===================================================================
# Space data generators (from sc_baseline_crossspace)
# ===================================================================

def make_scalar_grid(N=64, seed=42):
    """T1: Scalar grid NxN with tile-mean coarse."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, N, endpoint=False)
    xx, yy = np.meshgrid(x, x)
    gt = (0.3 * np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy)
          + 0.8 * (xx > 0.45).astype(float)
          + 0.2 * np.sin(2 * np.pi * 12 * xx) * np.cos(2 * np.pi * 10 * yy)
          + rng.standard_normal((N, N)) * 0.01)
    T = 8; NT = N // T
    coarse = np.zeros_like(gt)
    for ti in range(NT):
        for tj in range(NT):
            s = slice(ti * T, (ti + 1) * T)
            cs = slice(tj * T, (tj + 1) * T)
            coarse[s, cs] = gt[s, cs].mean()
    return gt, coarse, {'type': 'scalar_grid', 'N': N, 'T': T}


def make_vector_grid(N=64, D=32, seed=42):
    """T2: Vector grid NxN dim=D."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, N, endpoint=False)
    xx, yy = np.meshgrid(x, x)
    gt = np.zeros((N, N, D))
    for d in range(D):
        freq = 2 + d * 0.5
        phase = rng.uniform(0, 2 * np.pi)
        amp = 0.5 / (1 + d * 0.1)
        gt[:, :, d] = amp * np.sin(2 * np.pi * freq * xx + phase) * np.cos(
            2 * np.pi * (freq * 0.7) * yy)
        if d % 3 == 0:
            gt[:, :, d] += 0.3 * (xx > rng.uniform(0.2, 0.8)).astype(float)
    gt += rng.standard_normal((N, N, D)) * 0.02
    T = 8; NT = N // T
    coarse = np.zeros_like(gt)
    for ti in range(NT):
        for tj in range(NT):
            s = slice(ti * T, (ti + 1) * T)
            cs = slice(tj * T, (tj + 1) * T)
            coarse[s, cs, :] = gt[s, cs, :].mean(axis=(0, 1), keepdims=True)
    return gt, coarse, {'type': 'vector_grid', 'N': N, 'D': D, 'T': T}


def make_graph(n_points=500, k=8, n_clusters=25, seed=42):
    """T3: Irregular graph with k-NN edges."""
    rng = np.random.default_rng(seed)
    positions = rng.random((n_points, 2))
    tree = cKDTree(positions)
    _, idx = tree.query(positions, k=k + 1)
    neighbors = {i: set(idx[i, 1:].tolist()) for i in range(n_points)}

    gt = (0.5 * np.sin(4 * np.pi * positions[:, 0])
          * np.cos(3 * np.pi * positions[:, 1])
          + 0.7 * (positions[:, 0] > 0.4).astype(float)
          + rng.standard_normal(n_points) * 0.03)

    # kmeans2 needs legacy rng
    _, labels = kmeans2(positions, n_clusters, minit='points', seed=seed)
    coarse = np.zeros(n_points)
    for c in range(n_clusters):
        mask = labels == c
        if mask.any():
            coarse[mask] = gt[mask].mean()

    info = {
        'type': 'graph',
        'n_points': n_points, 'k': k, 'n_clusters': n_clusters,
        'positions': positions, 'neighbors': neighbors, 'labels': labels,
    }
    return gt, coarse, info


def make_tree(depth=8, seed=42):
    """T4: Binary tree with depth levels."""
    rng = np.random.default_rng(seed)
    n_nodes = 2 ** depth - 1
    coarse_depth = max(3, depth // 2)

    gt = np.zeros(n_nodes)
    for i in range(n_nodes):
        d = int(np.log2(i + 1))
        gt[i] = 0.5 * np.sin(i * 0.3) / (1 + d * 0.2) + rng.standard_normal() * 0.05
        if d >= 3:
            gt[i] += 0.3 * ((i % 7) > 3)

    coarse = np.zeros(n_nodes)
    for i in range(n_nodes):
        d = int(np.log2(i + 1))
        if d < coarse_depth:
            coarse[i] = gt[i]
        else:
            ancestor = i
            while int(np.log2(ancestor + 1)) >= coarse_depth and ancestor > 0:
                ancestor = (ancestor - 1) // 2
            subtree = _subtree_nodes(ancestor, n_nodes)
            coarse[i] = np.mean([gt[j] for j in subtree])

    info = {
        'type': 'tree', 'depth': depth,
        'n_nodes': n_nodes, 'coarse_depth': coarse_depth,
    }
    return gt, coarse, info


def _subtree_nodes(root, n_nodes):
    nodes = []
    q = [root]
    while q:
        curr = q.pop()
        if curr >= n_nodes:
            continue
        nodes.append(curr)
        left = 2 * curr + 1
        right = 2 * curr + 2
        if left < n_nodes:
            q.append(left)
        if right < n_nodes:
            q.append(right)
    return nodes


# ===================================================================
# Data generation: produce D_parent records per space, seed, depth
# ===================================================================

# Depth level simulation for grids: vary tile size to get different "levels"
GRID_LEVEL_TILE = {1: 16, 2: 8, 3: 4}

# Depth level simulation for graphs: vary n_clusters
GRAPH_LEVEL_CLUSTERS = {1: 10, 2: 25, 3: 50}

# Depth level simulation for trees: vary coarse_depth
TREE_LEVEL_COARSE_DEPTH = {1: 2, 2: 4, 3: 6}


def _generate_scalar_grid_at_level(level, seed):
    """T1: Generate GT/coarse at a given depth level (tile size)."""
    N = 64
    T = GRID_LEVEL_TILE[level]
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, N, endpoint=False)
    xx, yy = np.meshgrid(x, x)
    gt = (0.3 * np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy)
          + 0.8 * (xx > 0.45).astype(float)
          + 0.2 * np.sin(2 * np.pi * 12 * xx) * np.cos(2 * np.pi * 10 * yy)
          + rng.standard_normal((N, N)) * 0.01)
    NT = N // T
    coarse = np.zeros_like(gt)
    for ti in range(NT):
        for tj in range(NT):
            s = slice(ti * T, (ti + 1) * T)
            cs = slice(tj * T, (tj + 1) * T)
            coarse[s, cs] = gt[s, cs].mean()
    info = {'type': 'scalar_grid', 'N': N, 'T': T}
    return gt, coarse, info


def _generate_vector_grid_at_level(level, seed):
    """T2: Generate GT/coarse at a given depth level."""
    N = 64; D = 32
    T = GRID_LEVEL_TILE[level]
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, N, endpoint=False)
    xx, yy = np.meshgrid(x, x)
    gt = np.zeros((N, N, D))
    for d in range(D):
        freq = 2 + d * 0.5
        phase = rng.uniform(0, 2 * np.pi)
        amp = 0.5 / (1 + d * 0.1)
        gt[:, :, d] = amp * np.sin(2 * np.pi * freq * xx + phase) * np.cos(
            2 * np.pi * (freq * 0.7) * yy)
        if d % 3 == 0:
            gt[:, :, d] += 0.3 * (xx > rng.uniform(0.2, 0.8)).astype(float)
    gt += rng.standard_normal((N, N, D)) * 0.02
    NT = N // T
    coarse = np.zeros_like(gt)
    for ti in range(NT):
        for tj in range(NT):
            s = slice(ti * T, (ti + 1) * T)
            cs = slice(tj * T, (tj + 1) * T)
            coarse[s, cs, :] = gt[s, cs, :].mean(axis=(0, 1), keepdims=True)
    info = {'type': 'vector_grid', 'N': N, 'D': D, 'T': T}
    return gt, coarse, info


def _generate_graph_at_level(level, seed):
    """T3: Generate graph at a given depth level (varying cluster count)."""
    n_points = 500; k = 8
    n_clusters = GRAPH_LEVEL_CLUSTERS[level]
    rng = np.random.default_rng(seed)
    positions = rng.random((n_points, 2))
    tree = cKDTree(positions)
    _, idx = tree.query(positions, k=k + 1)
    neighbors = {i: set(idx[i, 1:].tolist()) for i in range(n_points)}

    gt = (0.5 * np.sin(4 * np.pi * positions[:, 0])
          * np.cos(3 * np.pi * positions[:, 1])
          + 0.7 * (positions[:, 0] > 0.4).astype(float)
          + rng.standard_normal(n_points) * 0.03)

    _, labels = kmeans2(positions, n_clusters, minit='points', seed=seed)
    coarse = np.zeros(n_points)
    for c in range(n_clusters):
        mask = labels == c
        if mask.any():
            coarse[mask] = gt[mask].mean()

    info = {
        'type': 'graph', 'n_points': n_points, 'k': k,
        'n_clusters': n_clusters, 'positions': positions,
        'neighbors': neighbors, 'labels': labels,
    }
    return gt, coarse, info


def _generate_tree_at_level(level, seed):
    """T4: Generate tree at a given depth level (varying coarse_depth)."""
    depth = 8
    n_nodes = 2 ** depth - 1
    coarse_depth = TREE_LEVEL_COARSE_DEPTH[level]
    rng = np.random.default_rng(seed)

    gt = np.zeros(n_nodes)
    for i in range(n_nodes):
        d = int(np.log2(i + 1))
        gt[i] = 0.5 * np.sin(i * 0.3) / (1 + d * 0.2) + rng.standard_normal() * 0.05
        if d >= 3:
            gt[i] += 0.3 * ((i % 7) > 3)

    coarse = np.zeros(n_nodes)
    for i in range(n_nodes):
        d = int(np.log2(i + 1))
        if d < coarse_depth:
            coarse[i] = gt[i]
        else:
            ancestor = i
            while int(np.log2(ancestor + 1)) >= coarse_depth and ancestor > 0:
                ancestor = (ancestor - 1) // 2
            subtree = _subtree_nodes(ancestor, n_nodes)
            coarse[i] = np.mean([gt[j] for j in subtree])

    info = {
        'type': 'tree', 'depth': depth,
        'n_nodes': n_nodes, 'coarse_depth': coarse_depth,
    }
    return gt, coarse, info


# Registry: space_name -> (data_generator_fn, R/Up_maker_fn)
SPACE_GENERATORS = {
    'T1_scalar': _generate_scalar_grid_at_level,
    'T2_vector': _generate_vector_grid_at_level,
    'T3_graph': _generate_graph_at_level,
    'T4_tree': _generate_tree_at_level,
}

LEVELS = [1, 2, 3]


def _make_ops(space_name, info):
    """Create R_fn, Up_fn for a given space type and info dict."""
    stype = info['type']
    if stype == 'scalar_grid':
        return make_grid_ops(3.0)
    elif stype == 'vector_grid':
        return make_vector_ops(3.0)
    elif stype == 'graph':
        gops = GraphOps(info['labels'], info['n_clusters'], info['n_points'])
        return gops.restrict, gops.prolong
    elif stype == 'tree':
        tops = TreeOps(info['n_nodes'], info['coarse_depth'])
        return tops.restrict, tops.prolong
    else:
        raise ValueError(f"Unknown space type: {stype}")


def generate_records(space_name, seeds, base_seed=42):
    """Generate D_parent records for one space type across all levels and seeds.

    Returns:
        list of dicts with keys:
            space, level, case_type, pos_type/neg_type, seed, d_parent
    """
    gen_fn = SPACE_GENERATORS[space_name]
    records = []

    for level in LEVELS:
        for seed in seeds:
            gt, coarse, info = gen_fn(level, seed)
            R_fn, Up_fn = _make_ops(space_name, info)

            # Positive baselines
            for pos_type, delta in [
                ('oracle', positive_oracle(gt, coarse)),
                ('scaled', positive_scaled(gt, coarse, 0.5, seed)),
                ('noisy', positive_noisy(gt, coarse, 0.005, seed)),
            ]:
                dp = d_parent_lf_frac(delta, coarse, R_fn, Up_fn)
                records.append({
                    'space': space_name,
                    'level': level,
                    'case_type': 'pos',
                    'pos_type': pos_type,
                    'neg_type': None,
                    'seed': int(seed),
                    'd_parent': float(dp),
                })

            # Negative baselines
            neg_generators = [
                ('lf_drift', lambda s: negative_lf_drift(gt, coarse, 0.3, 0.5, s)),
                ('random_lf', lambda s: negative_random_lf(coarse, sigma=8.0,
                                                           amplitude=0.5, seed=s)),
                ('semant_wrong', lambda s: negative_semant_wrong(coarse, 1.0)),
                ('cs_coherent', lambda s: negative_coarse_shift_coherent(
                    gt, coarse, 0.2, s, info)),
                ('cs_block', lambda s: negative_coarse_shift_block(
                    gt, coarse, 0.2, s, info)),
                ('cs_gradient', lambda s: negative_coarse_shift_gradient(
                    gt, coarse, 0.2, s, info)),
            ]

            for neg_type, gen_fn_neg in neg_generators:
                delta = gen_fn_neg(seed)
                dp = d_parent_lf_frac(delta, coarse, R_fn, Up_fn)
                records.append({
                    'space': space_name,
                    'level': level,
                    'case_type': 'neg',
                    'pos_type': None,
                    'neg_type': neg_type,
                    'seed': int(seed),
                    'd_parent': float(dp),
                })

    return records


# ===================================================================
# Threshold finding methods
# ===================================================================

def find_optimal_threshold(pos_vals, neg_vals, method='youden_j'):
    """Find optimal threshold to separate positive (low D_parent) from
    negative (high D_parent) cases.

    D_parent convention: lower = scale-consistent, higher = violation.
    We classify as violation if d_parent >= threshold.

    Args:
        pos_vals: array of D_parent for positive (consistent) cases
        neg_vals: array of D_parent for negative (violation) cases
        method: one of 'youden_j', 'f1_optimal', 'sensitivity_at_90'

    Returns:
        dict with: threshold, sensitivity, specificity, f1, accuracy,
                   youden_j (for method context)
    """
    from sklearn.metrics import roc_curve

    pos = np.asarray(pos_vals)
    neg = np.asarray(neg_vals)

    if len(pos) < 2 or len(neg) < 2:
        return {
            'threshold': float('nan'),
            'sensitivity': float('nan'),
            'specificity': float('nan'),
            'f1': float('nan'),
            'accuracy': float('nan'),
            'youden_j': float('nan'),
            'method': method,
            'n_pos': len(pos),
            'n_neg': len(neg),
        }

    # Labels: neg cases are "positive class" (violations we want to detect)
    # higher D_parent = more likely violation
    y_true = np.concatenate([np.zeros(len(pos)), np.ones(len(neg))])
    y_score = np.concatenate([pos, neg])

    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    if method == 'youden_j':
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        threshold = float(thresholds[best_idx])

    elif method == 'f1_optimal':
        best_f1 = -1.0
        threshold = float(thresholds[0])
        for t in thresholds:
            preds = (y_score >= t).astype(int)
            tp = np.sum((preds == 1) & (y_true == 1))
            fp = np.sum((preds == 1) & (y_true == 0))
            fn = np.sum((preds == 0) & (y_true == 1))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)
            if f1 > best_f1:
                best_f1 = f1
                threshold = float(t)

    elif method == 'sensitivity_at_90':
        # Find lowest threshold achieving >= 90% sensitivity (TPR >= 0.9)
        mask = tpr >= 0.90
        if mask.any():
            # Among those achieving >=90% sensitivity, pick the one with
            # highest specificity (lowest FPR), i.e. highest threshold
            valid_indices = np.where(mask)[0]
            # roc_curve returns thresholds in decreasing order for
            # positive-class direction; pick the first valid (highest threshold)
            best_idx = valid_indices[0]
            threshold = float(thresholds[best_idx])
        else:
            # Cannot achieve 90% sensitivity; use best available
            best_idx = np.argmax(tpr)
            threshold = float(thresholds[best_idx])

    else:
        raise ValueError(f"Unknown method: {method}")

    # Compute metrics at the chosen threshold
    preds = (y_score >= threshold).astype(int)
    tp = np.sum((preds == 1) & (y_true == 1))
    fp = np.sum((preds == 1) & (y_true == 0))
    tn = np.sum((preds == 0) & (y_true == 0))
    fn = np.sum((preds == 0) & (y_true == 1))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = sensitivity
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return {
        'threshold': threshold,
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'precision': float(precision),
        'f1': float(f1),
        'accuracy': float(accuracy),
        'youden_j': float(sensitivity + specificity - 1),
        'method': method,
        'n_pos': len(pos),
        'n_neg': len(neg),
    }


# ===================================================================
# Depth-specific thresholds
# ===================================================================

def compute_depth_specific_thresholds(records, method='youden_j'):
    """Group D_parent values by depth level, find optimal threshold per level.

    Args:
        records: list of record dicts (from generate_records)
        method: threshold-finding method

    Returns:
        dict: {level: {threshold, sensitivity, specificity, ...}}
    """
    by_level = defaultdict(lambda: {'pos': [], 'neg': []})

    for r in records:
        key = r['level']
        by_level[key][r['case_type']].append(r['d_parent'])

    thresholds = {}
    for level in sorted(by_level.keys()):
        pos_vals = by_level[level]['pos']
        neg_vals = by_level[level]['neg']
        result = find_optimal_threshold(pos_vals, neg_vals, method)
        result['level'] = level
        thresholds[level] = result

    return thresholds


# ===================================================================
# Space-specificity check
# ===================================================================

def check_space_specificity(thresholds_by_space):
    """Compare tau_parent[L] across space types.

    If thresholds differ >2x between any two spaces for the same level,
    flag for architect review.

    Args:
        thresholds_by_space: dict of {space_name: {level: {threshold, ...}}}

    Returns:
        dict with:
            ratios: {level: {(space_a, space_b): ratio}}
            flag: bool (True if any ratio > 2.0)
            flagged_pairs: list of (level, space_a, space_b, ratio)
    """
    spaces = sorted(thresholds_by_space.keys())
    levels = set()
    for sp in spaces:
        levels.update(thresholds_by_space[sp].keys())
    levels = sorted(levels)

    ratios = {}
    flagged_pairs = []

    for level in levels:
        ratios[level] = {}
        for i, sp_a in enumerate(spaces):
            for sp_b in spaces[i + 1:]:
                tau_a = thresholds_by_space[sp_a].get(level, {}).get('threshold', np.nan)
                tau_b = thresholds_by_space[sp_b].get(level, {}).get('threshold', np.nan)

                if np.isnan(tau_a) or np.isnan(tau_b) or tau_a <= 0 or tau_b <= 0:
                    ratio = float('nan')
                else:
                    ratio = max(tau_a, tau_b) / min(tau_a, tau_b)

                ratios[level][(sp_a, sp_b)] = float(ratio)

                if not np.isnan(ratio) and ratio > 2.0:
                    flagged_pairs.append((level, sp_a, sp_b, float(ratio)))

    return {
        'ratios': {lvl: {f"{a}_vs_{b}": v for (a, b), v in pairs.items()}
                   for lvl, pairs in ratios.items()},
        'flag': len(flagged_pairs) > 0,
        'flagged_pairs': flagged_pairs,
    }


# ===================================================================
# Leave-one-space-out cross-validation
# ===================================================================

def leave_one_space_out_cv(n_train_seeds=10, n_val_seeds=10, base_seed=42,
                           method='youden_j', verbose=True):
    """4-fold cross-validation: train on 3 spaces, validate on held-out.

    For each fold:
      - Generate training data from 3 spaces x n_train_seeds
      - Find thresholds per level on training data
      - Generate validation data from held-out space x n_val_seeds
      - Evaluate accuracy on held-out
      - Check kill criterion: accuracy drop > 15pp

    Args:
        n_train_seeds: number of random seeds for training data
        n_val_seeds: number of random seeds for validation data
        base_seed: base random seed
        method: threshold-finding method
        verbose: print progress

    Returns:
        dict with per-fold results, training accuracy, validation accuracy,
        and kill criterion assessment.
    """
    spaces = ['T1_scalar', 'T2_vector', 'T3_graph', 'T4_tree']
    train_seeds = list(range(base_seed, base_seed + n_train_seeds))
    val_seeds = list(range(base_seed + 1000, base_seed + 1000 + n_val_seeds))

    if verbose:
        print("\n" + "=" * 70)
        print("Leave-One-Space-Out Cross-Validation")
        print(f"Method: {method}  |  Train seeds: {n_train_seeds}  |  "
              f"Val seeds: {n_val_seeds}")
        print("=" * 70)

    fold_results = {}

    for held_out in spaces:
        train_spaces = [s for s in spaces if s != held_out]

        if verbose:
            print(f"\n  Fold: held-out = {held_out}")
            print(f"  Training on: {train_spaces}")

        # Generate training data
        train_records = []
        for sp in train_spaces:
            if verbose:
                print(f"    Generating training data for {sp}...", end=" ",
                      flush=True)
            recs = generate_records(sp, train_seeds, base_seed=base_seed)
            train_records.extend(recs)
            if verbose:
                print(f"{len(recs)} records")

        # Find thresholds on training data
        train_thresholds = compute_depth_specific_thresholds(
            train_records, method=method)

        # Training accuracy (per level)
        train_accuracies = {}
        for level, tinfo in train_thresholds.items():
            train_accuracies[level] = tinfo['accuracy']

        # Generate validation data
        if verbose:
            print(f"    Generating validation data for {held_out}...", end=" ",
                  flush=True)
        val_records = generate_records(held_out, val_seeds,
                                       base_seed=base_seed + 500)
        if verbose:
            print(f"{len(val_records)} records")

        # Evaluate on validation data using training thresholds
        val_accuracies = {}
        val_details = {}
        for level in sorted(train_thresholds.keys()):
            tau = train_thresholds[level]['threshold']
            level_recs = [r for r in val_records if r['level'] == level]
            if not level_recs or np.isnan(tau):
                val_accuracies[level] = float('nan')
                val_details[level] = {
                    'n': 0, 'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
                continue

            tp = fp = tn = fn = 0
            for r in level_recs:
                is_violation = r['case_type'] == 'neg'
                predicted_violation = r['d_parent'] >= tau
                if is_violation and predicted_violation:
                    tp += 1
                elif is_violation and not predicted_violation:
                    fn += 1
                elif not is_violation and predicted_violation:
                    fp += 1
                else:
                    tn += 1

            total = tp + fp + tn + fn
            acc = (tp + tn) / total if total > 0 else 0.0
            val_accuracies[level] = float(acc)
            val_details[level] = {
                'n': total, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
                'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            }

        # Check kill criterion per level
        kill_flags = {}
        for level in sorted(train_thresholds.keys()):
            train_acc = train_accuracies.get(level, 0.0)
            val_acc = val_accuracies.get(level, 0.0)
            drop = train_acc - val_acc
            kill_flags[level] = {
                'train_acc': float(train_acc),
                'val_acc': float(val_acc),
                'drop_pp': float(drop * 100),
                'kill': float(drop * 100) > 15.0,
            }

        fold_kill = any(kf['kill'] for kf in kill_flags.values()
                        if not np.isnan(kf['drop_pp']))

        fold_results[held_out] = {
            'train_thresholds': {
                int(k): {kk: vv for kk, vv in v.items()}
                for k, v in train_thresholds.items()
            },
            'train_accuracies': {int(k): float(v)
                                 for k, v in train_accuracies.items()},
            'val_accuracies': {int(k): float(v)
                               for k, v in val_accuracies.items()},
            'val_details': {int(k): v for k, v in val_details.items()},
            'kill_flags': {int(k): v for k, v in kill_flags.items()},
            'fold_kill': fold_kill,
        }

        if verbose:
            for level in sorted(train_thresholds.keys()):
                kf = kill_flags[level]
                status = "KILL" if kf['kill'] else "OK"
                print(f"    L={level}: tau={train_thresholds[level]['threshold']:.4f}  "
                      f"train_acc={kf['train_acc']:.3f}  "
                      f"val_acc={kf['val_acc']:.3f}  "
                      f"drop={kf['drop_pp']:.1f}pp  [{status}]")

    # Overall kill assessment
    any_kill = any(fr['fold_kill'] for fr in fold_results.values())

    if verbose:
        print(f"\n  Overall kill: {'YES — DO NOT deploy thresholds' if any_kill else 'NO — thresholds generalize'}")

    return {
        'folds': fold_results,
        'any_kill': any_kill,
        'method': method,
        'n_train_seeds': n_train_seeds,
        'n_val_seeds': n_val_seeds,
    }


# ===================================================================
# Holm-Bonferroni correction for multiple comparisons
# ===================================================================

def holm_bonferroni(p_values, alpha=0.05):
    """Apply Holm-Bonferroni correction to a list of p-values.

    Args:
        p_values: list of (label, p_value) tuples
        alpha: family-wise error rate

    Returns:
        list of (label, p_value, adjusted_alpha, reject) tuples
    """
    sorted_pvs = sorted(p_values, key=lambda x: x[1])
    m = len(sorted_pvs)
    results = []
    for i, (label, pv) in enumerate(sorted_pvs):
        adjusted_alpha = alpha / (m - i)
        reject = pv < adjusted_alpha
        results.append((label, float(pv), float(adjusted_alpha), reject))
    return results


# ===================================================================
# ROC curve plotting
# ===================================================================

def plot_roc_curves(all_records_by_space, output_path, method='youden_j'):
    """Plot ROC curves per level, overlay all spaces.

    One subplot per level. Each space type as a separate curve.

    Args:
        all_records_by_space: dict of {space: list_of_records}
        output_path: Path for the output PNG
        method: method used for threshold annotation
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    levels = sorted(LEVELS)
    spaces = sorted(all_records_by_space.keys())
    colors = {'T1_scalar': '#1f77b4', 'T2_vector': '#ff7f0e',
              'T3_graph': '#2ca02c', 'T4_tree': '#d62728'}

    fig, axes = plt.subplots(1, len(levels), figsize=(5 * len(levels), 5))
    if len(levels) == 1:
        axes = [axes]

    for ax, level in zip(axes, levels):
        ax.set_title(f"Level L={level}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Chance')

        for sp in spaces:
            records = all_records_by_space[sp]
            level_recs = [r for r in records if r['level'] == level]
            pos_vals = [r['d_parent'] for r in level_recs
                        if r['case_type'] == 'pos']
            neg_vals = [r['d_parent'] for r in level_recs
                        if r['case_type'] == 'neg']

            if len(pos_vals) < 2 or len(neg_vals) < 2:
                continue

            y_true = np.concatenate([np.zeros(len(pos_vals)),
                                     np.ones(len(neg_vals))])
            y_score = np.concatenate([pos_vals, neg_vals])

            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)

            ax.plot(fpr, tpr, color=colors.get(sp, 'gray'),
                    label=f"{sp} (AUC={roc_auc:.3f})", linewidth=1.5)

        ax.legend(loc='lower right', fontsize=8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.2)

    fig.suptitle(f"exp12a: ROC curves for D_parent by depth level (method={method})",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ===================================================================
# Main experiment
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="exp12a_tau_parent: Data-driven thresholds per depth level")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--n-train-seeds", type=int, default=10,
                        help="Number of training seeds per fold")
    parser.add_argument("--n-val-seeds", type=int, default=10,
                        help="Number of validation seeds per fold")
    parser.add_argument("--base-seed", type=int, default=42,
                        help="Base random seed")
    args = parser.parse_args()

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Curiosity -- exp12a_tau_parent")
    print("Data-driven thresholds tau_parent[L] per depth level")
    print(f"Train seeds: {args.n_train_seeds}  |  Val seeds: {args.n_val_seeds}  |  "
          f"Base seed: {args.base_seed}")
    print(f"Output: {out_dir}")
    print("=" * 70)

    spaces = ['T1_scalar', 'T2_vector', 'T3_graph', 'T4_tree']
    methods = ['youden_j', 'f1_optimal', 'sensitivity_at_90']
    all_seeds = list(range(args.base_seed, args.base_seed + args.n_train_seeds))

    # ---------------------------------------------------------------
    # Step 1: Generate D_parent records for all spaces and levels
    # ---------------------------------------------------------------
    print("\n[Step 1] Generating D_parent records for all spaces...")
    all_records_by_space = {}
    all_records = []
    for sp in spaces:
        print(f"  {sp}...", end=" ", flush=True)
        recs = generate_records(sp, all_seeds, base_seed=args.base_seed)
        all_records_by_space[sp] = recs
        all_records.extend(recs)
        n_pos = sum(1 for r in recs if r['case_type'] == 'pos')
        n_neg = sum(1 for r in recs if r['case_type'] == 'neg')
        print(f"{len(recs)} records ({n_pos} pos, {n_neg} neg)")

    # ---------------------------------------------------------------
    # Step 2: Compute thresholds per method, per level, per space
    # ---------------------------------------------------------------
    print("\n[Step 2] Computing thresholds per method, level, space...")
    full_results = {}

    for method in methods:
        print(f"\n  Method: {method}")
        full_results[method] = {}

        # Global thresholds (all spaces pooled)
        global_thresholds = compute_depth_specific_thresholds(all_records, method)
        full_results[method]['global'] = {
            int(k): v for k, v in global_thresholds.items()
        }

        for level, tinfo in sorted(global_thresholds.items()):
            print(f"    Global L={level}: tau={tinfo['threshold']:.4f}  "
                  f"acc={tinfo['accuracy']:.3f}  "
                  f"sens={tinfo['sensitivity']:.3f}  "
                  f"spec={tinfo['specificity']:.3f}  "
                  f"F1={tinfo['f1']:.3f}")

        # Per-space thresholds
        thresholds_by_space = {}
        for sp in spaces:
            sp_thresholds = compute_depth_specific_thresholds(
                all_records_by_space[sp], method)
            thresholds_by_space[sp] = sp_thresholds
            full_results[method][sp] = {
                int(k): v for k, v in sp_thresholds.items()
            }

        # Space-specificity check
        specificity_check = check_space_specificity(thresholds_by_space)
        full_results[method]['space_specificity'] = specificity_check

        if specificity_check['flag']:
            print(f"\n    WARNING: Space-specific threshold variation > 2x detected!")
            for lvl, sp_a, sp_b, ratio in specificity_check['flagged_pairs']:
                print(f"      L={lvl}: {sp_a} vs {sp_b}  ratio={ratio:.2f}")
        else:
            print(f"    Space-specificity check: OK (all ratios <= 2x)")

    # ---------------------------------------------------------------
    # Step 3: Leave-one-space-out cross-validation (best method)
    # ---------------------------------------------------------------
    # Run CV for all methods
    cv_results = {}
    for method in methods:
        cv = leave_one_space_out_cv(
            n_train_seeds=args.n_train_seeds,
            n_val_seeds=args.n_val_seeds,
            base_seed=args.base_seed,
            method=method,
            verbose=True,
        )
        cv_results[method] = cv

    # ---------------------------------------------------------------
    # Step 4: Statistical significance with Holm-Bonferroni
    # ---------------------------------------------------------------
    print("\n[Step 4] Statistical significance tests...")
    from scipy.stats import mannwhitneyu

    significance_results = {}
    for level in LEVELS:
        p_values = []
        for sp in spaces:
            recs = all_records_by_space[sp]
            level_recs = [r for r in recs if r['level'] == level]
            pos = [r['d_parent'] for r in level_recs if r['case_type'] == 'pos']
            neg = [r['d_parent'] for r in level_recs if r['case_type'] == 'neg']
            if len(pos) >= 2 and len(neg) >= 2:
                stat, pv = mannwhitneyu(neg, pos, alternative='greater')
                p_values.append((f"L{level}_{sp}", float(pv)))

        corrected = holm_bonferroni(p_values, alpha=0.05)
        significance_results[level] = corrected
        print(f"  Level L={level}:")
        for label, pv, adj_alpha, reject in corrected:
            status = "REJECT H0" if reject else "fail to reject"
            print(f"    {label}: p={pv:.2e}  adj_alpha={adj_alpha:.4f}  [{status}]")

    # ---------------------------------------------------------------
    # Step 5: Select best method and produce recommended thresholds
    # ---------------------------------------------------------------
    print("\n[Step 5] Selecting best method...")

    # Best = method with highest mean cross-validation accuracy and no kills
    best_method = None
    best_mean_acc = -1.0
    for method in methods:
        cv = cv_results[method]
        if cv['any_kill']:
            continue
        accs = []
        for fold_name, fold_data in cv['folds'].items():
            for lvl, acc in fold_data['val_accuracies'].items():
                if not np.isnan(acc):
                    accs.append(acc)
        mean_acc = np.mean(accs) if accs else 0.0
        if mean_acc > best_mean_acc:
            best_mean_acc = mean_acc
            best_method = method

    if best_method is None:
        # All methods have kills; pick the one with fewest kills
        min_kills = float('inf')
        for method in methods:
            cv = cv_results[method]
            n_kills = sum(1 for fd in cv['folds'].values() if fd['fold_kill'])
            if n_kills < min_kills:
                min_kills = n_kills
                best_method = method
        print(f"  WARNING: All methods have kill-criterion violations!")
        print(f"  Selecting {best_method} with fewest kills ({min_kills})")
    else:
        print(f"  Best method: {best_method} (mean CV accuracy: {best_mean_acc:.4f})")

    # Recommended thresholds: global thresholds from the best method
    recommended = {}
    global_best = full_results[best_method]['global']
    for level in LEVELS:
        if level in global_best:
            recommended[level] = {
                'tau_parent': global_best[level]['threshold'],
                'method': best_method,
                'accuracy': global_best[level]['accuracy'],
                'sensitivity': global_best[level]['sensitivity'],
                'specificity': global_best[level]['specificity'],
                'f1': global_best[level]['f1'],
            }

    print("\n  Recommended thresholds:")
    for level in sorted(recommended.keys()):
        r = recommended[level]
        print(f"    L={level}: tau_parent = {r['tau_parent']:.4f}  "
              f"(acc={r['accuracy']:.3f}  F1={r['f1']:.3f})")

    # ---------------------------------------------------------------
    # Step 6: Save results
    # ---------------------------------------------------------------
    print("\n[Step 6] Saving results...")

    # Full results
    full_output = {
        'thresholds_by_method': full_results,
        'cross_validation': cv_results,
        'significance': {
            int(k): [(label, pv, adj, rej)
                     for label, pv, adj, rej in v]
            for k, v in significance_results.items()
        },
        'best_method': best_method,
        'recommended': {int(k): v for k, v in recommended.items()},
        'params': {
            'n_train_seeds': args.n_train_seeds,
            'n_val_seeds': args.n_val_seeds,
            'base_seed': args.base_seed,
            'levels': LEVELS,
            'spaces': spaces,
            'methods': methods,
            'R_operator': 'gaussian_blur_sigma3.0 + decimation (grids); '
                          'cluster-mean (graph); subtree-mean (tree)',
            'd_parent_formula': 'lf_frac = ||Up(R(delta))|| / ||delta||',
        },
    }

    full_path = out_dir / "exp12a_results.json"
    with open(full_path, "w") as f:
        json.dump(full_output, f, indent=2, default=_json_default)
    print(f"  [Saved] {full_path}")

    # Recommended thresholds (compact)
    thresh_output = {
        'method': best_method,
        'thresholds': {f"L{k}": v['tau_parent']
                       for k, v in recommended.items()},
        'metrics': {f"L{k}": {kk: vv for kk, vv in v.items()
                               if kk != 'tau_parent'}
                    for k, v in recommended.items()},
        'kill_criterion_passed': not cv_results[best_method]['any_kill'],
    }

    thresh_path = out_dir / "exp12a_thresholds.json"
    with open(thresh_path, "w") as f:
        json.dump(thresh_output, f, indent=2, default=_json_default)
    print(f"  [Saved] {thresh_path}")

    # ROC curves
    try:
        roc_path = out_dir / "exp12a_roc_curves.png"
        plot_roc_curves(all_records_by_space, roc_path, method=best_method)
        print(f"  [Saved] {roc_path}")
    except ImportError:
        print("  [Skip] matplotlib not available; no ROC plot generated.")

    print("\n[Done]")
    return full_output


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
