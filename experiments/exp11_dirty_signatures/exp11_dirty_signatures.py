#!/usr/bin/env python3
"""
exp11 — Dirty Signatures: 12-bit packed signature + debounce tracker
for structural change detection across 4 space types.

Question: 12-bit dirty signature + debounce -> AUC > 0.8?
Kill criteria: AUC < 0.8 on any space type = FAIL.

Signature layout (12 bits total):
  bits [11:8]  seam_risk  — quantized SeamScore (4 bits, 0-15)
  bits  [7:4]  uncert     — quantized instability  (4 bits, 0-15)
  bits  [3:0]  mass       — quantized D_parent / lf_frac (4 bits, 0-15)

DebounceTracker fires after 2 consecutive threshold crossings
(Hamming distance >= 3 bits OR any single component change >= 4 levels).

Three test scenarios per (space_type x seed):
  1. Noise    — Gaussian perturbation, should NOT trigger (negative label)
  2. Structural — step function / discontinuity insert, MUST trigger (positive label)
  3. Drift    — gradual systematic shift, triggers with latency (positive label)

Evaluation: ROC / AUC, Holm-Bonferroni across 4 space types, 10 seeds each.

Outputs:
  exp11_results.json     — AUC per space, overall verdict
  exp11_roc_curves.png   — ROC curve per space type
  exp11_blast_radius.png — blast radius distribution
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Imports from project ────────────────────────────────────────────
# We import the space adapter classes and metric utilities from existing
# experiment modules.  The adapters provide a uniform interface for
# setup / get_units / refine_unit / seam_score / unit_error.

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from p2a_sensitivity.exp_p2a_sweep import (
    ScalarGridAdapter, VectorGridAdapter,
    IrregularGraphAdapter, TreeAdapter,
    run_probe_diagnostics, robust_jump,
)
from sc_baseline.metrics import d_parent, compute_beta
from sc_baseline.operators import (
    restrict_scalar, prolong_scalar,
    restrict_vector, prolong_vector,
    restrict_graph, prolong_graph,
    restrict_tree, prolong_tree,
)

EPS = 1e-10
N_SEEDS = 10
N_STEPS = 20          # simulation steps per scenario
HALO = 2              # halo size for grid adapters
HALO_HOPS = 1         # halo hops for graph / tree adapters


# ═══════════════════════════════════════════════════════════════════════
# Space factories & scene generator
# ═══════════════════════════════════════════════════════════════════════

SPACE_CONFIGS = {
    "scalar_grid":      lambda: ScalarGridAdapter(N=64, tile=8),
    "vector_grid":      lambda: VectorGridAdapter(N=32, tile=8, D=16),
    "irregular_graph":  lambda: IrregularGraphAdapter(n_points=200, k=6, n_clusters=10),
    "tree_hierarchy":   lambda: TreeAdapter(depth=6),
}


def make_scene(N, seed):
    """Generate a 2D ground-truth field used to initialise every space adapter."""
    rng = np.random.RandomState(seed)
    x = np.linspace(0, 1, N, endpoint=False)
    xx, yy = np.meshgrid(x, x)
    f = np.zeros((N, N))
    for _ in range(5):
        cx, cy = rng.uniform(0.1, 0.9, 2)
        sigma = rng.uniform(0.05, 0.2)
        f += rng.uniform(0.3, 1.0) * np.exp(
            -((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
    f += 0.5 * (xx > rng.uniform(0.3, 0.7))
    return f


# ═══════════════════════════════════════════════════════════════════════
# DirtySignature  (12-bit packed)
# ═══════════════════════════════════════════════════════════════════════

class DirtySignature:
    """Pack three 4-bit components into a 12-bit integer.

    Layout:
      bits [11:8]  seam_risk   (0-15)
      bits  [7:4]  uncert      (0-15)
      bits  [3:0]  mass        (0-15)
    """

    # Saturation ceilings for quantisation — values above these are
    # clipped to 15.
    SEAM_MAX  = 5.0     # SeamScore above 5 is very bad
    UNCERT_MAX = 1.0    # instability in [0, 1]
    MASS_MAX   = 1.0    # D_parent (lf_frac) in [0, 1]

    @staticmethod
    def _quantize(value, vmax):
        """Map a float in [0, vmax] to an int in [0, 15]."""
        clamped = max(0.0, min(float(value), vmax))
        return int(round(clamped / vmax * 15))

    @staticmethod
    def _dequantize(level, vmax):
        return level / 15.0 * vmax

    @classmethod
    def pack(cls, seam_risk, uncert, mass):
        """Pack three floats into a 12-bit int."""
        s = cls._quantize(seam_risk, cls.SEAM_MAX)
        u = cls._quantize(uncert, cls.UNCERT_MAX)
        m = cls._quantize(mass, cls.MASS_MAX)
        return (s << 8) | (u << 4) | m

    @staticmethod
    def unpack(sig):
        """Unpack a 12-bit int into (seam_risk_level, uncert_level, mass_level)."""
        s = (sig >> 8) & 0xF
        u = (sig >> 4) & 0xF
        m = sig & 0xF
        return s, u, m

    @classmethod
    def compute_signature(cls, space, state, unit, restrict_fn, prolong_fn,
                          coarse, rng_seed):
        """Compute all 3 components for *unit* and pack.

        Parameters
        ----------
        space : adapter object with .seam_score(), .unit_error(), etc.
        state : current field state (ndarray)
        unit  : unit identifier (tuple or int depending on space type)
        restrict_fn, prolong_fn : R / Up operators for this space type
        coarse : coarse-level reference state
        rng_seed : int — seed for instability jitter

        Returns
        -------
        int : 12-bit packed signature
        """
        # 1. seam_risk — SeamScore for this unit
        seam_risk = space.seam_score(state, unit)

        # 2. uncert — instability: jitter state, measure rank-flip fraction
        rng = np.random.RandomState(rng_seed)
        units = space.get_units()
        n_units = len(units)
        errors_orig = np.array([space.unit_error(state, u) for u in units])
        k_top = max(1, int(0.3 * n_units))
        top_orig = set(np.argsort(errors_orig)[-k_top:])

        if isinstance(state, np.ndarray):
            scale = (state.max() - state.min()) + 1e-10
            jittered = state + rng.randn(*state.shape) * 0.005 * scale
        else:
            jittered = state  # fallback
        errors_jit = np.array([space.unit_error(jittered, u) for u in units])
        top_jit = set(np.argsort(errors_jit)[-k_top:])
        uncert = len(top_orig.symmetric_difference(top_jit)) / (2 * k_top + EPS)

        # 3. mass — D_parent (lf_frac) via restrict/prolong operators
        #    We compute d_parent of the per-unit delta.
        state_after = space.refine_unit(state, unit)
        delta = state_after - state
        if delta.ndim >= 1 and np.linalg.norm(delta.ravel()) > 1e-12:
            dp = d_parent(delta, coarse, restrict_fn)
            mass = min(dp, cls.MASS_MAX)
        else:
            mass = 0.0

        return cls.pack(seam_risk, uncert, mass)


# ═══════════════════════════════════════════════════════════════════════
# DebounceTracker
# ═══════════════════════════════════════════════════════════════════════

class DebounceTracker:
    """Tracks per-unit signature history; fires after 2 consecutive
    threshold crossings.

    A crossing is detected when:
      - Hamming distance (in bits) between old and new signature >= 3, OR
      - Any single component changes by >= 4 quantised levels.
    """

    def __init__(self, hamming_threshold=3, component_threshold=4,
                 consecutive_needed=2):
        self.hamming_threshold = hamming_threshold
        self.component_threshold = component_threshold
        self.consecutive_needed = consecutive_needed

        # Per-unit state
        self._prev_sig = {}       # unit_id -> last signature
        self._consec_count = {}   # unit_id -> consecutive crossing count
        self._trigger_log = {}    # unit_id -> list of (step, sig) when triggered

    def reset(self):
        self._prev_sig.clear()
        self._consec_count.clear()
        self._trigger_log.clear()

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _hamming12(a, b):
        """Hamming distance between two 12-bit ints."""
        xor = a ^ b
        return bin(xor).count("1")

    @staticmethod
    def _component_diff(a, b):
        """Max absolute difference among unpacked components."""
        sa, ua, ma = DirtySignature.unpack(a)
        sb, ub, mb = DirtySignature.unpack(b)
        return max(abs(sa - sb), abs(ua - ub), abs(ma - mb))

    def _is_crossing(self, old_sig, new_sig):
        if self._hamming12(old_sig, new_sig) >= self.hamming_threshold:
            return True
        if self._component_diff(old_sig, new_sig) >= self.component_threshold:
            return True
        return False

    # ── public API ───────────────────────────────────────────────────

    def update(self, unit_id, new_sig, step=None):
        """Record a new signature for *unit_id*.

        Returns True if a trigger fires on this update.
        """
        if unit_id not in self._prev_sig:
            # First observation — no comparison possible
            self._prev_sig[unit_id] = new_sig
            self._consec_count[unit_id] = 0
            return False

        old_sig = self._prev_sig[unit_id]
        if self._is_crossing(old_sig, new_sig):
            self._consec_count[unit_id] = self._consec_count.get(unit_id, 0) + 1
        else:
            self._consec_count[unit_id] = 0

        self._prev_sig[unit_id] = new_sig

        if self._consec_count[unit_id] >= self.consecutive_needed:
            self._trigger_log.setdefault(unit_id, []).append(
                (step, new_sig))
            return True
        return False

    def should_trigger(self, unit_id):
        """Check (without updating) whether the unit would fire."""
        return self._consec_count.get(unit_id, 0) >= self.consecutive_needed

    def get_triggers(self, unit_id):
        return self._trigger_log.get(unit_id, [])


# ═══════════════════════════════════════════════════════════════════════
# Per-space-type operator helpers
# ═══════════════════════════════════════════════════════════════════════

def _make_operators(space_name, space):
    """Return (restrict_fn, prolong_fn, coarse_ref) appropriate for the
    space type, where each function works on the full state array."""

    if space_name == "scalar_grid":
        return restrict_scalar, prolong_scalar, space.coarse

    elif space_name == "vector_grid":
        return restrict_vector, prolong_vector, space.coarse

    elif space_name == "irregular_graph":
        labels = space.labels
        n_clusters = space.n_clusters
        n_pts = space.n_pts

        def r_fn(x):
            return restrict_graph(x, labels, n_clusters)

        def p_fn(x_c, target_shape):
            return prolong_graph(x_c, labels, n_pts)

        return r_fn, p_fn, space.coarse

    elif space_name == "tree_hierarchy":
        n_nodes = space.n
        coarse_depth = space.coarse_depth

        def r_fn(x):
            return restrict_tree(x, n_nodes, coarse_depth)

        def p_fn(x_c, target_shape):
            return prolong_tree(x_c, n_nodes, coarse_depth)

        return r_fn, p_fn, space.coarse

    else:
        raise ValueError(f"Unknown space: {space_name}")


# ═══════════════════════════════════════════════════════════════════════
# Scenario generators
# ═══════════════════════════════════════════════════════════════════════

def apply_noise(state, rng, step, noise_std=0.02):
    """Transient Gaussian noise (scenario 1: negative label)."""
    return state + rng.randn(*state.shape) * noise_std


def apply_structural(state, rng, step, space_name, event_step=5):
    """Insert a persistent structural discontinuity at *event_step*
    (scenario 2: positive label after event_step)."""
    if step < event_step:
        return state.copy()
    out = state.copy()
    if state.ndim == 2:
        H, W = state.shape
        mid = H // 2
        out[mid:, :] += 0.5  # step function across midpoint
    elif state.ndim == 3:
        H, W, D = state.shape
        mid = H // 2
        out[mid:, :, :] += 0.5
    elif state.ndim == 1:
        mid = len(state) // 2
        out[mid:] += 0.5
    return out


def apply_drift(state, rng, step, drift_rate=0.015):
    """Gradual systematic shift (scenario 3: positive label,
    triggers with latency)."""
    out = state.copy()
    if state.ndim == 2:
        H, W = state.shape
        out[:H // 2, :] += drift_rate * step
    elif state.ndim == 3:
        H, W, D = state.shape
        out[:H // 2, :, :] += drift_rate * step
    elif state.ndim == 1:
        mid = len(state) // 2
        out[:mid] += drift_rate * step
    return out


# ═══════════════════════════════════════════════════════════════════════
# Single-space trial
# ═══════════════════════════════════════════════════════════════════════

def run_trial(space_name, seed):
    """Run all 3 scenarios for one (space_name, seed) pair.

    Returns
    -------
    dict with keys:
      labels        : list[int]    — 0 = negative (noise), 1 = positive (struct/drift)
      scores        : list[float]  — continuous score (max signature distance from baseline)
      triggers      : list[bool]   — did debounce trigger?
      blast_radii   : list[int]    — how many units triggered per scenario
      latencies     : list[float]  — steps from event to first trigger (or NaN)
      trigger_steps : list[list]   — per-unit trigger step lists (for burstiness)
    """
    space = SPACE_CONFIGS[space_name]()
    scene = make_scene(64, seed)
    space.setup(scene)

    if hasattr(space, "coarse"):
        base_state = space.coarse.copy()
    else:
        base_state = np.zeros_like(space.gt)

    units = space.get_units()
    restrict_fn, prolong_fn, coarse_ref = _make_operators(space_name, space)

    results = {
        "labels": [],
        "scores": [],
        "triggers": [],
        "blast_radii": [],
        "latencies": [],
        "trigger_steps": [],
    }

    scenarios = [
        ("noise",      0, None),    # label=0, no event step
        ("structural", 1, 5),       # label=1, event at step 5
        ("drift",      1, None),    # label=1, gradual
    ]

    # Compute baseline signatures (from unperturbed state) using a fixed
    # rng_seed so every scenario shares the same reference point.
    baseline_sigs = {}
    for uid in units:
        baseline_sigs[uid] = DirtySignature.compute_signature(
            space, base_state, uid,
            restrict_fn, prolong_fn, coarse_ref,
            rng_seed=seed * 1000)

    for scenario_name, label, event_step in scenarios:
        # Fresh RNG per scenario so results are independent and reproducible.
        # Use a deterministic offset per scenario (not hash, which is randomised
        # across Python processes).
        scenario_offset = {"noise": 0, "structural": 1, "drift": 2}[scenario_name]
        rng_legacy = np.random.RandomState(seed * 10 + scenario_offset)

        tracker = DebounceTracker()
        state = base_state.copy()

        per_unit_first_trigger = {}   # unit_id -> step
        all_trigger_steps = []

        # Track max signature distance from baseline per unit
        max_baseline_dist = {}        # uid -> max hamming distance from baseline

        for step in range(N_STEPS):
            # Apply scenario perturbation
            if scenario_name == "noise":
                state_perturbed = apply_noise(state, rng_legacy, step)
            elif scenario_name == "structural":
                state_perturbed = apply_structural(
                    state, rng_legacy, step, space_name, event_step=event_step)
            else:  # drift
                state_perturbed = apply_drift(state, rng_legacy, step)

            # Compute signature for each unit.
            # Use a FIXED rng_seed (not step-dependent) so that the uncert
            # jitter is deterministic given the same state — this prevents
            # the jitter itself from creating spurious signature changes.
            for uid in units:
                sig = DirtySignature.compute_signature(
                    space, state_perturbed, uid,
                    restrict_fn, prolong_fn, coarse_ref,
                    rng_seed=seed * 1000)
                fired = tracker.update(uid, sig, step=step)
                if fired and uid not in per_unit_first_trigger:
                    per_unit_first_trigger[uid] = step
                    all_trigger_steps.append(step)

                # Track distance from baseline signature
                bdist = DebounceTracker._hamming12(baseline_sigs[uid], sig)
                cdist = DebounceTracker._component_diff(baseline_sigs[uid], sig)
                dist = bdist + cdist   # combined distance metric
                prev = max_baseline_dist.get(uid, 0)
                if dist > prev:
                    max_baseline_dist[uid] = dist

        # Aggregate
        any_trigger = len(per_unit_first_trigger) > 0
        blast = len(per_unit_first_trigger)

        # Score: use the maximum baseline distance across all units as the
        # primary decision variable.  Structural events cause large persistent
        # deviations from baseline; noise causes small transient deviations.
        # Also include debounce trigger count as a secondary signal.
        max_dist = max(max_baseline_dist.values(), default=0)
        # Mean distance across all units (captures breadth of change)
        mean_dist = (sum(max_baseline_dist.values()) / max(len(max_baseline_dist), 1))
        score = float(max_dist) + 0.5 * mean_dist + 0.1 * blast

        if scenario_name == "structural" and event_step is not None:
            if per_unit_first_trigger:
                latency = min(per_unit_first_trigger.values()) - event_step
            else:
                latency = float("nan")
        elif scenario_name == "drift":
            if per_unit_first_trigger:
                latency = float(min(per_unit_first_trigger.values()))
            else:
                latency = float("nan")
        else:
            latency = float("nan")

        results["labels"].append(label)
        results["scores"].append(score)
        results["triggers"].append(any_trigger)
        results["blast_radii"].append(blast)
        results["latencies"].append(latency)
        results["trigger_steps"].append(all_trigger_steps)

    return results


# ═══════════════════════════════════════════════════════════════════════
# ROC / AUC computation
# ═══════════════════════════════════════════════════════════════════════

def compute_roc_auc(labels, scores):
    """Compute ROC curve and AUC from labels and continuous scores.

    Returns (fpr, tpr, thresholds, auc).
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)

    # Sort by decreasing score
    desc = np.argsort(-scores)
    labels_sorted = labels[desc]
    scores_sorted = scores[desc]

    # Distinct thresholds
    thresholds = np.unique(scores_sorted)[::-1]

    tpr_list = [0.0]
    fpr_list = [0.0]

    n_pos = labels.sum()
    n_neg = len(labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        return np.array([0, 1]), np.array([0, 1]), np.array([0.0]), 0.5

    for thr in thresholds:
        predicted_pos = scores >= thr
        tp = (predicted_pos & (labels == 1)).sum()
        fp = (predicted_pos & (labels == 0)).sum()
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)

    # Ensure endpoint
    tpr_list.append(1.0)
    fpr_list.append(1.0)

    fpr_arr = np.array(fpr_list)
    tpr_arr = np.array(tpr_list)

    # Sort by fpr for proper integration
    order = np.argsort(fpr_arr)
    fpr_arr = fpr_arr[order]
    tpr_arr = tpr_arr[order]

    auc = float(np.trapz(tpr_arr, fpr_arr))
    return fpr_arr, tpr_arr, thresholds, auc


# ═══════════════════════════════════════════════════════════════════════
# Burstiness metric
# ═══════════════════════════════════════════════════════════════════════

def compute_burstiness(trigger_steps_list):
    """Ratio of clustered triggers vs isolated triggers.

    A trigger at step t is 'clustered' if another trigger happened at
    t-1 or t+1.  burstiness = n_clustered / n_total.
    """
    all_steps = []
    for ts in trigger_steps_list:
        all_steps.extend(ts)
    if len(all_steps) == 0:
        return 0.0
    step_set = set(all_steps)
    n_clustered = sum(
        1 for s in all_steps if (s - 1) in step_set or (s + 1) in step_set)
    return n_clustered / len(all_steps)


# ═══════════════════════════════════════════════════════════════════════
# Holm-Bonferroni correction
# ═══════════════════════════════════════════════════════════════════════

def holm_bonferroni(p_values, alpha=0.05):
    """Apply Holm-Bonferroni correction.

    Parameters
    ----------
    p_values : dict[str, float]
    alpha    : family-wise error rate

    Returns
    -------
    dict[str, dict] with keys: p_raw, p_adjusted, reject
    """
    m = len(p_values)
    sorted_items = sorted(p_values.items(), key=lambda kv: kv[1])
    results = {}
    for rank_0, (name, p_raw) in enumerate(sorted_items):
        rank = rank_0 + 1
        adjusted_alpha = alpha / (m - rank + 1)
        p_adj = min(p_raw * (m - rank + 1), 1.0)
        results[name] = {
            "p_raw": p_raw,
            "p_adjusted": p_adj,
            "reject": p_raw <= adjusted_alpha,
        }
    return results


def bootstrap_auc_ci(labels, scores, n_boot=2000, alpha=0.05, seed=42):
    """Bootstrap confidence interval for AUC."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    n = len(labels)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        _, _, _, a = compute_roc_auc(labels[idx], scores[idx])
        aucs.append(a)
    aucs = np.sort(aucs)
    lo = float(np.percentile(aucs, 100 * alpha / 2))
    hi = float(np.percentile(aucs, 100 * (1 - alpha / 2)))
    return lo, hi


def permutation_test_auc(labels, scores, n_perm=5000, seed=42):
    """Permutation test: H0 is AUC = 0.5 (random classifier)."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    _, _, _, observed_auc = compute_roc_auc(labels, scores)
    count = 0
    for _ in range(n_perm):
        perm_labels = rng.permutation(labels)
        _, _, _, perm_auc = compute_roc_auc(perm_labels, scores)
        if perm_auc >= observed_auc:
            count += 1
    p_value = (count + 1) / (n_perm + 1)
    return p_value


# ═══════════════════════════════════════════════════════════════════════
# Main experiment driver
# ═══════════════════════════════════════════════════════════════════════

def main():
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    space_names = list(SPACE_CONFIGS.keys())
    all_labels = {sn: [] for sn in space_names}
    all_scores = {sn: [] for sn in space_names}
    all_blast = {sn: [] for sn in space_names}
    all_latency = {sn: [] for sn in space_names}
    all_trigger_steps = {sn: [] for sn in space_names}
    all_triggers = {sn: [] for sn in space_names}

    print("=" * 70)
    print("exp11 — Dirty Signatures: 12-bit signature + debounce")
    print("=" * 70)

    for sn in space_names:
        print(f"\n[{sn}]", end="", flush=True)
        for seed in range(N_SEEDS):
            res = run_trial(sn, seed)
            all_labels[sn].extend(res["labels"])
            all_scores[sn].extend(res["scores"])
            all_blast[sn].extend(res["blast_radii"])
            all_latency[sn].extend(res["latencies"])
            all_trigger_steps[sn].extend(res["trigger_steps"])
            all_triggers[sn].extend(res["triggers"])
            print(".", end="", flush=True)
        print(" done")

    # ── Compute AUC per space type ──────────────────────────────────
    roc_data = {}
    auc_per_space = {}
    ci_per_space = {}
    p_values = {}

    for sn in space_names:
        labels = np.array(all_labels[sn])
        scores = np.array(all_scores[sn], dtype=float)
        fpr, tpr, thr, auc = compute_roc_auc(labels, scores)
        roc_data[sn] = (fpr, tpr)
        auc_per_space[sn] = auc
        ci_lo, ci_hi = bootstrap_auc_ci(labels, scores, seed=42)
        ci_per_space[sn] = (ci_lo, ci_hi)
        p_val = permutation_test_auc(labels, scores, seed=42)
        p_values[sn] = p_val

    # Holm-Bonferroni correction
    hb_results = holm_bonferroni(p_values)

    # ── Blast radius & burstiness ───────────────────────────────────
    blast_summary = {}
    burstiness_summary = {}
    latency_summary = {}

    for sn in space_names:
        blasts = np.array(all_blast[sn])
        # Separate by label
        labels_arr = np.array(all_labels[sn])
        noise_blast = blasts[labels_arr == 0]
        struct_blast = blasts[labels_arr == 1]
        blast_summary[sn] = {
            "noise_mean": float(noise_blast.mean()) if len(noise_blast) else 0.0,
            "noise_max": int(noise_blast.max()) if len(noise_blast) else 0,
            "struct_mean": float(struct_blast.mean()) if len(struct_blast) else 0.0,
            "struct_max": int(struct_blast.max()) if len(struct_blast) else 0,
        }

        burstiness_summary[sn] = compute_burstiness(all_trigger_steps[sn])

        lats = [l for l in all_latency[sn] if not (isinstance(l, float) and
                np.isnan(l))]
        latency_summary[sn] = {
            "mean": float(np.mean(lats)) if lats else float("nan"),
            "median": float(np.median(lats)) if lats else float("nan"),
            "min": float(np.min(lats)) if lats else float("nan"),
            "max": float(np.max(lats)) if lats else float("nan"),
        }

    # ── Verdict ─────────────────────────────────────────────────────
    all_pass = all(auc >= 0.8 for auc in auc_per_space.values())
    overall_verdict = "PASS" if all_pass else "FAIL"

    # ── Print summary ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"{'Space':<20s} {'AUC':>8s} {'CI_lo':>8s} {'CI_hi':>8s} "
          f"{'p_adj':>8s} {'Reject':>8s} {'Verdict':>8s}")
    print("-" * 70)
    for sn in space_names:
        auc = auc_per_space[sn]
        ci = ci_per_space[sn]
        hb = hb_results[sn]
        v = "PASS" if auc >= 0.8 else "FAIL"
        print(f"{sn:<20s} {auc:>8.3f} {ci[0]:>8.3f} {ci[1]:>8.3f} "
              f"{hb['p_adjusted']:>8.4f} {'Yes' if hb['reject'] else 'No':>8s} "
              f"{v:>8s}")

    print(f"\nOverall: {overall_verdict}")

    print("\nBlast radius (mean triggered units):")
    for sn in space_names:
        bs = blast_summary[sn]
        print(f"  {sn:<20s} noise={bs['noise_mean']:.1f}  "
              f"structural={bs['struct_mean']:.1f}")

    print("\nLatency to trigger (steps from event):")
    for sn in space_names:
        ls = latency_summary[sn]
        print(f"  {sn:<20s} mean={ls['mean']:.1f}  median={ls['median']:.1f}")

    print("\nBurstiness:")
    for sn in space_names:
        print(f"  {sn:<20s} {burstiness_summary[sn]:.3f}")

    # ── Save JSON ───────────────────────────────────────────────────
    results_json = {
        "verdict": overall_verdict,
        "auc_per_space": {sn: round(auc_per_space[sn], 4) for sn in space_names},
        "ci_per_space": {sn: [round(ci_per_space[sn][0], 4),
                               round(ci_per_space[sn][1], 4)]
                         for sn in space_names},
        "holm_bonferroni": {sn: {k: (round(v, 6) if isinstance(v, float) else v)
                                 for k, v in hb_results[sn].items()}
                            for sn in space_names},
        "blast_radius": blast_summary,
        "latency": latency_summary,
        "burstiness": burstiness_summary,
        "config": {
            "n_seeds": N_SEEDS,
            "n_steps": N_STEPS,
            "kill_criterion": "AUC < 0.8",
            "signature_bits": 12,
            "debounce_consecutive": 2,
            "hamming_threshold": 3,
            "component_threshold": 4,
        },
    }

    with open(out_dir / "exp11_results.json", "w") as f:
        json.dump(results_json, f, indent=2, default=str)
    print(f"\nSaved: {out_dir / 'exp11_results.json'}")

    # ── Plots ───────────────────────────────────────────────────────
    _plot_roc(roc_data, auc_per_space, ci_per_space, out_dir)
    _plot_blast_radius(all_blast, all_labels, space_names, out_dir)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    return results_json


# ═══════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════

def _plot_roc(roc_data, auc_per_space, ci_per_space, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("exp11 — Dirty Signature ROC Curves", fontsize=14,
                 fontweight="bold")

    colors = {"scalar_grid": "#2ca02c", "vector_grid": "#1f77b4",
              "irregular_graph": "#ff7f0e", "tree_hierarchy": "#d62728"}

    for idx, sn in enumerate(roc_data):
        ax = axes[idx // 2, idx % 2]
        fpr, tpr = roc_data[sn]
        auc = auc_per_space[sn]
        ci = ci_per_space[sn]
        ax.plot(fpr, tpr, color=colors.get(sn, "black"), lw=2,
                label=f"AUC={auc:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.axhline(0.8, color="red", ls=":", alpha=0.5, label="AUC=0.8 target")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(sn)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    fig.savefig(out_dir / "exp11_roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_dir / 'exp11_roc_curves.png'}")


def _plot_blast_radius(all_blast, all_labels, space_names, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("exp11 — Blast Radius Distribution", fontsize=14,
                 fontweight="bold")

    for idx, sn in enumerate(space_names):
        ax = axes[idx // 2, idx % 2]
        blasts = np.array(all_blast[sn])
        labels = np.array(all_labels[sn])

        noise_b = blasts[labels == 0]
        struct_b = blasts[labels == 1]

        max_val = max(blasts.max(), 1)
        bins = np.arange(-0.5, max_val + 1.5, 1)

        ax.hist(noise_b, bins=bins, alpha=0.6, color="#2ca02c",
                label="Noise (neg)", density=True)
        ax.hist(struct_b, bins=bins, alpha=0.6, color="#d62728",
                label="Structural (pos)", density=True)
        ax.set_xlabel("Blast radius (units triggered)")
        ax.set_ylabel("Density")
        ax.set_title(sn)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "exp11_blast_radius.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_dir / 'exp11_blast_radius.png'}")


# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
