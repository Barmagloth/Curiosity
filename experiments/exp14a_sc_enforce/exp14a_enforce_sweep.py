#!/usr/bin/env python3
"""
Curiosity -- exp14a_enforce_sweep: SC-Enforce validation sweep.

Tests the SC-Enforce module on positive baselines (good deltas that should PASS)
and negative baselines (bad deltas that should be DAMPED or REJECTED), sweeping
damp_factor across [0.3, 0.5, 0.7, 1.0] for each of 4 space types.

Kill criteria:
  1. Quality loss on positive baselines: >5% rejected -> FAIL
  2. Catch rate on negative baselines: <80% damped/rejected -> FAIL
  3. Damp convergence: >=95% cases converge in <=3 iterations -> else FAIL

CPU-only.  Should complete in under 10 minutes.
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
EXP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Add parent experiments dir so we can import from sibling packages
EXPERIMENTS_DIR = EXP_DIR.parent
sys.path.insert(0, str(EXPERIMENTS_DIR))

# Add sc_baseline to path for operators/baselines
SC_BASELINE_DIR = EXPERIMENTS_DIR / "sc_baseline"
sys.path.insert(0, str(SC_BASELINE_DIR))

# ---------------------------------------------------------------------------
# Imports from exp12a
# ---------------------------------------------------------------------------
from exp12a_tau_parent.exp12a_tau_parent import (
    make_scalar_grid, make_vector_grid, make_graph, make_tree,
    make_grid_ops, make_vector_ops, GraphOps, TreeOps,
    _generate_scalar_grid_at_level, _generate_vector_grid_at_level,
    _generate_graph_at_level, _generate_tree_at_level,
    GRID_LEVEL_TILE, GRAPH_LEVEL_CLUSTERS, TREE_LEVEL_COARSE_DEPTH,
)

# ---------------------------------------------------------------------------
# Imports from baselines_v2
# ---------------------------------------------------------------------------
from baselines_v2 import (
    positive_oracle, positive_scaled, positive_noisy,
    negative_lf_drift, negative_random_lf, negative_semant_wrong,
    negative_coarse_shift_coherent,
)

# ---------------------------------------------------------------------------
# Imports from sc_enforce
# ---------------------------------------------------------------------------
sys.path.insert(0, str(EXP_DIR))
from sc_enforce import SCEnforcer, load_thresholds, EnforceResult, StrictnessTracker, WasteBudget

# ---------------------------------------------------------------------------
# Threshold file
# ---------------------------------------------------------------------------
THRESHOLD_PATH = EXPERIMENTS_DIR / "exp12a_tau_parent" / "results" / "exp12a_thresholds.json"

# ---------------------------------------------------------------------------
# Sweep parameters
# ---------------------------------------------------------------------------
DAMP_FACTORS = [0.3, 0.5, 0.7, 1.0]
LEVELS = [1, 2, 3]
N_SEEDS = 10
BASE_SEED = 7700
MAX_DAMP_ITERATIONS = 3

SPACE_TYPES = ["T1_scalar", "T2_vector", "T3_graph", "T4_tree"]


# ===================================================================
# Operator builders per space type
# ===================================================================

def build_ops(space_type, info):
    """Return (R_fn, Up_fn) for a given space type and generation info."""
    if space_type == "T1_scalar":
        return make_grid_ops(sigma=3.0)
    elif space_type == "T2_vector":
        return make_vector_ops(sigma=3.0)
    elif space_type == "T3_graph":
        labels = info["labels"]
        n_clusters = info["n_clusters"]
        n_points = info["n_points"]
        gops = GraphOps(labels, n_clusters, n_points)
        R_fn = gops.restrict
        Up_fn = lambda xc, shape, _g=gops: _g.prolong(xc, shape)
        return R_fn, Up_fn
    elif space_type == "T4_tree":
        n_nodes = info["n_nodes"]
        coarse_depth = info["coarse_depth"]
        tops = TreeOps(n_nodes, coarse_depth)
        R_fn = tops.restrict
        Up_fn = lambda xc, shape, _t=tops: _t.prolong(xc, shape)
        return R_fn, Up_fn
    else:
        raise ValueError(f"Unknown space type: {space_type}")


# ===================================================================
# Data generators per space type
# ===================================================================

GENERATORS = {
    "T1_scalar": _generate_scalar_grid_at_level,
    "T2_vector": _generate_vector_grid_at_level,
    "T3_graph": _generate_graph_at_level,
    "T4_tree": _generate_tree_at_level,
}


# ===================================================================
# Baseline generators (adapted for all space types)
# ===================================================================

def make_positive_deltas(gt, coarse, info, seed):
    """Generate 3 positive baseline deltas."""
    space_t = info.get("type", "")
    deltas = {}

    # Oracle: gt - coarse (works for all types)
    deltas["pos_oracle"] = positive_oracle(gt, coarse)

    # Scaled
    deltas["pos_scaled"] = positive_scaled(gt, coarse, scale=0.5, seed=seed)

    # Noisy
    deltas["pos_noisy"] = positive_noisy(gt, coarse, noise_std=0.02, seed=seed)

    return deltas


def make_negative_deltas(gt, coarse, info, seed, R_fn, Up_fn):
    """Generate 3 negative baseline deltas."""
    space_t = info.get("type", "")
    deltas = {}

    # LF drift -- works for all ndims via baselines_v2
    try:
        deltas["neg_lf_drift"] = negative_lf_drift(gt, coarse, amplitude=0.3, seed=seed)
    except Exception:
        # Fallback: pure LF component
        oracle = gt - coarse
        deltas["neg_lf_drift"] = Up_fn(R_fn(oracle), gt.shape)

    # Random LF
    try:
        deltas["neg_random_lf"] = negative_random_lf(coarse, sigma=8.0, amplitude=0.5, seed=seed)
    except Exception:
        # Fallback: random + low-pass
        rng = np.random.RandomState(seed)
        raw = rng.randn(*gt.shape) * 0.3
        deltas["neg_random_lf"] = Up_fn(R_fn(raw), gt.shape)

    # Semantically wrong
    try:
        deltas["neg_semant_wrong"] = negative_semant_wrong(coarse, scale=1.0)
    except Exception:
        deltas["neg_semant_wrong"] = -2.0 * coarse

    return deltas


# ===================================================================
# Main sweep
# ===================================================================

def run_sweep():
    print("=" * 72)
    print("exp14a_enforce_sweep: SC-Enforce validation sweep")
    print("=" * 72)

    tau_parent = load_thresholds(str(THRESHOLD_PATH))
    print(f"Loaded {len(tau_parent)} thresholds from {THRESHOLD_PATH}")

    all_results = {}
    t0_global = time.time()

    for space_type in SPACE_TYPES:
        print(f"\n{'-' * 60}")
        print(f"  Space: {space_type}")
        print(f"{'-' * 60}")

        gen_fn = GENERATORS[space_type]
        space_rows = []
        t0_space = time.time()

        for level in LEVELS:
            for seed_idx in range(N_SEEDS):
                seed = BASE_SEED + seed_idx * 100 + level

                # Generate data
                gt, coarse, info = gen_fn(level, seed)
                R_fn, Up_fn = build_ops(space_type, info)

                # Generate baselines
                pos_deltas = make_positive_deltas(gt, coarse, info, seed)
                neg_deltas = make_negative_deltas(gt, coarse, info, seed, R_fn, Up_fn)

                for damp_factor in DAMP_FACTORS:
                    enforcer = SCEnforcer(
                        tau_parent=tau_parent,
                        R_fn=R_fn,
                        Up_fn=Up_fn,
                        space_type=space_type,
                        damp_factor=damp_factor,
                        max_damp_iterations=MAX_DAMP_ITERATIONS,
                    )

                    # --- Positive baselines ---
                    for bname, delta in pos_deltas.items():
                        result = enforcer.check_and_enforce(delta, coarse, level)
                        row = {
                            "space_type": space_type,
                            "level": level,
                            "seed": seed,
                            "damp_factor": damp_factor,
                            "baseline": bname,
                            "polarity": "positive",
                            "action": result.action,
                            "d_parent_before": round(result.original_d_parent, 6),
                            "d_parent_after": round(result.d_parent_value, 6),
                            "damp_iterations": result.damp_iterations,
                        }
                        space_rows.append(row)

                    # --- Negative baselines ---
                    for bname, delta in neg_deltas.items():
                        result = enforcer.check_and_enforce(delta, coarse, level)
                        row = {
                            "space_type": space_type,
                            "level": level,
                            "seed": seed,
                            "damp_factor": damp_factor,
                            "baseline": bname,
                            "polarity": "negative",
                            "action": result.action,
                            "d_parent_before": round(result.original_d_parent, 6),
                            "d_parent_after": round(result.d_parent_value, 6),
                            "damp_iterations": result.damp_iterations,
                        }
                        space_rows.append(row)

        dt_space = time.time() - t0_space
        print(f"  {space_type}: {len(space_rows)} trials in {dt_space:.1f}s")

        # Save incrementally
        out_path = RESULTS_DIR / f"sweep_{space_type}.json"
        with open(out_path, "w") as f:
            json.dump({"space_type": space_type, "rows": space_rows}, f, indent=2)
        print(f"  Saved -> {out_path}")

        all_results[space_type] = space_rows

    dt_global = time.time() - t0_global
    print(f"\nTotal sweep time: {dt_global:.1f}s")

    # ===================================================================
    # Aggregate and evaluate kill criteria
    # ===================================================================
    summary = evaluate_kill_criteria(all_results)
    summary["total_time_s"] = round(dt_global, 2)

    out_path = RESULTS_DIR / "sweep_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved -> {out_path}")

    print_summary_table(summary)
    return summary


def evaluate_kill_criteria(all_results):
    """Compute kill criteria from sweep results."""
    summary = {
        "per_space": {},
        "kill_criteria": {},
    }

    global_pos_total = 0
    global_pos_rejected = 0
    global_neg_total = 0
    global_neg_caught = 0
    global_damp_total = 0
    global_damp_converged = 0

    for space_type, rows in all_results.items():
        pos_rows = [r for r in rows if r["polarity"] == "positive"]
        neg_rows = [r for r in rows if r["polarity"] == "negative"]

        # Kill criterion 1: positive rejection rate
        pos_total = len(pos_rows)
        pos_rejected = sum(1 for r in pos_rows if r["action"] == "rejected")
        pos_passed = sum(1 for r in pos_rows if r["action"] == "pass")
        pos_damped = sum(1 for r in pos_rows if r["action"] == "damped")
        pos_reject_rate = pos_rejected / max(pos_total, 1)

        # Kill criterion 2: negative catch rate (damped + rejected)
        neg_total = len(neg_rows)
        neg_caught = sum(1 for r in neg_rows if r["action"] in ("damped", "rejected"))
        neg_catch_rate = neg_caught / max(neg_total, 1)

        # Kill criterion 3: damp convergence
        # Among all cases that entered damping, how many converged in <= max_damp_iterations?
        damp_cases = [r for r in rows if r["action"] == "damped"]
        damp_total = len(damp_cases)
        damp_converged = sum(1 for r in damp_cases
                             if r["damp_iterations"] <= MAX_DAMP_ITERATIONS)

        # Per-damp-factor breakdown
        per_df = {}
        for df in DAMP_FACTORS:
            df_rows = [r for r in rows if r["damp_factor"] == df]
            df_pos = [r for r in df_rows if r["polarity"] == "positive"]
            df_neg = [r for r in df_rows if r["polarity"] == "negative"]
            per_df[str(df)] = {
                "pos_total": len(df_pos),
                "pos_pass": sum(1 for r in df_pos if r["action"] == "pass"),
                "pos_damped": sum(1 for r in df_pos if r["action"] == "damped"),
                "pos_rejected": sum(1 for r in df_pos if r["action"] == "rejected"),
                "neg_total": len(df_neg),
                "neg_caught": sum(1 for r in df_neg if r["action"] in ("damped", "rejected")),
                "neg_pass": sum(1 for r in df_neg if r["action"] == "pass"),
            }

        summary["per_space"][space_type] = {
            "pos_total": pos_total,
            "pos_pass": pos_passed,
            "pos_damped": pos_damped,
            "pos_rejected": pos_rejected,
            "pos_reject_rate": round(pos_reject_rate, 4),
            "neg_total": neg_total,
            "neg_caught": neg_caught,
            "neg_catch_rate": round(neg_catch_rate, 4),
            "damp_total": damp_total,
            "damp_converged": damp_converged,
            "damp_convergence_rate": round(damp_converged / max(damp_total, 1), 4),
            "per_damp_factor": per_df,
        }

        global_pos_total += pos_total
        global_pos_rejected += pos_rejected
        global_neg_total += neg_total
        global_neg_caught += neg_caught
        global_damp_total += damp_total
        global_damp_converged += damp_converged

    # Global kill criteria
    global_pos_reject_rate = global_pos_rejected / max(global_pos_total, 1)
    global_neg_catch_rate = global_neg_caught / max(global_neg_total, 1)
    global_damp_conv_rate = global_damp_converged / max(global_damp_total, 1)

    kc1_pass = global_pos_reject_rate <= 0.05
    kc2_pass = global_neg_catch_rate >= 0.80
    kc3_pass = global_damp_conv_rate >= 0.95

    summary["kill_criteria"] = {
        "KC1_pos_reject_rate": {
            "value": round(global_pos_reject_rate, 4),
            "threshold": 0.05,
            "condition": "<=",
            "passed": kc1_pass,
        },
        "KC2_neg_catch_rate": {
            "value": round(global_neg_catch_rate, 4),
            "threshold": 0.80,
            "condition": ">=",
            "passed": kc2_pass,
        },
        "KC3_damp_convergence": {
            "value": round(global_damp_conv_rate, 4),
            "threshold": 0.95,
            "condition": ">=",
            "passed": kc3_pass,
        },
        "all_passed": kc1_pass and kc2_pass and kc3_pass,
    }

    return summary


def print_summary_table(summary):
    """Print a human-readable summary table."""
    print("\n" + "=" * 80)
    print("  SC-ENFORCE SWEEP SUMMARY")
    print("=" * 80)

    header = f"{'Space':<12} {'Pos.Total':>10} {'Pos.Rej%':>10} {'Neg.Total':>10} {'Neg.Catch%':>10} {'Damp.Conv%':>10}"
    print(header)
    print("-" * 80)

    for space_type in SPACE_TYPES:
        s = summary["per_space"][space_type]
        print(f"{space_type:<12} "
              f"{s['pos_total']:>10d} "
              f"{s['pos_reject_rate'] * 100:>9.1f}% "
              f"{s['neg_total']:>10d} "
              f"{s['neg_catch_rate'] * 100:>9.1f}% "
              f"{s['damp_convergence_rate'] * 100:>9.1f}%")

    print("-" * 80)

    # Per-damp-factor detail
    print("\n  Per damp_factor breakdown:")
    print(f"  {'Space':<12} {'damp_f':>6} {'P.pass':>7} {'P.damp':>7} {'P.rej':>6} {'N.catch':>8} {'N.pass':>7}")
    print("  " + "-" * 60)
    for space_type in SPACE_TYPES:
        pdf = summary["per_space"][space_type]["per_damp_factor"]
        for df_str in sorted(pdf.keys(), key=float):
            d = pdf[df_str]
            print(f"  {space_type:<12} {df_str:>6} "
                  f"{d['pos_pass']:>7d} {d['pos_damped']:>7d} {d['pos_rejected']:>6d} "
                  f"{d['neg_caught']:>8d} {d['neg_pass']:>7d}")

    # Kill criteria
    print("\n" + "=" * 80)
    print("  KILL CRITERIA")
    print("=" * 80)
    kc = summary["kill_criteria"]
    for name in ["KC1_pos_reject_rate", "KC2_neg_catch_rate", "KC3_damp_convergence"]:
        k = kc[name]
        verdict = "PASS" if k["passed"] else "FAIL"
        print(f"  {name}: {k['value']:.4f} {k['condition']} {k['threshold']} -> [{verdict}]")

    overall = "PASS" if kc["all_passed"] else "FAIL"
    print(f"\n  Overall: [{overall}]")
    print("=" * 80)


if __name__ == "__main__":
    run_sweep()
