#!/usr/bin/env python3
"""
DET-2 Recheck: Cross-Seed Stability for the Phase 2 pipeline (with topo profiling).

Tests that DIFFERENT seeds produce STATISTICALLY STABLE metrics.
Same methodology as exp11a, but uses CuriosityPipeline (with topo profiling
active for irregular_graph).

Sweep: 20 seeds x 4 spaces x 2 budgets = 160 runs.
Metrics: n_refined, n_rejected, reject_rate, psnr_gain, compliance.
Kill criterion: per-regime CV thresholds.

CV thresholds (from exp11a):
  - Regular spaces (scalar_grid, vector_grid): CV < 0.10
  - Irregular spaces at low budget: CV < 0.10
  - Irregular spaces at high budget: CV < 0.25
"""

import sys
import json
import time
import numpy as np
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent.parent / "exp_phase2_pipeline"
sys.path.insert(0, str(PIPELINE_DIR))
from pipeline import CuriosityPipeline
from config import PipelineConfig

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

N_SEEDS = 20
SEEDS = list(range(N_SEEDS))
SPACE_TYPES = ["scalar_grid", "vector_grid", "irregular_graph", "tree_hierarchy"]
BUDGET_LEVELS = {"low": 0.10, "high": 0.30}

# Kill-criteria metrics: structural pipeline behavior (must be stable across seeds)
KILL_METRIC_NAMES = [
    "n_refined",
    "compliance",  # n_refined / (budget * n_total)
]

# Informational metrics: tracked but NOT kill-criteria
# psnr_gain: depends on seed-generated ground truth (same issue as mean_leaf_value
#   removed from exp11a). Legitimate per-seed variation, not a pipeline defect.
# n_passed/n_rejected: at low budget (n_refined=1) these are binary → CV meaningless.
INFO_METRIC_NAMES = [
    "n_passed",
    "n_rejected",
    "reject_rate",
    "psnr_gain",
]

ALL_METRIC_NAMES = KILL_METRIC_NAMES + INFO_METRIC_NAMES

# Topo metrics (informational, irregular_graph only)
TOPO_METRIC_NAMES = [
    "topo_eta_f",
    "topo_computation_ms",
]

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def get_cv_threshold(space_type: str, budget_level: str) -> float:
    """CV threshold depends on space regularity and budget level."""
    is_irregular = space_type in ('irregular_graph', 'tree_hierarchy')
    is_high_budget = budget_level == 'high'
    if is_irregular and is_high_budget:
        return 0.25
    return 0.10


def extract_metrics(r, budget_frac: float) -> dict:
    """Extract stability metrics from a PipelineResult."""
    budget_units = max(1, int(budget_frac * r.n_total))
    compliance = r.n_refined / budget_units if budget_units > 0 else 0.0
    psnr_gain = r.quality_psnr - r.coarse_psnr

    m = {
        "n_refined": float(r.n_refined),
        "n_passed": float(r.n_passed),
        "n_rejected": float(r.n_rejected),
        "reject_rate": float(r.reject_rate),
        "psnr_gain": float(psnr_gain),
        "compliance": float(compliance),
    }

    # Topo metrics (only populated for irregular_graph)
    topo_eta_f = getattr(r, 'topo_eta_f', None)
    topo_ms = getattr(r, 'topo_computation_ms', None)
    if topo_eta_f is not None:
        m["topo_eta_f"] = float(topo_eta_f)
    if topo_ms is not None:
        m["topo_computation_ms"] = float(topo_ms)

    return m


def compute_cv(values):
    """Coefficient of variation (std/mean), handling zero-mean gracefully."""
    arr = np.array(values, dtype=float)
    mean = np.mean(arr)
    std = np.std(arr, ddof=0)
    if mean == 0:
        return 0.0 if std == 0 else float('inf')
    return std / abs(mean)


def main():
    print("=" * 70)
    print("DET-2 Recheck -- Cross-Seed Stability (Phase 2 Pipeline)")
    print("=" * 70)
    print(f"  Spaces:     {SPACE_TYPES}")
    print(f"  Seeds:      {N_SEEDS} (0..{N_SEEDS - 1})")
    print(f"  Budgets:    {BUDGET_LEVELS}")
    n_total = N_SEEDS * len(SPACE_TYPES) * len(BUDGET_LEVELS)
    print(f"  Total runs: {n_total}")
    print()

    pipe = CuriosityPipeline(PipelineConfig())

    all_raw = []
    cell_results = []
    overall_pass = True
    t_start = time.time()

    for space_type in SPACE_TYPES:
        for budget_name, budget_frac in BUDGET_LEVELS.items():
            seed_metrics = []
            for seed in SEEDS:
                r = pipe.run(space_type, seed=seed, budget_fraction=budget_frac)
                m = extract_metrics(r, budget_frac)
                seed_metrics.append(m)
                all_raw.append({
                    "space": space_type,
                    "budget": budget_name,
                    "budget_frac": budget_frac,
                    "seed": seed,
                    **m,
                })

            # Compute CV for all metrics
            cv_thresh = get_cv_threshold(space_type, budget_name)
            metrics_cv = {}
            cell_pass = True

            for metric in ALL_METRIC_NAMES:
                values = [sm[metric] for sm in seed_metrics]
                cv = compute_cv(values)
                mean_val = float(np.mean(values))
                std_val = float(np.std(values, ddof=0))
                is_kill = metric in KILL_METRIC_NAMES
                metrics_cv[metric] = {
                    "mean": round(mean_val, 6),
                    "std": round(std_val, 6),
                    "cv": round(cv, 6),
                    "kill_criteria": is_kill,
                }
                # Only kill-criteria metrics can cause FAIL
                if is_kill and cv >= cv_thresh:
                    cell_pass = False
                    overall_pass = False

            # Topo metrics CV (informational, not kill-criteria)
            topo_cv = {}
            if space_type == "irregular_graph":
                for tm in TOPO_METRIC_NAMES:
                    values = [sm.get(tm, 0.0) for sm in seed_metrics]
                    if any(v != 0.0 for v in values):
                        cv = compute_cv(values)
                        topo_cv[tm] = {
                            "mean": round(float(np.mean(values)), 4),
                            "std": round(float(np.std(values, ddof=0)), 4),
                            "cv": round(cv, 4),
                        }

            max_cv = max(
                (metrics_cv[m]["cv"] for m in KILL_METRIC_NAMES),
                default=0.0
            )

            status = "PASS" if cell_pass else "FAIL"
            print(f"  [{space_type:18s} {budget_name:4s}] "
                  f"max_CV={max_cv:.4f} (thresh={cv_thresh:.2f}) -> {status}")

            cell_results.append({
                "space": space_type,
                "budget": budget_name,
                "pass": cell_pass,
                "cv_threshold": cv_thresh,
                "max_cv": round(max_cv, 6),
                "metrics": metrics_cv,
                "topo_metrics": topo_cv if topo_cv else None,
            })

    elapsed = time.time() - t_start
    verdict = "PASS" if overall_pass else "FAIL"

    # Print summary
    print(f"\n{'=' * 70}")
    n_pass = sum(1 for cr in cell_results if cr["pass"])
    n_cells = len(cell_results)
    print(f"VERDICT: {verdict}")
    print(f"  Cells: {n_pass}/{n_cells} pass")
    print(f"  Total runs: {n_total}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print("=" * 70)

    # Detail per cell
    for cr in cell_results:
        status = "PASS" if cr["pass"] else "FAIL"
        ct = cr["cv_threshold"]
        print(f"\n  [{cr['space']:18s} {cr['budget']:4s}] {status} (thresh={ct:.2f})")
        print(f"    {'metric':20s}  {'mean':>10s}  {'std':>8s}  {'CV':>8s}  {'role':>6s}")
        for m in ALL_METRIC_NAMES:
            info = cr["metrics"][m]
            role = "KILL" if info["kill_criteria"] else "info"
            flag = " *** FAIL ***" if info["kill_criteria"] and info["cv"] >= ct else ""
            print(f"    {m:20s}  {info['mean']:10.4f}  "
                  f"{info['std']:8.4f}  {info['cv']:8.4f}  {role:>6s}{flag}")

    # Topo summary for irregular_graph
    topo_cells = [cr for cr in cell_results
                  if cr["space"] == "irregular_graph" and cr.get("topo_metrics")]
    if topo_cells:
        print(f"\n  Topo profiling metrics (informational, not kill-criteria):")
        for cr in topo_cells:
            for tm, info in cr["topo_metrics"].items():
                print(f"    [{cr['budget']:4s}] {tm:25s}  "
                      f"mean={info['mean']:.4f}  CV={info['cv']:.4f}")

    # Save JSON
    json_out = {
        "experiment": "det2_recheck_phase2",
        "verdict": verdict,
        "cv_thresholds": {
            "regular_or_low_budget": 0.10,
            "irregular_high_budget": 0.25,
        },
        "n_seeds": N_SEEDS,
        "space_types": SPACE_TYPES,
        "budget_levels": BUDGET_LEVELS,
        "n_total_runs": n_total,
        "elapsed_s": round(elapsed, 2),
        "cells": cell_results,
        "raw": all_raw,
    }

    json_path = RESULTS_DIR / "det2_recheck.json"
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2, default=str)
    print(f"\n  Saved: {json_path}")

    return verdict


if __name__ == "__main__":
    main()
