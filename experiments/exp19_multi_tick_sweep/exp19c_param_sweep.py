#!/usr/bin/env python3
"""
exp19c — Parameter sweep for multi-tick pipeline.

Sweeps: convergence_window, pilot_ticks, pilot_thresh_factor, min_roi_fraction.
Uses max_ticks=5 (exp19a optimum), medium scale, 5 seeds.

Design: one-at-a-time sweep (vary one param, hold others at default).
This gives 4 sweeps × ~4 values × 4 spaces × 5 seeds = 320 configs.
Much cheaper than full grid (4320 configs) and identifies main effects.
"""

import sys
import json
import time
import numpy as np
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENTS_DIR / "exp_phase2_pipeline"))
sys.path.insert(0, str(EXPERIMENTS_DIR / "exp10d_seed_determinism"))

from exp10d_seed_determinism import (
    ScalarGridSpace, VectorGridSpace, IrregularGraphSpace, TreeHierarchySpace,
)
from pipeline import CuriosityPipeline, PipelineConfig
import space_registry as _sr

# =====================================================================
# Config
# =====================================================================

SPACE_FACTORIES = {
    "scalar_grid":     lambda: ScalarGridSpace(N=128, tile=8),    # 256 units
    "vector_grid":     lambda: VectorGridSpace(N=64, tile=8, D=16),  # 64 units
    "irregular_graph": lambda: IrregularGraphSpace(n_points=500, k=6),
    "tree_hierarchy":  lambda: TreeHierarchySpace(depth=8),       # 32 units
}

SPACE_TYPES = ["scalar_grid", "vector_grid", "irregular_graph", "tree_hierarchy"]
N_SEEDS = 5
BUDGET = 0.30
MAX_TICKS = 5  # from exp19a

# Default values (baseline)
DEFAULTS = {
    "convergence_window": 2,
    "pilot_ticks": 3,
    "pilot_thresh_factor": 0.7,
    "min_roi_fraction": 0.15,
}

# Sweep ranges (one-at-a-time)
SWEEPS = {
    "convergence_window":   [1, 2, 3, 4, 5],
    "pilot_ticks":          [0, 1, 2, 3, 5],
    "pilot_thresh_factor":  [0.3, 0.5, 0.7, 0.9, 1.2],
    "min_roi_fraction":     [0.0, 0.05, 0.10, 0.15, 0.25, 0.40],
}


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# =====================================================================
# Runner
# =====================================================================

def run_config(space_type: str, seed: int, **overrides) -> dict:
    """Run one config with parameter overrides."""
    params = dict(DEFAULTS)
    params.update(overrides)

    cfg = PipelineConfig(
        max_ticks=MAX_TICKS,
        budget_fraction=BUDGET,
        convergence_window=params["convergence_window"],
        pilot_ticks=params["pilot_ticks"],
        pilot_thresh_factor=params["pilot_thresh_factor"],
        min_roi_fraction=params["min_roi_fraction"],
        topo_profiling_enabled=(space_type == "irregular_graph"),
    )
    pipe = CuriosityPipeline(cfg)

    # Override space factory
    old = _sr.SPACE_FACTORIES.get(space_type)
    _sr.SPACE_FACTORIES[space_type] = SPACE_FACTORIES[space_type]

    try:
        t0 = time.time()
        r = pipe.run(space_type, seed=seed, budget_fraction=BUDGET)
        wall = time.time() - t0
    finally:
        if old is not None:
            _sr.SPACE_FACTORIES[space_type] = old

    return {
        "space_type": space_type,
        "seed": seed,
        "psnr_gain": r.quality_psnr - r.coarse_psnr,
        "n_refined": r.n_refined,
        "n_total": r.n_total,
        "n_ticks_executed": r.n_ticks_executed,
        "convergence_reason": r.convergence_reason,
        "reject_rate": r.reject_rate,
        "wall_time": wall,
        **params,
    }


def run_sweep(param_name: str, values: list) -> list:
    """Sweep one parameter, holding others at default."""
    results = []
    for val in values:
        overrides = {param_name: val}
        for space_type in SPACE_TYPES:
            for seed in range(N_SEEDS):
                try:
                    r = run_config(space_type, seed, **overrides)
                    r["sweep_param"] = param_name
                    r["sweep_value"] = val
                    results.append(r)
                except Exception as e:
                    results.append({
                        "sweep_param": param_name,
                        "sweep_value": val,
                        "space_type": space_type,
                        "seed": seed,
                        "error": str(e),
                    })
    return results


def print_sweep_summary(param_name: str, results: list):
    """Print summary table for one sweep."""
    valid = [r for r in results if "error" not in r]
    if not valid:
        print(f"  [{param_name}] No valid results!")
        return

    values = sorted(set(r["sweep_value"] for r in valid))

    print(f"\n  === {param_name} ===")
    print(f"  {'Value':<12}", end="")
    for st in SPACE_TYPES:
        print(f" {st[:12]:>12}", end="")
    print()

    for val in values:
        print(f"  {str(val):<12}", end="")
        for st in SPACE_TYPES:
            subset = [r for r in valid if r["sweep_value"] == val and r["space_type"] == st]
            if subset:
                median_psnr = np.median([r["psnr_gain"] for r in subset])
                print(f" {median_psnr:>+11.2f}", end="")
            else:
                print(f" {'n/a':>12}", end="")
        print()


def main():
    total = sum(len(v) for v in SWEEPS.values()) * len(SPACE_TYPES) * N_SEEDS
    print("=" * 70)
    print("exp19c — Parameter sweep (one-at-a-time)")
    print(f"  max_ticks: {MAX_TICKS}, scale: medium, seeds: {N_SEEDS}")
    print(f"  sweeps: {list(SWEEPS.keys())}")
    print(f"  total configs: {total}")
    print("=" * 70)

    all_results = []
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    for param_name, values in SWEEPS.items():
        print(f"\n--- Sweeping {param_name}: {values} ---")
        t0 = time.time()
        results = run_sweep(param_name, values)
        dt = time.time() - t0
        all_results.extend(results)

        n_errors = sum(1 for r in results if "error" in r)
        print(f"  Done in {dt:.1f}s. {len(results)} results, {n_errors} errors.")

        print_sweep_summary(param_name, results)

        # Save incrementally
        out_path = out_dir / f"exp19c_{param_name}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)

    # Save combined
    with open(out_dir / "exp19c_combined.json", "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)

    # Summary: find best value per param per space
    print("\n" + "=" * 70)
    print("  RECOMMENDATIONS")
    print("=" * 70)

    valid_all = [r for r in all_results if "error" not in r]
    for param_name, values in SWEEPS.items():
        print(f"\n  {param_name}:")
        for st in SPACE_TYPES:
            best_val = None
            best_psnr = -999
            for val in values:
                subset = [r for r in valid_all
                          if r.get("sweep_param") == param_name
                          and r["sweep_value"] == val
                          and r["space_type"] == st]
                if subset:
                    median = np.median([r["psnr_gain"] for r in subset])
                    if median > best_psnr:
                        best_psnr = median
                        best_val = val
            if best_val is not None:
                print(f"    {st}: best={best_val} (PSNR +{best_psnr:.2f} dB)")

    print("\n" + "=" * 70)
    print("  exp19c COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
