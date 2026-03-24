#!/usr/bin/env python3
"""
exp19a — max_ticks scaling law.

Question: How should max_ticks scale with n_total?
Sweep: 3 scales x 7 max_ticks x 4 spaces x 10 seeds = 840 configs.
Kill: multi-tick PSNR >= 95% of single-tick PSNR at same budget.

Outputs results incrementally to results/exp19a_chunk_*.json.
"""

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

# Setup paths
EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENTS_DIR / "exp_phase2_pipeline"))
sys.path.insert(0, str(EXPERIMENTS_DIR / "exp10d_seed_determinism"))

from exp10d_seed_determinism import (
    ScalarGridSpace, VectorGridSpace, IrregularGraphSpace, TreeHierarchySpace,
)
from pipeline import CuriosityPipeline, PipelineConfig

# =====================================================================
# Scale configurations
# =====================================================================

SCALES = {
    "small": {
        "scalar_grid":    lambda: ScalarGridSpace(N=64, tile=8),    # 64 units
        "vector_grid":    lambda: VectorGridSpace(N=32, tile=8, D=16),  # 16 units
        "irregular_graph": lambda: IrregularGraphSpace(n_points=200, k=6),  # ~10 units
        "tree_hierarchy":  lambda: TreeHierarchySpace(depth=6),     # 8 units
    },
    "medium": {
        "scalar_grid":    lambda: ScalarGridSpace(N=128, tile=8),   # 256 units
        "vector_grid":    lambda: VectorGridSpace(N=64, tile=8, D=16),  # 64 units
        "irregular_graph": lambda: IrregularGraphSpace(n_points=500, k=6),  # ~25 units
        "tree_hierarchy":  lambda: TreeHierarchySpace(depth=8),     # 32 units
    },
    "large": {
        "scalar_grid":    lambda: ScalarGridSpace(N=256, tile=8),   # 1024 units
        "vector_grid":    lambda: VectorGridSpace(N=128, tile=8, D=16),  # 256 units
        "irregular_graph": lambda: IrregularGraphSpace(n_points=2000, k=6),  # ~100 units
        "tree_hierarchy":  lambda: TreeHierarchySpace(depth=10),    # 128 units
    },
}

MAX_TICKS_VALUES = [1, 2, 3, 5, 8, 13, 20]
N_SEEDS = 10
BUDGET = 0.30

SPACE_TYPES = ["scalar_grid", "vector_grid", "irregular_graph", "tree_hierarchy"]


# =====================================================================
# Runner
# =====================================================================

def run_single(space_factory, space_type: str, max_ticks: int,
               seed: int, budget: float) -> dict:
    """Run one configuration and return metrics."""
    cfg = PipelineConfig(
        max_ticks=max_ticks,
        budget_fraction=budget,
        # Disable topo profiling for speed on large graphs
        # (already validated in Phase 2)
        topo_profiling_enabled=(space_type == "irregular_graph"),
    )
    pipe = CuriosityPipeline(cfg)

    # Override space factory in pipeline
    # We need to monkey-patch because pipeline uses SPACE_FACTORIES
    import exp_phase2_pipeline.pipeline as _pipe_mod
    # Actually, CuriosityPipeline.run() uses SPACE_FACTORIES from space_registry
    # We need to temporarily override it
    from space_registry import SPACE_FACTORIES as _orig_factories
    import space_registry as _sr
    old_factory = _sr.SPACE_FACTORIES.get(space_type)
    _sr.SPACE_FACTORIES[space_type] = space_factory

    try:
        t0 = time.time()
        r = pipe.run(space_type, seed=seed, budget_fraction=budget)
        wall = time.time() - t0
    finally:
        # Restore original factory
        if old_factory is not None:
            _sr.SPACE_FACTORIES[space_type] = old_factory

    return {
        "psnr_final": r.quality_psnr,
        "psnr_coarse": r.coarse_psnr,
        "psnr_gain": r.quality_psnr - r.coarse_psnr,
        "n_refined": r.n_refined,
        "n_total": r.n_total,
        "n_ticks_executed": r.n_ticks_executed,
        "convergence_reason": r.convergence_reason,
        "reject_rate": r.reject_rate,
        "wall_time": wall,
        "gate_stage": r.gate_stage,
    }


def run_chunk(scale: str, space_type: str, chunk_id: int) -> dict:
    """Run all seeds x max_ticks for one (scale, space_type) combo."""
    factory = SCALES[scale][space_type]
    results = []

    for max_ticks in MAX_TICKS_VALUES:
        for seed in range(N_SEEDS):
            try:
                r = run_single(factory, space_type, max_ticks, seed, BUDGET)
                r["scale"] = scale
                r["space_type"] = space_type
                r["max_ticks"] = max_ticks
                r["seed"] = seed
                results.append(r)
            except Exception as e:
                results.append({
                    "scale": scale,
                    "space_type": space_type,
                    "max_ticks": max_ticks,
                    "seed": seed,
                    "error": str(e),
                })

    # Save incrementally
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"exp19a_chunk_{chunk_id}_{scale}_{space_type}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    valid = [r for r in results if "error" not in r]
    if valid:
        # Compare multi-tick vs single-tick (max_ticks=1)
        single = [r for r in valid if r["max_ticks"] == 1]
        if single:
            single_psnr = np.median([r["psnr_gain"] for r in single])
            print(f"  [{scale}/{space_type}] single-tick median PSNR gain: {single_psnr:.2f} dB")
            for mt in MAX_TICKS_VALUES[1:]:
                mt_results = [r for r in valid if r["max_ticks"] == mt]
                if mt_results:
                    mt_psnr = np.median([r["psnr_gain"] for r in mt_results])
                    ratio = mt_psnr / max(single_psnr, 0.01)
                    n_units = mt_results[0]["n_total"]
                    ticks_used = np.median([r["n_ticks_executed"] for r in mt_results])
                    print(f"    mt={mt:2d}: PSNR gain {mt_psnr:+.2f} dB "
                          f"({ratio:.0%} of single) "
                          f"ticks_used={ticks_used:.0f} "
                          f"n_units={n_units}")

    return {"chunk_id": chunk_id, "n_results": len(results),
            "n_errors": len([r for r in results if "error" in r])}


def main():
    parser = argparse.ArgumentParser(description="exp19a: max_ticks scaling law")
    parser.add_argument("--scale", choices=["small", "medium", "large", "all"],
                        default="all")
    parser.add_argument("--space", choices=SPACE_TYPES + ["all"], default="all")
    args = parser.parse_args()

    scales = [args.scale] if args.scale != "all" else ["small", "medium", "large"]
    spaces = [args.space] if args.space != "all" else SPACE_TYPES

    print("=" * 70)
    print("exp19a — max_ticks scaling law")
    print(f"  scales: {scales}")
    print(f"  spaces: {spaces}")
    print(f"  max_ticks: {MAX_TICKS_VALUES}")
    print(f"  seeds: {N_SEEDS}, budget: {BUDGET}")
    total = len(scales) * len(spaces) * len(MAX_TICKS_VALUES) * N_SEEDS
    print(f"  total configs: {total}")
    print("=" * 70)

    chunk_id = 0
    for scale in scales:
        for space_type in spaces:
            print(f"\n--- Chunk {chunk_id}: {scale} / {space_type} ---")
            t0 = time.time()
            summary = run_chunk(scale, space_type, chunk_id)
            dt = time.time() - t0
            print(f"  Done in {dt:.1f}s. "
                  f"Results: {summary['n_results']}, errors: {summary['n_errors']}")
            chunk_id += 1

    print("\n" + "=" * 70)
    print(f"  exp19a COMPLETE. {chunk_id} chunks saved.")
    print("=" * 70)


if __name__ == "__main__":
    main()
