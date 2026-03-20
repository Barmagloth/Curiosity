#!/usr/bin/env python3
"""DET-1 Recheck: same seed = same result (bitwise) for the Phase 2 pipeline."""
import sys, json
import numpy as np
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent.parent / "exp_phase2_pipeline"
sys.path.insert(0, str(PIPELINE_DIR))
from pipeline import CuriosityPipeline
from config import PipelineConfig

SPACES = ["scalar_grid", "vector_grid", "irregular_graph", "tree_hierarchy"]
SEEDS = list(range(10))  # 10 seeds
BUDGET = 0.30
RESULTS_DIR = Path(__file__).resolve().parent / "results"

pipe = CuriosityPipeline(PipelineConfig())
results = []
n_pass = 0
n_fail = 0

for space in SPACES:
    for seed in SEEDS:
        # Run twice with identical inputs
        r1 = pipe.run(space, seed=seed, budget_fraction=BUDGET)
        r2 = pipe.run(space, seed=seed, budget_fraction=BUDGET)

        # Compare final states bitwise
        match = np.array_equal(r1.final_state, r2.final_state)

        # Also compare split decisions
        decisions_match = (r1.n_refined == r2.n_refined and
                          r1.n_passed == r2.n_passed and
                          r1.n_rejected == r2.n_rejected)

        overall = match and decisions_match

        if overall:
            n_pass += 1
        else:
            n_fail += 1
            print(f"FAIL: {space} seed={seed}: state_match={match}, decisions_match={decisions_match}")

        results.append({
            "space": space, "seed": seed,
            "state_bitwise_match": bool(match),
            "decisions_match": bool(decisions_match),
            "overall": bool(overall),
        })

total = n_pass + n_fail
summary = {
    "total": total,
    "pass": n_pass,
    "fail": n_fail,
    "pass_rate": f"{n_pass}/{total}",
    "verdict": "PASS" if n_fail == 0 else "FAIL",
}

with open(RESULTS_DIR / "det1_recheck.json", "w") as f:
    json.dump({"results": results, "summary": summary}, f, indent=2)

print(f"\nDET-1 Recheck: {summary['pass_rate']} — {summary['verdict']}")
