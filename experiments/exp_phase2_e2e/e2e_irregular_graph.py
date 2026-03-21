#!/usr/bin/env python3
"""E2E validation: irregular_graph — 20 seeds x 3 budgets."""
import sys, json, time
import numpy as np
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent.parent / "exp_phase2_pipeline"
sys.path.insert(0, str(PIPELINE_DIR))
from pipeline import CuriosityPipeline
from config import PipelineConfig

SPACE = "irregular_graph"
SEEDS = list(range(20))
BUDGETS = [0.10, 0.30, 0.60]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

pipe = CuriosityPipeline(PipelineConfig())
results = []
for seed in SEEDS:
    for budget in BUDGETS:
        try:
            r = pipe.run(SPACE, seed=seed, budget_fraction=budget)
            rec = {
                "space": SPACE, "seed": seed, "budget": budget,
                "psnr_coarse": round(r.coarse_psnr, 4),
                "psnr_final": round(r.quality_psnr, 4),
                "psnr_gain": round(r.quality_psnr - r.coarse_psnr, 4),
                "mse_coarse": r.coarse_mse, "mse_final": r.quality_mse,
                "n_refined": r.n_refined, "n_total": r.n_total,
                "n_passed": r.n_passed, "n_damped": r.n_damped, "n_rejected": r.n_rejected,
                "reject_rate": round(r.reject_rate, 4),
                "waste_exhausted": r.waste_budget_exhausted,
                "gate_stage": r.gate_stage,
                "wall_time": round(r.wall_time_seconds, 4),
                # Topo profiling fields (irregular_graph only)
                "topo_zone": getattr(r, 'topo_zone', None),
                "topo_eta_f": getattr(r, 'topo_eta_f', None),
                "topo_tau_factor": getattr(r, 'topo_tau_factor', None),
                "topo_computation_ms": getattr(r, 'topo_computation_ms', None),
            }
            results.append(rec)
            fname = f"{SPACE}_seed{seed:02d}_budget{int(budget*100):03d}.json"
            with open(RESULTS_DIR / fname, "w") as f:
                json.dump(rec, f, indent=2)
        except Exception as e:
            print(f"FAIL: seed={seed} budget={budget}: {e}")
            results.append({"space": SPACE, "seed": seed, "budget": budget, "error": str(e)})

psnr_gains = [r["psnr_gain"] for r in results if "psnr_gain" in r]
reject_rates = [r["reject_rate"] for r in results if "reject_rate" in r]
wall_times = [r["wall_time"] for r in results if "wall_time" in r]
topo_zones = [r.get("topo_zone") for r in results if r.get("topo_zone")]
topo_eta_fs = [r["topo_eta_f"] for r in results if r.get("topo_eta_f") is not None]
topo_tau_factors = [r["topo_tau_factor"] for r in results if r.get("topo_tau_factor") is not None]
topo_times = [r["topo_computation_ms"] for r in results if r.get("topo_computation_ms") is not None]
summary = {
    "space": SPACE, "n_configs": len(results),
    "n_errors": sum(1 for r in results if "error" in r),
    "psnr_gain_median": round(float(np.median(psnr_gains)), 4) if psnr_gains else None,
    "psnr_gain_iqr": [round(float(np.percentile(psnr_gains, 25)), 4),
                      round(float(np.percentile(psnr_gains, 75)), 4)] if psnr_gains else None,
    "all_psnr_positive": all(g > 0 for g in psnr_gains) if psnr_gains else False,
    "reject_rate_median": round(float(np.median(reject_rates)), 4) if reject_rates else None,
    "reject_rate_max": round(float(max(reject_rates)), 4) if reject_rates else None,
    "wall_time_max": round(float(max(wall_times)), 4) if wall_times else None,
    "wall_time_median": round(float(np.median(wall_times)), 4) if wall_times else None,
    # Topo profiling summary
    "topo_zone_counts": {z: topo_zones.count(z) for z in set(topo_zones)} if topo_zones else None,
    "topo_eta_f_median": round(float(np.median(topo_eta_fs)), 4) if topo_eta_fs else None,
    "topo_tau_factor_median": round(float(np.median(topo_tau_factors)), 4) if topo_tau_factors else None,
    "topo_computation_ms_median": round(float(np.median(topo_times)), 2) if topo_times else None,
    "topo_computation_ms_max": round(float(max(topo_times)), 2) if topo_times else None,
}
with open(RESULTS_DIR / f"summary_{SPACE}.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n=== {SPACE} E2E Summary ===")
for k, v in summary.items():
    print(f"  {k}: {v}")
print(f"\nKill criteria:")
print(f"  PSNR > 0 dB: {'PASS' if summary['all_psnr_positive'] else 'FAIL'}")
print(f"  Reject rate < 5%: {'PASS' if (summary['reject_rate_max'] or 0) < 0.05 else 'FAIL'}")
print(f"  Runtime < 60s: {'PASS' if (summary['wall_time_max'] or 0) < 60 else 'FAIL'}")
