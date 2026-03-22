#!/usr/bin/env python3
"""
Enox integration comparison test.

Runs the pipeline in a standardized config (4 spaces x 5 seeds x budget=0.30)
and saves a fingerprint JSON. Run BEFORE and AFTER Enox integration, then
compare with --compare flag.

Usage:
    python enox_comparison.py --tag baseline     # before Enox
    python enox_comparison.py --tag enox         # after Enox
    python enox_comparison.py --compare baseline enox
"""

import sys
import json
import time
import hashlib
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import CuriosityPipeline
from config import PipelineConfig

SPACES = ["scalar_grid", "vector_grid", "irregular_graph", "tree_hierarchy"]
SEEDS = [0, 1, 2, 7, 42]
BUDGET = 0.30
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def state_hash(arr: np.ndarray) -> str:
    """SHA256 of array bytes for bitwise comparison."""
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def run_fingerprint(tag: str):
    """Run pipeline on standard config and save fingerprint."""
    RESULTS_DIR.mkdir(exist_ok=True)
    pipe = CuriosityPipeline(PipelineConfig())

    records = []
    t_total = time.time()

    for space in SPACES:
        for seed in SEEDS:
            r = pipe.run(space, seed=seed, budget_fraction=BUDGET)
            rec = {
                "space": space,
                "seed": seed,
                "psnr_coarse": round(r.coarse_psnr, 4),
                "psnr_final": round(r.quality_psnr, 4),
                "psnr_gain": round(r.quality_psnr - r.coarse_psnr, 4),
                "mse_final": r.quality_mse,
                "n_refined": r.n_refined,
                "n_total": r.n_total,
                "n_passed": r.n_passed,
                "n_damped": r.n_damped,
                "n_rejected": r.n_rejected,
                "reject_rate": round(r.reject_rate, 4),
                "gate_stage": r.gate_stage,
                "waste_exhausted": r.waste_budget_exhausted,
                "wall_time": round(r.wall_time_seconds, 6),
                "state_hash": state_hash(r.final_state),
                "topo_zone": r.topo_zone,
                "topo_eta_f": round(r.topo_eta_f, 4) if r.topo_eta_f else 0.0,
                "topo_tau_factor": r.topo_tau_factor,
            }
            records.append(rec)
            print(f"  [{space:18s} seed={seed:2d}] "
                  f"PSNR +{rec['psnr_gain']:6.2f} dB  "
                  f"reject={rec['reject_rate']:.2f}  "
                  f"t={rec['wall_time']:.4f}s  "
                  f"hash={rec['state_hash']}")

    # DET-1 spot check: run seed=0 twice for each space
    det_results = []
    for space in SPACES:
        r1 = pipe.run(space, seed=0, budget_fraction=BUDGET)
        r2 = pipe.run(space, seed=0, budget_fraction=BUDGET)
        match = np.array_equal(r1.final_state, r2.final_state)
        det_results.append({"space": space, "bitwise_match": match})
        status = "PASS" if match else "FAIL"
        print(f"  DET-1 {space}: {status}")

    elapsed = time.time() - t_total

    # Aggregate stats
    psnr_gains = [r["psnr_gain"] for r in records]
    wall_times = [r["wall_time"] for r in records]

    output = {
        "tag": tag,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_runs": len(records),
        "budget": BUDGET,
        "seeds": SEEDS,
        "aggregate": {
            "psnr_gain_median": round(float(np.median(psnr_gains)), 4),
            "psnr_gain_mean": round(float(np.mean(psnr_gains)), 4),
            "psnr_gain_min": round(float(min(psnr_gains)), 4),
            "wall_time_median": round(float(np.median(wall_times)), 6),
            "wall_time_mean": round(float(np.mean(wall_times)), 6),
            "wall_time_max": round(float(max(wall_times)), 6),
            "all_psnr_positive": all(g > 0 for g in psnr_gains),
            "any_rejects": any(r["reject_rate"] > 0 for r in records),
            "det1_all_pass": all(d["bitwise_match"] for d in det_results),
        },
        "det1": det_results,
        "runs": records,
        "total_elapsed_s": round(elapsed, 2),
    }

    out_path = RESULTS_DIR / f"enox_comparison_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Saved: {out_path}")
    print(f"  Total: {elapsed:.1f}s, {len(records)} runs")
    print(f"  PSNR gain median: +{output['aggregate']['psnr_gain_median']:.2f} dB")
    print(f"  Wall time median: {output['aggregate']['wall_time_median']*1000:.1f}ms")
    print(f"  DET-1: {'PASS' if output['aggregate']['det1_all_pass'] else 'FAIL'}")
    print(f"  All PSNR positive: {output['aggregate']['all_psnr_positive']}")
    return output


def compare(tag_a: str, tag_b: str):
    """Compare two fingerprint files."""
    path_a = RESULTS_DIR / f"enox_comparison_{tag_a}.json"
    path_b = RESULTS_DIR / f"enox_comparison_{tag_b}.json"

    if not path_a.exists():
        print(f"ERROR: {path_a} not found. Run with --tag {tag_a} first.")
        return
    if not path_b.exists():
        print(f"ERROR: {path_b} not found. Run with --tag {tag_b} first.")
        return

    with open(path_a) as f:
        a = json.load(f)
    with open(path_b) as f:
        b = json.load(f)

    print("=" * 75)
    print(f"COMPARISON: {tag_a} vs {tag_b}")
    print("=" * 75)

    # Aggregate comparison
    agg_a = a["aggregate"]
    agg_b = b["aggregate"]

    print(f"\n  {'Metric':<30s}  {'[' + tag_a + ']':>12s}  {'[' + tag_b + ']':>12s}  {'Delta':>12s}")
    print("  " + "-" * 70)

    for key in ["psnr_gain_median", "psnr_gain_mean", "psnr_gain_min",
                "wall_time_median", "wall_time_mean", "wall_time_max"]:
        va = agg_a[key]
        vb = agg_b[key]
        if "wall_time" in key:
            delta_str = f"{(vb - va)*1000:+.2f}ms"
            print(f"  {key:<30s}  {va*1000:10.2f}ms  {vb*1000:10.2f}ms  {delta_str:>12s}")
        else:
            delta_str = f"{vb - va:+.4f}"
            print(f"  {key:<30s}  {va:12.4f}  {vb:12.4f}  {delta_str:>12s}")

    for key in ["all_psnr_positive", "any_rejects", "det1_all_pass"]:
        va = agg_a[key]
        vb = agg_b[key]
        status = "same" if va == vb else "CHANGED"
        print(f"  {key:<30s}  {str(va):>12s}  {str(vb):>12s}  {status:>12s}")

    # Per-run hash comparison
    runs_a = {(r["space"], r["seed"]): r for r in a["runs"]}
    runs_b = {(r["space"], r["seed"]): r for r in b["runs"]}

    hash_matches = 0
    hash_mismatches = 0
    psnr_diffs = []

    print(f"\n  Per-run comparison:")
    print(f"  {'space':18s} {'seed':>4s}  {'hash_match':>10s}  {'PSNR_a':>8s}  {'PSNR_b':>8s}  {'delta':>8s}")
    print("  " + "-" * 60)

    for key in sorted(runs_a.keys()):
        if key not in runs_b:
            continue
        ra = runs_a[key]
        rb = runs_b[key]
        hmatch = ra["state_hash"] == rb["state_hash"]
        if hmatch:
            hash_matches += 1
        else:
            hash_mismatches += 1
        psnr_d = rb["psnr_gain"] - ra["psnr_gain"]
        psnr_diffs.append(psnr_d)
        flag = "  " if hmatch else " *"
        print(f"  {key[0]:18s} {key[1]:4d}  "
              f"{'SAME' if hmatch else 'DIFF':>10s}  "
              f"{ra['psnr_gain']:8.4f}  {rb['psnr_gain']:8.4f}  "
              f"{psnr_d:+8.4f}{flag}")

    print(f"\n  Hash: {hash_matches} same, {hash_mismatches} different")
    if psnr_diffs:
        print(f"  PSNR delta: median={np.median(psnr_diffs):+.4f}, "
              f"mean={np.mean(psnr_diffs):+.4f}, "
              f"max_abs={max(abs(d) for d in psnr_diffs):.4f}")

    # Verdict
    print(f"\n  {'=' * 50}")
    regression = False
    if not agg_b["all_psnr_positive"]:
        print("  REGRESSION: not all PSNR positive!")
        regression = True
    if not agg_b["det1_all_pass"]:
        print("  REGRESSION: DET-1 failed!")
        regression = True
    if agg_b["psnr_gain_median"] < agg_a["psnr_gain_median"] - 0.5:
        print(f"  REGRESSION: PSNR median dropped by "
              f"{agg_a['psnr_gain_median'] - agg_b['psnr_gain_median']:.2f} dB")
        regression = True

    if regression:
        print("  VERDICT: REGRESSION DETECTED")
    else:
        print("  VERDICT: NO REGRESSION")
    print(f"  {'=' * 50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enox comparison test")
    parser.add_argument("--tag", help="Run fingerprint with this tag")
    parser.add_argument("--compare", nargs=2, metavar=("TAG_A", "TAG_B"),
                        help="Compare two tags")
    args = parser.parse_args()

    if args.tag:
        run_fingerprint(args.tag)
    elif args.compare:
        compare(args.compare[0], args.compare[1])
    else:
        parser.print_help()
