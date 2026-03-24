#!/usr/bin/env python3
"""
exp19b — WeightedRhoGate stress test.

Question: Do EMA weights converge under perturbations?
Tests gate behavior under 4 instability profiles:
  - stable: instability constant ~0.1
  - ramp: instability 0.1 → 0.8 linearly
  - step: instability jumps 0.1 → 0.8 at tick 5
  - oscillating: instability alternates 0.1 / 0.8

Sweep: 4 profiles x 4 alpha values x 10 seeds = 160 configs.
Kill: weight variance over last 3 ticks < 0.05.
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENTS_DIR / "exp_phase2_pipeline"))

from pipeline import WeightedRhoGate, TickState, PipelineConfig


# =====================================================================
# Instability profiles
# =====================================================================

def profile_stable(tick: int, n_ticks: int) -> float:
    return 0.1

def profile_ramp(tick: int, n_ticks: int) -> float:
    return 0.1 + 0.7 * (tick / max(n_ticks - 1, 1))

def profile_step(tick: int, n_ticks: int) -> float:
    return 0.1 if tick < n_ticks // 2 else 0.8

def profile_oscillating(tick: int, n_ticks: int) -> float:
    return 0.1 if tick % 2 == 0 else 0.8

PROFILES = {
    "stable": profile_stable,
    "ramp": profile_ramp,
    "step": profile_step,
    "oscillating": profile_oscillating,
}

ALPHA_VALUES = [0.1, 0.2, 0.3, 0.5]
N_TICKS = 15
N_SEEDS = 10


# =====================================================================
# Runner
# =====================================================================

def simulate_gate(profile_name: str, alpha: float, seed: int) -> dict:
    """Simulate gate over N_TICKS with given instability profile."""
    profile_fn = PROFILES[profile_name]

    cfg = PipelineConfig(
        max_ticks=N_TICKS,
        ema_weight_alpha=alpha,
        pilot_ticks=3,
        pilot_thresh_factor=0.7,
        pilot_fsr_floor=0.05,
    )
    gate = WeightedRhoGate(cfg)
    tick_state = TickState()

    # Cold-start thresholds (simulate for "n/a" topo zone)
    tick_state.calibrated_instab_thresh = 0.25
    tick_state.calibrated_fsr_thresh = 0.05

    history = []
    for tick in range(N_TICKS):
        instability = profile_fn(tick, N_TICKS)
        fsr = instability * 0.3  # correlated FSR

        info = gate.evaluate(fsr, instability, tick_state, tick, probe_median_gain=0.5)
        history.append({
            "tick": tick,
            "instability": instability,
            "fsr": fsr,
            "w_resid": tick_state.w_resid,
            "w_hf": tick_state.w_hf,
            "w_var": tick_state.w_var,
            "stage": info["stage"],
        })

    # Analyze convergence: variance of weights in last 3 ticks
    last3 = history[-3:]
    w_resid_var = np.var([h["w_resid"] for h in last3])
    w_hf_var = np.var([h["w_hf"] for h in last3])
    w_var_var = np.var([h["w_var"] for h in last3])
    max_weight_var = max(w_resid_var, w_hf_var, w_var_var)

    converged = max_weight_var < 0.05
    final_weights = {
        "w_resid": history[-1]["w_resid"],
        "w_hf": history[-1]["w_hf"],
        "w_var": history[-1]["w_var"],
    }

    return {
        "profile": profile_name,
        "alpha": alpha,
        "seed": seed,
        "converged": converged,
        "max_weight_variance_last3": float(max_weight_var),
        "final_weights": final_weights,
        "n_healthy_ticks": sum(1 for h in history if h["stage"] == "healthy"),
        "n_combo_ticks": sum(1 for h in history if h["stage"] == "combo"),
        "history": history,
    }


def main():
    print("=" * 70)
    print("exp19b — WeightedRhoGate stress test")
    print(f"  profiles: {list(PROFILES.keys())}")
    print(f"  alpha values: {ALPHA_VALUES}")
    print(f"  n_ticks: {N_TICKS}, seeds: {N_SEEDS}")
    total = len(PROFILES) * len(ALPHA_VALUES) * N_SEEDS
    print(f"  total configs: {total}")
    print("=" * 70)

    results = []
    for profile_name in PROFILES:
        for alpha in ALPHA_VALUES:
            profile_results = []
            for seed in range(N_SEEDS):
                r = simulate_gate(profile_name, alpha, seed)
                profile_results.append(r)
                results.append(r)

            # Summary for this (profile, alpha) combo
            n_converged = sum(1 for r in profile_results if r["converged"])
            avg_var = np.mean([r["max_weight_variance_last3"] for r in profile_results])
            avg_w_resid = np.mean([r["final_weights"]["w_resid"] for r in profile_results])
            print(f"  [{profile_name}/alpha={alpha}] "
                  f"converged: {n_converged}/{N_SEEDS}  "
                  f"avg_var: {avg_var:.4f}  "
                  f"avg_w_resid: {avg_w_resid:.3f}")

    # Save
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save summary (without per-tick history)
    def _json_safe(obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return obj

    summary = []
    for r in results:
        s = {}
        for k, v in r.items():
            if k == "history":
                continue
            s[k] = _json_safe(v)
        summary.append(s)
    with open(out_dir / "exp19b_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save full results with history (using custom encoder for numpy types)
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

    with open(out_dir / "exp19b_full.json", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    # Kill criteria check
    all_converged = all(r["converged"] for r in results)
    oscillating_converged = all(
        r["converged"] for r in results if r["profile"] == "oscillating"
    )
    print(f"\n  Kill: all converged = {all_converged}")
    print(f"  Kill: oscillating converged = {oscillating_converged}")
    if not all_converged:
        failed = [r for r in results if not r["converged"]]
        for r in failed[:5]:
            print(f"    FAIL: {r['profile']}/alpha={r['alpha']}/seed={r['seed']} "
                  f"var={r['max_weight_variance_last3']:.4f}")

    print("\n" + "=" * 70)
    print("  exp19b COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
