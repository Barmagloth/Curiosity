#!/usr/bin/env python3
"""
exp14 — P1-B3: Anchors + Periodic Rebuild.

Compares three strategies for tree maintenance across evolving data:
  A) No rebuild (baseline): start from previous state, refine only dirty units.
  B) Periodic rebuild: full from-coarse pipeline every K steps (K=5,10,20,50).
  C) Dirty-triggered rebuild: full rebuild when >X% of units are dirty.

"Evolving data": each step shifts seed by +1, creating incremental GT change.
Key insight: the pipeline always starts from coarse. A "local update" means
starting from the PREVIOUS final_state and only refining dirty units.

Divergence metric:
  div = ||state_local - state_fullrebuild|| / (||state_fullrebuild|| + eps)

Kill criteria: divergence < 0.05 → local updates sufficient.

4 spaces × 20 seeds × (1 baseline + 4 periodic + 4 dirty) = 720 configs.
Reduced to N_STEPS=20 to fit timeout.

Usage:
  python exp14_anchors.py                 # run all
  python exp14_anchors.py --chunk 0 4     # run chunk 0 of 4
"""

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict

# --- path setup ---
EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENTS_DIR / "exp_phase2_pipeline"))
sys.path.insert(0, str(EXPERIMENTS_DIR / "exp10d_seed_determinism"))
sys.path.insert(0, str(EXPERIMENTS_DIR / "exp13_segment_compression"))
sys.path.insert(0, str(EXPERIMENTS_DIR / "sc_baseline"))
sys.path.insert(0, str(EXPERIMENTS_DIR / "exp12a_tau_parent"))
sys.path.insert(0, str(EXPERIMENTS_DIR / "exp14a_sc_enforce"))

from config import PipelineConfig
from pipeline import CuriosityPipeline
from space_registry import SPACE_FACTORIES
from segment_compress import hamming12, component_diff

# ===================================================================
# Constants
# ===================================================================

SPACE_TYPES = ["scalar_grid", "vector_grid", "irregular_graph", "tree_hierarchy"]
N_SEEDS = 20
N_STEPS = 20
PERIODIC_K = [5, 10, 20, 50]
DIRTY_X = [0.05, 0.10, 0.20, 0.50]
BUDGET = 0.30

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ===================================================================
# Dirty signature (12-bit)
# ===================================================================

def compute_sig(space, state, unit, rho_val: float) -> int:
    """12-bit dirty signature for a unit."""
    seam = min(15, int(rho_val * 30))

    if hasattr(unit, '__len__'):
        ti, tj = unit
        T = space.T
        s = slice(ti * T, (ti + 1) * T)
        cs = slice(tj * T, (tj + 1) * T)
        local_diff = state[s, cs] - space.coarse[s, cs] if state.ndim == 2 else \
                     state[s, cs, :] - space.coarse[s, cs, :]
    elif isinstance(unit, int) and hasattr(space, 'labels'):
        pts = np.where(space.labels == unit)[0]
        local_diff = state[pts] - space.coarse[pts] if len(pts) > 0 else np.array([0.0])
    elif isinstance(unit, int):
        local_diff = np.array([state[unit] - space.coarse[unit]])
    else:
        local_diff = np.array([0.0])

    uncert = min(15, int(float(np.var(local_diff)) * 150))
    mass = min(15, int(float(np.mean(np.abs(local_diff))) * 30))
    return (seam << 8) | (uncert << 4) | mass


def is_dirty(sig_now: int, sig_baseline: int) -> bool:
    """Check if signature changed enough to be considered dirty."""
    return hamming12(sig_now, sig_baseline) >= 3 or component_diff(sig_now, sig_baseline) >= 4


# ===================================================================
# Local update: refine only dirty units from previous state
# ===================================================================

def local_update(space, prev_state: np.ndarray, units: list,
                 dirty_set: set, budget_fraction: float) -> np.ndarray:
    """Refine only dirty units from previous state.

    Unlike pipe.run() which starts from coarse, this starts from prev_state
    and only updates units in dirty_set.
    """
    state = prev_state.copy()
    n_budget = max(1, int(budget_fraction * len(units)))

    # Compute rho on carried state with new GT
    rho_values = np.array([space.unit_rho(state, u) for u in units])

    # Sort by descending rho (same as canonical traversal)
    order = np.argsort(-rho_values)

    refined = 0
    for idx in order:
        if refined >= n_budget:
            break
        u = units[idx]
        u_str = str(u)

        # Only refine if dirty OR in top rho
        if u_str not in dirty_set and rho_values[idx] < np.median(rho_values):
            continue

        # Refine
        if hasattr(u, '__len__'):
            state_new = space.refine_unit(state, u, halo=2)
        elif isinstance(u, int) and hasattr(space, 'labels'):
            state_new = space.refine_unit(state, u, halo_hops=1)
        elif isinstance(u, int):
            state_new = space.refine_unit(state, u, halo_hops=0)
        else:
            continue

        state = state_new
        refined += 1

    return state


# ===================================================================
# Single evolution run
# ===================================================================

@dataclass
class RunResult:
    space_type: str
    seed_base: int
    strategy: str
    param: float
    n_steps: int
    divergences: List[float]
    final_divergence: float
    mean_divergence: float
    max_divergence: float
    total_rebuilds: int
    psnr_local: List[float]
    psnr_full: List[float]
    dirty_fractions: List[float]
    total_wall_seconds: float


def run_evolution(space_type: str, seed_base: int,
                  strategy: str, param: float) -> RunResult:
    """Run N_STEPS of evolving data with a given rebuild strategy."""
    t0 = time.time()

    pipe = CuriosityPipeline(config=PipelineConfig(budget_fraction=BUDGET))

    divergences = []
    psnr_local_list = []
    psnr_full_list = []
    dirty_fracs = []
    total_rebuilds = 0

    # State tracking
    carried_state = None
    baseline_sigs: Dict[str, int] = {}

    for step in range(N_STEPS):
        seed = seed_base + step

        # --- Always compute full rebuild (ground truth) ---
        result_full = pipe.run(space_type, seed=seed, budget_fraction=BUDGET)
        full_state = result_full.final_state

        # --- Setup space for this step ---
        space = SPACE_FACTORIES[space_type]()
        space.setup(seed)
        units = space.get_units()

        # --- Determine dirty set ---
        dirty_set = set()
        if carried_state is not None and baseline_sigs:
            for u in units:
                u_str = str(u)
                rho = space.unit_rho(carried_state, u)
                sig_now = compute_sig(space, carried_state, u, rho)
                if u_str in baseline_sigs and is_dirty(sig_now, baseline_sigs[u_str]):
                    dirty_set.add(u_str)

        dirty_frac = len(dirty_set) / max(len(units), 1)
        dirty_fracs.append(dirty_frac)

        # --- Decide: rebuild or local update ---
        do_rebuild = False
        if step == 0:
            do_rebuild = True
        elif strategy == "no_rebuild":
            do_rebuild = False
        elif strategy.startswith("periodic"):
            K = int(param)
            if step % K == 0:
                do_rebuild = True
        elif strategy.startswith("dirty"):
            if dirty_frac > param:
                do_rebuild = True

        # --- Execute ---
        if do_rebuild:
            total_rebuilds += 1
            carried_state = full_state.copy()
        else:
            # LOCAL UPDATE: start from previous state, refine dirty units
            carried_state = local_update(
                space, carried_state, units, dirty_set, BUDGET)

        # --- Divergence ---
        norm = float(np.linalg.norm(full_state)) + 1e-12
        div = float(np.linalg.norm(carried_state - full_state)) / norm
        divergences.append(div)

        # --- PSNR ---
        gt = space.gt
        mse_local = float(np.mean((gt - carried_state) ** 2))
        mse_full = float(np.mean((gt - full_state) ** 2))
        data_range = float(gt.max() - gt.min()) + 1e-12
        psnr_l = 10.0 * np.log10(data_range**2 / max(mse_local, 1e-15))
        psnr_f = 10.0 * np.log10(data_range**2 / max(mse_full, 1e-15))
        psnr_local_list.append(float(psnr_l))
        psnr_full_list.append(float(psnr_f))

        # --- Update baseline signatures ---
        baseline_sigs = {}
        for u in units:
            rho = space.unit_rho(carried_state, u)
            baseline_sigs[str(u)] = compute_sig(space, carried_state, u, rho)

    wall = time.time() - t0
    divs = np.array(divergences)

    return RunResult(
        space_type=space_type,
        seed_base=seed_base,
        strategy=strategy,
        param=param,
        n_steps=N_STEPS,
        divergences=divergences,
        final_divergence=float(divs[-1]),
        mean_divergence=float(divs.mean()),
        max_divergence=float(divs.max()),
        total_rebuilds=total_rebuilds,
        psnr_local=psnr_local_list,
        psnr_full=psnr_full_list,
        dirty_fractions=dirty_fracs,
        total_wall_seconds=wall,
    )


# ===================================================================
# Build configs
# ===================================================================

def build_configs():
    configs = []
    for st in SPACE_TYPES:
        for si in range(N_SEEDS):
            seed_base = 1000 + si * 100
            configs.append((st, seed_base, "no_rebuild", 0.0))
            for K in PERIODIC_K:
                configs.append((st, seed_base, f"periodic_{K}", float(K)))
            for X in DIRTY_X:
                configs.append((st, seed_base, f"dirty_{X}", X))
    return configs


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="exp14: Anchors + Periodic Rebuild")
    parser.add_argument("--chunk", nargs=2, type=int, default=None)
    args = parser.parse_args()

    configs = build_configs()
    print(f"Total configs: {len(configs)}")

    if args.chunk is not None:
        ci, nc = args.chunk
        cs = (len(configs) + nc - 1) // nc
        configs = configs[ci*cs : min((ci+1)*cs, len(configs))]
        print(f"Running chunk {ci}/{nc}: {len(configs)} configs")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for i, (st, sb, strat, param) in enumerate(configs):
        print(f"  [{i+1}/{len(configs)}] {st} seed={sb} {strat}", end=" ", flush=True)
        try:
            r = run_evolution(st, sb, strat, param)
            results.append(asdict(r))
            print(f"max_div={r.max_divergence:.4f} rebuilds={r.total_rebuilds} "
                  f"dirty_max={max(r.dirty_fractions):.2f} ({r.total_wall_seconds:.1f}s)")
        except Exception as e:
            print(f"FAILED: {e}")
            results.append({"space_type": st, "seed_base": sb,
                            "strategy": strat, "param": param, "error": str(e)})

        # Incremental save
        if (i + 1) % 20 == 0 or i == len(configs) - 1:
            sfx = f"_chunk{args.chunk[0]}" if args.chunk else ""
            out = RESULTS_DIR / f"exp14_results{sfx}.json"
            with open(out, "w") as f:
                json.dump(results, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("exp14 Summary — Anchors + Periodic Rebuild")
    print("=" * 70)
    strategies = ["no_rebuild"] + \
        [f"periodic_{K}" for K in PERIODIC_K] + \
        [f"dirty_{X}" for X in DIRTY_X]
    for strat in strategies:
        sub = [r for r in results if r.get("strategy") == strat and "error" not in r]
        if sub:
            md = [r["max_divergence"] for r in sub]
            ad = [r["mean_divergence"] for r in sub]
            rb = [r["total_rebuilds"] for r in sub]
            print(f"  {strat:20s}: max_div={np.max(md):.4f}  mean_div={np.mean(ad):.4f}  "
                  f"rebuilds={np.mean(rb):.1f}  PASS={np.max(md) < 0.05}")
    print("=" * 70)


if __name__ == "__main__":
    main()
