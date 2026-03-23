#!/usr/bin/env python3
"""
exp18 -- Basin Membership vs Feature Similarity.

Hypothesis: the Curiosity tree is an RG-flow, not a metric space.
Proximity = "same basin of attraction" (convergence to same fixed point),
not "small tree distance" (which failed in exp15, Spearman < 0.3).

For each space type x seed:
  1. Run the pipeline to get refined state + rho trajectory
  2. Identify "basins" -- groups of units that converge to the same
     fixed-point ancestor (a node whose children gain < tau_gain for
     K consecutive refinement steps)
  3. For each random pair (i, j):
       same_basin(i,j) = 1 if same fixed-point ancestor, 0 otherwise
       feature_similarity(i,j) = 1 - ||f_i - f_j|| / max_dist
  4. Point-biserial correlation between same_basin and feature_similarity

Kill criteria: point-biserial r > 0.3 (same threshold as exp15).

4 spaces x 20 seeds = 80 configs. 1000 random pairs per run.
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from scipy import stats as sp_stats

# --- path setup ---
EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENTS_DIR / "exp_phase2_pipeline"))
sys.path.insert(0, str(EXPERIMENTS_DIR / "exp10d_seed_determinism"))

from config import PipelineConfig
from pipeline import CuriosityPipeline
from space_registry import SPACE_FACTORIES

# ===================================================================
# Constants
# ===================================================================

SPACE_TYPES = ["scalar_grid", "vector_grid", "irregular_graph", "tree_hierarchy"]
N_SEEDS = 20
N_PAIRS = 1000
BUDGET = 0.30

# Fixed-point detection: a node is a "fixed point" if refinement gain
# drops below this fraction of parent's rho (gain negligible).
FIXED_POINT_RATIO = 0.10  # child_rho < 0.1 * parent_rho

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ===================================================================
# Basin identification
# ===================================================================

def compute_rho_per_unit(space, state_initial, state_final, units):
    """Compute initial and final rho for each unit."""
    rho_initial = np.array([space.unit_rho(state_initial, u) for u in units])
    rho_final = np.array([space.unit_rho(state_final, u) for u in units])
    return rho_initial, rho_final


def identify_basins_grid(space, rho_initial, rho_final, units, space_type):
    """Identify basins for grid spaces via quadtree aggregation.

    Strategy: build a virtual quadtree. A subtree is a "basin" if the
    rho-ratio (final/initial) of all its leaves is below FIXED_POINT_RATIO,
    meaning they've all converged to the same fixed point.

    Fallback: cluster by rho_final magnitude into quantile-based groups.
    """
    NT = space.NT
    n_units = len(units)

    # Compute refinement gain ratio per unit
    gain_ratio = np.zeros(n_units)
    for i in range(n_units):
        if rho_initial[i] > 1e-12:
            gain_ratio[i] = rho_final[i] / rho_initial[i]
        else:
            gain_ratio[i] = 0.0  # already converged

    # Build quadtree: group by 2x2 super-tiles, then 4x4, etc.
    # Each grouping level: tiles that share the same super-tile ancestor.
    # A basin = the coarsest level where ALL children have gain_ratio < threshold.

    # Map each unit to its grid position
    unit_to_idx = {str(u): i for i, u in enumerate(units)}
    pos_grid = {}
    for u in units:
        ti, tj = u
        pos_grid[str(u)] = (ti, tj)

    # Try hierarchical grouping at multiple scales
    best_basins = {}
    for level in range(1, 5):  # 2^level grouping
        scale = 2 ** level
        if scale > NT:
            break

        groups = {}
        for u in units:
            ti, tj = u
            gi, gj = ti // scale, tj // scale
            key = (level, gi, gj)
            if key not in groups:
                groups[key] = []
            groups[key].append(str(u))

        # Check each group: is it a "basin" (all members converged)?
        for key, members in groups.items():
            ratios = [gain_ratio[unit_to_idx[m]] for m in members]
            # Basin if median gain ratio is low (most units converged)
            if np.median(ratios) < FIXED_POINT_RATIO:
                for m in members:
                    if m not in best_basins:
                        best_basins[m] = key

    # Assign remaining units to singleton basins
    basin_labels = {}
    basin_counter = 0
    key_to_label = {}
    for u in units:
        u_str = str(u)
        if u_str in best_basins:
            key = best_basins[u_str]
            if key not in key_to_label:
                key_to_label[key] = basin_counter
                basin_counter += 1
            basin_labels[u_str] = key_to_label[key]
        else:
            # Fallback: use rho_final quantile as basin
            idx = unit_to_idx[u_str]
            basin_labels[u_str] = basin_counter
            basin_counter += 1

    return basin_labels


def identify_basins_tree(space, rho_initial, rho_final, units):
    """Identify basins for tree_hierarchy space.

    Direct tree structure: walk up from each leaf, find the first ancestor
    where the subtree rho drops below the fixed-point threshold.
    """
    n_units = len(units)

    # Compute gain ratio
    gain_ratio = np.zeros(n_units)
    for i in range(n_units):
        if rho_initial[i] > 1e-12:
            gain_ratio[i] = rho_final[i] / rho_initial[i]
        else:
            gain_ratio[i] = 0.0

    # For tree: group by parent at various depths
    # A "basin" = subtree whose root's children all have low gain
    basin_labels = {}
    # Try grouping at each depth level
    for depth_cut in range(1, 8):
        # Ancestor at depth_cut levels up
        for i, u in enumerate(units):
            u_str = str(u)
            if u_str in basin_labels:
                continue
            ancestor = u
            for _ in range(depth_cut):
                if ancestor > 0:
                    ancestor = (ancestor - 1) // 2
            # Check if all siblings under this ancestor have low gain
            # (simplified: just check this unit's gain)
            if gain_ratio[i] < FIXED_POINT_RATIO:
                basin_labels[u_str] = f"basin_{ancestor}"

    # Remaining: singleton
    for i, u in enumerate(units):
        u_str = str(u)
        if u_str not in basin_labels:
            basin_labels[u_str] = f"singleton_{u}"

    # Convert to integer labels
    label_map = {}
    counter = 0
    int_labels = {}
    for u_str, lab in basin_labels.items():
        if lab not in label_map:
            label_map[lab] = counter
            counter += 1
        int_labels[u_str] = label_map[lab]

    return int_labels


def identify_basins_graph(space, rho_initial, rho_final, units):
    """Identify basins for irregular_graph space.

    Use community structure + convergence: units in the same community
    that have both converged (low gain ratio) share a basin.
    """
    labels = space.labels
    n_units = len(units)

    gain_ratio = np.zeros(n_units)
    for i in range(n_units):
        if rho_initial[i] > 1e-12:
            gain_ratio[i] = rho_final[i] / rho_initial[i]
        else:
            gain_ratio[i] = 0.0

    # Group by community label + convergence status
    basin_labels = {}
    for i, u in enumerate(units):
        u_str = str(u)
        community = int(labels[u]) if hasattr(labels, '__getitem__') else u
        converged = gain_ratio[i] < FIXED_POINT_RATIO
        if converged:
            basin_labels[u_str] = f"c{community}_converged"
        else:
            basin_labels[u_str] = f"c{community}_active"

    # Convert to int labels
    label_map = {}
    counter = 0
    int_labels = {}
    for u_str, lab in basin_labels.items():
        if lab not in label_map:
            label_map[lab] = counter
            counter += 1
        int_labels[u_str] = label_map[lab]

    return int_labels


def identify_basins(space, rho_initial, rho_final, units, space_type):
    """Dispatch to space-specific basin identification."""
    if space_type in ("scalar_grid", "vector_grid"):
        return identify_basins_grid(space, rho_initial, rho_final, units, space_type)
    elif space_type == "tree_hierarchy":
        return identify_basins_tree(space, rho_initial, rho_final, units)
    elif space_type == "irregular_graph":
        return identify_basins_graph(space, rho_initial, rho_final, units)
    else:
        raise ValueError(f"Unknown space type: {space_type}")


# ===================================================================
# Feature extraction (reused from exp15)
# ===================================================================

def extract_leaf_features(space, state, units):
    """Extract feature vectors for each unit from the refined state."""
    features = {}
    for u in units:
        if hasattr(u, '__len__'):
            # Grid
            ti, tj = u
            T = space.T
            s = slice(ti * T, (ti + 1) * T)
            cs = slice(tj * T, (tj + 1) * T)
            if state.ndim == 3:
                feat = state[s, cs, :].ravel()
            else:
                feat = state[s, cs].ravel()
        elif isinstance(u, int):
            if hasattr(space, 'labels'):
                # Graph: cluster mean feature
                pts = np.where(space.labels == u)[0]
                feat = state[pts] if len(pts) > 0 else np.array([0.0])
            else:
                # Tree
                feat = np.array([state[u]])
        else:
            feat = np.array([0.0])
        features[str(u)] = feat
    return features


# ===================================================================
# Single run
# ===================================================================

@dataclass
class BasinRunResult:
    space_type: str
    seed: int
    n_units: int
    n_basins: int
    n_pairs: int
    same_basin_frac: float       # fraction of pairs in same basin
    point_biserial_r: float      # point-biserial correlation
    point_biserial_p: float
    spearman_r: float            # for comparison with exp15
    spearman_p: float
    mean_feat_sim_same: float    # mean feature similarity within basins
    mean_feat_sim_diff: float    # mean feature similarity across basins
    separation: float            # same - diff (effect size)
    wall_seconds: float


def run_basin_analysis(space_type: str, seed: int) -> BasinRunResult:
    """Run one basin-membership analysis."""
    t0 = time.time()

    # 1. Run pipeline
    cfg = PipelineConfig(budget_fraction=BUDGET)
    pipe = CuriosityPipeline(config=cfg)
    result = pipe.run(space_type, seed=seed, budget_fraction=BUDGET)

    # 2. Set up space for feature extraction
    space = SPACE_FACTORIES[space_type]()
    space.setup(seed)
    units = space.get_units()
    n_units = len(units)

    # 3. Compute rho values and identify basins
    rho_initial, rho_final = compute_rho_per_unit(
        space, result.initial_state, result.final_state, units)
    basin_labels = identify_basins(
        space, rho_initial, rho_final, units, space_type)

    n_basins = len(set(basin_labels.values()))

    # 4. Extract features
    features = extract_leaf_features(space, result.final_state, units)

    # 5. Sample random pairs
    rng = np.random.default_rng(seed + 1818)
    n_pairs = min(N_PAIRS, n_units * (n_units - 1) // 2)
    pair_set = set()
    attempts = 0
    while len(pair_set) < n_pairs and attempts < n_pairs * 10:
        i, j = rng.integers(0, n_units, size=2)
        if i != j:
            pair_set.add((min(i, j), max(i, j)))
        attempts += 1

    # 6. Compute same_basin and feature_similarity
    same_basin_arr = []
    feat_sim_arr = []
    unit_strs = [str(u) for u in units]

    # Precompute all feature norms for normalization
    all_dists = []
    sample_pairs_for_max = list(pair_set)[:min(500, len(pair_set))]
    for i, j in sample_pairs_for_max:
        fi = features.get(unit_strs[i], np.array([0.0]))
        fj = features.get(unit_strs[j], np.array([0.0]))
        min_len = min(len(fi), len(fj))
        d = float(np.linalg.norm(fi[:min_len] - fj[:min_len]))
        all_dists.append(d)
    max_dist = max(all_dists) if all_dists and max(all_dists) > 1e-12 else 1.0

    for i, j in pair_set:
        u_str_i = unit_strs[i]
        u_str_j = unit_strs[j]

        # Same basin?
        sb = 1 if basin_labels.get(u_str_i) == basin_labels.get(u_str_j) else 0
        same_basin_arr.append(sb)

        # Feature similarity
        fi = features.get(u_str_i, np.array([0.0]))
        fj = features.get(u_str_j, np.array([0.0]))
        min_len = min(len(fi), len(fj))
        d = float(np.linalg.norm(fi[:min_len] - fj[:min_len]))
        sim = 1.0 - d / max_dist
        feat_sim_arr.append(sim)

    same_basin_arr = np.array(same_basin_arr)
    feat_sim_arr = np.array(feat_sim_arr)

    # 7. Point-biserial correlation
    if (len(same_basin_arr) > 2
            and np.std(same_basin_arr) > 0
            and np.std(feat_sim_arr) > 0
            and len(np.unique(same_basin_arr)) == 2):
        pb_r, pb_p = sp_stats.pointbiserialr(same_basin_arr, feat_sim_arr)
    else:
        pb_r, pb_p = 0.0, 1.0

    # 8. Spearman for comparison
    if (len(same_basin_arr) > 2
            and np.std(same_basin_arr) > 0
            and np.std(feat_sim_arr) > 0):
        sr, sp = sp_stats.spearmanr(same_basin_arr, feat_sim_arr)
    else:
        sr, sp = 0.0, 1.0

    # 9. Effect size: mean similarity within vs across basins
    same_mask = same_basin_arr == 1
    diff_mask = same_basin_arr == 0
    mean_sim_same = float(feat_sim_arr[same_mask].mean()) if same_mask.any() else 0.0
    mean_sim_diff = float(feat_sim_arr[diff_mask].mean()) if diff_mask.any() else 0.0
    separation = mean_sim_same - mean_sim_diff

    same_frac = float(same_basin_arr.mean())
    wall = time.time() - t0

    return BasinRunResult(
        space_type=space_type,
        seed=seed,
        n_units=n_units,
        n_basins=n_basins,
        n_pairs=n_pairs,
        same_basin_frac=same_frac,
        point_biserial_r=float(pb_r),
        point_biserial_p=float(pb_p),
        spearman_r=float(sr),
        spearman_p=float(sp),
        mean_feat_sim_same=mean_sim_same,
        mean_feat_sim_diff=mean_sim_diff,
        separation=separation,
        wall_seconds=wall,
    )


# ===================================================================
# Main
# ===================================================================

def main():
    configs = [(st, 42 + s) for st in SPACE_TYPES for s in range(N_SEEDS)]
    print(f"exp18 -- Basin Membership vs Feature Similarity")
    print(f"Total configs: {len(configs)}")
    print(f"Fixed-point ratio threshold: {FIXED_POINT_RATIO}")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for idx, (st, seed) in enumerate(configs):
        print(f"  [{idx+1}/{len(configs)}] {st} seed={seed}", end=" ", flush=True)
        try:
            r = run_basin_analysis(st, seed)
            results.append(asdict(r))
            print(f"pb_r={r.point_biserial_r:+.3f} basins={r.n_basins} "
                  f"same_frac={r.same_basin_frac:.2f} sep={r.separation:+.3f} "
                  f"({r.wall_seconds:.1f}s)")
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "space_type": st, "seed": seed, "error": str(e),
            })

        # Incremental save every 10 runs
        if (idx + 1) % 10 == 0 or idx == len(configs) - 1:
            out_path = RESULTS_DIR / "exp18_results.json"
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)

    # ---------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------
    print("\n" + "=" * 80)
    print("exp18 Summary -- Basin Membership vs Feature Similarity")
    print("=" * 80)
    print(f"{'space_type':20s} | {'pb_r':>8s} | {'pb_p':>8s} | {'spear_r':>8s} "
          f"| {'basins':>6s} | {'same%':>6s} | {'sep':>8s} | {'PASS':>5s}")
    print("-" * 80)

    all_pb = []
    per_space = {}
    for st in SPACE_TYPES:
        subset = [r for r in results if r.get("space_type") == st and "error" not in r]
        if not subset:
            print(f"  {st:20s} | no data")
            continue

        pb_vals = [r["point_biserial_r"] for r in subset]
        pb_p_vals = [r["point_biserial_p"] for r in subset]
        sr_vals = [r["spearman_r"] for r in subset]
        basin_vals = [r["n_basins"] for r in subset]
        same_vals = [r["same_basin_frac"] for r in subset]
        sep_vals = [r["separation"] for r in subset]

        mean_pb = np.mean(pb_vals)
        mean_pp = np.mean(pb_p_vals)
        mean_sr = np.mean(sr_vals)
        mean_basins = np.mean(basin_vals)
        mean_same = np.mean(same_vals)
        mean_sep = np.mean(sep_vals)
        passed = mean_pb > 0.3

        per_space[st] = {
            "point_biserial_r_mean": float(mean_pb),
            "point_biserial_r_std": float(np.std(pb_vals)),
            "point_biserial_p_mean": float(mean_pp),
            "spearman_r_mean": float(mean_sr),
            "spearman_r_std": float(np.std(sr_vals)),
            "n_basins_mean": float(mean_basins),
            "same_basin_frac_mean": float(mean_same),
            "separation_mean": float(mean_sep),
            "separation_std": float(np.std(sep_vals)),
            "n_runs": len(subset),
            "PASS": bool(passed),
        }

        all_pb.extend(pb_vals)
        tag = "PASS" if passed else "FAIL"
        print(f"  {st:20s} | {mean_pb:+.4f} | {mean_pp:.4f} | {mean_sr:+.4f} "
              f"| {mean_basins:6.1f} | {mean_same:5.2f} | {mean_sep:+.4f} | {tag:>5s}")

    print("-" * 80)
    if all_pb:
        overall_pb = np.mean(all_pb)
        overall_tag = "PASS" if overall_pb > 0.3 else "FAIL"
        print(f"  {'OVERALL':20s} | {overall_pb:+.4f} |          |          "
              f"|        |       |          | {overall_tag:>5s}")
    print("=" * 80)

    # Save summary
    summary = {
        "experiment": "exp18_basin_membership",
        "hypothesis": "Basin membership (RG-flow convergence) correlates with feature similarity better than LCA-distance",
        "kill_criteria": "point-biserial r > 0.3",
        "fixed_point_ratio": FIXED_POINT_RATIO,
        "n_seeds": N_SEEDS,
        "n_pairs": N_PAIRS,
        "per_space": per_space,
        "overall_point_biserial_r": float(np.mean(all_pb)) if all_pb else None,
        "overall_PASS": bool(float(np.mean(all_pb)) > 0.3) if all_pb else False,
    }
    summary_path = RESULTS_DIR / "exp18_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults: {RESULTS_DIR / 'exp18_results.json'}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
