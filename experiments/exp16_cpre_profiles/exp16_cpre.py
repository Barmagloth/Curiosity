#!/usr/bin/env python3
"""
exp16 — C-pre: Trajectory Profile Clustering.

Tests whether refinement units follow discrete behavioral profiles.

Approach: Run pipeline with DecisionJournal enabled over multiple
"evolution steps" (seed-shift). For each unit, build a trajectory feature
vector from the journal entries across steps:
  - mean/std rho across steps
  - mean/std d_parent
  - fraction of steps where passed / damped / rejected
  - mean damp_iterations
  - first/last step where unit was refined

Then cluster these trajectory vectors and check if natural clusters exist.

Kill criteria: Gap statistic > 1.0 AND Silhouette > 0.3 → Track C unfreezes.

4 spaces × 20 seeds = 80 runs.
"""

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

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

# ===================================================================
# Constants
# ===================================================================

SPACE_TYPES = ["scalar_grid", "vector_grid", "irregular_graph", "tree_hierarchy"]
N_SEEDS = 20
N_EVOLUTION_STEPS = 10  # steps per run for trajectory building
BUDGET = 0.30

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ===================================================================
# Gap statistic
# ===================================================================

def gap_statistic(X: np.ndarray, k_range=range(1, 11), n_ref=10, seed=42) -> Dict:
    """Compute gap statistic for k-means clustering.

    Gap(k) = E[log(W_ref(k))] - log(W(k))
    where W(k) = within-cluster dispersion for k clusters.

    Returns dict with best_k, gap_values, s_values.
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape

    def _dispersion(data, labels, k):
        w = 0.0
        for c in range(k):
            mask = labels == c
            if mask.sum() > 1:
                w += np.sum((data[mask] - data[mask].mean(axis=0))**2)
        return w

    log_w = []
    log_w_ref = []
    s_values = []

    for k in k_range:
        if k >= n:
            break

        # Actual data
        km = KMeans(n_clusters=k, random_state=seed, n_init=5)
        labels = km.fit_predict(X)
        w = _dispersion(X, labels, k)
        log_w.append(np.log(w + 1e-15))

        # Reference (uniform) data
        ref_logs = []
        x_min = X.min(axis=0)
        x_max = X.max(axis=0)
        for _ in range(n_ref):
            X_ref = rng.uniform(x_min, x_max, size=(n, d))
            km_ref = KMeans(n_clusters=k, random_state=seed, n_init=3)
            labels_ref = km_ref.fit_predict(X_ref)
            w_ref = _dispersion(X_ref, labels_ref, k)
            ref_logs.append(np.log(w_ref + 1e-15))

        log_w_ref.append(np.mean(ref_logs))
        s_values.append(np.std(ref_logs) * np.sqrt(1 + 1.0/n_ref))

    gaps = [ref - act for ref, act in zip(log_w_ref, log_w)]
    ks = list(k_range)[:len(gaps)]

    # Find best k: smallest k where gap(k) >= gap(k+1) - s(k+1)
    best_k = ks[0]
    for i in range(len(gaps) - 1):
        if gaps[i] >= gaps[i+1] - s_values[i+1]:
            best_k = ks[i]
            break
    else:
        best_k = ks[np.argmax(gaps)]

    return {
        "best_k": best_k,
        "max_gap": float(max(gaps)) if gaps else 0.0,
        "gap_values": {k: float(g) for k, g in zip(ks, gaps)},
    }


# ===================================================================
# Trajectory feature extraction
# ===================================================================

def extract_trajectory_features(
        space_type: str, seed_base: int, n_steps: int
) -> tuple:
    """Run pipeline over n_steps and extract per-unit trajectory features.

    Returns (X, unit_ids, step_data) where:
      X: (n_units, n_features) trajectory feature matrix
      unit_ids: list of unit string IDs
      step_data: list of per-step raw data
    """
    cfg = PipelineConfig(
        budget_fraction=BUDGET,
        enforce_enabled=True,
        enox_journal_enabled=True,
        enox_include_uri_map=True,
    )
    pipe = CuriosityPipeline(config=cfg)

    # Collect per-unit data across steps
    unit_histories: Dict[str, Dict] = {}  # unit_str -> {rhos, d_parents, decisions, ...}
    step_data = []

    for step in range(n_steps):
        seed = seed_base + step
        result = pipe.run(space_type, seed=seed, budget_fraction=BUDGET)

        space = SPACE_FACTORIES[space_type]()
        space.setup(seed)
        units = space.get_units()

        # Compute rho for all units
        rhos = {str(u): space.unit_rho(result.final_state, u) for u in units}

        # Build enforce lookup
        dp_lookup = {}
        action_lookup = {}
        damp_lookup = {}
        for entry in result.enforce_log:
            dp_lookup[entry["unit"]] = entry.get("d_parent", 0.0)
            action_lookup[entry["unit"]] = entry.get("action", "pass")
            damp_lookup[entry["unit"]] = entry.get("damp_iterations", 0)

        # Track per unit
        for u in units:
            u_str = str(u)
            if u_str not in unit_histories:
                unit_histories[u_str] = {
                    "rhos": [], "d_parents": [], "decisions": [],
                    "damp_iters": [], "steps_refined": [],
                }
            h = unit_histories[u_str]
            h["rhos"].append(rhos.get(u_str, 0.0))
            h["d_parents"].append(dp_lookup.get(u_str, 0.0))
            h["decisions"].append(action_lookup.get(u_str, "none"))
            h["damp_iters"].append(damp_lookup.get(u_str, 0))
            h["steps_refined"].append(step)

        step_data.append({
            "step": step, "seed": seed,
            "psnr": result.quality_psnr,
            "n_refined": result.n_refined,
            "reject_rate": result.reject_rate,
        })

    # Build feature matrix
    unit_ids = sorted(unit_histories.keys())
    n_units = len(unit_ids)
    # Features per unit:
    #  [0] mean_rho, [1] std_rho, [2] mean_d_parent, [3] std_d_parent,
    #  [4] frac_pass, [5] frac_damped, [6] frac_rejected, [7] frac_none,
    #  [8] mean_damp_iters, [9] rho_trend (slope)
    n_features = 10
    X = np.zeros((n_units, n_features))

    for idx, u_str in enumerate(unit_ids):
        h = unit_histories[u_str]
        rhos = np.array(h["rhos"])
        dps = np.array(h["d_parents"])
        decs = h["decisions"]
        damps = np.array(h["damp_iters"])

        X[idx, 0] = rhos.mean()
        X[idx, 1] = rhos.std()
        X[idx, 2] = dps.mean()
        X[idx, 3] = dps.std()

        n_dec = len(decs)
        X[idx, 4] = sum(1 for d in decs if d == "pass") / max(n_dec, 1)
        X[idx, 5] = sum(1 for d in decs if d == "damped") / max(n_dec, 1)
        X[idx, 6] = sum(1 for d in decs if d == "rejected") / max(n_dec, 1)
        X[idx, 7] = sum(1 for d in decs if d == "none") / max(n_dec, 1)

        X[idx, 8] = damps.mean()

        # Rho trend: simple linear regression slope
        if len(rhos) > 1:
            t = np.arange(len(rhos))
            slope = np.polyfit(t, rhos, 1)[0]
            X[idx, 9] = slope
        else:
            X[idx, 9] = 0.0

    return X, unit_ids, step_data


# ===================================================================
# Single run
# ===================================================================

@dataclass
class CPreRunResult:
    space_type: str
    seed_base: int
    n_units: int
    n_features: int
    n_steps: int
    # Gap statistic
    gap_best_k: int
    gap_max: float
    gap_values: Dict
    # Clustering results
    best_silhouette: float
    best_method: str
    best_k: int
    all_clusterings: List[dict]
    best_labels: List[int]
    # Step data
    psnr_trajectory: List[float]
    wall_seconds: float


def run_cpre(space_type: str, seed_base: int) -> CPreRunResult:
    """Run C-pre analysis for one space × seed."""
    t0 = time.time()

    X, unit_ids, step_data = extract_trajectory_features(
        space_type, seed_base, N_EVOLUTION_STEPS)

    if X.shape[0] < 4:
        return CPreRunResult(
            space_type=space_type, seed_base=seed_base,
            n_units=X.shape[0], n_features=X.shape[1],
            n_steps=N_EVOLUTION_STEPS,
            gap_best_k=1, gap_max=0.0, gap_values={},
            best_silhouette=0.0, best_method="none", best_k=0,
            all_clusterings=[], best_labels=[],
            psnr_trajectory=[s["psnr"] for s in step_data],
            wall_seconds=time.time() - t0,
        )

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Gap statistic
    gap = gap_statistic(X_scaled, k_range=range(1, min(11, X.shape[0])))

    # Clustering
    clusterings = []

    for k in range(2, min(11, X.shape[0])):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        if len(set(labels)) < 2:
            continue
        sil = silhouette_score(X_scaled, labels)
        clusterings.append({
            "method": f"kmeans_{k}", "k": k,
            "silhouette": float(sil), "labels": labels.tolist(),
        })

    for eps in [0.3, 0.5, 0.8, 1.0, 1.5]:
        db = DBSCAN(eps=eps, min_samples=max(2, X.shape[0] // 10))
        labels = db.fit_predict(X_scaled)
        n_cl = len(set(labels)) - (1 if -1 in labels else 0)
        if n_cl >= 2:
            mask = labels >= 0
            if mask.sum() >= 2:
                sil = silhouette_score(X_scaled[mask], labels[mask])
                clusterings.append({
                    "method": f"dbscan_eps{eps}", "k": n_cl,
                    "silhouette": float(sil), "labels": labels.tolist(),
                })

    if clusterings:
        best = max(clusterings, key=lambda c: c["silhouette"])
    else:
        best = {"method": "none", "k": 0, "silhouette": 0.0, "labels": []}

    wall = time.time() - t0

    return CPreRunResult(
        space_type=space_type,
        seed_base=seed_base,
        n_units=X.shape[0],
        n_features=X.shape[1],
        n_steps=N_EVOLUTION_STEPS,
        gap_best_k=gap["best_k"],
        gap_max=gap["max_gap"],
        gap_values=gap["gap_values"],
        best_silhouette=best["silhouette"],
        best_method=best["method"],
        best_k=best["k"],
        all_clusterings=[{k: v for k, v in c.items() if k != "labels"}
                         for c in clusterings],
        best_labels=best["labels"],
        psnr_trajectory=[s["psnr"] for s in step_data],
        wall_seconds=wall,
    )


# ===================================================================
# Cross-run stability
# ===================================================================

def compute_stability(results_by_space: Dict[str, List[CPreRunResult]]) -> Dict[str, float]:
    """ARI stability across seeds."""
    stability = {}
    for st, runs in results_by_space.items():
        same_k = [r for r in runs if r.best_k > 0 and len(r.best_labels) > 0]
        if len(same_k) < 2:
            stability[st] = 0.0
            continue
        aris = []
        for i in range(len(same_k)):
            for j in range(i + 1, len(same_k)):
                li = same_k[i].best_labels
                lj = same_k[j].best_labels
                if len(li) == len(lj) and len(li) > 0:
                    aris.append(adjusted_rand_score(li, lj))
        stability[st] = float(np.mean(aris)) if aris else 0.0
    return stability


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="exp16: C-pre Trajectory Profiles")
    parser.add_argument("--chunk", nargs=2, type=int, default=None)
    args = parser.parse_args()

    configs = [(st, 1000 + s * 100) for st in SPACE_TYPES for s in range(N_SEEDS)]
    print(f"Total configs: {len(configs)}")

    if args.chunk is not None:
        ci, nc = args.chunk
        cs = (len(configs) + nc - 1) // nc
        configs = configs[ci * cs : min((ci + 1) * cs, len(configs))]
        print(f"Running chunk {ci}/{nc}: {len(configs)} configs")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    results_by_space: Dict[str, List[CPreRunResult]] = {st: [] for st in SPACE_TYPES}

    for i, (st, sb) in enumerate(configs):
        print(f"  [{i+1}/{len(configs)}] {st} seed={sb}", end=" ", flush=True)
        try:
            r = run_cpre(st, sb)
            results.append(asdict(r))
            results_by_space[st].append(r)
            print(f"gap_k={r.gap_best_k} gap={r.gap_max:.2f} "
                  f"sil={r.best_silhouette:.3f} method={r.best_method} "
                  f"({r.wall_seconds:.1f}s)")
        except Exception as e:
            print(f"FAILED: {e}")
            results.append({"space_type": st, "seed_base": sb, "error": str(e)})

        if (i + 1) % 10 == 0 or i == len(configs) - 1:
            sfx = f"_chunk{args.chunk[0]}" if args.chunk else ""
            out = RESULTS_DIR / f"exp16_results{sfx}.json"
            with open(out, "w") as f:
                json.dump(results, f, indent=2)

    # Stability
    stability = compute_stability(results_by_space)
    with open(RESULTS_DIR / "exp16_stability.json", "w") as f:
        json.dump(stability, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("exp16 Summary — C-pre Trajectory Profiles")
    print("=" * 70)
    for st in SPACE_TYPES:
        subset = [r for r in results if r.get("space_type") == st and "error" not in r]
        if subset:
            sils = [r["best_silhouette"] for r in subset]
            gaps = [r["gap_max"] for r in subset]
            ks = [r["gap_best_k"] for r in subset]
            ari = stability.get(st, 0.0)
            gap_pass = np.mean(gaps) > 1.0
            sil_pass = np.mean(sils) > 0.3
            print(f"  {st:20s}: gap={np.mean(gaps):.2f}±{np.std(gaps):.2f}  "
                  f"sil={np.mean(sils):.3f}±{np.std(sils):.3f}  "
                  f"k_mode={max(set(ks), key=ks.count)}  "
                  f"ARI={ari:.3f}  "
                  f"GAP_PASS={gap_pass}  SIL_PASS={sil_pass}")
    print("=" * 70)
    overall_gaps = [r["gap_max"] for r in results if "error" not in r]
    overall_sils = [r["best_silhouette"] for r in results if "error" not in r]
    if overall_gaps:
        print(f"  Kill criteria: gap>{1.0}={'PASS' if np.mean(overall_gaps) > 1.0 else 'FAIL'}  "
              f"sil>{0.3}={'PASS' if np.mean(overall_sils) > 0.3 else 'FAIL'}")
        print(f"  Track C: {'UNFREEZE' if np.mean(overall_gaps) > 1.0 and np.mean(overall_sils) > 0.3 else 'STAYS FROZEN'}")


if __name__ == "__main__":
    main()
