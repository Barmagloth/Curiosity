#!/usr/bin/env python3
"""
exp15b — P3b: Bush Clustering (Leaf-Path Clusters).

For each space type, runs the pipeline with Journal enabled, extracts
per-leaf feature vectors, and tests whether natural clusters ("bushes") exist.

Feature vector per leaf:
  - Path encoding: position in hierarchy (binary for tree, Morton for grid,
    community ID for graph)
  - Dirty signature components: seam_risk, uncert, mass (3 floats)
  - ρ value (informativity)
  - D_parent (LF leakage — from enforce_log)
  - Gate decision (1-hot: stage1=0, stage2=1)

Clustering methods: k-means (k=2..10), DBSCAN, agglomerative.
Metrics: Silhouette, Davies-Bouldin, Calinski-Harabasz.
Stability: ARI between clustering results across 20 seeds.

Kill criteria: Silhouette > 0.4 AND cross-run ARI > 0.6.

4 spaces × 20 seeds = 80 runs.

Usage:
  python exp15b_bushes.py                 # run all
  python exp15b_bushes.py --chunk 0 2     # run chunk 0 of 2
"""

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score,
)
from sklearn.preprocessing import StandardScaler

# --- path setup ---
EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENTS_DIR / "exp_phase2_pipeline"))
sys.path.insert(0, str(EXPERIMENTS_DIR / "exp10d_seed_determinism"))
sys.path.insert(0, str(EXPERIMENTS_DIR / "exp13_segment_compression"))

from config import PipelineConfig
from pipeline import CuriosityPipeline
from space_registry import SPACE_FACTORIES
from segment_compress import unpack_signature

# ===================================================================
# Constants
# ===================================================================

SPACE_TYPES = ["scalar_grid", "vector_grid", "irregular_graph", "tree_hierarchy"]
N_SEEDS = 20
K_RANGE = range(2, 11)  # k-means k=2..10
BUDGET = 0.30

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ===================================================================
# Feature extraction per leaf
# ===================================================================

def compute_dirty_sig_float(space, state, unit, rho_val: float) -> Tuple[float, float, float]:
    """Compute dirty signature components as floats (0-1 normalized)."""
    # seam_risk
    seam = min(1.0, rho_val * 2.0)

    # Get local diff
    if hasattr(unit, '__len__'):
        ti, tj = unit
        T = space.T
        s = slice(ti * T, (ti + 1) * T)
        cs = slice(tj * T, (tj + 1) * T)
        if state.ndim == 3:
            local_diff = state[s, cs, :] - space.coarse[s, cs, :]
        else:
            local_diff = state[s, cs] - space.coarse[s, cs]
    elif isinstance(unit, int):
        if hasattr(space, 'labels'):
            pts = np.where(space.labels == unit)[0]
            local_diff = state[pts] - space.coarse[pts] if len(pts) > 0 else np.array([0.0])
        else:
            local_diff = np.array([state[unit] - space.coarse[unit]])
    else:
        local_diff = np.array([0.0])

    uncert = min(1.0, float(np.var(local_diff)) * 10.0)
    mass = min(1.0, float(np.mean(np.abs(local_diff))) * 2.0)

    return seam, uncert, mass


def extract_leaf_features_for_clustering(
        space, state: np.ndarray, units: list,
        rho_values: np.ndarray, enforce_log: list,
        gate_stage: str) -> np.ndarray:
    """Extract feature matrix (n_units × n_features) for clustering.

    Features per unit:
      [0] position_encoding (normalized hierarchy position)
      [1] seam_risk
      [2] uncert
      [3] mass
      [4] rho
      [5] d_parent (from enforce_log, 0 if not available)
      [6] gate_stage (0=stage1, 1=stage2)
    """
    n = len(units)
    X = np.zeros((n, 7))

    # Build d_parent lookup from enforce log
    dp_lookup = {}
    for entry in enforce_log:
        dp_lookup[entry["unit"]] = entry.get("d_parent", 0.0)

    for idx, unit in enumerate(units):
        # Position encoding
        if hasattr(unit, '__len__'):
            # Grid: Morton-normalized
            ti, tj = unit
            NT = space.NT
            X[idx, 0] = (ti * NT + tj) / max(NT * NT, 1)
        elif isinstance(unit, int):
            if hasattr(space, 'labels'):
                X[idx, 0] = unit / max(space.n_clusters, 1)
            else:
                X[idx, 0] = unit / max(space.n, 1)
        else:
            X[idx, 0] = 0.0

        # Dirty signature components
        rho_val = float(rho_values[idx]) if idx < len(rho_values) else 0.0
        seam, uncert, mass_val = compute_dirty_sig_float(space, state, unit, rho_val)
        X[idx, 1] = seam
        X[idx, 2] = uncert
        X[idx, 3] = mass_val

        # Rho
        X[idx, 4] = rho_val

        # D_parent
        X[idx, 5] = dp_lookup.get(str(unit), 0.0)

        # Gate stage
        X[idx, 6] = 0.0 if gate_stage == "stage1_healthy" else 1.0

    return X


# ===================================================================
# Clustering analysis
# ===================================================================

@dataclass
class ClusteringResult:
    method: str
    k: int
    silhouette: float
    davies_bouldin: float
    calinski_harabasz: float
    labels: List[int]


def run_clustering(X: np.ndarray) -> List[ClusteringResult]:
    """Run multiple clustering methods on feature matrix X."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    results = []

    # k-means sweep
    for k in K_RANGE:
        if k >= len(X_scaled):
            continue
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        if len(set(labels)) < 2:
            continue
        sil = silhouette_score(X_scaled, labels)
        db = davies_bouldin_score(X_scaled, labels)
        ch = calinski_harabasz_score(X_scaled, labels)
        results.append(ClusteringResult(
            method=f"kmeans_{k}", k=k,
            silhouette=float(sil), davies_bouldin=float(db),
            calinski_harabasz=float(ch), labels=labels.tolist(),
        ))

    # DBSCAN (auto-detect k)
    for eps in [0.3, 0.5, 0.8, 1.0]:
        db = DBSCAN(eps=eps, min_samples=max(2, len(X_scaled) // 20))
        labels = db.fit_predict(X_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters >= 2:
            # Filter out noise points for metrics
            mask = labels >= 0
            if mask.sum() >= 2:
                sil = silhouette_score(X_scaled[mask], labels[mask])
                db_score = davies_bouldin_score(X_scaled[mask], labels[mask])
                ch = calinski_harabasz_score(X_scaled[mask], labels[mask])
                results.append(ClusteringResult(
                    method=f"dbscan_eps{eps}", k=n_clusters,
                    silhouette=float(sil), davies_bouldin=float(db_score),
                    calinski_harabasz=float(ch), labels=labels.tolist(),
                ))

    # Agglomerative
    for k in [2, 3, 4, 5]:
        if k >= len(X_scaled):
            continue
        agg = AgglomerativeClustering(n_clusters=k)
        labels = agg.fit_predict(X_scaled)
        if len(set(labels)) < 2:
            continue
        sil = silhouette_score(X_scaled, labels)
        db_score = davies_bouldin_score(X_scaled, labels)
        ch = calinski_harabasz_score(X_scaled, labels)
        results.append(ClusteringResult(
            method=f"agglom_{k}", k=k,
            silhouette=float(sil), davies_bouldin=float(db_score),
            calinski_harabasz=float(ch), labels=labels.tolist(),
        ))

    return results


# ===================================================================
# Single run
# ===================================================================

@dataclass
class BushRunResult:
    space_type: str
    seed: int
    n_units: int
    n_features: int
    best_method: str
    best_k: int
    best_silhouette: float
    best_davies_bouldin: float
    best_calinski_harabasz: float
    all_clusterings: List[dict]  # [{method, k, silhouette, ...}]
    best_labels: List[int]
    wall_seconds: float


def run_bush_analysis(space_type: str, seed: int) -> BushRunResult:
    """Run one bush clustering analysis."""
    t0 = time.time()

    cfg = PipelineConfig(
        budget_fraction=BUDGET,
        enforce_enabled=True,
        enox_journal_enabled=True,
    )
    pipe = CuriosityPipeline(config=cfg)
    result = pipe.run(space_type, seed=seed, budget_fraction=BUDGET)

    # Get space
    space = SPACE_FACTORIES[space_type]()
    space.setup(seed)
    units = space.get_units()
    n_units = len(units)

    # Compute rho
    rho_values = np.array([space.unit_rho(result.final_state, u) for u in units])

    # Extract features
    X = extract_leaf_features_for_clustering(
        space, result.final_state, units,
        rho_values, result.enforce_log, result.gate_stage)

    # Run clustering
    clusterings = run_clustering(X)

    # Find best by silhouette
    if clusterings:
        best = max(clusterings, key=lambda c: c.silhouette)
        all_c = [{"method": c.method, "k": c.k,
                   "silhouette": c.silhouette,
                   "davies_bouldin": c.davies_bouldin,
                   "calinski_harabasz": c.calinski_harabasz}
                  for c in clusterings]
    else:
        best = ClusteringResult("none", 0, 0.0, 999.0, 0.0, [])
        all_c = []

    wall = time.time() - t0

    return BushRunResult(
        space_type=space_type,
        seed=seed,
        n_units=n_units,
        n_features=X.shape[1],
        best_method=best.method,
        best_k=best.k,
        best_silhouette=best.silhouette,
        best_davies_bouldin=best.davies_bouldin,
        best_calinski_harabasz=best.calinski_harabasz,
        all_clusterings=all_c,
        best_labels=best.labels,
        wall_seconds=wall,
    )


# ===================================================================
# Cross-run stability (ARI)
# ===================================================================

def compute_stability(results_by_space: Dict[str, List[BushRunResult]]) -> Dict[str, float]:
    """Compute pairwise ARI between best clusterings across seeds."""
    stability = {}
    for st, runs in results_by_space.items():
        if len(runs) < 2:
            stability[st] = 0.0
            continue

        # Only compare runs with the same best_k
        # Find most common best_k
        k_counts = {}
        for r in runs:
            k_counts[r.best_k] = k_counts.get(r.best_k, 0) + 1
        best_k = max(k_counts, key=k_counts.get) if k_counts else 0

        # Filter runs with that k
        same_k = [r for r in runs if r.best_k == best_k and len(r.best_labels) > 0]
        if len(same_k) < 2:
            stability[st] = 0.0
            continue

        # Pairwise ARI
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
    parser = argparse.ArgumentParser(description="exp15b: Bush Clustering")
    parser.add_argument("--chunk", nargs=2, type=int, default=None,
                        help="Chunk index and total chunks")
    args = parser.parse_args()

    configs = [(st, 42 + s) for st in SPACE_TYPES for s in range(N_SEEDS)]
    print(f"Total configs: {len(configs)}")

    if args.chunk is not None:
        chunk_idx, n_chunks = args.chunk
        chunk_size = (len(configs) + n_chunks - 1) // n_chunks
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, len(configs))
        configs = configs[start:end]
        print(f"Running chunk {chunk_idx}/{n_chunks}: configs [{start}:{end}]")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    results_by_space: Dict[str, List[BushRunResult]] = {st: [] for st in SPACE_TYPES}

    for i, (st, seed) in enumerate(configs):
        print(f"  [{i+1}/{len(configs)}] {st} seed={seed}", end=" ", flush=True)
        try:
            r = run_bush_analysis(st, seed)
            results.append(asdict(r))
            results_by_space[st].append(r)
            print(f"best={r.best_method} sil={r.best_silhouette:.3f} "
                  f"k={r.best_k} ({r.wall_seconds:.1f}s)")
        except Exception as e:
            print(f"FAILED: {e}")
            results.append({
                "space_type": st, "seed": seed, "error": str(e),
            })

        # Incremental save
        if (i + 1) % 10 == 0 or i == len(configs) - 1:
            suffix = f"_chunk{args.chunk[0]}" if args.chunk else ""
            out_path = RESULTS_DIR / f"exp15b_results{suffix}.json"
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)

    # Stability analysis
    stability = compute_stability(results_by_space)

    # Summary
    print("\n" + "=" * 60)
    print("exp15b Summary — Bush Clustering")
    print("=" * 60)
    for st in SPACE_TYPES:
        subset = [r for r in results if r.get("space_type") == st and "error" not in r]
        if subset:
            sils = [r["best_silhouette"] for r in subset]
            ks = [r["best_k"] for r in subset]
            ari = stability.get(st, 0.0)
            sil_pass = np.mean(sils) > 0.4
            ari_pass = ari > 0.6
            print(f"  {st:20s}: sil={np.mean(sils):.3f}±{np.std(sils):.3f}  "
                  f"k_mode={max(set(ks), key=ks.count)}  "
                  f"ARI={ari:.3f}  "
                  f"PASS={'YES' if sil_pass and ari_pass else 'NO'}")
    print("=" * 60)

    # Save stability
    stability_path = RESULTS_DIR / "exp15b_stability.json"
    with open(stability_path, "w") as f:
        json.dump(stability, f, indent=2)
    print(f"Stability saved to {stability_path}")


if __name__ == "__main__":
    main()
