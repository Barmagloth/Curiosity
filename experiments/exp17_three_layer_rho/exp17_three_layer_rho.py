#!/usr/bin/env python3
"""Exp17: Three-Layer rho Architecture.

Separates monolithic rho into three independent layers:
  L0: Topology (data-independent) — space structure
  L1: Presence (data-dependent, query-independent) — where data exists
  L2: Query (task-specific) — where THIS metric needs refinement

Compares against: naive (no topo), single-pass (current), industry baselines.

Usage:
  python exp17_three_layer_rho.py                    # all configs
  python exp17_three_layer_rho.py --chunk 0 8        # chunk 0 of 8
  python exp17_three_layer_rho.py --test-only scalar_grid  # single space quick test
"""
import os
import sys
import json
import time
import argparse
import traceback
import numpy as np
from pathlib import Path
from dataclasses import asdict

# Ensure UTF-8 output on Windows
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / 'exp_phase2_pipeline'))
sys.path.insert(0, str(SCRIPT_DIR.parent / 'exp10d_seed_determinism'))

from exp10d_seed_determinism import SPACE_FACTORIES
from layers import ThreeLayerPipeline, IndustryBaselines
from config17 import (Exp17Config, grid_N_for_units,
                       graph_points_for_units, tree_depth_for_units)

RESULTS_DIR = SCRIPT_DIR / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


# =====================================================================
# Space factory with variable scale
# =====================================================================

def create_space(space_type: str, seed: int, n_units: int = 100):
    """Create a space with approximately n_units units."""
    space = SPACE_FACTORIES[space_type]()

    # Override size parameters based on scale
    if space_type in ("scalar_grid", "vector_grid"):
        N = grid_N_for_units(n_units, tile_size=8)
        space.N = N
        space.T = 8
        space.NT = N // 8

    elif space_type == "irregular_graph":
        n_points = graph_points_for_units(n_units)
        space.n_points = min(n_points, 100000)  # cap for memory
        space.k = 6 if n_points <= 10000 else 8

    elif space_type == "tree_hierarchy":
        depth = tree_depth_for_units(n_units)
        space.depth = depth
        space.n = 2 ** depth - 1

    space.setup(seed)
    return space


def create_sparsity_mask(space_type: str, n_units_shape, fraction: float,
                          rng: np.random.RandomState):
    """Create a boolean mask where True = zeroed unit."""
    if space_type in ("scalar_grid", "vector_grid"):
        # n_units_shape = (NT, NT)
        mask = rng.random(n_units_shape) < fraction
        return mask
    return None  # No sparsity mask for graph/tree


# =====================================================================
# Run a single configuration
# =====================================================================

def run_config(space_type: str, seed: int, approach: str,
               n_units: int = 100, cfg: Exp17Config = None):
    """Run a single experiment configuration. Returns a result dict."""
    if cfg is None:
        cfg = Exp17Config()

    result = {
        'space_type': space_type,
        'seed': seed,
        'approach': approach,
        'n_units_target': n_units,
        'error': None,
    }

    try:
        # Create space
        space = create_space(space_type, seed, n_units)
        units = space.get_units()
        state = space.coarse.copy()
        n_actual = len(units)
        result['n_units_actual'] = n_actual

        pipeline = ThreeLayerPipeline()

        # Sparsity mask for L1 testing
        # Actually zero GT for masked units so ALL approaches see the same data
        rng = np.random.RandomState(seed + 9999)
        sp_mask = None
        if space_type in ("scalar_grid", "vector_grid"):
            sp_mask = create_sparsity_mask(
                space_type, (space.NT, space.NT), cfg.sparsity_fraction, rng)
            # Zero out GT and coarse for masked tiles
            T = space.T
            for ti in range(space.NT):
                for tj in range(space.NT):
                    if sp_mask[ti, tj]:
                        s = slice(ti * T, (ti + 1) * T)
                        cs = slice(tj * T, (tj + 1) * T)
                        if space.gt.ndim == 2:
                            space.gt[s, cs] = 0.0
                            space.coarse[s, cs] = 0.0
                        else:
                            space.gt[s, cs, :] = 0.0
                            space.coarse[s, cs, :] = 0.0
            state = space.coarse.copy()  # refresh state after zeroing

        # ---- Run the selected approach ----

        if approach == "naive_full":
            res = pipeline.run_single_pass(
                space, state, units, space_type,
                budget_fraction=cfg.budget_fraction,
                halo_width=cfg.halo_width, halo_hops=cfg.halo_hops,
                use_topo=False)
            result.update({
                'total_ms': res['total_ms'],
                'build_ms': 0.0,
                'query_ms': res['total_ms'],
                'psnr': res['psnr'],
                'mse': res['mse'],
                'n_refined': res['n_refined'],
                'index_bytes': 0,
                'n_after_l0': n_actual,
                'n_after_l1': n_actual,
                'l0_prune_pct': 0.0,
                'l1_prune_pct': 0.0,
            })

        elif approach == "single_pass":
            res = pipeline.run_single_pass(
                space, state, units, space_type,
                budget_fraction=cfg.budget_fraction,
                halo_width=cfg.halo_width, halo_hops=cfg.halo_hops,
                use_topo=True)
            result.update({
                'total_ms': res['total_ms'],
                'build_ms': 0.0,
                'query_ms': res['total_ms'],
                'psnr': res['psnr'],
                'mse': res['mse'],
                'n_refined': res['n_refined'],
                'index_bytes': 0,
                'n_after_l0': n_actual,
                'n_after_l1': n_actual,
                'l0_prune_pct': 0.0,
                'l1_prune_pct': 0.0,
            })

        elif approach == "three_layer":
            t0 = time.perf_counter()

            frozen, l0_res, l1_res = pipeline.build_frozen_tree(
                space, state, units, space_type, seed,
                l0_threshold=cfg.l0_threshold,
                l1_min_survival=cfg.l1_min_survival,
                sparsity_mask=sp_mask)

            l2_res = pipeline.run_query_on_frozen(
                space, state, units, frozen, l1_res, l0_res,
                query_fn="mse",
                budget_fraction=cfg.budget_fraction,
                halo_width=cfg.halo_width, halo_hops=cfg.halo_hops)

            total_ms = (time.perf_counter() - t0) * 1000.0

            result.update({
                'total_ms': total_ms,
                'build_ms': frozen.l0_ms + frozen.l1_ms,
                'l0_ms': frozen.l0_ms,
                'l1_ms': frozen.l1_ms,
                'query_ms': l2_res.computation_ms,
                'psnr': l2_res.psnr_final,
                'mse': l2_res.mse_final,
                'n_refined': l2_res.n_refined,
                'index_bytes': frozen.memory_bytes,
                'n_after_l0': l0_res.n_surviving,
                'n_after_l1': l1_res.n_surviving,
                'l0_prune_pct': 100.0 * (1 - l0_res.n_surviving / max(l0_res.n_total, 1)),
                'l1_prune_pct': 100.0 * l1_res.n_pruned_l1 / max(l1_res.n_input, 1),
                'zone': frozen.zone,
            })

        elif approach == "three_layer_reuse":
            t0 = time.perf_counter()

            frozen, l0_res, l1_res = pipeline.build_frozen_tree(
                space, state, units, space_type, seed,
                l0_threshold=cfg.l0_threshold,
                l1_min_survival=cfg.l1_min_survival,
                sparsity_mask=sp_mask)

            build_ms = frozen.l0_ms + frozen.l1_ms

            # Run 3 different queries on frozen tree
            query_results = {}
            for qf in cfg.query_functions:
                l2_res = pipeline.run_query_on_frozen(
                    space, state, units, frozen, l1_res, l0_res,
                    query_fn=qf,
                    budget_fraction=cfg.budget_fraction,
                    halo_width=cfg.halo_width, halo_hops=cfg.halo_hops)
                query_results[qf] = {
                    'psnr': l2_res.psnr_final,
                    'mse': l2_res.mse_final,
                    'query_ms': l2_res.computation_ms,
                    'n_refined': l2_res.n_refined,
                }

            total_ms = (time.perf_counter() - t0) * 1000.0

            result.update({
                'total_ms': total_ms,
                'build_ms': build_ms,
                'l0_ms': frozen.l0_ms,
                'l1_ms': frozen.l1_ms,
                'query_results': query_results,
                'psnr': query_results['mse']['psnr'],  # primary = MSE query
                'mse': query_results['mse']['mse'],
                'n_refined': query_results['mse']['n_refined'],
                'index_bytes': frozen.memory_bytes,
                'n_after_l0': l0_res.n_surviving,
                'n_after_l1': l1_res.n_surviving,
                'l0_prune_pct': 100.0 * (1 - l0_res.n_surviving / max(l0_res.n_total, 1)),
                'l1_prune_pct': 100.0 * l1_res.n_pruned_l1 / max(l1_res.n_input, 1),
                'zone': frozen.zone,
            })

        elif approach == "industry_kdtree":
            res = IndustryBaselines.run_kdtree(
                space, state, units,
                budget_fraction=cfg.budget_fraction,
                halo_width=cfg.halo_width, halo_hops=cfg.halo_hops)
            result.update({
                'total_ms': res['total_ms'],
                'build_ms': res['build_ms'],
                'query_ms': res['query_ms'],
                'psnr': res['psnr'],
                'mse': res['mse'],
                'n_refined': res['n_refined'],
                'index_bytes': res['index_bytes'],
                'n_after_l0': n_actual,
                'n_after_l1': n_actual,
                'l0_prune_pct': 0.0,
                'l1_prune_pct': 0.0,
            })

        elif approach == "industry_quadtree":
            if space_type not in ("scalar_grid", "vector_grid"):
                result['error'] = f"quadtree not applicable to {space_type}"
                result['skipped'] = True
                return result
            res = IndustryBaselines.run_quadtree(
                space, state, units,
                budget_fraction=cfg.budget_fraction,
                halo_width=cfg.halo_width)
            result.update({
                'total_ms': res['total_ms'],
                'build_ms': res['build_ms'],
                'query_ms': res['query_ms'],
                'psnr': res['psnr'],
                'mse': res['mse'],
                'n_refined': res['n_refined'],
                'index_bytes': res['index_bytes'],
                'n_after_l0': n_actual,
                'n_after_l1': n_actual,
                'l0_prune_pct': 0.0,
                'l1_prune_pct': 0.0,
            })

        elif approach == "industry_leiden_bf":
            if space_type != "irregular_graph":
                result['error'] = f"leiden_bf not applicable to {space_type}"
                result['skipped'] = True
                return result
            res = IndustryBaselines.run_leiden_bf(
                space, state, units,
                budget_fraction=cfg.budget_fraction,
                halo_hops=cfg.halo_hops)
            result.update({
                'total_ms': res['total_ms'],
                'build_ms': res['build_ms'],
                'query_ms': res['query_ms'],
                'psnr': res['psnr'],
                'mse': res['mse'],
                'n_refined': res['n_refined'],
                'index_bytes': res['index_bytes'],
                'n_after_l0': n_actual,
                'n_after_l1': n_actual,
                'l0_prune_pct': 0.0,
                'l1_prune_pct': 0.0,
            })

        elif approach == "industry_wavelet":
            if space_type not in ("scalar_grid",):
                result['error'] = f"wavelet not applicable to {space_type}"
                result['skipped'] = True
                return result
            res = IndustryBaselines.run_wavelet(
                space, state, units,
                budget_fraction=cfg.budget_fraction,
                halo_width=cfg.halo_width)
            result.update({
                'total_ms': res['total_ms'],
                'build_ms': res['build_ms'],
                'query_ms': res['query_ms'],
                'psnr': res['psnr'],
                'mse': res['mse'],
                'n_refined': res['n_refined'],
                'index_bytes': res['index_bytes'],
                'n_after_l0': n_actual,
                'n_after_l1': n_actual,
                'l0_prune_pct': 0.0,
                'l1_prune_pct': 0.0,
            })

        else:
            result['error'] = f"Unknown approach: {approach}"

    except Exception as e:
        result['error'] = f"{type(e).__name__}: {e}"
        result['traceback'] = traceback.format_exc()

    return result


# =====================================================================
# Build experiment configurations
# =====================================================================

def build_configs(cfg: Exp17Config, chunk: int = None, n_chunks: int = None):
    """Build list of (space_type, seed, approach, n_units) configs."""
    configs = []

    # Applicable approaches per space type
    applicable = {
        "scalar_grid": [
            "naive_full", "single_pass", "three_layer", "three_layer_reuse",
            "industry_kdtree", "industry_quadtree", "industry_wavelet"],
        "vector_grid": [
            "naive_full", "single_pass", "three_layer", "three_layer_reuse",
            "industry_kdtree", "industry_quadtree"],
        "irregular_graph": [
            "naive_full", "single_pass", "three_layer", "three_layer_reuse",
            "industry_kdtree", "industry_leiden_bf"],
        "tree_hierarchy": [
            "naive_full", "single_pass", "three_layer", "three_layer_reuse",
            "industry_kdtree"],
    }

    for scale in cfg.scale_levels:
        n_seeds = cfg.n_seeds if scale <= 1000 else 5  # fewer seeds at 10K
        for st in cfg.space_types:
            for s_idx in range(n_seeds):
                seed = cfg.base_seed + s_idx * 100 + scale
                for approach in applicable.get(st, []):
                    configs.append((st, seed, approach, scale))

    # Apply chunking
    if chunk is not None and n_chunks is not None:
        chunk_size = len(configs) // n_chunks + 1
        start = chunk * chunk_size
        end = min(start + chunk_size, len(configs))
        configs = configs[start:end]

    return configs


# =====================================================================
# Summary computation
# =====================================================================

def compute_summary(results: list) -> dict:
    """Compute aggregated summary from all results."""
    summary = {
        'n_total': len(results),
        'n_errors': sum(1 for r in results if r.get('error')),
        'n_skipped': sum(1 for r in results if r.get('skipped')),
        'per_approach': {},
        'per_space': {},
        'kill_criteria': {},
    }

    # Group by (space_type, approach, scale)
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        if r.get('error') or r.get('skipped'):
            continue
        key = (r['space_type'], r['approach'], r.get('n_units_target', 100))
        groups[key].append(r)

    for (st, approach, scale), runs in groups.items():
        psnrs = [r['psnr'] for r in runs if 'psnr' in r]
        times = [r['total_ms'] for r in runs if 'total_ms' in r]
        key = f"{st}/{approach}/{scale}"
        summary['per_approach'][key] = {
            'n_runs': len(runs),
            'psnr_mean': float(np.mean(psnrs)) if psnrs else None,
            'psnr_std': float(np.std(psnrs)) if psnrs else None,
            'time_mean_ms': float(np.mean(times)) if times else None,
            'time_std_ms': float(np.std(times)) if times else None,
        }

    # Kill criteria checks
    # 1. Reusability: min(psnr_frozen / psnr_fresh) >= 0.80
    reuse_ratios = []
    for r in results:
        if r.get('approach') == 'three_layer_reuse' and 'query_results' in r:
            for qf, qr in r['query_results'].items():
                # Compare with single_pass PSNR for same space/seed/scale
                sp_results = [x for x in results
                              if x.get('approach') == 'single_pass'
                              and x.get('space_type') == r['space_type']
                              and x.get('seed') == r['seed']
                              and x.get('n_units_target') == r.get('n_units_target')]
                if sp_results:
                    sp_psnr = sp_results[0].get('psnr', 1.0)
                    if sp_psnr > 0:
                        reuse_ratios.append(qr['psnr'] / sp_psnr)

    summary['kill_criteria']['reusability_min_ratio'] = (
        float(min(reuse_ratios)) if reuse_ratios else None)
    summary['kill_criteria']['reusability_pass'] = (
        min(reuse_ratios) >= 0.80 if reuse_ratios else None)

    return summary


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description='Exp17: Three-Layer rho')
    parser.add_argument('--chunk', type=int, nargs=2, metavar=('IDX', 'TOTAL'),
                        help='Run chunk IDX of TOTAL')
    parser.add_argument('--test-only', type=str, metavar='SPACE',
                        help='Quick test with one space type, 2 seeds')
    args = parser.parse_args()

    cfg = Exp17Config()

    if args.test_only:
        cfg.space_types = [args.test_only]
        cfg.n_seeds = 2
        cfg.scale_levels = [100]
        cfg.save_every = 5

    chunk_idx = args.chunk[0] if args.chunk else None
    n_chunks = args.chunk[1] if args.chunk else None

    configs = build_configs(cfg, chunk_idx, n_chunks)
    print(f"Exp17: {len(configs)} configs"
          f"{f' (chunk {chunk_idx}/{n_chunks})' if chunk_idx is not None else ''}")

    out_suffix = f"_chunk{chunk_idx}" if chunk_idx is not None else ""
    out_path = RESULTS_DIR / f"exp17_results{out_suffix}.json"

    results = []
    t_start = time.time()

    for i, (st, seed, approach, scale) in enumerate(configs):
        r = run_config(st, seed, approach, n_units=scale, cfg=cfg)
        r['config_index'] = i
        results.append(r)

        status = "OK" if not r.get('error') else f"ERR: {r['error'][:50]}"
        skip = " [SKIP]" if r.get('skipped') else ""
        psnr_str = f"PSNR={r.get('psnr', 0):.2f}" if 'psnr' in r and r['psnr'] else ""
        time_str = f"{r.get('total_ms', 0):.1f}ms" if 'total_ms' in r else ""

        print(f"  [{i+1}/{len(configs)}] {st:20s} seed={seed} "
              f"{approach:25s} scale={scale:5d} "
              f"{psnr_str:12s} {time_str:10s} {status}{skip}")

        # Incremental save
        if (i + 1) % cfg.save_every == 0 or i == len(configs) - 1:
            with open(out_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)

    elapsed = time.time() - t_start
    print(f"\nDone: {len(results)} results in {elapsed:.1f}s")
    print(f"Errors: {sum(1 for r in results if r.get('error') and not r.get('skipped'))}")
    print(f"Skipped: {sum(1 for r in results if r.get('skipped'))}")

    # Compute and save summary
    summary = compute_summary(results)
    summary_path = RESULTS_DIR / f"exp17_summary{out_suffix}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults: {out_path}")
    print(f"Summary: {summary_path}")

    # Print kill criteria
    kc = summary.get('kill_criteria', {})
    if kc.get('reusability_min_ratio') is not None:
        status = "PASS" if kc['reusability_pass'] else "FAIL"
        print(f"\nKill: reusability min ratio = "
              f"{kc['reusability_min_ratio']:.3f} ({status})")


if __name__ == '__main__':
    main()
