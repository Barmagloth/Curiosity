# exp19 — Multi-Tick Parameter Sweep + Benchmark

Phase 4 validation of the multi-tick pipeline (WeightedRhoGate, issues 1-7).
2050 configs total, 0 errors. All sub-experiments COMPLETE (25 March 2026).

## Summary

| Sub-exp | Configs | Key Result |
|---------|---------|------------|
| **19a** | 840 | mt=2-3 optimal. vector_grid: multi-tick +6-19% vs single. Scaling: `min(5, max(2, ceil(n_budget/50)))` |
| **19b** | 160 | Gate 160/160 PASS. alpha=0.3 recommended. Convergence even under oscillating instability |
| **19c** | 420 | On clean synthetic: multi-tick features = overhead (residual stable). Pilot/ROI unnecessary on clean data |
| **19d** | 150 | CIFAR-10: mt=3 = 96-97% of single-tick. Real graphs: 100%. Wall overhead <5% |
| **19e** | 480 | **Noisy: multi-tick BEATS single-tick +2-7%. Hetero clean: -26-30%. Mixed σ≥0.10: +4-7%** |

**Conclusion:** Multi-tick is beneficial under noise (σ≥0.05). On clean data, single-tick is optimal. The gate adapts (w_resid drops to 0.84 at σ=0.40), ROI filters noise-dominated units, convergence saves budget. However, issue 9 (noise-fitting: refinement copies noisy GT) is NOT addressed — this is Phase 5 (P5-noise, exp20).

## Sub-experiments

### exp19a: max_ticks scaling law (DONE — 840 configs)
- **Question:** How should max_ticks scale with n_total?
- **Sweep:** 3 scales (small/medium/large) × 7 max_ticks {1,2,3,5,8,13,20} × 4 spaces × 10 seeds
- **Kill:** multi-tick PSNR >= 95% of single-tick at same budget
- **Result:**
  - scalar_grid: mt=2 optimal (100% on large, 97% on small)
  - vector_grid: mt=5 optimal, **BEATS** single-tick by 6-19% across all scales
  - irregular_graph: mt=3-5 optimal (100%)
  - tree_hierarchy: mt=2 sufficient (100%, convergence stops early)
- **Scaling law:** `max_ticks = min(5, max(2, ceil(n_budget / 50)))` — at least 50 units per tick, max 5 ticks
- **ROI fix discovered:** global MSE reduction too small per unit → switched to local unit_rho reduction. Result: mt=3 went from 37% to 99% of single-tick PSNR.

### exp19b: WeightedRhoGate stress test (DONE — 160 configs, 160/160 PASS)
- **Question:** Do EMA weights converge under perturbations?
- **Sweep:** 4 alpha values {0.1,0.2,0.3,0.5} × 4 profiles (stable/ramp/step/oscillating) × 10 seeds
- **Kill:** weight variance over last 3 ticks < 0.05
- **Result:** ALL 160/160 converged. Max variance = 0.0020 (oscillating, alpha=0.5).
  - alpha=0.1: conservative, slow to adapt (w_resid=0.85 on stable)
  - alpha=0.3: good balance (recommended default)
  - alpha=0.5: fast but higher variance under oscillation

### exp19c: parameter sweep (DONE — 420 configs)
- **Design:** One-at-a-time sweep (vary one param, hold others at default). Medium scale, 5 seeds.
- **Sweep:** convergence_window {1-5} × pilot_ticks {0-5} × pilot_thresh_factor {0.3-1.2} × min_roi_fraction {0.0-0.40}
- **Result on clean synthetic:** Best = convergence_window=1, pilot_ticks=0, pilot_thresh_factor=0.3, min_roi_fraction=0.0.
- **Interpretation:** On clean data, all multi-tick features are overhead — residual is stable, ROI filter and pilot are unnecessary. These parameters matter under noise/degradation (see exp19e).

### exp19d: real data benchmark (DONE — 150 configs)
- **Data:** CIFAR-10 (scalar 128×128, vector 64×64×3) + networkx graphs (karate, les_miserables, florentine)
- **Sweep:** mt={1,3,5} × 10 seeds × 5 data configs
- **Result:**
  - CIFAR scalar: mt=3 = 96% of single-tick (+7.74 vs +8.07 dB)
  - CIFAR vector: mt=3 = 97% (+6.27 vs +6.44 dB)
  - Real graphs: mt=3,5 = 100% (convergence on tick 1, too few clusters)
  - Wall time overhead: <5%

### exp19e: noisy + heterogeneous stress test (DONE — 480 configs)
- **Data:** 16 scenarios: noisy (σ={0.05,0.10,0.20,0.40}), heterogeneous, mixed (hetero+noise)
- **Sweep:** mt={1,3,5} × 10 seeds × medium+large scales
- **Result:**
  - **Noisy: multi-tick BEATS single-tick by 2-7%.** Gate adapts w_resid from 1.0→0.84.
  - **Hetero (clean): multi-tick LOSES 26-30%.** Residual stable → overhead from budget splitting.
  - **Mixed σ≥0.10: multi-tick BEATS by 4-7%.** Gate correctly identifies unstable residual.
  - **vs clean GT: multi-tick slightly worse** (−2.6 vs −2.4 dB). Issue 9 (noise-fitting) NOT addressed.
- **Key finding:** multi-tick gate adapts correctly under noise but doesn't prevent noise-copying.

## Files

```
exp19a_scaling.py       — max_ticks scaling runner (840 configs)
exp19a_aggregate.py     — aggregation + summary tables
exp19b_gate_stress.py   — gate convergence test (160 configs)
exp19c_param_sweep.py   — one-at-a-time parameter sweep (420 configs)
exp19d_real_data.py     — CIFAR + real graphs benchmark (150 configs)
exp19e_noisy_hetero.py  — noisy/hetero stress test (480 configs)
results/                — JSON results per sub-experiment
```

## Dependencies
- Phase 4 multi-tick pipeline (`exp_phase2_pipeline/pipeline.py`)
- CIFAR-10 dataset (auto-downloaded to `data/cifar10/`)
- scipy (for image resizing), networkx (real graphs), leidenalg (optional, for graph clustering)
