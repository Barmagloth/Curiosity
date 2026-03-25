# Curiosity Architecture (Phase 4, March 2026)

Current state: Phase 4 multi-tick pipeline complete (exp19, 2050 configs, 0 errors). Phase 5 (robustness / noise) next. All decisions backed by experimental results.

---

## 1. System Overview

Curiosity is an adaptive refinement system for arbitrary state spaces. The core idea: dimensionality is not a fixed number but a depth of refinement; refinement occurs only where it is justified informationally and budget-wise.

The system operates on a state space X (features, latent representations, activations, data). An informativeness function rho(x) determines where refinement makes sense. The refinement tree logs all split decisions. The pipeline is domain-agnostic and validated on four space types: scalar_grid, vector_grid, irregular_graph, tree_hierarchy.

**Goal structure:**
- **Track A (instrument)** -- build the adaptive refinement pipeline. Active. Phases 0-4 complete.
- **Track B (research)** -- study the structure of refinement trees. After Instrument Readiness Gate.
- **Track C (generalization)** -- non-spatial domains. UNFROZEN after exp16 (C-pre trajectory profiles PASS).

**Technology:** Python, NumPy (CPU-only core logic), PyTorch/CUDA for GPU paths. Environments: `.venv` (CPU, Python 3.13) + `.venv-gpu` (DirectML/CUDA).

---

## 2. Pipeline Architecture (Phase 4, Multi-Tick)

The pipeline executes in two phases: one-time setup (Phase A) and multi-tick refinement loop (Phase B).

```
Phase A (one-time):
  Init space -> Layout selection -> Compression guard (tree only)
  -> Topo profiling (graph only) -> Enox infrastructure init
  -> SC-enforce setup -> Cold-start threshold calibration

Phase B (per tick, up to max_ticks):
  Compute weighted rho -> Select candidates (quantile + governor strictness)
  -> Probe allocation (unevaluated tiles only)
  -> Refine selected units -> SC-enforce (damp/reject)
  -> ROI gating (local unit_rho reduction) -> Halo blending
  -> GovernorEMA update -> WeightedRhoGate update (EMA weights)
  -> Convergence check -> Next tick or stop
```

**Multi-tick results (exp19, 2050 configs):** mt=2-3 optimal. Beneficial under noise (gate adapts w_resid 1.0 to 0.84, +2-7% quality). On clean synthetic data, single-tick optimal (multi-tick features = overhead). Real data (CIFAR): mt=3 achieves 96-97% of single-tick PSNR. Scaling law: `min(5, max(2, ceil(n_budget/50)))`.

**Convergence detector:** stops when zero candidates accepted for K consecutive ticks (convergence_window=2). Prevents wasted computation.

---

## 3. WeightedRhoGate

Replaces the discrete TwoStageGate (Phase 2) with smooth EMA weight transitions. Single function: `rho = sum(w_i * signal_i)`.

**Behavior:**
- When residual is healthy (instability <= thresh AND FSR <= thresh): w_resid approaches 1.0, w_hf and w_var approach 0.0.
- When residual is unstable: EMA shifts weights toward combo proportional to instability excess. HF receives 60% of combo weight, variance 40%.
- Minimum guaranteed residual weight: resid_min_weight (default 0.20).
- EMA update rate: ema_weight_alpha (default 0.3, validated in exp19b: 160/160 PASS).

**Key invariant:** SC-enforce is unaffected by weight changes. rho determines WHO is refined, not HOW. delta = refine_unit() does not depend on rho.

**Cold-start thresholds:** initial instability/FSR thresholds computed from topo_zone (GREEN=0.15, YELLOW=0.25, RED=0.40) blended with CV(initial_rho). Pilot ticks (first K=3) fine-tune from observed data. Replaces blind pilot calibration.

---

## 4. Three-Layer rho Decomposition (L0 / L1 / L2)

Monolithic rho conflates three orthogonal concerns (exp17, Phase 3.5). Solution -- cascading decomposition:

| Layer | Name | Purpose | Recomputation |
|-------|------|---------|---------------|
| L0 | Topology | Structural properties: curvature, connectivity, clusters | Once (on tree rebuild) |
| L1 | Presence | Data existence: dirty signatures, cascade quotas | Every step (incremental) |
| L2 | Query | Task-specific: residual, HF, utility-weighted combo | Every query |

**Cascade quotas (Variant C):** each L0 cluster guarantees `min(1, ceil(cluster_size * budget_fraction))` survivors at L1. No region is extinguished. Solves the region extinction problem (fixed threshold killed 97% of units on scalar_grid at scale 1000).

**FrozenTree:** serializable snapshot of L0+L1. Reusable spatial index (analogous to R-tree). Built once (expensive), then different L2 queries run on top (cheap).

**Streaming pipeline:** L0 clusters processed sequentially (most important first), each through full L0->L1->L2 cycle. Global budget cap enforced. First results appear after first cluster. Validated: 1080 configs, reusability 12/12 PASS (min 0.838).

---

## 5. Budget Control (Three Orthogonal Mechanisms)

Three independent mechanisms ensure budget compliance:

**Mechanism 1 -- L1 Cascade Quotas (structural):**
Topology-driven. Each L0 cluster receives a quota proportional to size. Information flows strictly top-down (L0->L1->L2). No region is extinguished.

**Mechanism 2 -- GovernorEMA (hardware-adaptive):**
Restored from exp08. EMA-based global strictness thermostat. Two-layer architecture: hardware parameter sets RANGE (leash length), EMA feedback moves WITHIN range. Corridor: [TARGET*0.5, TARGET*1.5]. Strictness clamp: 0.05. Warmup: 3 ticks. NOT applicable in streaming mode (cross-cluster bleed). Batch/reuse only.

**Mechanism 3 -- StrictnessTracker + WasteBudget (self-tightening noose):**
Per-unit reputation system. StrictnessTracker maintains per-unit multipliers: escalation x1.5 on reject, decay x0.9 per clean step. WasteBudget computes R_max = floor(B_step * omega); each reject costs strictness_multiplier units; force-stops when waste >= R_max. Safety fuse: hard_cap = 3x target.

**Streaming budget:** Institutional Inequality Formula: `W_cluster = N_units * (1 - ECR)^gamma`. GREEN clusters get ~90%, RED ~42% of nominal quota. Forward carry (C): RED gets strict minimum but receives leftover from GREEN if anomalously clean. gamma sweep planned: {1.0..4.0}.

---

## 6. SC-Enforce (Scale-Consistency)

Guarantees step_delta does not redefine the semantics of the parent scale. Parent coarse is the anchor, step_delta is the subordinate correction.

**Formal requirement:** `D_parent = ||R(step_delta)|| / (||step_delta|| + epsilon) < tau_parent`

**Operator pair (R, Up):** R = gaussian blur (sigma=3.0) + decimation. Up = bilinear upsampling. Fixed before experiments.

**Enforcement actions:** D_parent > tau_parent triggers damp step_delta (factor 0.5, up to 3 iterations) or reject split. Rejected units increase local strictness via StrictnessTracker.

**Per-space thresholds (tau_parent):** set independently per space type via Youden's J (exp12a). Grids: D_parent ~0.42-0.50, graph: ~0.08, tree: ~0.19. Single global threshold is impossible due to different R/Up dynamic ranges.

**Cross-space validation (Phase 0):** D_parent AUC: T1=1.000, T2=1.000, T3=1.000, T4=0.824. SC-baseline (SC-0..SC-4) COMPLETE.

---

## 7. Halo + Probe

### Halo (Boundary-Aware Blending)

Hard insertion of a refined tile creates a step discontinuity. The Laplacian flags this as a false HF signal. Halo fixes it via cosine feathering (overlap >= 3 elements) relative to parent coarse.

**Applicability rule:** boundary parallelism >= 3 AND no context leakage.

| Topology | Halo | Reason |
|----------|------|--------|
| Grid (tile >= 3) | Always | Wide boundary strip, no leakage |
| k-NN graph | Apply | min_cut typically >> 3 |
| Tree / Forest | Never | min_cut=1, context leakage into sibling subtrees |
| DAG | Per-case | Check boundary parallelism and leakage |

### Probe (Exploration)

Budget: 5-10% of total (probe_fraction=0.10). Targets unevaluated, unrefined tiles only (Phase 4 fix, issue 7 -- no overlap with rejected tiles). Deterministic: seed = f(tile_coordinates, depth_level, global_seed).

**Priorities:** coarse residual/variance, uncertainty, time since last check. Probe is insurance against structural blindness and false fixed points -- mandatory even at apparent convergence.

---

## 8. Topo Profiling (irregular_graph)

Multi-stage pre-runtime analysis performed during IrregularGraphSpace initialization, before the first pipeline tick.

**Hybrid curvature engine:** (1) Forman-Ricci for ALL edges (O(1) per edge, cheap); (2) sort by |Forman| anomaly; (3) upgrade top-N anomalous edges to Ollivier-Ricci (expensive, O(W^3) via EMD). N = floor(topo_budget / t_ollivier). Default topo_budget_ms=50.

**Hardware calibration:** Synthetic Transport Probe at session start. Generates synthetic EMD problem (d_test=10), measures median solve time, extrapolates kappa_max. One-time measurement (~52ms).

**Key metrics:** sigma_F (Forman std dev), eta_F = sigma_F / sqrt(2*mean_degree) -- topological entropy index. Threshold 0.70 from dead zone [0.60, 0.76] in 35-graph corpus.

**Topo zone classification:** GREEN (kappa_mean > 0, dense cliques, ECR < 5%), YELLOW (kappa < 0, Gini < 0.12, eta_F <= 0.70, regular lattices), RED (structural chaos, ECR > 30%). Determines runtime policy: tau_eff factor (GREEN=1.3, YELLOW=1.0, RED=0.7), split budget, SC-enforce strictness, cold-start gate thresholds.

---

## 9. Enox Infrastructure

Four observation-only patterns. Pure annotation -- never modify pipeline state. All default OFF for backward compatibility.

| Pattern | Purpose | Key detail |
|---------|---------|------------|
| **RegionURI** | Deterministic SHA256 address per unit | SHA256(parent_id\|op_type\|child_idx), 16 hex chars |
| **DecisionJournal** | Append-only log of gate/enforcement decisions | region_id, tick, stage, decision, thresholds |
| **MultiStageDedup** | Three-level dedup: exact hash, metric-distance, policy | epsilon=0.0 default; scaffolding for multi-pass |
| **PostStepSweep** | Post-step scan for identical siblings | Dirty signature comparison, threshold 5% |

Enox patterns provide debugging, audit trails, and merge-candidate identification. Active in production only when explicitly enabled via config flags.

---

## 10. Compression Guard (tree_hierarchy)

Tree_hierarchy has degree-2 chain nodes that waste budget without adding information. Compression guard decides whether chain compression is viable before the pipeline runs.

**Decision function:** `should_compress(n_active, budget_step, n_stable_d2)`. Uses N_CRITICAL_D2 threshold. If viable, degree-2 nodes are added to d2_skip_set and excluded from refinement candidates.

**Layout policy for tree_hierarchy:** Hybrid per-level. D_direct where occupancy p_l < 0.40 AND operator is compute-heavy (matmul-like); A_bitset elsewhere. Upper levels (small N_l, high occupancy) use A_bitset. Lower levels (large N_l, low occupancy, heavy compute) use D_direct. Stencil ops: D saves memory but NEVER wins time, so A_bitset. Validated: exp10j, 158K trials.

---

## 11. Layout Policy

Layout is determined by a vector of space properties, not by type name. Three axes:

1. **Topological isotropy I(X):** entropy of degree distribution H(D). I=0 (regular grid) -> D_direct optimal. As I grows, indirect addressing becomes unavoidable.
2. **Metric gap M(X, L):** Kendall rank correlation between BFS-rank and linear address. Property of (topology, linearization) pair. tau -> 1: block packing viable. tau -> 0: padding waste.
3. **Dynamic density p:** occupancy k/N. Above p* (~0.40): A_bitset cheaper. Below: D_direct/D_blocked. p* stability likely due to hardware quantization (warp=32, mem transactions=128B).

**Production layouts:**

| Space type | Layout | Status |
|------------|--------|--------|
| scalar_grid | D_direct (packed tiles + tile_map) | Production (exp10g PASS) |
| vector_grid | D_direct | Production (exp10h: 72/72 PASS) |
| irregular_graph / spatial | D_blocked (cbr <= 0.35) | Production (exp10i) |
| irregular_graph / scale-free | A_bitset fallback | cbr=0.66 rejects D_blocked (exp10i) |
| tree_hierarchy | Hybrid per-level (D_direct + A_bitset) | Production (exp10j: 158K trials) |

**Law:** `layout = argmin C(I, M, p)`. Current policy table = tabulated values. New space type: compute (I, M, p), find nearest row.

---

## 12. Convergence + ROI

### Convergence Detector

Stops the multi-tick loop when zero candidates are accepted for convergence_window (default 2) consecutive ticks. Prevents wasted computation on converged regions. Convergence reason is logged: "zero_accepted", "max_ticks", or "budget_exhausted".

### ROI Gating

Each refinement is evaluated by local unit_rho reduction (not global MSE). `ROI = rho_before - rho_after` for the refined unit's region. Refinements with ROI below min_roi_fraction * reference_median_gain are marked as low-value.

**Critical fix (exp19):** global MSE -> local unit_rho reduction. mt=3 went from 37% to 99% of single-tick PSNR. Global MSE change from one unit is negligible on large spaces, causing false rejection of useful refinements.

---

## 13. Determinism (DET-1 / DET-2)

### DET-1 (Seed Determinism) -- Hard Constraint

Identical data + rho + seed + budget = identical tree (bitwise match). Three components:
1. **Canonical traversal order:** Z-order/Morton index. Eliminates hash table iteration order and thread race dependence.
2. **Deterministic probe:** seed = f(tile_coordinates, depth_level, global_seed).
3. **Governor isolation:** EMA update strictly after the full step, not during processing.

**Status:** DET-1 PASS (Phase 1).

### DET-2 (Cross-Seed Stability) -- Soft Constraint

Across different seeds, metrics (PSNR, cost, compliance, SeamScore) are statistically stable: CV < tau_cv for each metric.

**Status:** DET-2 PASS (Phase 1).

---

## 14. Modules A-H

| Module | Component | Implementation | Status |
|--------|-----------|---------------|--------|
| A | WeightedRhoGate | `pipeline.py` WeightedRhoGate class | Phase 4 DONE. exp19b: 160/160 PASS |
| B | Three-Layer rho (L0/L1/L2) | `layers.py` ThreeLayerPipeline | Phase 3.5 DONE. 1080 configs, 12/12 PASS |
| C | SC-Enforce | `sc_enforce.py` SCEnforcer | Phase 2 DONE. Cross-space D_parent AUC 0.824-1.000 |
| D | Halo | `space_registry.py` per-space blending | Phase 0 DONE. Topology rule validated |
| E | Budget (3 mechanisms) | `pipeline.py` GovernorEMA, StrictnessTracker, WasteBudget | Phase 4 DONE. Governor EMA sweep pending |
| F | Topo Profiling | `topo_features.py` extract_topo_features | Phase 2 DONE. Hybrid curvature engine |
| G | Enox Infrastructure | `enox_infra.py` 4 patterns | DONE. Observation-only, never modifies state |
| H | Compression Guard | `pipeline.py` should_compress + d2_skip_set | Phase 2 DONE. tree_hierarchy only |

**Supporting components:**
- Layout policy: `space_registry.py` LAYOUT_POLICY (Phase 1 DONE, P0 CLOSED)
- Convergence + ROI: `pipeline.py` _check_convergence, _compute_unit_roi (Phase 4 DONE)
- Determinism: canonical traversal, deterministic probe, governor isolation (Phase 1 DONE)
- SeamScore: `SeamScore = Jumpout / (Jumpin + eps)`. Validated on 4 space types. Production-ready.

---

## 15. Validation Status

### Completed Phases

| Phase | Date | Key Result |
|-------|------|------------|
| Phase 0 | March 18, 2026 | Cross-space validation. SC-baseline (SC-0..SC-4) PASS |
| Phase 1 | March 20, 2026 | All streams PASS. P0 Layout CLOSED. DET-1 PASS. DET-2 PASS |
| Phase 2 | March 20, 2026 | Pipeline assembled, SC-enforce integrated, E2E validated |
| Enox | March 21, 2026 | Four observation-only patterns. Pure annotation |
| Phase 3 | March 22, 2026 | Exp14 anchors: grid PASS, graph/tree FAIL. Exp16 C-pre: PASS -> Track C UNFREEZE |
| Phase 3.5 | March 23, 2026 | Three-layer rho. 1080 configs, reusability 12/12 PASS. Industry benchmarks |
| Exp18 | March 23, 2026 | Basin membership FAIL (r=0.019). Deferred to post-multi-pass |
| Phase 4 | March 25, 2026 | Multi-tick pipeline. exp19: 2050 configs, 0 errors |

### Open Items

- **Phase 4 (continued):** P4a downstream consumer test, P4b matryoshka, MultiStageDedup test, Governor EMA sweep (4320 configs planned). Issue 8 (post-refinement quality feedback) open.
- **Phase 5 (robustness):** Issue 9 (noise-fitting) -- system optimizes to noisy observations, not true signal. exp20a: 6 denoising approaches x 3 sigma levels x 10 seeds. exp20b: composite from Pareto-best. exp20c: pipeline integration + DET re-check.
- **Deferred:** Bushes revisit after Track C. RG-flow after multi-pass. Sigma-squared estimation for T3/T4 is open research.

### Known Architectural Issues (from viz testbed, March 24)

9 issues exposed by interactive testbed. Issues 1-7 IMPLEMENTED in Phase 4:
1. Missing convergence detector -> DONE
2. Gate health thresholds not data-driven -> DONE (cold-start)
3. FSR inflated by refined tiles -> DONE (exclusion)
4. Gate oscillation without hysteresis -> DONE (EMA weights)
5. EMA weights uncalibrated on first ticks -> DONE (cold-start + pilot)
6. Gain/cost incommensurability -> DONE (local unit_rho ROI)
7. Probe/reject overlap -> DONE (unevaluated-only targeting)
8. No post-refinement quality feedback -> OPEN (Phase 4+)
9. Noise-fitting -> OPEN (Phase 5)
