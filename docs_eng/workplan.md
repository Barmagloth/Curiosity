# Work Plan (Curiosity)

> **Status:** Modules A-H described. As of 25 March 2026:
> - Module C (informativeness rho): validated (Exp0.4-0.7, two-stage gate)
> - Module D (tree + split/merge): validated (Exp0.1-0.3, Exp0.8 governor)
> - Module E (deltas + boundaries): validated (halo w∈[2,4], SeamScore, Phase 1/2)
> - Module F (benchmark): conducted (exp10 series, 158K+ trials, layout policy fixed)
> - Module G (Scale-Consistency): validated (SC-baseline AUC 0.82-1.0, exp12a tau_parent PASS)
> - **Module H (three-layer rho): validated** (exp17, 1080 configs, reusability 12/12 PASS, cascade quotas, streaming pipeline)
> - Modules A, B (canonicalization, cache): NOT implemented (not on critical path)
> - Phase 1 completed (20.03.2026). Phase 2 completed (21.03.2026). Phase 3 completed (22.03.2026). Phase 3.5 completed (23.03.2026). Phase 4 multi-tick completed (25.03.2026).
> - Next step: C-optimization scoring (roadmap) + Phase 5 (robustness/noise).

## Core Logic
1. Only modules that provide standalone benefit survive: cache, detector, recomputation scheduler, profiling.
2. Start not from a "smart tree" but from **data identity**: canonicalization → hash → cache.
3. A tree without stability and metrics is a decorative bush. First define rules for change and measurement, then structure.

---

## A. Canonicalization + Content Hash of Tiles (First Viable Building Block)
**Goal:** any data region (tile/patch/block) gets a stable ID and a "meaning" hash, not a "memory layout" hash.

**Deliverables:**
- `RegionPath`: path in tree (root → quadrants), stable byte serialization.
- `TileSpec`: shape, dtype, contiguous/stride normalization, padding policy.
- `Hash(tile)`: identical data → identical hash regardless of allocations.
- Table: `hash -> cached_result` + hit/miss counters.

**Tests:**
- Same tile in different buffers → same hash.
- Micro-change in data → hash changes (unless a separate "soft hash" mode is enabled).

**Standalone value:** content-addressable cache for any pipeline.

---

## B. Change-Driven Recomputation Scheduler (Incrementality)
**Goal:** skip work if input hasn't changed.

**Deliverables:**
- API: `compute(tile) -> result`, wrapper `cached_compute(tile, key=hash)`.
- Task queue: recompute only changed tiles.
- Log: which tiles were recomputed and why.

**Metrics:**
- Fraction of tiles recomputed on local change (expected to be small).
- Latency/throughput with and without cache.

**Standalone value:** incremental recomputation as an independent optimization.

---

## C. "Interestingness" as a Measurable Function (No Trees Yet)
**Goal:** determine "where to compute" through measurements, not belief.

**Deliverables:**
- Tile-level scoring set:
  - gradient / high-frequency energy,
  - reconstruction error (if baseline/teacher available),
  - activation/gradient variance.
- Normalization of scores to a comparable scale.
- Thresholds derived from distribution quantiles, not guesswork.

**Standalone value:** ROI attention mask for compression, logging, adaptive upsampling, etc.

---

## D. Region Tree (Quadtree/Octree) + Split/Merge Rules
**Goal:** formalize space partitioning so the scheduler can work with addresses.

**Deliverables:**
- Node: `path, level, bbox, hash, score, state`.
- Operations:
  - `split(node) -> 4 children`,
  - `merge(children) -> parent`.
- Stability:
  - hysteresis (different split/merge thresholds),
  - minimum "node age" before collapse,
  - depth and/or change-rate limits.

**Standalone value:** adaptive complexity markup for scenes/data.

---

## E. Delta-Only Recomputation and Boundary Stitching
**Goal:** new leaves compute refinement, old ones are untouched, boundaries are reconciled.

**Deliverables:**
- Explicit step_delta definition (e.g., residual relative to parent_coarse).
- Recomputation policy:
  - `new_leaves`,
  - `changed_hash_leaves`,
  - `boundary_neighbors`.
- Level stitching:
  - overlap/padding,
  - boundary blending (seamless).

**Standalone value:** boundary-only refinement as an independent optimization.

---

## F. Benchmark + Management Overhead (Anti-Self-Deception)
**Goal:** measure whether tree/queue/hash overhead eats the gain.

**Deliverables:**
- Per-stage profiling: hashing / scheduling / compute / merge.
- Balance map: time and memory by component.
- Scenarios:
  - smooth scene (should be cheap),
  - one sharp object (local subdivision),
  - noise (should not subdivide infinitely).

---

## G. Scale-Consistency: Don't Break Parent-Scale Semantics (v1.6; updated v1.7)
**Goal:** guarantee that step_delta does not redefine parent_coarse by smuggling — refinement adds detail but does not push new LF meaning upward.

**Deliverables:**
- Implementation of operator pair (R, Up): `gaussian blur + decimation` / `bilinear upsampling`.
- Compute D_parent and D_hf metrics per tree node.
- Baseline experiment: collect D_parent/D_hf distributions on positive (correct refinement) and negative (artificial LF drift) cases. Evaluate separability (AUC, effect size, quantile separation).
- Data-driven thresholds τ_parent[L] from baseline results.
- Enforcement: damp step_delta / reject split / increase local strictness when D_parent > τ_parent.
- Integration of D_parent as a contextual signal in ρ (not self-sufficient).

**Kill criterion:** if separability between positive and negative is insufficient — revisit metrics or the (R, Up) pair, not the thresholds.

**Protocol:** `scale_consistency_verification_protocol_v1.0.md`.

**Standalone value:** cross-scale semantic coherence diagnostics for any hierarchical representation.

---

## Phase 4: MultiStageDedup Testing (Planned)

**Goal:** activate MultiStageDedup (3-level deduplication from Enox infrastructure) with epsilon > 0 in multi-pass/iterative refinement mode. In the current single-pass (epsilon=0.0) no level fires — this is scaffolding for Phase 4.

**Dependencies:** P4a (downstream consumer test) — multi-pass context is required for dedup to be meaningful.

**Kill criteria:** dedup reduces budget waste on repeatedly processed units without PSNR loss > 0.5 dB.

---

## H. Three-Layer Rho Decomposition (exp17, March 22-23, 2026)

**Status:** experimentally validated. Architecture works, reusability PASS across all spaces and scales (100 / 1K / 10K).

### Core Idea

The monolithic rho is replaced by a cascade of three layers:

```
Layer 0: TOPOLOGY       — "how the space is structured" (data-independent)
Layer 1: DATA YES/NO    — "where non-trivial signal exists" (data-dependent, query-independent)
Layer 2: QUERY          — "of what exists — where is what I need" (task-specific)
```

Each layer narrows the working set for the next. Reusability increases bottom-up: topology is immutable, the data map updates rarely, the query changes constantly.

### Key Architectural Decisions

1. **Cascade quotas (Variant C):** L1 does not cut by a fixed threshold — each L0 cluster guarantees a minimum number of survivors proportional to cluster size. The survival threshold is tied to budget_fraction. Topology dictates quotas. No region goes extinct.

2. **Streaming pipeline:** instead of L0(all) -> L1(all) -> L2(all) — per-cluster processing: each L0 cluster goes through L0 -> L1 -> L2 before moving to the next. Clusters processed in L0 priority score order. Advantages:
   - First results appear after 1 cluster, not after the full map.
   - L1 pruning genuinely reduces the number of refinements (budget per-cluster).
   - Can stop early — partial results are already useful.

### Experimental Results (exp17, 1080 configs)

- **Reusability:** 12/12 PASS (min ratio = 0.838, threshold = 0.80). Frozen tree is reusable across different queries.
- **PSNR:** 2-4 dB below single_pass/kdtree on grid (L1 pruning cost), parity on graph/tree.
- **Timing (single query):** streaming faster than batch by 10-20%, but kdtree (scipy C) is faster than both.
- **Amortized:** three_layer beats single_pass starting from 2 queries on tree_hierarchy.

### Roadmap: C/Cython Optimization (Phase 5+)

**Rationale:** the current bottleneck is Python overhead in scoring phases. Refinement (numpy) is already near-C speed. Rewriting scoring in C will have a multiplicative effect specifically for streaming, because streaming scores FEWER units.

**Projected speedups (C vs Python):**

| Component | Python (current) | C (projected) | Speedup |
|-----------|-----------------|--------------|---------|
| L0 topo extraction (graph) | 70ms | ~2-5ms | 15-35x |
| L1 presence scoring | 5ms | ~0.5ms | 10x |
| L2 query scoring | 8ms | ~0.8ms | 10x |
| Refinement (numpy) | 19ms | ~19ms | 1x (already fast) |
| **TOTAL streaming (1K grid)** | **33ms** | **~22ms** | **1.5x** |
| **TOTAL streaming (graph)** | **68ms** | **~22ms** | **3x** |

**Expected outcome:** streaming in C would beat kdtree on graph/tree spaces and match it on grid. While retaining side data (topo features, zones, cluster structure, decision journal) that kdtree does not provide.

**Implementation priority:**
1. L0 topo in C/Cython (maximum ROI — 70ms -> 5ms)
2. L1 + L2 scoring in C (single vectorized pass)
3. Refinement batching (grouping adjacent tiles for cache locality)

---

## I. Semantic Observables Logging (Non-Binding)
**Goal:** accumulate data for Track B without affecting Track A kill criteria.

**Deliverables:**
- Log tree topology per run (for topology stability analysis).
- Log LCA-distance vs. feature-distance scatter data.
- Log cluster purity if labeled inputs are available.
- Storage: append-only log alongside experiment results.

**Standalone value:** if Track B opens, this data is ready. If not — zero cost beyond storage.

---

## Mini-Roadmap (Each Step = A Standalone Trophy)
1. Hash+Cache for tiles (canonicalization, stable key, hit/miss, log).
2. Incremental scheduler (recompute only what changed).
3. Interestingness scoring (metrics + normalization + quantile thresholds).
4. Tree + hysteresis (split/merge without jitter).
5. Recomputation policy (new/changed/boundary) + gain measurement.
6. Step_delta + boundary stitching (overlap/blend for seamless output).
7. **Determinism** (canonical traversal order, deterministic probe, governor isolation → bitwise reproducibility at fixed seed + statistical stability across seeds).
8. Scale-Consistency baseline (R/Up, D_parent/D_hf, separability, τ_parent).
9. Scale-Consistency enforcement (damp/reject/strictness + integration into ρ).
10. (R, Up) sensitivity probe (Exp0.10, after step 8).
11. Semantic observables logging (non-binding, parallel with steps 8–10).
12. Optimizations (batching, kernels, GPU specifics) — only after stabilization.

---

## Future Research (Post-Phase 4)

### RG-flow Verification

RG-flow verification (post-Phase 4): basin membership requires multi-pass to form basins. Exp18 showed r=0.019 in single-pass (FAIL). Connected to: Exp0.10 (R,Up) sensitivity (concept section 8.10).

### Governor EMA Restoration + Sweep

Reconnect EMA feedback from exp0.8 as global strictness thermostat (lost during Phase 2 pipeline assembly). Two-layer architecture: hardware parameter sets range, EMA feedback controls within range. Applicable to batch and frozen reuse modes; NOT applicable to streaming (cross-cluster bleed).

### Rule: Magic Numbers Documentation

> Every numeric constant in config.py must reference the experiment/sweep from which it was derived. If no reference exists, mark with `# UNVALIDATED, needs sweep (exp19)`. Rule introduced March 25, 2026.

### Convergence Detector (missing)

**Discovered:** runtime visualization (viz/index.html) showed that after refining all useful tiles the system continues spinning empty ticks — budget not exhausted, but no candidates remain or all are below threshold. Governor regulates *intensity* of refinement (strictness) but does not decide to *stop*.

**Problem:** no formal convergence detector. Current stop conditions:
- Budget exhaustion (hard stop)
- Waste >= R_max (force-stop current tick, not session)
- Missing: "accepted == 0 for last K ticks -> stop session"

**Required:**
```
Convergence rule:
  IF accepted == 0 for last K ticks (K >= 2):
    STOP session, return remaining budget as unspent
  IF accepted < probe_count for last K ticks:
    STOP session (only probe discovers anything = diminishing returns)
```

**Dependency:** fix during Phase 4 pipeline assembly. No new experiments needed — purely engineering at harness loop level.

**Status (March 25, 2026):** IMPLEMENTED in pipeline.py Phase 4 multi-tick. accepted==0 for convergence_window ticks -> stop.

### Gate Health Thresholds — Data-Driven Calibration (open problem)

**Discovered:** when implementing the two-stage gate in the visualization (viz/index.html), it turned out that instability/FSR thresholds for switching Stage 1 <-> Stage 2 cannot be fixed. Hardcoded values from P2a sweep (0.3/0.2) don't work on data with different statistics — instability metric (CV residual) scales differently depending on data, tile size, refinement stage, and noise level.

**Solution — composite approach (two mechanisms simultaneously):**

**(A) Pilot calibration.** First K ticks (K=2-3) — always Stage 2 (statistics collection). After pilot: threshold = percentile(observed) x factor. Provides absolute threshold calibrated to specific data.

**(B) Relative threshold.** Switch to Stage 1 when instability < EMA(instability over last K ticks) x factor. Adapts to trend: if instability steadily drops (refinement progressing), threshold tightens.

**Composite A+B:** threshold = min(pilot_threshold, ema_relative_threshold). Pilot sets upper bound (worst-case from initial data). EMA pulls down as data stabilizes.

**Dependency:** implement during Phase 4 pipeline assembly. Requires sweep for factor validation in A and B.

**Status (March 25, 2026):** IMPLEMENTED in pipeline.py Phase 4 multi-tick.

### FSR Metric — Refined Tiles Inflate Sign-Flip Rate (bug)

**Discovered:** viz testbed. FSR (fraction of sign-flips) counts all tiles including just-refined ones. Refined tile -> residual~0 -> guaranteed sign-flip. At 40 tiles/tick out of 1024 this is ~4% baseline FSR even on ideal data. If fsrThresh < 4% -> gate stuck in Stage 2 forever.

**Fix:** FSR must be computed **only on non-refined tiles**. Implemented in viz, needs verification in real pipeline (exp06, exp07).

**Dependency:** verify during Phase 4 assembly.

**Status (March 25, 2026):** IMPLEMENTED in pipeline.py Phase 4 multi-tick.

### Gate Oscillation — Missing Hysteresis

**Discovered:** viz testbed. When instability ~ threshold, gate jumps Stage 1->2->1->2 every tick. Each tick uses a different rho function (residual-only vs combo), breaking DET-2 (metric stability across seeds): one seed may land on Stage 1, another on Stage 2 in the same tick.

**Required:** hysteresis band. Switch to Stage 1 when instab < threshold x 0.8, back to Stage 2 when instab > threshold x 1.2.

**Dependency:** Phase 4 pipeline.

**Status (March 25, 2026):** IMPLEMENTED in pipeline.py Phase 4 multi-tick.

### EMA Weights Uncalibrated on First Ticks

**Discovered:** viz testbed. On tick 1 prevGains=null -> EMA cannot adapt weights -> first ticks of Stage 2 operate with arbitrary initial weights (0.4/0.35/0.25). Decisions based on them are irreversible — refined tiles cannot be unrefined.

**Problem:** pilot calibration solves the **threshold** problem but not the **weights** problem. First K ticks of Stage 2 operate blind.

**Dependency:** Phase 4. Related to pilot calibration threshold — can be unified into a single pilot phase.

**Status (March 25, 2026):** IMPLEMENTED in pipeline.py Phase 4 multi-tick.

### Gain Threshold — Gain/Cost Incommensurability

**Discovered:** viz testbed. When good candidates are exhausted (80%+ tiles refined), the relative threshold (quantile of current candidates) passes tiles with negligible gain because threshold is relative. An absolute threshold is needed, but gain (MSE) and cost (compute units) are incommensurable — direct comparison gain > cost is meaningless.

**Broader problem:** no single "currency" for gain and cost. Governor operates in compute units, rho in normalized scores, gain in MSE. Three different spaces.

**Dependency:** Phase 4. Fundamental architectural question.

**Status (March 25, 2026):** IMPLEMENTED in pipeline.py Phase 4 multi-tick.

### Probe Overlaps with Rejected Tiles

**Discovered:** viz testbed. Probe targets lowest-rho unselected tiles. But governor may evaluate and consciously reject a tile (gain < threshold). Probe can target the same tile — wasting probe budget on a tile governor just rejected.

**Fix:** probe must target **unevaluated** tiles (not in evalCount), not merely unselected. Probe = exploration of the unknown, not re-evaluation of the rejected.

**Dependency:** Phase 4 pipeline. Simple fix at probe selection logic level.

**Status (March 25, 2026):** IMPLEMENTED in pipeline.py Phase 4 multi-tick.

### No Post-Refinement Quality Feedback

**Discovered:** viz testbed. Tile accepted -> refined -> marked forever. Nobody checks whether refinement actually improved quality. If the operator (or in AdaHiMem: compression/decompression) reconstructs poorly, tile is considered "refined" forever, governor gets no signal about poor quality.

**Problem:** per-tile strictness accounts only for accept/reject decision, not post-refinement quality. Tile with high gain but poor refinement — blind spot.

**Dependency:** Phase 4+. Requires additional residual pass after refinement. Increases compute cost by ~10-20%.

### Noise-Fitting — System Optimizes to Noise, Not Signal (fundamental problem)

**Discovered:** viz testbed with noise on the source field (sigma > 0). All experiments (exp00a-exp18) used clean synthetic data or data with negligible noise. The system minimizes residual = |current - observed_data|. If observed_data is noisy, the system spends budget reproducing noise exactly.

**Dependency:** fundamental problem. Not Phase 4 — Phase 5 (robustness). But Phase 4 experiments must not run on noisy data without noise-awareness.

**Plan:** sweep of denoising variants (A-D + SureShrink + BayesShrink + SURE-Bayes blend), then composite from Pareto-best. Details in `docs/experiment_hierarchy.md` -> P5-noise (exp20a/b/c).

### Streaming Budget Control (B+C)

Smooth budget control for streaming mode (currently only binary go/stop):
- **(B) L0-informed allocation (Institutional Inequality Formula).** Cluster budget weight:

  $$W_{cluster} = N_{units} \times (1 - ECR)^{\gamma}, \quad \gamma \geq 2$$

  Derivation from StrictnessTracker thermodynamics: $$E[\Delta S] = (1 - ECR) \times 0.9 + ECR \times 1.5$$. GREEN (ECR=0.05): E[ΔS]=0.93 → thrives. YELLOW (ECR=0.15): E[ΔS]=0.975 → equilibrium. RED (ECR=0.33): E[ΔS]=1.098 → dies from WasteBudget. Quadratic (γ=2) matches actual throughput capacity: GREEN ~90%, YELLOW ~72%, RED ~42% of nominal. γ to be validated in sweep: γ ∈ {1.0, 1.5, 2.0, 2.5, 3.0, 4.0} (from linear baseline to aggressive).

- **(C) Adaptive redistribution:** unspent cluster budget flows to subsequent clusters (forward carry). RED gets a strict minimum, but if anomalously clean, receives leftover from GREEN.

### Governor + B+C Sweep Test

Sweep: 3 modes (batch/reuse/streaming) x 3 hardware profiles (low/mid/high) x 6 γ ∈ {1.0, 1.5, 2.0, 2.5, 3.0, 4.0} x 4 spaces x 20 seeds. Metrics: PSNR, time, reject rate, compliance, budget utilization. Kill: batch/reuse — EMA improves compliance; streaming — B+C(γ*) >= equal-allocation baseline.

---

## What Remains Useful Even If "The Whole Thing" Doesn't Pan Out
- Content-addressable cache.
- Incremental recomputation.
- ROI "interestingness" mask.
- Stable adaptive complexity markup.
- Cross-scale coherence diagnostics (D_parent / D_hf).
