# Work Plan (Curiosity)

> **Status:** Modules A-G described as implementation plan. As of 20 March 2026:
> - Module C (informativeness ρ): validated (Exp0.4-0.7, two-stage gate)
> - Module D (tree + split/merge): validated (Exp0.1-0.3, Exp0.8 governor)
> - Module E (deltas + boundaries): validated (halo w∈[2,4], SeamScore, Phase 1/2)
> - Module F (benchmark): conducted (exp10 series, 158K+ trials, layout policy fixed)
> - Module G (Scale-Consistency): validated (SC-baseline AUC 0.82-1.0, exp12a τ_parent PASS)
> - Modules A, B (canonicalization, cache): NOT implemented (not on critical path)
> - P0 Layout CLOSED. DET-1 PASS. DET-2 PASS. Phase 1 completed 20 March 2026.
> - Phase 2 COMPLETED (March 21, 2026). Pipeline assembled, SC-enforce integrated, E2E validated.
> - Enox infrastructure: 4 observation-only patterns — ✅ DONE. NO REGRESSION.
> - Next step: Phase 3.

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

## H. (R, Up) Sensitivity Probe (Exp0.10)
**Goal:** verify that system behavior is qualitatively stable across different (R, Up) pairs.

**Dependency:** after SC-baseline (module G baseline experiment).

**Deliverables:**
- Test 4 pairs: gaussian+bilinear (default), box+nearest, Lanczos+bicubic, haar wavelet.
- Compare D_parent/D_hf distributions, tree topology divergence, PSNR ceiling, SC-baseline separability (ROC-AUC).
- Decision: default justified, or pair selection mechanism needed.

**Kill criterion:** topology + D_parent stable (±20%) → default stands. Divergence > 50% → new open question.

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

## What Remains Useful Even If "The Whole Thing" Doesn't Pan Out
- Content-addressable cache.
- Incremental recomputation.
- ROI "interestingness" mask.
- Stable adaptive complexity markup.
- Cross-scale coherence diagnostics (D_parent / D_hf).
