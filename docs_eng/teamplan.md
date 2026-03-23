# Curiosity: Parallel Team Work Plan

## Context

The Curiosity project has completed the Exp0.1–0.8 series. Ahead are P0–P4 + SC-baseline + cross-space Halo validation. The goal is to distribute work across 4+ parallel executors, with Barmagloth acting as architect (makes decisions at forks, reviews results, does not write code).

**Constraint**: GPU — NVIDIA RTX 2070 8 GB (CUDA 12.8, PC 2). Uses PyTorch 2.10.0+cu128 (Python 3.12). Previously: AMD Radeon 780M (PC 1, DirectML) — not used since Phase 1.

---

## Streams of Work (Streams)

### ✅ Phase 0: COMPLETED (March 18, 2026)

Four streams, all independent of each other:

| Stream | Executor | Task | Type | Dependencies |
|--------|----------|------|------|--------------|
| **S1: Environment** | Executor A | Set up ROCm + PyTorch on AMD GPU. Verify compatibility of existing code (exp07, exp08). Write `environment_1.md` with versions. | Infrastructure | None — starts immediately |
| **S2: Halo cross-space** | Executor B | Extend `phase2_probe_seam/exp_seam_crossspace.py` — validate Halo (cosine feathering, ≥3 elements) across all 4 space types (scalar grid, vector grid, irregular graph, tree hierarchy). CPU-only. | Validation | None — reuses phase2 code |
| **S3: P2a sweep design** | Executor C | Implement sensitivity sweep of thresholds (instability_threshold, FSR_threshold) across 5 scenes (clean/noise/blur/spatvar/jpeg) **× 4 space types** (scalar grid, vector grid, irregular graph, tree hierarchy). CPU-only, reuses exp07/exp08 code + phase2 cross-space infrastructure. | Experiment | None — data and code are available |
| **S4: SC-baseline scaffold** | Executor D | Implement the SC-baseline scaffold per the `scale_consistency_verification_protocol_v1.0.md` protocol: SC-0 (idempotence R), SC-1 (positive/negative baselines), SC-2 (D_parent, D_hf computation). CPU-only. | Code + validation | None — protocol is ready |

**Phase 0 Results:**
- S1: CPU venv (Python 3.13) + GPU venv (Python 3.12 + DirectML, Radeon 780M, 2-3× speedup at 2048+)
- S2: Halo FAIL on trees (0.56×). Rule: parallelism ≥ 3 AND no context leakage. Grid/graph: ok. Tree: never.
- S3: P2a sweep code ready (20K configurations), not yet executed
- S4: SC-baseline passed. D_parent updated: ‖R(δ)‖ / (‖δ‖ + ε), R=σ3.0. AUC 0.824–1.000 across 4 spaces.

**Additionally completed (out of plan):**
- coarse_shift generator fixed (spatially coherent sign fields)
- D_parent combo sweep: 66 combinations tested
- abs_vs_signed auxiliary rejected (counterproductive with fixed generator)

**Forks for the architect (end of Phase 0):** ✅ All resolved
- S2 result: Halo works across all 4 spaces → "mandatory invariant" status confirmed. If not → decision on Halo modification.
- S1 result: ROCm works → proceed to GPU experiments. If not → fallback to cloud / WSL + CUDA.

---

### ✅ Phase 1: COMPLETED (March 20, 2026)

| Stream | Executor | Task | Type | Dependencies |
|--------|----------|------|------|--------------|
| **S1: P0 — Exp0.9b0** | Executor A | Buffer-scaling probe: O(k) vs O(M) overhead. Grid vs compact. Kill compact if overhead > 20%. Requires GPU. | Experiment (GPU) | S1 Phase 0 (environment) |
| **S1b: DET-1** | Executor A | Seed determinism: canonical traversal order (Z-order tie-break), deterministic probe, governor isolation. Absorbs 0.9h. Kill: any divergence = fail. **Blocker for Phase 2.** | Validation | S1 P0 (layout determines traversal order) |
| **S2: P1-B2 prototype** | Executor B | Dirty signatures: 12-bit signature (seam_risk + uncert + mass), debounce, AUC > 0.8. CPU prototype, later ported to GPU. | Code + experiment | None (CPU) |
| **S3: P2a execution** | Executor C | Run sensitivity sweep (from Phase 0) across 5 scenes × 4 spaces. Determine: ridge width > 30% → manual thresholds ok; < 10% → P2b needed. **Important:** if ridge width differs across spaces — this is itself a significant result requiring an architect decision. | Experiment | S3 Phase 0 (code ready) |
| **S4: SC-baseline completion** | Executor D | SC-0..SC-4 passed. Remaining: SC-5 — set data-driven τ_parent[L]. Prepare SC-enforce (Phase 2). | Validation | S4 Phase 0 (scaffold) ✅ |
| **S5: Deferred revisit** | Executor E | Re-investigation of Morton layout / block-sparse / phase schedule with a different approach. Literature review + new ideas. Not an experiment — a research note with proposals. | Research | None |

**Phase 1 Results (all PASS):**
- S1 exp10 series: layout policy fixed (D_direct for grids, hybrid for trees, D_blocked conditional for spatial graphs, A_bitset fallback for scale-free)
- S1b exp10d: DET-1 PASS 240/240 bitwise match CPU+CUDA
- S2 exp11: dirty signatures PASS (AUC 0.91-1.0)
- S3 P2a: sensitivity PASS — ridge 100%, manual thresholds ok, P2b not needed
- S4 exp12a: tau_parent PASS (per-space thresholds)
- DET-2: PASS 8/8 (per-regime CV)

**Gate: Phase 1 → Phase 2:** PASSED. P0 layout CLOSED. DET-1 PASSED.

**Forks for the architect (end of Phase 1):** ✅ All resolved
- Layout: D_direct for grids, hybrid for trees, D_blocked conditional for spatial graphs, A_bitset fallback for scale-free
- P2b needed? No — ridge 100%
- SC-5: thresholds found, per-space
- Morton/block-sparse: Morton killed, block addressing viable only for spatial graphs

---

### ✅ Phase 2: End-to-End Pipeline Validation — COMPLETED (March 21, 2026)

| Stream | Executor | Task | Dependencies |
|--------|----------|------|--------------|
| **S1: P0 completion + DET-2** | Executor A | Exp0.9b (end-to-end if compact survives). DET-2 (cross-seed stability, 20 seeds × 4 spaces × 2 budgets). | P0 + DET-1 (Phase 1) |
| **S2: P1-B1 compression** | Executor B | Segment compression (degree-2 + signature-stable + length cap). Compression ratio > 50%, overhead < 10%. | P1-B2 (Phase 1) + P0 layout + DET-1 |
| **S3: P2b (conditional)** | Executor C | Online percentile estimation for adaptive threshold. Only if P2a showed narrow ridge. Otherwise — assists other streams. | P2a result |
| **S4: SC-enforce** | Executor D | Damp delta / reject split when D_parent > τ_parent. Integration of enforcement into the pipeline. | SC-baseline pass |
| **S5: Enox infra** | — | Four observation-only patterns (RegionURI, DecisionJournal, MultiStageDedup, PostStepSweep). Pure annotation, zero functional change. Groundwork for Phase 3. | Phase 2 pipeline |

**Phase 2 Results:**
- S1 (Pipeline Assembly): CuriosityPipeline assembled — gate + governor + SC-enforce + probe + traversal. ✅ DONE
- S2 (SC-Enforce): Three-tier pass/damp/reject + strictness-weighted waste budget + adaptive τ T4(N). ✅ DONE
- S3 (Segment Compression): Thermodynamic guards (N_critical=12, bombardment). Compression 60-66% on d7/d8. ✅ DONE
- S4 (E2E Validation): 240 configs, 4 spaces, DET-1 40/40 + DET-2 8/8. Topo profiling integrated. ✅ DONE
- S5 (Enox Infra): 4 observation-only patterns. ✅ DONE. Comparison: NO REGRESSION (15/20 bitwise SAME). DET-1 PASS.

**Gate: Phase 2 → Phase 3: ✅ PASSED**
- Pipeline assembled and E2E validated
- SC-enforce integrated
- Topo profiling integrated
- Enox infrastructure: ✅ DONE. NO REGRESSION

**Forks for the architect (end of Phase 2): ✅ All resolved**
- SC-enforce works: three tiers (pass/damp/reject), adaptive τ for trees
- Compression: profitable on d7/d8, guards cut unprofitable cases
- Enox: ADOPT RegionURI hash, ADAPT decision journal/dedup/sweep/provenance, SKIP phase sep/perspectives

---

### Phase 3: Semantics + Rebuild (Week 9-11) — COMPLETED (March 22, 2026)

| Stream | Executor | Task | Dependencies |
|--------|----------|------|--------------|
| **S1: P1-B3 anchors** | Executor A/B | Periodic rebuild + anchor insertion. Divergence < 5% vs full rebuild. | P1-B1 |
| **S2: P3a LCA-distance** | Executor C | Correlation of LCA-distance with feature similarity. Correlation > 0.3 → tree is semantic. | P1-B1 (compressed tree) |
| **S3: P3b bushes** | Executor D | Clusters of leaf-paths. Silhouette > 0.4 + stability. | P3a (can run in parallel) |
| **S4: C-pre (exp16)** | — | Trajectory profiles: 2-7 natural clusters, Gap>1.0, Silhouette>0.3 across 4 spaces. | Pipeline ready |

**Phase 3 Results:**
- S1 (Exp14 anchors): grid PASS, graph/tree FAIL. Divergence > 5% on irregular topologies.
- S2 (Exp15 LCA-distance): FAIL. Correlation < 0.3.
- S3 (Exp15b bushes): FAIL. Silhouette > 0.4 (clusters are real), but ARI < 0.6 (not stable across seeds). Revisit planned after Track C.
- S4 (Exp16 C-pre): PASS. Trajectory profiles found → **Track C UNFREEZE**.

**Gate: Phase 3 -> Phase 3.5: PASSED** (partial — anchors/LCA/bushes FAIL, but C-pre PASS opened Track C and motivated three-layer rho decomposition).

---

### Phase 3.5: Rho Decomposition (March 23, 2026) — DONE

> Not originally planned — emerged from Phase 3 results (anchors FAIL → "why isn't rho reusable?" → three-layer architecture).

| Stream | Task | Result |
|--------|------|--------|
| **Exp17: three-layer rho** | Decompose rho into L0 (topology) -> L1 (presence) -> L2 (query) | 1080 configs, reusability 12/12 PASS |
| **Cascade quotas** | Adaptive L1 threshold tied to L0 clusters | Fixed scalar_grid 1000: 0.725 FAIL -> 0.928 PASS |
| **Streaming pipeline** | Per-cluster L0->L1->L2 processing with L0-priority ordering | 10-20% faster than batch on grids |
| **Industry benchmarks** | kdtree, quadtree, wavelets, leiden comparison | kdtree faster on single query, 3L wins at >=2 queries on tree |
| **Roadmap C-opt** | Projected speedups from C/Cython rewrite of scoring | Recorded in workplan.md, section H |

**Key findings:**
- Three-layer architecture validated, frozen tree is reusable
- Bottleneck = refinement (numpy), not scoring → C-optimization of scoring will have multiplicative effect
- Bushes (exp15b) revisit planned: leaf-path similarity for merge candidates / downstream features

---

### Phase 4: Integration (Week 12-14)

| Stream | Executor | Task | Dependencies |
|--------|----------|------|--------------|
| **S1: P4a downstream** | Executor A | Classifier/autoencoder on adaptive-refined vs dense vs coarse. Metric loss < 2%. | All P0-P3.5 + SC |
| **S2: P4b matryoshka** | Executor B | Each nesting level is valid for downstream. | P4a |
| **S3: C-pre** | — | DONE (exp16, Phase 3). Track C UNFREEZE. | — |
| **S4: MultiStageDedup testing** | Executor C | Test MultiStageDedup with epsilon > 0 in multi-pass/iterative refinement mode. In single-pass (epsilon=0.0) dedup never fires — it is scaffolding that comes alive with multi-pass. Kill criteria: dedup must reduce budget waste on repeated units without PSNR loss > 0.5 dB. | P4a (multi-pass context) |

---

### Post-Phase 4 / Research Track

| Experiment | Description | Dependencies |
|------------|-------------|--------------|
| **Exp RG-flow verification** | Test whether refinement tree behaves as RG-flow trajectory. Basin membership as semantic metric (exp18 showed basins don't form in single-pass at 30% budget — needs multi-pass). Verify: convergence to fixed points, basin stability, universality under different (R, Up) pairs. | Multi-pass pipeline from Phase 4 |
| **Governor EMA restoration** | Reconnect EMA feedback from exp0.8 as global strictness thermostat. Orthogonal to StrictnessTracker (per-unit reputation) and WasteBudget (kill switch). Provides smooth intake regulation, not just emergency stop. Priority: optimization, not critical. | Phase 4 pipeline |
| **Streaming budget control (B+C)** | Smooth budget control for streaming mode (currently only binary go/stop). Two mechanisms: **(B)** L0-informed allocation — budget per cluster proportional to expected utility (GREEN → more, RED → less), not just size; **(C)** Adaptive redistribution — unspent cluster budget flows to subsequent clusters (forward carry). Test in sweep alongside Governor EMA. | Phase 4 pipeline + L0 zones |
| **Exp Governor-sweep** | Validate Governor EMA + streaming B+C on emulated hardware configs. Sweep: 3 pipeline modes (batch, frozen reuse, streaming) x 3 emulated hardware profiles (low/mid/high budget) x 4 spaces x 20 seeds. Metrics: PSNR, wall time, reject rate, waste exhaustion rate, compliance. Kill criteria: (1) batch+reuse: Governor EMA improves compliance vs no-governor; (2) streaming B+C: PSNR >= equal-allocation baseline. Dependencies: Governor EMA restoration + streaming B+C. | Governor EMA + B+C |

---

## Critical Path

```
Phase 0: S1(env) ──→ Phase 1: S1(P0) → S1b(DET-1) ──→ Phase 2: S2(P1-B1) ──→ Phase 3: S1(P1-B3) ──→ Phase 4: S1(P4)
                                                    └──→ Phase 2: S1(P0 finish + DET-2)
```

**Critical path:** Env → P0 → DET-1 → P1(B2→B1→B3) → P4 = ~14 weeks

**Parallel streams reduce actual elapsed time:**
- Halo cross-space (Phase 0) — not on the critical path, but blocks confidence in invariants
- P2, SC-baseline — parallel with P1, do not extend the critical path
- DET-2 — parallel with P1-B1 (after DET-1), blocks Track B but not Phase 3
- Deferred revisit — pure research, blocks nothing

---

## Architect Decision Points

| When | What to decide | Input data |
|------|---------------|------------|
| End of Phase 0 | ✅ Resolved | Result |
|---|---|---|
| ROCm? | CPU + DirectML (ROCm does not support Windows iGPU) |
| Halo cross-space? | Partially: grid/graph yes, tree no. Rule derived. |
| D_parent fail? | Fixed: σ=3.0 + lf_frac normalization |
| coarse_shift? | Generator fixed to spatially coherent |
| End of Phase 1 | ✅ Resolved: layout policy fixed, P2b not needed, SC pass, Morton killed | All streams PASS |
| End of Phase 2 | ✅ Resolved (March 21, 2026). SC-enforce: pass/damp/reject. Compression: 60-66%. Enox: 4 observation-only patterns. | All streams PASS |
| End of Phase 3 | Resolved (March 22, 2026). Anchors: grid PASS, graph/tree FAIL. LCA-distance FAIL. Bushes FAIL (ARI<0.6). C-pre PASS → Track C UNFREEZE. | Exp14-16 reports |
| End of Phase 4 | Instrument Readiness Gate passed? Transition to Track B? | P4a, P4b (C-pre already DONE) |

---

## Key Files

- `docs/experiment_hierarchy.md` — dependency graph, kill criteria
- `docs/workplan.md` — modules A-H
- `docs/scale_consistency_verification_protocol_v1.0.md` — SC protocol
- `docs/concept_v1.8.md` — canonical concept (current)
- `experiments/phase2_probe_seam/` — code for reuse in Halo cross-space
- `experiments/exp07_gate/`, `experiments/exp08_schedule/` — code for P2a sweep

---

## Verification

After each phase:
1. All kill criteria verified (numbers, not opinions)
2. Holm-Bonferroni corrections for multiple comparisons
3. 10–20 seeds for reproducibility
4. Results recorded in `docs/experiment_results.md` (append)
5. Architect reviews before proceeding to the next phase

**Cross-space validation principle:** Any experiment claiming a statement about "arbitrary spaces" MUST be verified on at least 4 space types (scalar grid, vector grid, irregular graph, tree hierarchy) — the same ones on which SeamScore has been validated. A result on a single space type (e.g., 2D pixel grid) is NOT considered sufficient for generalization.
