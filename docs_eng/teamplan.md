# Curiosity: Parallel Team Work Plan

## Context

The Curiosity project has completed the Exp0.1–0.8 series. Ahead are P0–P4 + SC-baseline + cross-space Halo validation. The goal is to distribute work across 4+ parallel executors, with Barmagloth acting as architect (makes decisions at forks, reviews results, does not write code).

**Constraint**: GPU is AMD Radeon 780M (no CUDA). Uses DirectML + PyTorch (Python 3.12). CPU venv on Python 3.13.

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

### Phase 1: P0 + DET-1 + Parallel CPU Tasks (Week 3–5)

| Stream | Executor | Task | Type | Dependencies |
|--------|----------|------|------|--------------|
| **S1: P0 — Exp0.9b0** | Executor A | Buffer-scaling probe: O(k) vs O(M) overhead. Grid vs compact. Kill compact if overhead > 20%. Requires GPU. | Experiment (GPU) | S1 Phase 0 (environment) |
| **S1b: DET-1** | Executor A | Seed determinism: canonical traversal order (Z-order tie-break), deterministic probe, governor isolation. Absorbs 0.9h. Kill: any divergence = fail. **Blocker for Phase 2.** | Validation | S1 P0 (layout determines traversal order) |
| **S2: P1-B2 prototype** | Executor B | Dirty signatures: 12-bit signature (seam_risk + uncert + mass), debounce, AUC > 0.8. CPU prototype, later ported to GPU. | Code + experiment | None (CPU) |
| **S3: P2a execution** | Executor C | Run sensitivity sweep (from Phase 0) across 5 scenes × 4 spaces. Determine: ridge width > 30% → manual thresholds ok; < 10% → P2b needed. **Important:** if ridge width differs across spaces — this is itself a significant result requiring an architect decision. | Experiment | S3 Phase 0 (code ready) |
| **S4: SC-baseline completion** | Executor D | SC-0..SC-4 passed. Remaining: SC-5 — set data-driven τ_parent[L]. Prepare SC-enforce (Phase 2). | Validation | S4 Phase 0 (scaffold) ✅ |
| **S5: Deferred revisit** | Executor E | Re-investigation of Morton layout / block-sparse / phase schedule with a different approach. Literature review + new ideas. Not an experiment — a research note with proposals. | Research | None |

**Gate: Phase 1 → Phase 2:**
- P0 layout locked (grid / compact).
- **DET-1 passed** (bitwise tree match at fixed seed). Without this, Phase 2 results are untestable.

**Forks for the architect (end of Phase 1):**
- P0 result → layout choice (grid / compact). Determines all of P1.
- P2a result → go/no-go for P2b (adaptive threshold). **Additionally:** if ridge width differs substantially across space types → may require space-dependent tuning (separate architect decision).
- SC-baseline → pass/fail. If fail → reconsider the (R, Up) pair.
- Deferred revisit → architect decides whether to bring Morton/block-sparse/schedule back into the plan.

---

### Phase 2: Compression + Enforcement + DET-2 (Week 6–8)

| Stream | Executor | Task | Dependencies |
|--------|----------|------|--------------|
| **S1: P0 completion + DET-2** | Executor A | Exp0.9b (end-to-end if compact survives). DET-2 (cross-seed stability, 20 seeds × 4 spaces × 2 budgets). | P0 + DET-1 (Phase 1) |
| **S2: P1-B1 compression** | Executor B | Segment compression (degree-2 + signature-stable + length cap). Compression ratio > 50%, overhead < 10%. | P1-B2 (Phase 1) + P0 layout + DET-1 |
| **S3: P2b (conditional)** | Executor C | Online percentile estimation for adaptive threshold. Only if P2a showed narrow ridge. Otherwise — assists other streams. | P2a result |
| **S4: SC-enforce** | Executor D | Damp delta / reject split when D_parent > τ_parent. Integration of enforcement into the pipeline. | SC-baseline pass |

---

### Phase 3: Semantics + Rebuild (Week 9–11)

| Stream | Executor | Task | Dependencies |
|--------|----------|------|--------------|
| **S1: P1-B3 anchors** | Executor A/B | Periodic rebuild + anchor insertion. Divergence < 5% vs full rebuild. | P1-B1 |
| **S2: P3a LCA-distance** | Executor C | Correlation of LCA-distance with feature similarity. Correlation > 0.3 → tree is semantic. | P1-B1 (compressed tree) |
| **S3: P3b bushes** | Executor D | Clusters of leaf-paths. Silhouette > 0.4 + stability. | P3a (can run in parallel) |

---

### Phase 4: Integration (Week 12–14)

| Stream | Executor | Task | Dependencies |
|--------|----------|------|--------------|
| **S1: P4a downstream** | Executor A | Classifier/autoencoder on adaptive-refined vs dense vs coarse. Metric loss < 2%. | All P0–P3 + SC |
| **S2: P4b matryoshka** | Executor B | Each nesting level is valid for downstream. | P4a |
| **S3: C-pre** | Executor C | Is there natural clustering in trajectory features? Go/no-go for Track C. | P3 results |

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
| End of Phase 1 | Layout (grid/compact)? P2b needed? SC pass? Morton/schedule return? | P0, P2a, SC, S5 reports |
| End of Phase 2 | SC-enforce works? Compression sufficient? | S2, S4 reports |
| End of Phase 3 | Tree is semantic? C unfreezes? | P3a, P3b reports |
| End of Phase 4 | Instrument Readiness Gate passed? Transition to Track B? | P4a, P4b, C-pre |

---

## Key Files

- `docs/experiment_hierarchy.md` — dependency graph, kill criteria
- `docs/workplan.md` — modules A–F
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
