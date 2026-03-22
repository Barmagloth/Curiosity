# Curiosity — Experiment Hierarchy (v2.1)

This document captures the current status, dependencies, and execution order of experiments.

v2.1: added DET level (determinism and reproducibility) — cross-infrastructure requirement.

Updated after Phase 0 (parallel validation: halo cross-space, SC-baseline, D_parent fix).

---

# Mapping: folders → questions → validation plan

| Folder | Question | Validation plan | Status |
|--------|----------|-----------------|--------|
| `exp01_poc/` | Does adaptive refinement work? | — | ✅ Yes (PoC) |
| `exp02_cifar_poc/` | (same, CIFAR) | — | ✅ Yes |
| `exp03_halo_diagnostic/` | Is halo required? | — | ✅ Yes |
| `phase1_halo/` | Halo: r_min, blending, hardened | §A (A1+A2+A3) | ✅ Closed |
| `exp04_combined_interest/` | Is combined interest needed? | — | ✅ Yes |
| `exp05_break_oracle/` | Oracle-free verification | — | ✅ Yes |
| `exp06_adaptive_switch/` | Auto-switch ρ | — | ✅ Yes |
| `exp07_gate/` | Two-stage gate | — | ✅ Yes |
| `exp08_schedule/` | Schedule + governor + probe | — | ✅ Closed |
| `phase2_probe_seam/` | Probe + SeamScore validation | §B (B1+B2) | ✅ Closed |
| `exp09a_layout_sandbox/` | Layout: grid vs compact (microbench) | §C (C3/Exp0.9a) | ✅ Partial |
| `halo_crossspace/` | Halo applicability across space types | Phase 0 | ✅ Closed (rule derived) |
| `sc_baseline/` | Scale-consistency D_parent/D_hf verification | Phase 0 (SC-0..SC-4) | ✅ Closed (SC-5 open) |
| `p2a_sensitivity/` | Sensitivity sweep of gate thresholds | P2a | ✅ PASS (ridge=100%, MANUAL_OK) |
| `exp10_buffer_scaling/` | Grid vs compact-with-reverse-map on GPU | P0 (0.9b0) | ✅ KILL compact (VRAM +38.6%). Grid baseline. |
| `exp10d_seed_determinism/` | Bitwise determinism at fixed seed | DET-1 | ✅ PASS (240/240) |
| `exp10e_tile_sparse/` | Tile-sparse layouts without global reverse_map | P0 (0.9b1) | ✅ A alive, B/C killed. See exp10f. |
| `exp10f_packed_lookup/` | Packed tiles + alternative lookup (hash/direct) | P0 (0.9b2) | ⚠️ D passes Contour A, fails Contour B (peak VRAM from conv2d workspace). E archived as contingency. |
| `exp10g_dual_benchmark/` | Dual-mode benchmark: manual stencil (layout cost) vs conv2d (operator cost). Resolves D's Contour B. | P0 (0.9b3) | ✅ D_direct PASS both contours. -54% to -80% time, -36% to -86% peak VRAM. |
| `exp10h_cross_space/` | D_direct on vector_grid and tree_hierarchy | P0 (0.9b3) | ✅ vector_grid 72/72 PASS. tree FAIL 0/108 (configs too small → exp10j). |
| `exp10i_graph_blocks/` | Graph block-based addressing with 3 partition strategies | P0 (0.9b3) | ✅ Spatial graphs conditional (cbr≤0.35). Scale-free rejected (cbr=0.66). |
| `exp10j_tree_perlevel/` | Per-level independent D_direct vs A_bitset benchmark for trees. Finds break-even thresholds per level. | P0 (0.9b3) | ✅ matmul: D wins at p<0.40 any N_l. stencil: D saves memory, never time. Contour B 45% PASS. |
| `exp11_dirty_signatures/` | 12-bit dirty signature + debounce | P1-B2 | ✅ PASS (AUC 0.91-1.0, baseline comparison + temporal ramp) |
| `exp11a_det2_stability/` | Cross-seed stability (DET-2) | DET-2 | ✅ PASS 8/8 (per-regime CV thresholds) |
| `exp12a_tau_parent/` | Data-driven τ_parent[L] per depth | SC-5 | ✅ PASS (per-space thresholds, specificity 1.000) |
| `exp13_segment_compression/` | Segment compression with thermodynamic guards (N_critical=12, bombardment guard) | P1-B1 | ✅ PASS (overhead eliminated on small trees) |
| `exp14a_sc_enforce/` | Scale-consistency enforcement: three-tier pass/damp/reject + strictness-weighted waste budget + adaptive τ T4(N)=τ_base*(1+β/√N) | SC-enforce | ✅ PASS |
| `exp_phase2_pipeline/` | Full pipeline assembly (gate + governor + SC-enforce + probe + traversal) + topological pre-runtime profiling (hybrid Forman/Ollivier curvature, three-zone classifier v3, η_F entropy index) + Enox infrastructure (RegionURI, DecisionJournal, MultiStageDedup, PostStepSweep — observation-only, all defaults False) | Phase 2 + Topo + Enox | ✅ PASS |
| `exp_phase2_e2e/` | End-to-end validation: 4 space types, 240 configs. DET-1 recheck 40/40 + DET-2 recheck 8/8 (with topo profiling). irregular_graph re-run with zone classification (GREEN 75%/RED 25%) | Phase 2 | ✅ PASS |
| `exp_deferred_revisit/` | Research note: Morton/block-sparse/schedule | — | ✅ Done |

**Phase 2 note:** Graph clustering upgraded from k-means to Leiden (community detection), validated on 10 pathological topologies: Swiss Roll, Barbell, Hub-Spoke, Ring of Cliques, Bipartite, Erdos-Renyi, Grid, Planar Delaunay, Mobius strip.

**Topo Profiling note (21.03.2026):** Topological pre-runtime profiling added to IrregularGraphSpace. Hybrid Forman/Ollivier curvature with hardware-calibrated budget (Synthetic Transport Probe). Three-zone classifier v3 (κ_mean + Gini(PageRank) + η_F) stamps each graph GREEN/YELLOW/RED before pipeline starts. η_F = σ_F / √(2⟨k⟩) — dimensionless entropy index normalized against Poisson noise floor of Erdős-Rényi random graph with same mean degree. Threshold η=0.70 selected from clean gap [0.60, 0.76] in corpus: all YELLOW graphs (Grid, Ladder, Planar, Möbius) have η < 0.60, all RED graphs (ER, Bipartite) have η > 0.76. Validated at 97% accuracy on 35-graph corpus. Pre-runtime overhead: P50=56ms, MAX=125ms.

**Enox note (21.03.2026):** Four observation-only infrastructure patterns from the Enox framework, implemented for the project's needs: (1) RegionURI — SHA256 unit address, (2) DecisionJournal — append-only decision log, (3) MultiStageDedup — 3-level deduplication (scaffolding for multi-pass Phase 3, epsilon=0.0 → never fires), (4) PostStepSweep — sibling dirty-sig detection in tree hierarchy. All patterns are pure annotation, never modify state. Defaults = False, zero overhead. Baseline fingerprint: 20 runs, DET-1 PASS. Integration complete. NO REGRESSION (15/20 bitwise SAME).

**Note:** §A/B/C are sections of the validation plan written between Exp0.3 and Phase 1.
In §B, "B1/B2" = probe scenes. In P1 below, "B1/B2/B3" = tree compression. Different contexts.

---

# Closed (No Further Experiments Needed)

| # | Question | Status | Source |
|---|----------|--------|--------|
| 1 | Does adaptive refinement work? | **Yes** | PoC, Exp0.1–0.2 |
| — | Is halo mandatory? | **Yes**, w ∈ [2,4], cosine feather | Exp0.2–0.3, Phase 1 |
| — | Is probe mandatory? | **Yes**, uncert, 5–10% budget | Exp0.8, Phase 2 |
| — | SeamScore as production metric | **Yes**, dual check works in 4 spaces | Phase 2 |
| — | Governor (EMA) for budget | **Yes**, StdCost −50%, penalty −85% | Exp0.8 |
| 2 | Is combined interestingness needed? | **Yes, under signal degradation.** Two-stage gate | Exp0.4–0.7b |
| 3 | Phase schedule by depth? | **No** under current conditions | Exp0.8v5 |
| — | Halo cross-space applicability | **Rule derived** (grid/graph: yes, tree: no). boundary parallelism >= 3 AND no context leakage | Phase 0 |
| — | Morton layout (element-level sort) | **Killed** (12-15x overhead, zero compute benefit). But Morton as tile key encoding for lookup is alive, see exp10e | 0.9a sandbox |
| — | Block-sparse layout (element-level B=8) | **Killed** (expansion ratio). But paged sparse tiles alive as candidate in exp10e | 0.9a sandbox |
| — | Compact-with-global-reverse-map | **Killed** (VRAM +38.6%). Dense bookkeeping kills sparse compute gains | exp10 |
| — | P2a: manual gate thresholds | **Ok**, ridge 100%, P2b not needed | P2a sweep |
| — | DET-1: seed determinism | **PASS** (240/240 bitwise match CPU+CUDA) | exp10d |
| — | Non-overlapping writes determinism | **Clean** (bitwise match) | 0.9a sandbox |

---

# Open — Current Hierarchy

## Level 0: Infrastructure Prerequisites

Without answers to these questions, everything above is decorative.

### P0. GPU Layout

No upward dependencies — this is the foundation.

```
P0. GPU Layout
├── 0.9b0 (exp10): buffer-scaling probe — grid vs compact-with-reverse-map
│         RESULT: KILL compact (VRAM +38.6%). Grid is baseline.
│         But: implementation killed, not the sparse principle. Compute O(k) faster than O(M).
│
├── 0.9b1 (exp10e): tile-sparse layouts without global reverse_map
│         RESULT:
│         A (bitset): ALIVE — time -27..31%, VRAM +18%. Not sparse-memory, but execution layout.
│         B (packed Morton + binary search): KILLED on time (+1700%). Storage idea alive.
│         C (paged): KILLED (+5000-30000%). Dead permanently.
│
├── 0.9b2 (exp10f): packed tiles + alternative lookup
│         D passes Contour A, fails Contour B (peak VRAM from conv2d workspace).
│         E archived as contingency.
│
├── 0.9b3 (exp10g): dual-mode benchmark
│         RESULT: D_direct PASS both contours. -54% to -80% time, -36% to -86% peak VRAM.
│         Manual stencil (Contour A) + conv2d (Contour B) both pass.
│
├── 0.9b3 (exp10h): cross-space D_direct (vector_grid + tree_hierarchy)
│         RESULT: vector_grid 72/72 PASS both contours. tree 0/108 FAIL.
│         Tree failure: configs too small, per-level tile_map not amortized → exp10j.
│
├── 0.9b3 (exp10i): graph block-based addressing (3 partition strategies)
│         RESULT: Spatial graphs (random_geometric, grid_graph) conditionally viable
│         with spatial partition, cbr≤0.35. Scale-free (barabasi-albert) REJECTED, cbr=0.66.
│
├── 0.9b3 (exp10j): per-level tree break-even (158K trials)
│         RESULT: matmul op: D wins at p_l<0.375-0.40 for ALL level sizes.
│         stencil op: D saves memory but NEVER wins on time. Contour B: 45% PASS.
│         Policy: D_direct per-level only when operator is compute-heavy AND p_l < 0.40.
│
├── 0.9b:  end-to-end pipeline (after exp10g)
│         finalist vs grid, 4 spaces × 2 budgets
│
└── 0.9h:  ⟶ ABSORBED INTO DET-1 (✅ PASS)
```

**Principle:** addressing at the operation's level of abstraction. Refinement operates on tiles —
addressing must be tile-level, not element-level. Sparse on the outside, dense inside the tile.

**Current P0 status: CLOSED.**

Final layout policy (all space types resolved):

| Space type | Layout | Status | Evidence |
|------------|--------|--------|----------|
| scalar_grid | D_direct (packed tiles + direct tile_map) | Production | exp10g: both contours PASS |
| vector_grid | D_direct (packed tiles + direct tile_map) | Production | exp10h: 72/72 PASS |
| tree_hierarchy | Hybrid: D_direct per-level where p_l<0.40 + matmul op; A_bitset elsewhere | Validated | exp10j: break-even found |
| irregular_graph / spatial | D_blocked (graph block addressing) conditional | Conditional | exp10i: spatial partition, cbr≤0.35 |
| irregular_graph / scale-free | A_bitset (dense grid + bitset mask) fallback | Fallback only | exp10i: blocks rejected, cbr=0.66 |

Layout naming glossary:
- **D_direct** = packed tiles + direct tile_map (O(1) lookup, no element-level reverse_map)
- **A_bitset** = dense grid + bitset mask (simple fallback)
- **D_blocked** = graph block addressing (block_map[block_id] -> slot, spatial graphs only)
- **E_hash** = hash table lookup (archived, dominated by D_direct)

Killed forever:
- Element-level reverse_map[M] (exp10: VRAM +38.6%)
- Binary search lookup on GPU (exp10e-B: +1700%)
- Paged sparse tiles (exp10e-C: +9000%)
- Hash as primary lookup (exp10f-E: dominated by D_direct)
- Fixed-size blocks for scale-free graphs (exp10i: cbr 0.64-0.99)

**P0 output:** layout locked per space type. D_direct is production for grids.
Hybrid D_direct/A_bitset per-level for trees. D_blocked conditional for spatial graphs.

### DET. Determinism and Reproducibility (v1.8)

Cross-infrastructure requirement. Without DET-1, stability pass (Instrument Readiness Gate) cannot be achieved. Without DET-2, Track B is blocked.

```
DET. Determinism
├── DET-1: Seed determinism (Hard Constraint)
│         Two runs, identical inputs + seed → bitwise tree match.
│         CPU and GPU separately.
│         Components: canonical traversal order (Z-order tie-break),
│                     deterministic probe (seed = f(coords, level, global_seed)),
│                     governor isolation (EMA update after full step).
│         Kill criterion: any divergence = fail.
│         Absorbs 0.9h (halo overlap determinism) as a special case.
│
└── DET-2: Cross-seed stability (Soft Constraint)
          N=20 seeds × 4 spaces × 2 budgets.
          Metrics: PSNR, cost, compliance, SeamScore.
          Kill criterion: CV > 0.10 for any metric = fail.
          (τ_cv=0.10 is preliminary, refined from baseline.)
```

**Dependencies:** DET-1 depends on P0 (layout determines traversal order). DET-2 depends on DET-1 (determinism is a precondition for meaningful stability measurement).

**DET output:** confirmed testability (DET-1) and reproducibility (DET-2). Without this, Instrument Readiness Gate cannot be passed.

---

## Level 1: Route Representation

Depends on P0 (layout determines how active_idx enters the pipeline). Does not depend on tree "meaning" — pure storage engineering.

### P1. Compression and Structure Maintenance

```
P1. Tree / route compression
├── B2: dirty signatures (12 bits: seam_risk + uncert + mass)
│       debounce (2 consecutive hits)
│       scenarios: noise / structural event / drift
│       metrics: blast radius, latency-to-trigger, burstiness
│       ↑ this is the foundation — without it B1 and B3 don't know when to fire
│
├── B1: segment compression (degree-2 + signature-stable + length cap)
│       depends on B2 (merge criterion = signature stability)
│       metrics: memory vs node-per-node, local update cost
│
└── B3: anchors + periodic rebuild
        depends on B1 + B2
        scenario: frequent local updates
        comparison: (a) local only (b) local + periodic rebuild
        metrics: total cost over N steps, "dirt" accumulation
```

**P1 output:** tree storage format (flat nodes vs segments), dirty detection mechanism, rebuild strategy.

---

## Level 2: Combined Signal Reliability

Depends on P0 (pipeline works), does not depend on P1 (compression is orthogonal).

### P2. Auto-Tuning and ρ Robustness

Two-stage gate confirmed (Exp0.7b), but thresholds (instability, FSR) are manual.

```
P2. ρ-gate auto-tuning
├── P2a: sensitivity analysis — sweep instability/FSR thresholds
│        on existing scenes (clean/noise/blur/spatvar/jpeg)
│        question: how flat is the "optimality ridge"?
│        if wide → manual thresholds are fine, auto-tuning not needed
│        if narrow → adaptive threshold needed
│
└── P2b: adaptive threshold (only if P2a shows narrow ridge)
         online estimation of instability/FSR percentiles
         metrics: PSNR stability across scenes, overhead
```

**P2 output:** either "manual thresholds ± 30% — no difference" (and the question is closed), or a concrete auto-tuning mechanism.

---

## Level 3: Tree Semantics

Depends on P0 (layout) + P1 (storage format).

### P3. Does the Tree Provide a Semantic Metric?

```
P3. Tree semantics
├── P3a: LCA-distance as feature
│        on a real tree from the pipeline: does LCA-distance
│        correlate with "semantic similarity" (‖feature_i − feature_j‖)?
│        if no → tree = just a log, not a metric
│
├── P3b: bushes — path clusters
│        are there natural clusters among leaf paths?
│        metrics: silhouette, stability across runs
│
└── C-pre: profile discreteness check
           trajectory features (EMA quantiles, split signatures, stability)
           question: is there cluster structure?
           if yes → C unfreezes
           if no → C is dead
```

**P3 output:** either "tree is purely a log, provides no semantic metric" (ok, not a problem), or a concrete method for extracting semantics.

---

## Level SC: Scale-Consistency Invariant (v1.7)

Partially formalizes the meta-question from v1.5 "how not to break features." Depends on P0 (pipeline), independent of P1/P2/P3. Can run in parallel.

### SC-baseline. Verification of Metrics D_parent / D_hf

```
SC-baseline. Scale-Consistency Verification
├── SC-0: fix pair (R, Up), verify R idempotence                          ✅ COMPLETE
├── SC-1: prepare positive (strong + empirical) and negative baselines    ✅ COMPLETE
├── SC-2: compute D_parent, D_hf across all cases                        ✅ COMPLETE
├── SC-3: analyze separability (AUC, effect size, quantile separation)    ✅ COMPLETE
│         globally + by depth + by structure type
├── SC-4: kill criterion — PASSED with updated D_parent formula
│         (R=gauss sigma=3.0, lf_frac normalization, AUC=0.853, d=1.491) ✅ COMPLETE
└── SC-5: set data-driven tau_parent[L] — needs data-driven threshold setting
```

**Kill criterion:** Global ROC-AUC >= 0.75, Depth-conditioned AUC >= 0.65, Effect size >= medium (d >= 0.5). If not met — change metrics, **do not** tweak thresholds.

**SC-4 result:** PASSED. Updated D_parent formula: `||R(delta)|| / (||delta|| + epsilon)`, R=gauss sigma=3.0. AUC=0.853, d=1.491. Cross-space: T1=1.000, T2=1.000, T3=1.000, T4=0.824 (all >= 0.75).

**SC-5 status:** tau_parent needs data-driven threshold setting.

**SC-baseline output:** validated thresholds tau_parent[L] or decision to revise metric construction.

Full protocol: `docs/scale_consistency_verification_protocol_v1.0.md`.

### SC-σ sweep. Optimization of R operator σ parameter

```
SC-σ. Fine-grained sweep of σ parameter
├── σ sweep: [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
│       at each σ: full SC-3 (AUC, Cohen's d, per-negative-type)
├── tile_size sweep: σ × tile_size ∈ {8, 16, 32, 64}
│       question: does σ_opt depend on tile_size? Is σ/tile_size ≈ const?
├── cross-space: σ sweep × 4 space types
│       question: is σ_opt the same across spaces or space-dependent?
└── output: formula/rule for σ selection, or fixed σ_opt
```

**Known limitation of current σ=3.0:** chosen as smallest integer in coarse sweep [0.5, 1.0, 2.0, 3.0]. Fine-grained search not performed. σ=2.5 may suffice, or σ=4.0 may be better. Optimum may depend on tile_size and space type.

**Dependencies:** none (can run in parallel with anything, reuses sc_baseline code).
**Priority:** low (σ=3.0 passes kill criteria; optimization is not a blocker).

---

### SC-enforce. Enforcement (after SC-baseline) — ✅ CLOSED (exp14a)

```
SC-enforce. Scale-Consistency Enforcement
├── damp delta / reject split / increase local strictness when D_parent > τ_parent
└── D_parent as contextual signal in ρ (not self-sufficient)
```

**Result (exp14a_sc_enforce):** Three-tier enforcement (pass/damp/reject) + strictness-weighted waste budget + adaptive τ T4(N) = τ_base * (1 + β/√N). Adaptive tau resolves high reject rate (~50%) on tree_hierarchy with tight T4 thresholds.

---

## Level 4: Global Coherence ("Don't Break Features")

Depends on **everything above** + SC-baseline. Meta-question from Concept v1.5, partially formalized through Scale-Consistency Invariant (Concept v1.7, section 8).

### P4. Representation Coherence Under Non-Uniform Depth

```
P4. "Don't break features"
├── P4a: downstream consumer test
│        feed adaptive-refined representation into a simple downstream
│        (classifier / autoencoder)
│        compare with dense-refined and coarse-only
│        question: does downstream break under non-uniform depth?
│        (with scale-consistency enforcement vs. without)
│
├── P4b: matryoshka invariant
│        verify that the representation at any "matryoshka" level
│        is valid as consumer input
│        (not just visually smooth, but functionally correct)
│
└── P4c: guarantee mechanism (if P4a/b show a problem)
         options: padding/projection layer, consistency loss,
         depth-aware normalization, stricter τ_parent
```

**P4 output:** either "non-uniform depth doesn't break downstream" (and the question is closed), or a concrete protection mechanism.

---

# Frozen

## C. DAG + Profiles

**Entry contract (all three simultaneously):**

1. At least two irreducible objectives (cannot be collapsed into a scalar without semantic loss)
2. A concrete downstream consumer that actually relies on these objectives
3. An observable conflict: different optimal solutions under different objectives on the same data

Pre-experiment C-pre (in P3) may signal unfreezing, but is not sufficient by itself.

If the contract is not met — freeze is indefinite.

---

# Dependency Graph

```
P0 (GPU layout)
 ├──→ DET-1 (seed determinism) ──→ DET-2 (cross-seed stability)
 │         │
 ├──→ P1 (tree compression)  ──→ P3 (tree semantics)
 │    └── exp11 (dirty sig) ──→ exp13 (segment compression)
 │                                       │
 ├──→ P2 (ρ auto-tuning)               ├──→ C-pre
 │                                        │
 └──→ SC-baseline (✅ SC-0..SC-4) ──→ SC-5 (exp12a) ──→ SC-enforce (exp14a) ──→ P4
                                                                │
                                                                ▼
 exp07 + exp10d + exp12a + exp14a ──→ exp_phase2_pipeline ──→ exp_phase2_e2e (✅ 240 configs, DET-1)
```

**Critical path:** P0 → DET-1 → P1 → P3 → P4.

**Phase 2 path (✅ CLOSED):** exp07 + exp10d + exp12a + exp14a → exp_phase2_pipeline → exp_phase2_e2e.

**Parallel branches:** P2 and SC-baseline — both run in parallel with P1, all are needed before P4. DET-2 parallel with P1 (after DET-1).

**Gate blockers:** DET-1 blocks stability pass. DET-2 blocks Track B.

---

# Working Order

1. **P0: 0.9b0** — buffer-scaling probe, kill/go for compact
2. **P0: 0.9b/0.9c** — if compact survives; otherwise lock in grid
3. **DET-1** — seed determinism (canonical order, deterministic probe, governor isolation). Absorbs 0.9h. Blocker for stability pass.
4. **P1-B2** — dirty signatures (parallel with DET-1, runs on CPU)
5. **DET-2** — cross-seed stability (20 seeds × 4 spaces × 2 budgets). Parallel with P1.
6. **P2a** — sensitivity sweep of gate thresholds, **5 scenes × 4 space types** (code ready, parallel with P1)
7. **SC-5** — set data-driven τ_parent[L] (SC-0..SC-4 ✅ complete; parallel with P1)
8. **P1-B1** — segment compression (after B2)
9. **P1-B3** — anchors + rebuild (after B1+B2)
10. **SC-enforce** — enforcement of scale-consistency (after SC-5)
11. **P3a/P3b** — tree semantics (after P1)
12. **C-pre** — profile cluster check (after P3, cheap)
13. **P4** — "don't break features" (after everything + SC + DET)

---

# Naming Convention (v3+)

Historically, numbering grew organically: chronological IDs (0.1–0.9a),
validation plan sections (§A/B/C), roadmap levels (P0–P4),
sub-experiments within levels (B1–B3 in P1). Result: confusion.

**Rules for new experiments:**

1. **Single sequential numbering.** Next experiment = `exp10`.
   Integer numbering, no dots (dots confused with sub-versions).
   Number = creation order. Never reused.

2. **Sub-experiments use lowercase letter.** `exp10a`, `exp10b`, `exp10c`.
   One series = one numeric root.

3. **Folder = `exp{N}{suffix}_{short_name}/`.** Examples:
   `exp10_buffer_scaling/`, `exp10a_synthetic_kernel/`, `exp11_dirty_signatures/`.

4. **Roadmap mapping — only in this document**, not in folder names.
   Folders don't contain "P0" or "B2" in their names.

5. **Each folder contains README.md** (short, 5–15 lines):
   - Question/hypothesis (one sentence)
   - Kill criteria
   - Link to roadmap level (P0/P1/P2/...)
   - Status (open / closed / killed)

6. **Legacy names are not renamed.** `phase1_halo/`, `phase2_probe_seam/`,
   `exp09a_layout_sandbox/` — historical legacy, linked to new numbering
   via the mapping table above.

**Working order → experiment number mapping:**

| Step | Roadmap | Description | Future exp# |
|------|---------|-------------|-------------|
| 1 | P0 | buffer-scaling probe (kill/go compact) | exp10 |
| 2 | P0 | end-to-end pipeline grid vs compact | exp10a/b/c |
| 3 | DET-1 | seed determinism (canonical order, det. probe, governor isolation). Absorbs 0.9h | exp10d |
| 4 | P1-B2 | dirty signatures | exp11 |
| 5 | DET-2 | cross-seed stability (20 seeds × 4 spaces × 2 budgets) | exp11a |
| 6 | P2a | sensitivity sweep of gate thresholds (5 scenes × 4 spaces) | exp12 |
| 7 | SC-5 | set data-driven τ_parent[L] (SC-0..SC-4 ✅) | exp12a |
| 8 | P1-B1 | segment compression (thermodynamic guards, N_critical=12) | exp13 ✅ |
| 9 | P1-B3 | anchors + rebuild | exp14 |
| 10 | SC-enforce | enforcement: three-tier + waste budget + adaptive τ | exp14a ✅ |
| — | SC-σ | fine-grained σ sweep × tile_size × 4 spaces (low priority) | exp14b |
| 10½ | Phase 2 | full pipeline assembly (gate+governor+SC-enforce+probe+traversal) | exp_phase2_pipeline ✅ |
| 10¾ | Phase 2 | end-to-end validation (4 spaces, 240 configs, DET-1 verified) | exp_phase2_e2e ✅ |
| 10⅞ | Topo | topological pre-runtime profiling: hybrid Forman/Ollivier curvature + three-zone classifier v3 (κ+Gini+η_F). 35-graph corpus, 97% accuracy. η_F=0.70 from gap [0.60, 0.76] | exp_phase2_pipeline (topo_features.py) ✅ |
| 11 | P3a/b | tree semantics | exp15 |
| 12 | C-pre | profile cluster check | exp16 |
| 13 | P4 | "don't break features" | exp17 |

Numbers are provisional. If an unplanned experiment arises between steps,
it gets the next free number.

---

# Instrument Readiness Gate

All experiments P0–P4 + SC + DET belong to **Track A** (building the instrument). Transition to **Track B** (researching tree structure) requires passing the Instrument Readiness Gate:

1. **Invariant pass** — all mandatory invariants hold (including DET-1: seed determinism)
2. **Overhead profile** — overhead does not consume the gain
3. **Stability pass** — DET-1 (bitwise match at fixed seed) + DET-2 (CV of metrics < τ_cv across seeds)
4. **One validated benchmark** — adaptive > random > coarse with confirmed numbers
5. **Attribution diagnostics** — each module's contribution measured (ablation)

Details: `docs/target_problem_definition_v1.1.md`.

After successful Track B, **Track C** opens (generalization to non-spatial domains: graphs, latent spaces, activations). Long-term ambition, not a current goal.

---

# Principles

* Judge numbers first, then ambitions.
* No "next stage" is locked in advance.
* Kill criteria are two-sided (speed + memory).
* Forensic-grade protocol: controls, Holm-Bonferroni, cost-fair comparisons.
