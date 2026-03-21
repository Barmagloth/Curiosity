# Session Handoff — Curiosity Phase 2

Document for a new AI orchestrator session. Contains full context for immediately continuing work.

## Where We Are

Curiosity project.

- Phase 0 **complete** (March 18, 2026).
- Phase 1 **complete** (March 20, 2026). All streams — PASS. P0 Layout **CLOSED**. DET-1 **PASS**. DET-2 **PASS**.
- Phase 2 **complete** (March 20, 2026). Pipeline assembled, SC-enforce integrated, E2E validated.
- **Next step — Phase 3.**

Workstation: **PC 2** (NVIDIA RTX 2070, 8 GB, CUDA 12.8). Working directory: `R:\Projects\Curiosity`.

## What to Read (in this order)

| # | File | Why |
|---|------|-----|
| 1 | `docs/session_handoff.md` | **This file** — current status, what to do next |
| 2 | `docs/concept_v1.8.md` | Canonical concept (current) |
| 3 | `docs/teamplan.md` | Plan with Phase 0, Phase 1 marks, Phase 2-4 descriptions |
| 4 | `docs/experiment_hierarchy.md` | Dependency graph, priorities, exp10+ numbering |
| 5 | `docs/environment_2.md` | How to activate .venv-gpu on PC 2 (CUDA) |
| 6 | `docs/phase1_plan.md` | Archive — Phase 1 plan and results (reference) |

## Phase 1 — Final Results (March 20, 2026)

### Main Streams

| Stream | Result | Status |
|--------|--------|--------|
| S1 exp10 | KILL compact-with-reverse-map (VRAM +38.6%). Grid — baseline. | PASS |
| S1b exp10d | DET-1 PASS (240/240 bitwise match CPU+CUDA) | PASS |
| S2 exp11 | PASS | PASS |
| S3 P2a | PASS — ridge 100%. Manual thresholds ok. P2b not needed. | PASS |
| S4 exp12a | PASS | PASS |
| S5 deferred | Research note done. | PASS |
| DET-2 | PASS (cross-seed stability) | PASS |

**Gate Phase 1 -> Phase 2: PASSED.** All streams PASS. P0 Layout closed. DET-1 and DET-2 passed.

---

## P0 LAYOUT — CLOSED (full exp10 series, March 19, 2026)

### Layout Dictionary

- **D_direct** ("packed tiles + direct tile_map") — active tiles in a compact array, tile_map[tile_id] -> slot for O(1) lookup. No element-level reverse_map. Winner for grids.
- **A_bitset** ("dense grid + bitset mask") — full-size data tensor + activation bitmask. Simple fallback.
- **D_blocked** ("block addressing for graphs") — graph nodes split into fixed blocks, block_map[block_id] -> slot. Works only for spatial graphs.
- **E_hash** ("hash table lookup") — archival fallback, dominated by D_direct at current scale. Resurrection triggers documented.

### Final Layout Policy

| Space type | Layout | Status | Evidence |
|------------|--------|--------|----------|
| scalar_grid | D_direct (packed tiles + tile_map) | Production | exp10g: both contours PASS |
| vector_grid | D_direct (packed tiles + tile_map) | Production | exp10h: 72/72 PASS |
| tree_hierarchy | Hybrid: D_direct per-level where occupancy < 40% + heavy compute; A_bitset otherwise | Validated | exp10j: break-even found |
| irregular_graph / spatial | D_blocked (block addressing) conditional | Conditional | exp10i: spatial partition, cbr<0.30 |
| irregular_graph / scale-free | A_bitset (dense grid + bitset mask) fallback | Fallback only | exp10i: blocks rejected, cbr=0.66 |

### Killed Permanently

- Element-level reverse_map[M] (exp10: VRAM +38.6%)
- Binary search on GPU (exp10e-B: +1700%)
- Paged sparse tiles (exp10e-C: +9000%)
- Hash as primary lookup (exp10f-E: dominated by D_direct)
- Fixed blocks for scale-free graphs (exp10i: cbr 0.64-0.99)

---

## What to Do — Phase 2

### Goal

End-to-end pipeline validation. Assemble the full pipeline (layout + halo + gate + governor + probe + SeamScore) and run on real tasks.

### Tasks

1. **Pipeline integration** — assemble all validated components into a single runtime:
   - Layout (D_direct / A_bitset / D_blocked / hybrid) by space type
   - Halo (cosine feathering, overlap >= 3)
   - Two-stage gate (residual-first + utility-weighted fallback)
   - Budget governor (EMA strictness controller)
   - Probe (5-10% budget for exploration)
   - SeamScore (Jumpout / (Jumpin + eps))

2. **End-to-end validation** — run the assembled pipeline on real tasks, all 4 space types

3. **SC-enforce** — enforcement integration (damp delta / reject split when D_parent > tau_parent)

### Critical Path

```
Phase 2 → Instrument Readiness Gate → Track A
```

---

## Open Questions for Phase 2

1. **Governor hysteresis for irregular high budget** — Track B. How to tune the EMA controller for spaces with irregular budget profiles.

2. **Graph-native sparse for scale-free** — Track C. CSR/COO instead of A_bitset fallback for scale-free graphs (barabasi-albert and similar).

3. **C(I,M,p) surface with correct metrics** — Track C. Building the curiosity = f(information, mass, position) surface with correct metrics for each space type.

---

## Phase 2 — Final Results (March 20, 2026)

### Streams

| Stream | Description | Status |
|--------|-------------|--------|
| A (Pipeline Assembly) | CuriosityPipeline: gate + governor + SC-enforce + probe + traversal | DONE |
| B (SC-Enforce) | Three-tier pass/damp/reject + strictness-weighted waste budget + adaptive tau T4(N) = tau_base*(1+beta/sqrt(N)) | DONE |
| C (Segment Compression) | Thermodynamic guards (N_critical=12, bombardment guard) — eliminate overhead on small trees | DONE |
| D (E2E Validation) | 240 configs, 4 space types, DET-1 + DET-2 recheck with topo | DONE |

### Key Findings

- **tree_hierarchy high reject rate (~50%)** due to tight T4 thresholds — resolved with adaptive tau T4(N) = tau_base * (1 + beta/sqrt(N))
- **Graph clustering** upgraded from k-means to Leiden (community detection), validated on 10 pathological topologies: Swiss Roll, Barbell, Hub-Spoke, Ring of Cliques, Bipartite, Erdos-Renyi, Grid, Planar Delaunay, Mobius strip
- **E2E irregular_graph re-run** (March 21, 2026) with topo profiling active: zone distribution GREEN 75%/RED 25%, PSNR −0.20 dB vs pre-topo (expected)

### E2E Results (with topo profiling, March 21, 2026)

| Space | PSNR gain median | Reject max | Wall max |
|-------|-----------------|------------|----------|
| scalar_grid | +7.34 dB | 0% | 23ms |
| vector_grid | +1.46 dB | 0% | 33ms |
| irregular_graph | +3.54 dB | 0% | 245ms |
| tree_hierarchy | +1.48 dB | 0% | 4ms |

### DET Rechecks (with topo profiling, March 21, 2026)

| Test | Scope | Result |
|------|-------|--------|
| DET-1 (bitwise determinism) | 4 spaces × 10 seeds = 40 | ✅ 40/40 PASS |
| DET-2 (cross-seed stability) | 4 spaces × 20 seeds × 2 budgets = 160 | ✅ 8/8 cells PASS |

DET-2 kill metrics (n_refined, compliance): CV≈0. psnr_gain CV=0.09–0.37 — informational, depends on seed GT (not kill-criteria).

**Gate Phase 2 -> Phase 3: PASSED.** All streams DONE. Pipeline assembled and validated end-to-end. Topo profiling integrated and DET-verified.

### Topological Pre-Runtime Profiling of Graph Spaces (March 21, 2026)

**Problem:** the pipeline had no visibility into space topology before starting. The rho-function allocated budget blindly — without accounting for bridges, hubs, or structural chaos.

**Solution:** multi-stage profiling during IrregularGraphSpace initialization, BEFORE the first pipeline tick:

1. **Hybrid curvature** — Forman-Ricci O(1) for ALL edges + Ollivier-Ricci (exact EMD via linprog HiGHS) for top-N anomalous edges. N = floor(topo_budget_ms / t_ollivier_ms), budget <= 50ms.
2. **Hardware calibration** — Synthetic Transport Probe: kappa_max = W_test * cbrt(tau_edge / t_test). One-time measurement at session start (52ms).
3. **Three-zone classifier (v3)** — assigns a label GREEN/YELLOW/RED to the graph:
   - **GREEN** (kappa_mean > 0): dense cliques, ECR < 5%. Budget carte blanche, relaxed tau_eff.
   - **YELLOW** (kappa < 0, Gini < 0.12, eta_F <= 0.70): uniform lattices, ECR 10-25%. Standard limits.
   - **RED** (kappa < 0, Gini >= 0.12 OR eta_F > 0.70): structural chaos, ECR > 30%. Maximum tau tightening, deep split blocking.
4. **eta_F = sigma_F / sqrt(2*mean(k))** — dimensionless topological entropy index. sigma_F = Forman curvature variance, sqrt(2*mean(k)) = noise floor of variance for an Erdos-Renyi random graph with the same mean degree. A graph whose structural variance is below the Poisson floor is regular. Above — it radiates chaos.

**eta_F = 0.70 threshold justification:**
- Threshold sweep over the kappa<0 subset of the corpus (22 graphs):
  - YELLOW graphs (Grid, Ladder, Planar, Mobius): eta_F < 0.60 — clearly below threshold
  - RED graphs (ER, Bipartite): eta_F > 0.76 — clearly above
  - Gap [0.60, 0.76] — dead zone, no graph in the corpus falls there
  - Threshold 0.70 — midpoint of the gap, maximum margin from both sides
- The only borderline case: Watts-Strogatz (eta_F = 1.02) -> RED. Classified correctly: WS with p=0.1 is a ring with injected Poisson chaos, eta > 1.0 = structural noise objectively exceeds random background. The fact that Leiden carves out chunks with ECR=15.7% is algorithmic luck, not a topological property.
- Stability: the range eta_thresh in [0.60, 0.75] yields identical 86% on the kappa<0 subset. The threshold is not on a knife edge.

**Validation:** 35-graph corpus (9 base + 26 scale/parameter variations). v3 accuracy = 97% (34/35, the only documented borderline case — Karate Club, 3 pp from threshold).

**Performance (pre-runtime, one-time setup):**
- P50 = 56ms, P90 = 85ms, MAX = 125ms (Swiss Roll 1000 nodes)
- Leiden: 1.2ms average (negligible)
- Topo features + classifier: 57.6ms average
- Overhead vs pipeline tick (500ms): 11.8% mean, 25% worst case

**Key files:**
- `exp_phase2_pipeline/topo_features.py` — core: CalibrationResult, compute_curvature_hybrid, extract_topo_features, topo_adjusted_rho
- `exp_phase2_pipeline/test_zone_classifier.py` — v1/v2/v3 validation on 35 graphs
- `exp_phase2_pipeline/bench_preruntime.py` — pre-runtime overhead benchmark

**Returned TopoFeatures contains:**
- Per-node: curvature (hybrid), pagerank, clustering_coeff, local_density, degree
- Per-cluster: mean/std/max aggregates + boundary curvature
- Profiling: sigma_F, eta_F, gini_pagerank, **topo_zone** (GREEN/YELLOW/RED)

---

## What Was Done Previously (Phase 0)

### Experiments
1. **Environment**: PC 1 (AMD Radeon 780M, DirectML) + PC 2 (RTX 2070, CUDA 12.8)
2. **Halo cross-space**: grid/graph OK, tree FAIL (0.56x). Rule: parallelism >= 3 AND no leakage.
3. **P2a sweep**: code ready (20K configurations), NOT RUN
4. **SC-baseline**: D_parent = ||R(delta)|| / (||delta|| + eps), R = gauss sigma=3.0. AUC 0.824-1.000 across 4 spaces.

### Key Architectural Decisions
- Halo: NOT a universal invariant — topology-dependent rule.
- D_parent: formula updated (lf_frac).
- Morton/block-sparse/phase schedule: DEFERRED, not rejected.

---

## Environment

```bash
# Activate venv (PC 2):
R:\Projects\Curiosity\.venv-gpu\Scripts\activate
# Python 3.12.11, PyTorch 2.10.0+cu128, CUDA 12.8

# Git auth:
gh auth setup-git
# Token: R:\Projects\.gh_tkn
```

## Principles

- **Cross-space validation** — 4 space types mandatory
- **Kill criteria before launch** — every experiment
- **Holm-Bonferroni** — for multiple comparisons
- **10-20 seeds** — for reproducibility
- **Barmagloth = architect** — makes decisions at forks, does not write code
