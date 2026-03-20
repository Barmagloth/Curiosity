# Session Handoff — Curiosity Phase 2

Document for a new AI orchestrator session. Contains full context for immediately continuing work.

## Where We Are

Curiosity project. Phase 0 **complete** (March 18, 2026). Phase 1 **complete** (March 20, 2026). All streams PASS. P0 Layout **CLOSED**. DET-1 **PASS**. DET-2 **PASS**. Next step — **Phase 2**.

Workstation: **PC 2** (NVIDIA RTX 2070, 8 GB, CUDA 12.8). Working directory: `R:\Projects\Curiosity`.

---

## Phase 1 Final Results (all PASS)

| Stream | Experiment | Result | Status |
|--------|-----------|--------|--------|
| S1 | exp10 series | Layout policy fixed: D_direct for grids, hybrid for trees, D_blocked conditional for spatial graphs, A_bitset fallback for scale-free | PASS |
| S1b | exp10d | DET-1 PASS 240/240 bitwise match CPU+CUDA | PASS |
| S2 | exp11 | Dirty signatures PASS (AUC 0.91-1.0) | PASS |
| S3 | P2a | Sensitivity PASS — ridge 100%, manual thresholds ok, P2b not needed | PASS |
| S4 | exp12a | tau_parent PASS — per-space thresholds found | PASS |
| DET-2 | — | PASS 8/8 (per-regime cross-validation) | PASS |

Gate Phase 1 -> Phase 2: **PASSED**.

---

## What to Do — Phase 2

### Goal

End-to-end pipeline validation. Assemble the full pipeline (layout + halo + gate + governor + probe + SeamScore) and run on real tasks.

### Critical Path

```
Phase 2 → Instrument Readiness Gate → Track A
```

### Key Tasks

1. **Assemble full pipeline** — integrate all validated components into a single runtime:
   - Layout (D_direct / hybrid / D_blocked / A_bitset per space type)
   - Halo (cosine feathering, topology-dependent rule)
   - Two-stage gate (residual-first with fallback)
   - Budget governor (EMA strictness controller)
   - Probe (5-10% budget for exploration)
   - SeamScore (Jumpout / (Jumpin + eps))
2. **End-to-end validation** — run assembled pipeline on real tasks across all 4 space types
3. **Integration testing** — verify component interactions under production conditions
4. **Instrument Readiness Gate** — confirm pipeline meets all criteria for Track A transition

---

## Open Questions for Phase 2

- exp11 dirty signatures: integration approach after redesign (was FAIL 3/4, now PASS after fix)
- exp12a tau_parent: per-space thresholds — enforce or advisory?
- SC-enforce: how to integrate D_parent enforcement into the runtime pipeline
- DET-2 stability guarantees: sufficient for production or additional validation needed?

---

## P0 LAYOUT — CLOSED (full exp10 series)

### Final Layout Policy

| Space type | Layout | Status | Evidence |
|------------|--------|--------|----------|
| scalar_grid | D_direct (packed tiles + tile_map) | Production | exp10g: both contours PASS |
| vector_grid | D_direct (packed tiles + tile_map) | Production | exp10h: 72/72 PASS |
| tree_hierarchy | Hybrid: D_direct per-level where occupancy < 40% + heavy compute; A_bitset otherwise | Validated | exp10j: break-even found |
| irregular_graph / spatial | D_blocked (block addressing) conditional | Conditional | exp10i: spatial partition, cbr<0.30 |
| irregular_graph / scale-free | A_bitset (dense grid + bitset mask) fallback | Fallback only | exp10i: blocks rejected, cbr=0.66 |

---

## What to Read (in this order)

| # | File | Why |
|---|------|-----|
| 1 | `docs/concept_v1.8.md` | Canonical concept — all validated decisions |
| 2 | `docs/experiment_hierarchy.md` | Dependency graph, priorities, exp10+ numbering |
| 3 | `docs_eng/phase1_plan.md` | Phase 1 plan and results (ARCHIVED) — full exp10 series details |
| 4 | `docs_eng/teamplan.md` | Team plan with Phase 0-1 complete, Phase 2 active |
| 5 | `docs/environment_2.md` | How to activate .venv-gpu on PC 2 (CUDA) |

---

## What Was Done Previously

### Phase 0 (March 18, 2026)

1. **Environment**: PC 1 (AMD Radeon 780M, DirectML) + PC 2 (RTX 2070, CUDA 12.8)
2. **Halo cross-space**: grid/graph OK, tree FAIL (0.56x). Rule: parallelism >= 3 AND no leakage.
3. **P2a sweep**: code ready (20K configurations), NOT RUN
4. **SC-baseline**: D_parent = ||R(delta)|| / (||delta|| + eps), R = gauss sigma=3.0. AUC 0.824-1.000 across 4 spaces.

### Phase 1 (March 19-20, 2026)

Full exp10 series (exp10 through exp10j) — layout policy resolved for all space types. DET-1 passed. All streams closed with PASS.

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
