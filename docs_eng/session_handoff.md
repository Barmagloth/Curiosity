# Session Handoff — Curiosity Phase 1

Document for a new AI orchestrator session. Contains full context for immediately continuing work.

## Where We Are

Curiosity project. Phase 0 (parallel validation + environment setup) is **complete**. All results committed and pushed to main.

## What Was Done in This Session

### Documentation
- Fixed broken references in handoff.md (concept_v1.5→v1.6→v1.7)
- Added recommended reading order to README
- Created teamplan.md (RU + ENG) — parallel team work plan
- Added .gitignore

### Phase 0 — Experiments and Code
1. **Environment**: .venv (CPU, Python 3.13) + .venv-gpu (DirectML, Python 3.12, AMD Radeon 780M)
2. **Halo cross-space**: Halo works on grid/graph, FAIL on tree (0.56×)
   - Rule: boundary parallelism ≥ 3 AND no context leakage
   - Code: experiments/halo_crossspace/
3. **P2a sweep**: code ready (20K configurations), NOT YET RUN
   - Code: experiments/p2a_sensitivity/
4. **SC-baseline**: D_hf pass, D_parent pass (after fixes)
   - Final formula: D_parent = ‖R(δ)‖ / (‖δ‖ + ε), R = gauss σ=3.0
   - Cross-space: T1=1.000, T2=1.000, T3=1.000, T4=0.824
   - coarse_shift generator fixed (coherent sign fields)
   - Code: experiments/sc_baseline/

### Key Architectural Decisions
- Halo: NOT a universal invariant. Applies only when parallelism≥3 + no leakage.
- D_parent: formula changed. Old one (α·‖coarse‖+β in denominator) — zero discriminative signal.
- Morton/block-sparse/phase schedule: DEFERRED, not rejected.
- Cross-space validation: mandatory for any claims about "arbitrary spaces" (4 types: scalar grid, vector grid, irregular graph, tree hierarchy).

## What to Do Next — Phase 1

Per the plan in docs/teamplan.md, "Phase 1" section. Barmagloth is the architect, executors are AI agents.

### Phase 1 Streams (parallel):
1. **P0: Exp0.9b0** — Buffer-scaling probe on GPU (DirectML). Grid vs compact. Kill compact if overhead >20%. Use .venv-gpu.
2. **P1-B2**: Dirty signatures prototype (CPU). 12-bit signature, debounce, AUC>0.8.
3. **P2a**: RUN sensitivity sweep (code already ready in experiments/p2a_sensitivity/). 5 scenes × 4 spaces.
4. **SC-baseline completion**: SC-5 — set data-driven τ_parent[L]. Prepare SC-enforce.
5. **Deferred revisit**: Research note on Morton/block-sparse/phase schedule with new approaches.

### Forks for the Architect (end of Phase 1):
- P0: grid or compact?
- P2a: ridge width — manual ok or adaptive needed?
- P2a cross-space: ridge the same or different across spaces?
- Deferred: bring back Morton/block-sparse/schedule?

## Key Files for Orientation

| File | Why to read |
|------|------------|
| docs/concept_v1.8.md | Canonical concept (current) |
| docs/teamplan.md | Team work plan with completion marks |
| docs/experiment_hierarchy.md | Dependency graph, priorities, roadmap |
| docs/experiment_results.md | All experiment numbers |
| docs/environment_1.md | How to activate the environment |
| experiments/halo_crossspace/results/APPLICABILITY_RULE.md | Halo applicability rule |
| experiments/sc_baseline/results/CROSSSPACE_SC_REPORT.md | Final SC results |

## Principles

- **Cross-space validation** — mandatory (4 space types)
- **Kill criteria before launch** — every experiment
- **Holm-Bonferroni** — for multiple comparisons
- **10–20 seeds** — for reproducibility
- **Git author**: Barmagloth <>
