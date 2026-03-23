# ARCHIVED — see session_handoff.md

# HANDOFF — Curiosity Project Transfer

Document for a new project participant (human or AI orchestrator). Contains everything needed to continue work without additional context.

---

## Status in One Sentence

Phase 0 (parallel validation) complete. Halo — applicability rule derived. D_parent — formula updated and validated on 4 space types. Environment configured (CPU + DirectML GPU). Ready for Phase 1 (P0 GPU layout).

---

## What Has Been Done

### Validated Components (do not change without strong reasons):

1. **Adaptive refinement** — selective tile refinement > random selection
2. **Halo** — cosine feathering, overlap >= 3 elements, mandatory. Applicability rule: boundary parallelism >= 3 AND no context leakage. Grid/graph: always. Tree/forest: never.
3. **Two-stage gate** — residual-first with fallback to utility-weighted combo
4. **Budget governor** — EMA strictness controller, mandatory
5. **Probe** — 5–10% budget for exploration, mandatory
6. **SeamScore** — `Jumpout / (Jumpin + eps)`, production-ready
7. **D_parent (scale-consistency)** — `||R(delta)|| / (||delta|| + epsilon)`, R=gauss sigma=3.0. Validated on 4 space types (AUC 0.824–1.000).

### What Was Deferred (requires re-evaluation with a different approach):

- Phase schedule (changing rho weights by step) — no gain in current formulation
- Morton layout — sorting overhead exceeds gain in current implementation
- Block-sparse layout — poor expansion ratio in current configuration

---

## First Task: Exp0.9b0

**Type:** Buffer-scaling probe
**Priority:** P0 (highest)
**Kill criteria:** defined (see experiment_hierarchy.md)
**Environment:** ready (DirectML on Radeon 780M)

---

## Key Documents

| Document | Contents |
|---|---|
| `docs/concept_v2.0.md` | Canonical concept, all validated decisions |
| `docs/glossary.md` | Project glossary |
| `docs/experiment_results.md` | Detailed Exp0.1–Exp0.8 results with numbers |
| `docs/experiment_hierarchy.md` | Dependency graph, P0–P4 priorities, roadmap |
| `docs/architecture.md` | System architecture, components, stack |
| `docs/teamplan.md` | Team work plan |
| `docs/environment_1.md` | Environment parameters, setup |

---

## What Is Missing (Known Gaps)

1. **Code** — experiment implementations Exp0.1–Exp0.9a. Without them, it's impossible to reproduce baselines, run the next experiment, or verify results.

2. **Raw data/results** — specific numbers exist in documentation, but raw data tables and experiment logs are not saved in the repository.

4. **Formatting conventions** — how to name experiments, where to write results, expected log format.

5. **Jupyter Notebooks** — experiment workbooks are not included.

7. **P2a sweep is ready (code) but has not been run.**

---

## Priority Hierarchy for Continuation

```
P0: GPU layout (Exp0.9b0 — buffer-scaling probe)
P1: Tree compression (segment compression)
P2: Auto-tuning (gate threshold auto-tuning)
P3: Tree semantics (bushes, LCA-distance)
P4: Downstream compatibility (don't break features)
```

Order is strict. No skipping ahead without closing dependencies.

---

## Working Principles

- **"Judge numbers, then ambitions"** — every claim backed by data
- **Kill criteria before execution** — if an experiment fails criteria, it is closed
- **Cost-fair comparisons** — computational overhead accounted for in every comparison
- **Observable-only diagnostics** — no oracle information in metrics
- **Holm-Bonferroni corrections** — for multiple comparisons

---

## Project Sources

Work was conducted in parallel across two environments:
- **Claude (Anthropic)** — primary documentation, Memory with full context, 3 project files
- **ChatGPT (OpenAI)** — additional discussions, work plan, code specifications

This repository consolidates data from both sources as of March 10, 2026.
