# Curiosity — Target Problem Definition (v1.1)

This document fixes why the project exists, what counts as success, and against what to compare.
Without this, "quality-per-cost" is fog in a cylinder.

Related to: `curiosity_concept_v1.6.md` (how it is built), `experiment_hierarchy.md` (what is being tested).

---

## Structure of Goals

The project has two sequential goals, not parallel ones. The order is not optional.

```
Track A — build the instrument
    ↓  (only after Instrument Readiness Gate)
Track B — research with the instrument
```

Track A can be successful even if Track B fails.
Track B cannot honestly begin until Track A reaches instrumental maturity.
Otherwise, we are studying artifacts of an unfinished apparatus, not the structure of the world.

---

## Track A — Instrument

### Question
Can we build a system that locally refines a representation under budget, does not break boundaries, does not become blind without exploration, and does not destroy downstream utility?

### Formulation
For tasks with locally heterogeneous information density, build an adaptive refinement pipeline that:
- computes only informative regions,
- preserves quality no worse than dense baseline at lower cost,
- does not create artifacts at boundaries (halo),
- does not become structurally blind (probe),
- keeps spending under control (governor),
- does not destroy the validity of representation under heterogeneous refinement depth (scale-consistency).

### Domain
Tasks with spatially heterogeneous structure:
- 2D/3D fields with sharp boundaries and flat regions
- Images with multiscale structure
- Volumetric / sparse scenes
- Multiscale signals with local anomalies

### Success criteria
| Metric | Requirement |
|---|---|
| PSNR / MSE | ≥ dense baseline at equal budget |
| HF metric (interior-only) | No worse than dense (no seam artifacts) |
| Cost compliance | StdCost / P95 within governor target |
| Feature consistency (D_parent) | < τ_parent on ≥ 90% of nodes |
| Winrate vs random selection | ≥ 90% on primary metric |
| Overhead | Management cost does not eat the gain |

### Cost model
Primary: number of refined tiles × cost of refine operation.
Budget is set as target_per_step (tiles/step). System overhead is included in budget (cost-fair).

### Baselines
1. **Dense baseline** — full recomputation at maximum resolution (oracle of quality)
2. **Same-budget random** — same budget, random tile selection
3. **Same-budget uniform coarse** — same budget, uniform coarse approximation

### Status
**Active goal.** Exp0.1–Exp0.8 closed, basic invariants confirmed.
Next step — Exp0.9b0 and baseline for scale-consistency.

---

## Instrument Readiness Gate

Track B begins only after passing all five criteria.

| Criterion | What is checked |
|---|---|
| **Invariant pass** | All mandatory invariants are satisfied: halo, probe, governor, scale-consistency. None is violated systematically. |
| **Overhead profile** | Real overhead of the control system is measured. It does not eat the gain from adaptive refinement. Cost-fair comparison shows positive result. |
| **Stability pass** | System is stable: governor does not oscillate, split decisions are reproducible, results are stable between runs. |
| **One validated benchmark** | At least one benchmark where adaptive > same-budget random > coarse, with confirmed numbers and fixed protocol. |
| **Attribution diagnostics** | It is clear what each module contributes. Ablation or attribution shows contribution of halo, probe, governor, ρ individually. No "black box". |

**Without any one of the five — Track B is premature.**

**With any five — Track B can begin, even if:**
- governor is not yet the best possible,
- ρ is not yet finally canonized,
- layout is not yet optimal,
- speed is still far from production.

Instrumental maturity ≠ perfection. This is the minimum at which the apparatus does not lie.

---

## Track B — Research

### Question
Does the tree structure of refinements carry its own meaning — or is it just bookkeeping of splits?

### Formulation
Use the built mechanism as an instrument to study whether the refinement tree gives meaningful geometry of representation.

### Research questions
1. Does LCA-distance in the tree correlate with semantic distance in task space?
2. Do bushes (sets of paths to the same meaning) carry real distinctness — or is this an artifact of ρ?
3. Is the map of semantic density (distribution of splits) predictive for downstream tasks?
4. Is the Scale-Consistency Invariant a sufficient condition for preserving downstream utility — or are additional constraints needed?
5. Does the phase schedule come alive under non-ideal refine, where phase switching is physically justified?

### Success criteria
- LCA-distance correlates with meaningful structure in data (measurable, not declarative)
- Tree metrics actually explain or predict something in the downstream task
- "Semantic geometry" is falsifiable: there are specific conditions under which the hypothesis is rejected

### Domain
Determined after reaching Instrument Readiness Gate, based on which downstream consumer is most natural for the built instrument.

### Status
**Research goal, not current.** Open questions are fixed in `curiosity_concept_v1.6.md` §13.

---

## What is Currently Honestly Not Fixed

| Question | Status |
|---|---|
| Specific downstream task for Track B | Not chosen. Chosen after Readiness Gate. |
| Feature consistency metric (downstream) | Not defined. D_parent is a proxy, not a direct measure. |
| Applicability to non-spatial domains (graphs, latent) | Theoretically stated, experimentally not verified. |
| Generality beyond spatial data | Long-term ambition, not current goal. |

---

## One Sentence

**Curiosity is a mechanism of adaptive refinement of representations under budget, which decides where to compute, computes only what is locally needed, preserves correctness at boundaries, and does not allow structural blindness or destruction of features.**

Currently the instrument is being built (Track A). Later it will be used to research structure (Track B).
