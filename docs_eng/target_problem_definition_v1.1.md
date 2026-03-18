# Curiosity — Target Problem Definition (v1.1)

This document fixes why the project exists, what counts as success, and against what to compare.
Without this, "quality-per-cost" is fog in a cylinder.

Related to: `concept_v1.8.md` (how it is built), `experiment_hierarchy.md` (what is being tested).

---

## Structure of Goals

The project has three sequential goals, not parallel ones. The order is not optional.

```
Track A — build the instrument
    ↓  (only after Instrument Readiness Gate)
Track B — research with the instrument
    ↓  (only after successful Track B)
Track C — generalize to non-spatial domains
```

Track A can be successful even if Track B fails.
Track B cannot honestly begin until Track A reaches instrumental maturity.
Track C cannot honestly begin until Track B confirms semantic geometry.
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
Tasks with heterogeneous information density in arbitrary computational spaces:
- 2D/3D scalar and vector fields with sharp boundaries and flat regions
- Images with multiscale structure
- Irregular graphs with cluster structure (k-NN, spatial graphs)
- Tree hierarchies (routing structures, decision trees)
- Volumetric / sparse scenes
- Multiscale signals with local anomalies

**Cross-space validation principle:** all claims of domain-agnostic generality must be verified on at least 4 space types (scalar grid, vector grid, irregular graph, tree hierarchy). Results on a single type are not sufficient.

**Halo limitation:** cosine feathering is applicable only to spaces with boundary parallelism ≥ 3 and no context leakage. Not applicable to tree/forest topologies (see concept_v1.8.md §6).

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

### Non-Binding Observables (Track A)

The following metrics are logged during Track A experiments but **do not** affect kill criteria or success evaluation. They accumulate data for Track B.

| Observable | What it measures | Cost |
|---|---|---|
| **Tree topology stability** | Do semantically similar inputs produce similar trees? (correlation of tree structure between runs) | Low (logging) |
| **LCA-distance vs. feature-distance** | Scatter: distance in tree vs. distance in feature space | Low (logging) |
| **Cluster purity** | If inputs have labels (e.g. CIFAR), do bushes align with classes? | Low (post-analysis) |

These are **observational**, not prescriptive. Track A is evaluated by reconstruction, cost, and invariant compliance — not by tree semantics.

### Status
**Active goal.** Exp0.1–Exp0.8 closed, basic invariants confirmed. Phase 0 (parallel validation) complete: Halo applicability rule derived, SC-baseline (D_parent/D_hf) validated on 4 space types, environment configured.
Next step — Phase 1: Exp0.9b0 (P0 GPU layout), P2a sweep, SC-5 (τ_parent).

---

## Instrument Readiness Gate

Track B begins only after passing all five criteria.

| Criterion | What is checked |
|---|---|
| **Invariant pass** | All mandatory invariants are satisfied: halo (with topology-dependent applicability rule), probe, governor, scale-consistency (D_parent < τ_parent), determinism (DET-1). Cross-space validation passed. None is violated systematically. |
| **Overhead profile** | Real overhead of the control system is measured. It does not eat the gain from adaptive refinement. Cost-fair comparison shows positive result. |
| **Stability pass** | System is deterministic at fixed seed (DET-1: bitwise tree match). Statistically stable across seeds (DET-2: CV of metrics < τ_cv). Governor does not oscillate. |
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
**Research goal, not current.** Open questions are fixed in `concept_v1.8.md` §13.

---

## Track C — Generalization

### Question
Does the mechanism work beyond spatial data — in arbitrary state spaces?

### Formulation
Verify Curiosity's applicability to non-spatial domains: graphs, latent spaces,
activations, feature hierarchies. Confirm the claimed domain-agnostic generality.

### Entry Condition
Track B is successful: semantic geometry of the tree confirmed on at least one domain.
Without this, there is nothing to generalize.

### Status
**Long-term ambition. Not a current goal.**

---

## What is Currently Honestly Not Fixed

| Question | Status |
|---|---|
| Specific downstream task for Track B | Not chosen. Chosen after Readiness Gate. |
| Feature consistency metric (downstream) | Not defined. D_parent is a proxy, not a direct measure. |
| Applicability to non-spatial domains (graphs, latent) | Partially verified: SeamScore, D_parent/D_hf validated on 4 space types (scalar grid, vector grid, irregular graph, tree hierarchy). Halo — limited applicability (grid/graph: yes, tree: no). Fixed as Track C. |
| Generality beyond spatial data | Long-term ambition. Entry condition — successful Track B. |

---

## One Sentence

**Curiosity is a mechanism of adaptive refinement of representations under budget, which decides where to compute, computes only what is locally needed, preserves correctness at boundaries, and does not allow structural blindness or destruction of features.**

Currently the instrument is being built (Track A). Later it will be used to research structure (Track B). Then generality is verified (Track C).
