# Curiosity — Conceptual Document (v1.4, Historical)

This document captures the project logic as it stood after Exp0.2–Exp0.3. It is preserved for historical reference. The canonical version is concept_v1.5.md.

Updated after experiments Exp0.2–Exp0.3.

---

# 0. Original Motivation

The project goal is not to speed up a neural network per se, nor to replace an optimizer.

The goal is to reshape the search space so that:

* it can be explored adaptively,
* computational resources are spent only where they yield information,
* feature interdependencies are preserved (no "broken features").

Key idea:

> Dimensionality is not a fixed number but a depth of refinement. Refinement should occur only where it is justified informationally and budget-wise.

---

# 1. Base Model

## 1.1 Space

There is a state space X (of arbitrary nature: features, latent, activations, data).

There is an informativeness function ρ(x).

ρ(x) measures where refinement makes sense.

## 1.2 Refinement

Process:

1. Start with a coarse approximation (coarse level).
2. Compute ρ.
3. Decide whether to split a region.
4. Refine only selected regions.
5. Repeat until stopping criteria.

Refinement always takes the form:

```
output = coarse + delta
```

where delta:

* is initialized to zero,
* is energy-bounded,
* disabling any current level returns a valid representation of the previous level,
* may degrade local consistency at boundaries.

Halo is applied at each refinement level.

---

# 2. Trees and "Bushes"

## 2.1 Tree

A tree is a log of refinement routes.

Each root-to-leaf path is a sequence of split decisions.

The structure must be GPU-friendly (flat packing, no pointer chasing).

## 2.2 Bush

Bush = a set of paths leading to the same meaning.

For the first prototype, a tree distance metric is used (via LCA / common prefix).

---

# 3. Informativeness Function ρ(x)

Tree semantics are defined by ρ.

ρ may reflect:

* task error,
* variance,
* disagreement,
* compressibility,
* teacher-guided signal.

ρ can be hierarchical.

---

# 4. Interestingness Policy

Signal balance depends on depth L.

## 4.1 Absolute Power

At coarse levels, priority goes to large structures.

Absolute power weight decays with depth.

## 4.2 Local Anomaly

Sensitivity to local dispersion increases with depth.

## 4.3 Payoff

Split is allowed only if gain exceeds cost.

## 4.4 Budget

Hard constraints:

* max_depth
* max_nodes
* max_cost
* max_boundary_budget

## 4.5 Strictness Auto-Tuning

Controller operates on EMA of spending.

## 4.6 Junk Detector

If average gain over K splits < δ (5–10% of previous level) — noise splitting is suppressed.

---

# 5. Boundary-Aware Refinement (Mandatory Requirement)

Experiment Exp0.3 showed:

Refinement without overlap / feather / halo during tile substitution is guaranteed to produce false high-frequency artifacts under sparse budgets.

Cause: Hard insertion of a refined tile creates a step discontinuity at the refined/non-refined boundary. HF metrics flag this as an artificial signal.

The smarter the adaptive tile selection, the stronger the artifact.

Conclusion: Halo / overlap is a necessary condition for correctness.

Minimum rule:

* overlap ≥ 3–4 pixels at tile_size=16
* blending performed relative to coarse

Without boundary-aware refinement, the system is penalized by high-frequency metrics at low budgets.

---

# 6. Probe-Budget Exploration (Mandatory Requirement)

The strategy "refine only found boundaries" is structurally blind.

Internal structure may exist within a domain that is not visible at the current level.

Each domain stores:

* a fixed set of internal probe tiles
* an internal score
* a check-recency counter

Each step:

* 90–95% budget → exploitation
* 5–10% budget → probe

Probe priority:

* coarse residual / variance
* uncertainty
* time since last check

If probe detects activity — the domain is activated.

Overlap does not replace exploration. Exploration does not fix seams. Both mechanisms are mandatory.

---

# 7. Boundaries

A boundary is a zone of informativeness decay.

Halo is applied on both sides of a boundary.

---

# 8. Map Construction Strategies

1. Shared density map.
2. Meaning-specific map.

---

# 9. Key Invariants

1. Refinement is additive.
2. Halo is mandatory.
3. Exploration is mandatory.
4. Cost is bounded by budget.
5. The system does not create artificial seams.
6. The system does not become structurally blind.

---

# 10. Core Principle of Curiosity

Curiosity is a system that:

1. Computes only where there is signal.
2. Does so without creating artificial seams.
3. Does not become blind to internal structure.
4. Manages budget consciously.

Formally:

Refinement must be boundary-aware and must include controlled exploration.

---

# 11. Experimental Status (Exp0.2–Exp0.3)

Confirmed:

* Adaptive tile selection yields MSE_rgb/PSNR gains.
* HF degradation is caused by seam artifacts.
* Halo ≥ 3–4 pixels eliminates HF degradation.
* Interior-only HF confirms the problem was in seams.
