# Curiosity — Conceptual Document (v1.5)

This document captures the current project logic so that the line of reasoning can be reconstructed a year from now without external context.

Updated after experiments Exp0.1–Exp0.8.

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

## 3.1 Components of ρ

ρ may include:

* Residual — current approximation error (primary signal).
* HF energy — structural energy (Laplacian, gradient).
* Variance — local dispersion / disagreement.
* Payoff — expected gain vs. cost.
* Teacher-guided signal.

ρ can be hierarchical (cheap criterion → expensive criterion).

## 3.2 Signal Combination (Exp0.4–Exp0.7b)

One signal = one bias. Residual is blind to hidden structure under noise; HF catches false frequencies; Variance loves noise.

Combination architecture — two-stage gate:

**Stage 1** — "Is residual healthy?" (binary check):

* If residual is stable (instability < threshold, FSR < threshold) → weights collapse to residual-only.
* If not → transition to Stage 2.

**Stage 2** — utility-based soft weights:

* Each expert (resid, var, hf) gets a weight by utility: U_i = median(gain) − λ·FSR − μ·instability.
* Residual has a minimum guaranteed weight (prior).
* Weights are EMA-smoothed with hysteresis.

Normalization — quantile (rank-based), not absolute.

**Experimental status:**

* Exp0.4: Residual-only = oracle on clean data.
* Exp0.5: Oracle breaks under noise (corr 0.90 → 0.54), degraded coarse.
* Exp0.6: Binary switch resid/combo works, but alias is a borderline case.
* Exp0.7: Soft gating wins on noise/spatvar/jpeg (+0.08–1.10 dB), but loses on clean/blur (−1.6–2.5 dB) due to softmax "democracy".
* Exp0.7b: Two-stage gate solves clean/blur (Δ ≈ 0) and retains gains on noise/spatvar (+0.77–1.49 dB). JPEG — only downside (−0.21 dB, threshold tuning issue).

**Conclusion:** the two-stage gate is the correct architecture. Residual rules while reliable; when it breaks — smoothly yields to the combination.

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

Hard constraints: max_depth, max_nodes, max_cost, max_boundary_budget.

## 4.5 Junk Detector

If the average gain over K splits < δ (5–10% of previous level) — noise splitting is suppressed.

---

# 5. Schedule: Strictness Control (Exp0.8)

## 5.1 Governor (EMA Controller) — Mandatory

**Control variable:** strictness — quantile threshold for selecting refine candidates. All tiles above threshold pass; spending fluctuates.

**Feedback signal:** EMA of actual cost/step (after warmup).

**Goal:** keep cost/step within a corridor around target_per_step (per-step, not "on average").

**Safeguards:**

* Δstrictness per step ≤ clamp (anti-oscillation).
* hard_cap_per_step = 3× target (safety fuse against spending explosion).
* Warmup: first N steps strictness doesn't move, but spending accrues.
* Compliance — asymmetric: overbudget penalty > underbudget penalty (overspend = SLA violation, underspend = missed opportunity).

**Experimental result (Exp0.8v5):**

* Governor halves StdCost (~5.15 → ~3.25), P95 from 11.0 to ~6.5, compliance penalty from ~3.2 to ~0.5 — across all 4 scenes.
* PSNR slightly lower (−0.24 dB clean, −0.68 dB shift): fixed threshold "eats" the best tiles upfront, governor spreads more evenly. At equal total cost, governor loses on peak quality but wins on spending predictability.
* **Conclusion:** governor is needed for budget control, not quality. Without it, the budget is a declaration.

## 5.2 Governor Applicability Condition

Governor works only if the strictness range actually changes the number of candidates.

If at max strictness n_candidates ≥ hard_cap — the controller hits the ceiling and is useless.

This is fixed by task parameterization (more tiles, different target, different range), not "yet another piece of magic".

## 5.3 Phase Schedule (ρ weights by step/depth) — Deferred

Exp0.8v5 showed no PSNR gain with depth-dependent weights (resid→anomaly across steps).

Hypothesis: with ideal refine and reliable ρ, changing priorities by phase is overengineering.

Keep as an optional extension for the future, when a real tree with non-ideal refine appears and phase-based signal switching becomes physically justified.

---

# 6. Boundary-Aware Refinement (Mandatory Requirement)

Exp0.2–Exp0.3 showed:

Refinement without overlap / feather / halo during tile substitution is guaranteed to produce false HF artifacts under sparse budgets.

Cause: hard insertion of a refined tile → step discontinuity at the boundary → Laplacian flags it as a false HF signal.

The smarter the adaptive tile selection, the stronger the artifact (adaptive concentrates refine in high-gradient zones where the step hits harder).

**Minimum rule:**

* overlap ≥ 3–4 pixels at tile_size=16.
* Blending (cosine feather) is performed relative to coarse.

Halo ≥ 3 pixels fully eliminates HF degradation. Interior-only HF metric confirmed: the problem was in seams, not in tile selection.

---

# 7. Probe-Budget Exploration (Mandatory Requirement)

The strategy "refine only found boundaries" is structurally blind.

Each step:

* 90–95% budget → exploitation.
* 5–10% budget → probe.

Probe priority: coarse residual / variance, uncertainty, time since last check.

If probe detects activity — the domain is activated.

**Trade-off:** on stationary scenes, probe may reduce PSNR (budget spent "in vain"), but it increases the chance of catching shifts/rare regions. Probe contribution (ablation) in Exp0.8 confirmed: probe's PSNR contribution is minimal on static data, but probe is insurance against structural blindness, not a quality optimization.

Overlap does not replace exploration. Exploration does not fix seams. Both mechanisms are mandatory.

---

# 8. Boundaries

A boundary is a zone of informativeness decay.

Halo is applied on both sides of a boundary.

---

# 9. Map Construction Strategies

1. Shared density map (reusable).
2. Meaning-specific map (fast, but single-use).

---

# 10. Key Invariants

1. Refinement is additive (residual).
2. Halo is mandatory.
3. Exploration is mandatory.
4. Cost is managed by the governor (EMA controller) within budget.
5. The system does not create artificial seams.
6. The system does not become structurally blind.
7. ρ defines map semantics; combination via two-stage gate.

---

# 11. Core Principle of Curiosity

Curiosity is a system that:

1. Computes only where there is signal.
2. Does so without creating artificial seams (halo).
3. Does not become blind to internal structure (probe).
4. Manages budget consciously (governor), not declaratively.
5. Adapts the interestingness function to conditions (two-stage gate), rather than relying on a single sensor.

Formally: refinement must be boundary-aware, must include controlled exploration, and must be budget-governed.

---

# 12. Experimental Status (Exp0.1–Exp0.8)

## Confirmed

| Experiment | Question | Result |
|---|---|---|
| Exp0.1 | Does adaptive refinement work? | Yes: adaptive > random by MSE/PSNR at any coverage. |
| Exp0.2 | Robust on real images? | Yes by MSE_rgb/PSNR (98–100% winrate). HF degradation at low budget → motivation for halo. |
| Exp0.3 | Is halo needed? | Yes: halo ≥ 3 px eliminates HF artifacts. Interior-only HF confirmed: problem was in seams. |
| Exp0.4 | Is combined ρ needed? | On clean data: residual-only = oracle. Combination not needed. |
| Exp0.5 | Does the oracle break? | Yes: under noise, blur, alias residual degrades (corr 0.90 → 0.54). |
| Exp0.6 | Does binary switch work? | Partially: clean/blur → resid, noise → combo. Alias — borderline case. |
| Exp0.7/0.7b | Is soft gating needed? | Two-stage gate: solves clean/blur (Δ ≈ 0), wins on noise/spatvar (+0.77–1.49 dB). |
| Exp0.8 | Is schedule needed? | Governor (EMA): needed for budget control (StdCost −50%, penalty −85%). Phase schedule: not needed under current conditions. |

## Open Questions (Next Verification Hierarchy)

* Is auto-tuning of gate thresholds needed? (currently manual instability/FSR thresholds)
* Is a complex data structure needed? (Morton / dynamic list vs. fixed grid)
* Does the tree provide a semantic metric? (bushes, LCA-distance as feature)
* How does the system behave with non-ideal refine? (delta ≠ GT − coarse)
* Does phase schedule come alive with non-ideal refine?
* How to ensure "no broken features"? Adaptive refinement by definition creates a non-uniform representation: part of the space is refined, part is not. If a downstream consumer (model, loss, next layer) expects a coherent feature vector, non-uniform refinement depth may destroy feature interdependencies. Halo fixes local seams but does not guarantee global coherence. A mechanism or invariant is needed to ensure that the representation at any "matryoshka" level remains a valid input for the consumer — not just visually smooth.
