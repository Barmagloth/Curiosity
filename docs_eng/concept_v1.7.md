# Curiosity — Conceptual Document (v1.7)

This document captures the current project logic so that the line of reasoning can be reconstructed a year from now without external context.

Updated after theoretical analysis following experiments Exp0.1–Exp0.8, and Phase 0 results (cross-space validation).

Changes relative to v1.6:
- Section 6 (Halo): added topology-based applicability rule
- Section 8 (SC): updated D_parent formula (lf_frac), R=gauss σ=3.0
- Section 8: added cross-space validation results for SC
- Section 11: added cross-space validation invariant
- Section 13: updated experimental status

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

1. Start with a coarse approximation (**root coarse** — the initial representation, the starting point of the refinement tree).
2. Compute ρ.
3. Decide whether to split a region.
4. Refine only selected regions.
5. Repeat until stopping criteria.

### Terminology: coarse and delta hierarchy

The project distinguishes two senses of "coarse" and two senses of "delta":

* **Root coarse** — the very first coarse representation from which the entire refinement tree grows. Global anchor.
* **Parent coarse** — the coarse representation of the parent scale for a given refinement step. At level L, parent coarse is the output of level L−1. At L=1, parent coarse coincides with root coarse.
* **Step delta** — the additive correction of a single refinement step: `refined_L = parent_coarse_L + step_delta_L`.
* **Cumulative delta** — the accumulated correction relative to root coarse: `cumulative_delta_L = refined_L − root_coarse↓L` (where ↓L denotes projection to the scale of level L). Needed for drift diagnostics, not for local refinement mechanics.

Refinement always takes the form:

```
refined = parent_coarse + step_delta
```

where step_delta:

* is initialized to zero,
* is energy-bounded,
* disabling any current level returns parent_coarse (a valid representation of the previous level),
* may degrade local consistency at boundaries.

Halo is applied at each refinement level (subject to the topology applicability rule, see section 6).

---

# 2. Trees and "Bushes"

## 2.1 Tree

A tree is a log of refinement routes.

Each root-to-leaf path is a sequence of split decisions.

The structure must be GPU-friendly (flat packing, no pointer chasing).

**Alternative interpretation (theoretical, not experimentally validated):**
A tree can be viewed as a trajectory of RG-flow (renormalization group flow). Nodes are system states at different scales; edges are coarse-graining operations. Fixed points of the flow are domains where further refinement ceases to change the effective description. This interpretation currently does not affect architecture, but motivates the Scale-Consistency Invariant (section 8).

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
* D_parent — inter-scale drift (see section 8). Contextual signal, not self-sufficient.

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

## 4.5 Garbage Detector

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

## 6.1 Halo Applicability Rule by Topology

Halo is applied **not unconditionally**, but when two conditions are met:

a) **Boundary parallelism ≥ 3** — at least 3 independent cross-edges at the tile boundary (min_cut ≥ 3).
b) **No context leakage** — halo expansion does not bleed into unrelated tiles.

Both conditions must hold simultaneously.

### By topology:

| Topology | Halo | Rationale |
|---|---|---|
| Grid (tile_size ≥ 3) | **ALWAYS** apply | Boundary parallelism guaranteed |
| k-NN graph with spatial clustering | **Apply** | min_cut >> 3 |
| Tree / Forest | **NEVER** | min_cut=1, context leakage guaranteed |
| DAG | **Check per-case** | Depends on specific structure |

### Root cause of tree failure

Trees have single-edge bottleneck (min_cut=1 between any two subtrees), sibling bleed (halo of one child bleeds into the subtree of a sibling), and extreme S/V asymmetry (surface/volume ratio). These three factors make halo on trees not merely useless but harmful.

## 6.2 Minimum Rule (for applicable topologies)

* overlap ≥ 3–4 elements at tile_size ≥ 16.
* Blending (cosine feather) is performed relative to parent coarse.

**Important:** halo is responsible for local geometric correctness at tile boundaries. This is orthogonal to the Scale-Consistency Invariant (section 8), which is responsible for inter-scale semantic correctness. One does not replace the other.

---

# 7. Probe-Budget Exploration (Mandatory Requirement)

Each step:

* 90–95% budget → exploitation.
* 5–10% budget → probe.

**Refined role of probe (v1.6):** probe is not simply exploration, but a defense mechanism against false fixed points. If refinement converges to a state "nothing interesting here", probe periodically checks this claim. By analogy with thermal fluctuations in physics: random perturbations preventing degeneration into a local minimum.

Overlap does not replace exploration. Exploration does not fix seams. Both mechanisms are mandatory.

---

# 8. Scale-Consistency Invariant (New Section, v1.6; updated v1.7)

## 8.1 Motivation

The open question from v1.5 "how to ensure no broken features" formalizes through a requirement of scale-consistency of representations.

Halo ensures local correctness at boundaries (geometric level). This is insufficient: a refined level can be smooth at seams, yet semantically contradict the parent coarse at its own scale.

## 8.2 Asymmetric Invariant

Key principle: **parent coarse is the anchor, step delta is a subordinate correction**.

`refined = parent_coarse + step_delta`

The direction is one-way: refined does not redefine parent coarse, but lives within its tolerance. Strict equality `parent_coarse = R(refined)` is a trap: it destroys the matryoshka and converts the tree into a pyramid of bottom-up dependencies.

**Formal requirement:**

```
‖R(step_delta)‖ / (‖step_delta‖ + ε) < τ_rel
```

Meaning: step_delta should be invisible from the level above. Refinement can add detail, but should not smuggle a new parent-scale meaning upward.

## 8.3 Coarse-Graining Operator

For the invariant to work, a consistent pair **(R, Up)** must be fixed:

* **R** — coarse-graining operator: `low-pass filter + decimation`. Choice: gaussian blur σ=3.0 + decimation (linear, GPU-cheap, consistent with parent_coarse + step_delta). **Updated v1.7:** σ increased from 1.0 to 3.0 — σ=1.0 was insufficiently aggressive as LP filter, positive deltas retained energy after R.
* **Up** — LF-component restoration operator in delta space: `bilinear upsampling`. This is **not** the inverse of R (it does not exist), but a projection of the coarse component back into the original scale.

Pair (R, Up) is fixed before the first experiment. Different pairs give different tree physics and different fixed points. The choice is architectural, not technical.

Idempotency requirement: `‖R(parent_coarse) − parent_coarse_downsampled‖ ≈ 0`, where parent_coarse_downsampled is parent_coarse at R's target scale. If not satisfied — the tree collapses. (R includes decimation, so the comparison is made in the smaller-scale space, not with the original parent_coarse.)

## 8.4 Metrics

Two non-equivalent diagnostic axes (different numerators, but both built via pair (R, Up) — statistical independence not guaranteed):

**D_parent** — step_delta leakage into parent scale (updated v1.7):
```
D_parent = ‖R(step_delta)‖ / (‖step_delta‖ + ε)
```

**Update v1.7:** old formula `D_parent = ‖R(step_delta)‖ / (α·‖parent_coarse‖ + β)` replaced. Reason: the denominator `α·‖parent_coarse‖ + β` is identical for all deltas on the same tile and carries zero discriminative signal. New interpretation: "what fraction of delta energy is low-frequency?" (lf_frac) — operationally correct measure.

**D_hf** — high-frequency purity of step_delta itself:
```
delta_HF = step_delta - Up(R(step_delta))   # HF residue after LF-component removal
D_hf = ‖delta_HF‖ / (‖step_delta‖ + ε)
```

Interpretation of combinations:

| D_parent | D_hf | Meaning |
|---|---|---|
| high | low | step_delta is mostly LF and leaks into parent. Invariant violation. |
| low | high | step_delta is detail-like, does not touch parent. Normal. |
| both medium | — | Borderline zone; check residual/gain/probe. |

### Cross-space validation results (v1.7)

| Space | AUC D_parent |
|---|---|
| T1 scalar grid | 1.000 |
| T2 vector grid | 1.000 |
| T3 irregular graph | 1.000 |
| T4 tree hierarchy | 0.824 |

No space-specific tuning needed. The D_parent (lf_frac) formula works out of the box across all 4 space types.

## 8.5 Threshold τ_rel / τ_parent

τ_rel is the conceptual relative threshold from the invariant formula. In operational implementation, it materializes as **τ_parent** — a data-driven threshold, possibly depth-dependent:

```
τ_rel  →  τ_parent[L]
```

Threshold is not a constant: relative normalization is embedded in the formula, but at fine levels step_delta should become increasingly HF-pure, so τ_parent[L] may decay with depth. This is determined from the baseline experiment, not prescribed in advance.

ε in the denominator is a stabilizer against division by zero in nearly empty regions.

**τ_parent is established data-driven** via baseline experiment (see section 8.6 and `scale_consistency_verification_protocol_v1_0.md`), not manually.

## 8.6 Baseline Experiment (Mandatory First Step)

Before enforcement is introduced, distributions of D_parent and D_hf must be collected on:

* **Positive baseline** — demonstrably correct refinement cases (clean synthetic data, near-GT delta).
* **Negative baseline** — demonstrably bad cases (artificial LF drift, intentional parent_coarse shift, semantically wrong step_delta).

Measure across slices: globally, by tree level, by structure type (smooth, boundary, texture, noise).

**Kill criterion (fixed before baseline runs):**

If separability between positive and negative is insufficient (by AUC / effect size / quantile separation) — **metrics or the pair (R, Up) are reconsidered**, not thresholds τ tuned. Poor separability means the signal does not carry the needed discriminative information, not that the threshold was chosen poorly.

**Status (v1.7): SC-baseline has been RUN and PASSED.**

* D_hf: AUC=0.806, d=1.34 (pass)
* D_parent (with formula and R fixes): AUC range [0.824, 1.000] across 4 spaces (pass)
* coarse_shift generator fixed (spatially coherent sign fields)
* Status: baseline complete, ready for SC-enforce

## 8.7 Enforcement

After baseline validation:

```python
if D_parent > τ_parent:
    damp step_delta
    reject split
    increase strictness locally
```

D_parent is also used as a contextual signal in ρ:
* high D_parent + high residual + stable gain → region is structurally interesting → raise probe priority
* high D_parent + low gain + unstable step_delta → refinement mechanics problem → discourage split

## 8.8 Scale-Stable Fixed Point

Local stopping of refinement is determined not only by budget, but by reaching a scale-stable fixed point.

A node is considered scale-stable if simultaneously:
1. `gain_from_split < τ_gain`
2. `D_parent < τ_parent`
3. This holds stably for K consecutive steps.

Meaning: refinement adds nothing meaningful either locally or at the level above.

**Probe remains mandatory** as insurance against false stabilization (a false fixed point is indistinguishable from a real one without external checks).

Budget governor remains as a global safeguard. Local stopping by fixed point and global budget are orthogonal mechanisms.

## 8.9 Step_delta Tolerance and Two-Sided Risk

τ_parent effectively defines **step_delta tolerance** — the permissible fraction of parent_coarse that step_delta is allowed to alter when projected back to the parent scale.

This creates a two-sided risk:

> **τ_parent too tight** → SC invariant cuts legitimate features that only emerge upon refinement. The system loses information it should have kept. Consequence: conservative tree that under-refines structurally rich regions.
>
> **τ_parent too loose** → The tree rewrites itself in a few steps. Step_delta accumulates across levels, and parent_coarse semantics drift. In the best case — gradual drift; in the worst case — full hierarchy degradation.

**Key clarification:** step_delta is computed relative to **parent_coarse** (the frozen working tile of the previous level), not relative to root_coarse. This limits the scale of the problem: each step_delta is small relative to its immediate parent, even if cumulative_delta relative to root_coarse is large.

**Open question:** automatic mechanism for choosing τ_parent that balances these two risks does not yet exist. The baseline experiment (section 8.6) sets τ_parent empirically from positive cases, but the optimal trade-off between sensitivity and stability is data-dependent and may require adaptive τ_parent(L, regime). This is deferred until after the SC-baseline experiment provides real distributions.

## 8.10 (R, Up) Sensitivity (Open)

The choice of pair (R, Up) is architectural and determines the entire physics of the tree: D_parent distributions, D_hf distributions, scale-stable fixed points, and τ_parent thresholds.

Current default (gaussian blur + decimation / bilinear upsampling) is chosen for simplicity and GPU efficiency. **Sensitivity of the system to this choice is assumed low but not validated experimentally.** If different (R, Up) pairs produce qualitatively different trees on the same data, the default may need justification or a selection mechanism.

See **Exp0.10: (R, Up) Sensitivity Probe** in `experiment_hierarchy.md`. Dependency: after SC-baseline.

---

# 9. Boundaries

A boundary is a zone of informativeness decay.

Halo is applied on both sides of a boundary (subject to the applicability rule, section 6).

---

# 10. Map Construction Strategies

1. Shared density map (reusable).
2. Meaning-specific map (fast, but single-use).

---

# 11. Key Invariants

1. Refinement is additive (residual).
2. Halo is mandatory (local boundary of tiles) — subject to topology applicability rule (section 6).
3. Exploration is mandatory (probe as defense against false fixed points).
4. Cost is managed by the governor (EMA controller) within budget.
5. The system does not create artificial seams.
6. The system does not become structurally blind.
7. ρ defines map semantics; combination via two-stage gate.
8. **Scale-consistency:** step_delta does not redefine the semantics of the parent scale. `‖R(step_delta)‖ / (‖step_delta‖ + ε) < τ_rel`. Pair (R, Up) is fixed. Thresholds are data-driven.
9. **Cross-space validation:** any claim about "arbitrary spaces" must be validated on ≥4 space types (scalar grid, vector grid, irregular graph, tree hierarchy). A result on a single type is NOT sufficient.

---

# 12. Core Principle of Curiosity

Curiosity is a system that:

1. Computes only where there is signal.
2. Does so without creating artificial seams (halo).
3. Does not become blind to internal structure (probe / defense against false fixed points).
4. Manages budget consciously (governor), not declaratively.
5. Adapts the informativeness function to conditions (two-stage gate), rather than relying on a single sensor.
6. **Does not destroy the semantics of the parent scale during refinement (scale-consistency).**

Two conservation laws:
* budget conservation → budget governor
* scale consistency → D_parent constraint + enforcement

Formally: refinement must be boundary-aware, must include controlled exploration, must be budget-governed, and must preserve scale-consistent representation.

---

# 13. Experimental Status

## Confirmed (Exp0.1–Exp0.8)

| Experiment | Question | Result |
|---|---|---|
| Exp0.1 | Does adaptive refinement work? | Yes: adaptive > random by MSE/PSNR at any coverage. |
| Exp0.2 | Robust on real images? | Yes by MSE_rgb/PSNR (98–100% winrate). HF degradation → motivation for halo. |
| Exp0.3 | Is halo needed? | Yes: halo ≥ 3 elements eliminates HF artifacts. |
| Exp0.4 | Is combined ρ needed? | On clean data: residual-only = oracle. |
| Exp0.5 | Does the oracle break? | Yes: under noise residual degrades (corr 0.90 → 0.54). |
| Exp0.6 | Does binary switch work? | Partially. Alias — borderline case. |
| Exp0.7/0.7b | Is soft gating needed? | Two-stage gate: solves clean/blur (Δ ≈ 0), wins on noise/spatvar (+0.77–1.49 dB). |
| Exp0.8 | Is schedule needed? | Governor (EMA): needed. Phase schedule: not needed under current conditions. |

## Phase 0: Cross-Space Validation (v1.7)

| Component | Result | Status |
|---|---|---|
| Halo cross-space | Validated on 3/4 spaces, topology applicability rule derived | pass (with constraints) |
| SC-baseline (D_hf) | AUC=0.806, d=1.34 | pass |
| SC-baseline (D_parent, lf_frac) | AUC [0.824, 1.000] across 4 spaces | pass |
| coarse_shift generator | Fixed: spatially coherent sign fields | fix applied |
| P2a sweep | Code ready, 20K configs | ready |

## Open Questions

* Is auto-tuning of gate thresholds needed? (currently manual instability/FSR thresholds)
* Is a complex data structure needed? (Morton / dynamic list vs. fixed grid)
* Does the tree provide a semantic metric? (bushes, LCA-distance as feature)
* How does the system behave with non-ideal refine? (step_delta ≠ GT − parent_coarse)
* Does phase schedule come alive with non-ideal refine?
* ~~**Scale-consistency baseline:** what is the separability of D_parent / D_hf on positive vs. negative cases? Is the chosen pair (R, Up) sufficiently discriminative?~~ → Closed: baseline completed, separability sufficient (AUC 0.806–1.000). D_parent formula updated to lf_frac, R updated to σ=3.0. coarse_shift generator fixed.
* **Fixed points:** are scale-stable fixed points a reliable local stopping criterion in practice, or is probe insufficient as the only safeguard?
* **(R, Up) sensitivity:** is the system's behavior qualitatively stable across different (R, Up) pairs? See section 8.10, Exp0.10.
* **Step_delta tolerance:** automatic mechanism for choosing τ_parent that balances feature loss vs. hierarchy drift. See section 8.9.
* **Track C spatial dependency:** transition to non-spatial domains (graphs, latent spaces) will require reworking spatially-dependent components: Halo (cosine feathering), R/Up (gaussian/bilinear), SeamScore (gradient energy), and parts of ρ (Laplacian-based HF energy). This is not incremental adaptation but partial redesign. Accepted as a known trade-off.
* ~~How to ensure "no broken features"?~~ → Closed: formalized via Scale-Consistency Invariant (section 8). Transitions to experimental queue as baseline experiment.
