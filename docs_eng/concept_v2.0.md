# Curiosity — Conceptual Document (v2.0)

v2.0 (March 23, 2026): Three-layer rho decomposition (L0 Topology → L1 Presence → L2 Query), cascade quotas, streaming pipeline, Phase 3/3.5 complete.

This document captures the current project logic so that the line of reasoning can be reconstructed a year from now without external context.

Updated after theoretical analysis following experiments Exp0.1–Exp0.8, and Phase 0 results (cross-space validation).

v2.0: added determinism invariants (section 8A) and reproducibility requirements.

v1.8.3: Layout Selection Invariant — layout choice as a function of three space properties, not a type name.

Changes relative to v1.6:
- Section 6 (Halo): added topology-based applicability rule
- Section 8 (SC): updated D_parent formula (lf_frac), R=gauss σ=3.0
- Section 8: added cross-space validation results for SC
- Section 11: added cross-space validation invariant
- Section 13: updated experimental status

Changes in v1.8.4:
- Section 14: added Enox observation patterns (RegionURI, DecisionJournal, MultiStageDedup, PostStepSweep)

Changes in v1.8.3:
- Section 2.1: added Layout Selection Invariant (three axes: isotropy, metric gap, dynamic density)
- Section 11: added invariant #12 (layout is determined by space properties)
- Section 12: added fourth conservation law (layout–topology correspondence)

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

**Addressing principle (v1.8.1):** addressing must be at the abstraction level of the operation. Refinement operates on tiles — addressing is tile-level, not element-level. Data within a tile is dense (dense intra-tile). Outside — sparse by tiles. A global element-level reverse_map is prohibited (exp10: VRAM +38.6%, constructive failure). Lookup — only on active/support set O(k), not on the entire space O(M).

**Layout policy by space types (v1.8.2, exp10e-j, FINAL):**
- scalar_grid → D_direct (packed tiles + tile_map) — production layout (exp10g: both contours PASS)
- vector_grid → D_direct — production layout (exp10h: 72/72 PASS both contours)
- tree_hierarchy → Hybrid per-level: D_direct where p_l < 0.40 AND operator is compute-heavy (matmul-like); A_bitset elsewhere. Upper levels (small N_l, high occupancy) → A_bitset. Lower levels (large N_l, low occupancy, heavy compute) → D_direct. Stencil ops: D saves memory but NEVER wins time → A_bitset. (exp10j: 158K trials, break-even stable across all branching factors)
- irregular_graph / spatial → D_blocked conditional (spatial partition, cbr ≤ 0.35) (exp10i)
- irregular_graph / scale-free → blocked layout rejected (cbr=0.66); A_bitset fallback (exp10i)

For graphs: fixed-size blocks are NOT a universal abstraction. Compute-path is healthy (Contour B 100%), data representation is sick (padding 50-97%). Graphs split into two classes based on presence of spatial structure.

**Layout Selection Invariant (v1.8.3)**

The table above is a precomputed cache of a fundamental law: layout is determined by a vector of space properties, not by its type name. Three axes:

**Axis 1 — Topological isotropy I(X).** A measure of local structure homogeneity. Formalization: entropy of the degree distribution H(D) = −Σ P(k) log P(k). When I = 0 (H(D) = 0, regular grid) direct addressing (D_direct) is optimal — memory offsets are fully predictable. As I grows, indirect addressing becomes unavoidable.

**Axis 2 — Metric gap M(X, L).** A measure of divergence between topological distance and physical addressing. Formalization: Kendall rank correlation τ between BFS-rank of neighbors and the difference of their linear addresses in the chosen linearization L. Crucially: M is a property of the pair (topology, linearization), not of topology in isolation. The same graph under Morton ordering has high τ, under random numbering — low τ. When τ → 1 block packing (D_blocked) is effective (a cache line covers neighbors). When τ → 0 any attempt at local packing turns into padding waste.

Corollary: cbr (cross-block ratio, exp10i) is an empirical projection of M(X, L) onto a specific blocking scheme.

**Axis 3 — Dynamic density p.** The only axis that depends not on space structure but on algorithm state. Occupancy p = k/N (fraction of active tiles). When p > p* the overhead of sparsity support (tile_map, block_map) exceeds the cost of computing over the entire universe (A_bitset). When p < p* the emptiness is too expensive to drag through the ALU.

Refinement: p* depends on (operator_weight, absolute_scale). The threshold p* ≈ 0.40 is stable across domain sizes in exp10j (2–4096 nodes), but amortization of tile_map overhead requires sufficient absolute N. On very small domains tile_map overhead never pays off regardless of p (exp10h: trees with 15–585 nodes → 0/108 FAIL). The stability of p* ≈ 0.40 is likely due to hardware quantization (warp size = 32, memory transactions = 128 bytes) — the continuous cost function collapses into discrete steps at the silicon level. Experimentally this is a hypothesis, not a fact.

**Law:** layout = argmin C(I, M, p), where C is the integral cost (addressing + padding + compute waste). The current policy table contains tabulated values of C for known space classes. When a new space type appears, it suffices to compute (I, M, p) and find the nearest table row, rather than writing a new if-branch.

Runtime profiler is NOT planned: in the current scope space types are known statically. Dynamic dispatch is Track C territory, not before.

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

## 3.3 Three-Layer Decomposition of ρ (exp17, Phase 3.5, 22–23 March 2026)

Phase 3 showed: monolithic ρ conflates three orthogonal concerns, making the refinement tree non-reusable. Solution — cascading decomposition:

**Layer 0 (Topology)** — "what is the structure of the space?" Data-independent. Computed once at initialization. For graph: Leiden communities + curvature + PageRank + boundary anomalies → per-unit structural score + cluster_ids. For grid: spatial quadrant blocks (trivial topology). For tree: depth-band grouping.

**Layer 1 (Presence)** — "where is there data?" Data-dependent, query-independent. Binary presence map: "is there non-trivial signal here?" Threshold via **cascade quotas** (Variant C) — each L0 cluster guarantees min(1, ceil(cluster_size × budget_fraction)) survivors. Topology dictates quotas; no region is extinguished.

**Layer 2 (Query)** — "where exactly does refinement pay off?" Task-specific. Three interchangeable query functions on the same frozen tree: MSE, max absolute error, HF residual. Cheap — the tree already exists.

**FrozenTree** — serializable snapshot of L0+L1. Reusable spatial index. Analogous to an R-tree in spatial databases.

**Two build modes:**
- **Batch**: L0(all) → L1(all) → L2(all) — traditional, baseline.
- **Streaming**: cluster-by-cluster processing with L0-priority ordering and global budget cap. First results appear after the first cluster is processed.

**Budget control** is NOT an EMA governor (section 5.1 describes the Exp0.8-era governor). The three-layer pipeline uses **StrictnessTracker + WasteBudget** (self-tightening noose): each rejected unit's cost equals its strictness multiplier (escalates ×1.5 per reject, decays ×0.9 per clean step). When weighted waste >= R_max = floor(B × ω), the step force-terminates.

**Historical note (v1.8):** the original monolithic ρ = all three layers fused into one number. It works but: non-reusable (different query = full rebuild), opaque (cannot tell topology from signal), mixes concerns (structural analysis entangled with task-specific scoring).

**Experimental status (exp17):** 1080 configs, 4 spaces × 3 scales (100/1K/10K) × 8 approaches × 20 seeds. Reusability: 12/12 PASS (min ratio 0.838). PSNR: 2–4 dB below single_pass on grid (price of pruning), parity on graph/tree. Streaming 10–20% faster than batch. kdtree (scipy C) faster on single query; three_layer wins at >=2 queries on tree_hierarchy. Roadmap: C/Cython scoring optimization for multiplicative speedup.

**Relationship with two-stage gate (3.2):** the gate remains inside Layer 2 — it determines which query signal to use (residual-only or combo). Layers 0–1 operate before the gate and do not depend on it.

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

## 5.1 Budget Governor (StrictnessTracker + WasteBudget) — Mandatory

**Implementation:** `StrictnessTracker` + `WasteBudget` (see `experiments/exp14a_sc_enforce/sc_enforce.py`, `pipeline.py` lines 421-439, 476). A self-tightening noose: per-unit multipliers with escalation (x1.5 on reject) and decay (x0.9 per clean step); WasteBudget: R_max = floor(B_step * omega), each reject costs strictness_multiplier units, force-stop when waste >= R_max.

**Note:** `GovernorIsolation` from exp10d is an EMA tracker for DET-1 step isolation (always receives 1.0, output never used for decisions). Do not confuse with Budget Governor.

**Two-layer Governor architecture (v2.0):**
1. **Hardware parameter** (static, at init): sets the RANGE of how much the governor can tighten/loosen. Powerful hardware → wide range. Weak hardware → narrow range. Input: hardware calibration (Synthetic Transport Probe).
2. **EMA feedback** (dynamic, at runtime): moves WITHIN the hardware-defined range based on live signals (waste rate, rejection rate, cost/step).

Metaphor: dog on leash. Hardware parameter = leash length. EMA = how far the dog actually walks. **Status:** EMA governor worked in exp0.8 but was lost during Phase 2 pipeline assembly. Needs restoration in Phase 4.

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
* PSNR slightly lower (−0.24 dB clean, −0.68 dB shift): fixed threshold "eats" the best tiles upfront, budget governor spreads more evenly. At equal total cost, budget governor loses on peak quality but wins on spending predictability.
* **Conclusion:** budget governor is needed for budget control, not quality. Without it, the budget is a declaration.

## 5.2 Governor Applicability Condition

Governor works only if the strictness range actually changes the number of candidates.

If at max strictness n_candidates ≥ hard_cap — the controller hits the ceiling and is useless.

This is fixed by task parameterization (more tiles, different target, different range), not "yet another piece of magic".

## 5.3 Phase Schedule (ρ weights by step/depth) — Deferred

Exp0.8v5 showed no PSNR gain with depth-dependent weights (resid→anomaly across steps).

Hypothesis: with ideal refine and reliable ρ, changing priorities by phase is overengineering.

Keep as an optional extension for the future, when a real tree with non-ideal refine appears and phase-based signal switching becomes physically justified.

## 5.4 Budget Control in Three-Layer Pipeline (v2.0, exp17)

The EMA governor (5.1) was designed for the monolithic ρ era. The three-layer pipeline (section 3.3) replaces it with a different mechanism:

**StrictnessTracker + WasteBudget** (self-tightening noose):
- Each rejected unit accumulates cost equal to its strictness multiplier.
- Multiplier escalates ×1.5 per consecutive reject, decays ×0.9 per clean (accepted) step.
- When weighted waste >= R_max = floor(B × ω), the step force-terminates.

Key difference from EMA governor: the waste budget is not a smoothed average but a hard cumulative cap with escalating penalties. This provides stricter guarantees in the streaming regime where cluster processing order affects budget distribution.

The EMA governor remains valid for monolithic-ρ use cases. The two mechanisms are not mixed.

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

* **R** — coarse-graining operator: `low-pass filter + decimation`. Choice: gaussian blur σ=3.0 + decimation (linear, GPU-cheap, consistent with parent_coarse + step_delta). **Updated v1.7:** σ increased from 1.0 to 3.0 — σ=1.0 was insufficiently aggressive as LP filter, positive deltas retained energy after R. **Known limitation:** σ=3.0 was chosen as the smallest integer value passing kill criteria in a coarse sweep [0.5, 1.0, 2.0, 3.0]. Fine-grained sweep not conducted; optimal σ may depend on tile_size and space type. A dedicated experiment (SC-σ sweep) is planned.
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

# 8A. Determinism and Reproducibility (new section, v2.0)

## 8A.1 Motivation

In v1.3 there existed an invariant: "The tree is built by a stable rule." By v1.7 it was lost — replaced by other invariants without explicit inheritance.

Without formalized determinism:
* **Testability** (Track A): impossible to catch regressions if the "correct" tree looks different every time. A code change is indistinguishable from statistical noise.
* **Reproducibility** (Track B): results that cannot be repeated under fixed conditions are not amenable to scientific analysis.
* **Content-addressable cache**: route hashing loses effectiveness with unstable paths (content cache is unaffected, but route reuse is).

## 8A.2 Two Levels of Requirements

### Level 1: Seed Determinism (Hard Constraint)

**Invariant:** identical data + ρ + seed + budget = identical tree (bitwise match).

Three mandatory components:

1. **Canonical traversal order (tie-break).** When ρ values are equal, split order is determined by canonical tile index (Z-order / Morton). Lower index goes first. No dependence on hash table iteration order or thread races.

2. **Deterministic probe.** Probe seed = f(tile_coordinates, depth_level, global_seed). Pseudo-random but reproducible. Exploration is predictable given a fixed seed.

3. **Governor isolation from processing order.** EMA accumulation and governor decisions do not depend on the order of processing sibling branches. Processing in canonical order, EMA update strictly after the full step.

**Enforcement:** Unit tests: two runs with identical inputs → bitwise tree match. Any divergence = fail.

### Level 2: Cross-Seed Statistical Stability (Soft Constraint)

**Invariant:** across different seeds, metrics (PSNR, cost, compliance, SeamScore) are statistically stable.

Formally: for N ≥ 10 seeds, coefficient of variation CV(metric) < τ_cv (τ_cv set data-driven from baseline).

Different seeds produce different trees — that is normal. But metrics must be stable. If one ρ with seed=42 yields PSNR 30 dB and with seed=43 yields 25 dB — the system is unstable and conclusions drawn from it are invalid.

**Already partially in place:** 10–20 seeds in experiment protocols, Holm-Bonferroni corrections. Formalization requires only recording as an invariant and defining τ_cv.

## 8A.3 Relationship with Other Mechanisms

| Mechanism | Source of Non-determinism | Solution |
|---|---|---|
| Probe | Stochastic tile selection | seed = f(coords, level, global_seed) |
| Governor (EMA) | Processing order affects accumulation | Canonical order + update after full step |
| Split at equal ρ | Dependence on traversal implementation | Tie-break by Z-order/Morton index |
| Halo overlap on GPU | Overlapping writes during parallel accumulation | Exp0.9h (absorbed into DET-1) |
| Two-stage gate | Expert evaluation order | Fixed order, no data dependence |

## 8A.4 Experimental Validation

Two experiments (see experiment_hierarchy.md):

1. **DET-1: Seed determinism.** Two runs, identical inputs, one seed → bitwise tree match. CPU and GPU separately. Kill criterion: any divergence = fail.

2. **DET-2: Cross-seed stability.** N=20 seeds, 4 spaces, 2 budgets. Measure CV for PSNR, cost, compliance, SeamScore. Kill criterion: CV > 0.10 for any metric = fail (threshold preliminary, refined from baseline).

DET-1 is a precondition for Track A (Instrument Readiness Gate, stability pass). DET-2 is a precondition for Track B.

---

# 9. Boundaries

A boundary is a zone of informativeness decay.

Halo is applied on both sides of a boundary (subject to the applicability rule, section 6).

---

# 10. Map Construction Strategies

## 10.1 Historical Approach (v1.8, monolithic ρ)

Two strategies existed:

1. **Shared density map** (reusable) — a single ρ map shared across queries.
2. **Meaning-specific map** (fast, but single-use) — ρ tailored to one task.

In practice, the monolithic ρ made strategy 1 illusory: because ρ fused topology, presence, and query concerns into one number, a "shared" map was still query-contaminated. Different queries required full rebuilds.

## 10.2 Current Architecture: Three-Layer Map (v2.0, exp17)

The three-layer decomposition (section 3.3) resolves this by separating concerns:

**Reusable layers (L0 + L1 → FrozenTree):**
- L0 (Topology): structural analysis — clusters, bridges, hubs, density, boundaries. Computed once per space. Data-independent.
- L1 (Presence): binary signal map — "is there non-trivial data here?" Data-dependent but query-independent. Cascade quotas from L0 guarantee minimum survivors per cluster.
- FrozenTree = serializable snapshot of L0+L1. Persists across queries.

**Disposable layer (L2 → Query map):**
- L2 (Query): task-specific refinement on the frozen tree. Three interchangeable query functions (MSE, max_abs, HF residual). Cheap to swap — the tree is already built.

**Build modes:**
- **Batch**: L0(all) → L1(all) → L2(all). Traditional full-pass.
- **Streaming**: cluster-by-cluster (L0→L1→L2 per cluster), L0-priority ordering, global budget cap. First results after first cluster.

**Budget control:** StrictnessTracker + WasteBudget (self-tightening noose). Each rejected unit's cost = its strictness multiplier (escalates ×1.5 per reject, decays ×0.9 per clean step). When weighted waste >= R_max = floor(B × ω), the step force-terminates. This replaces the EMA governor (section 5.1) for the three-layer pipeline.

**Result:** strategy 1 (shared reusable map) is now real — FrozenTree is genuinely query-independent. Strategy 2 (meaning-specific) corresponds to a full L0+L1+L2 pass for a single query. The cost of adding a second query drops to L2-only.

---

# 11. Key Invariants

1. Refinement is additive (residual).
2. Halo is mandatory (local boundary of tiles) — subject to topology applicability rule (section 6).
3. Exploration is mandatory (probe as defense against false fixed points).
4. Cost is managed within budget: EMA governor (monolithic ρ) or StrictnessTracker + WasteBudget (three-layer pipeline). See sections 5.1 and 5.4.
5. The system does not create artificial seams.
6. The system does not become structurally blind.
7. ρ defines map semantics; combination via two-stage gate.
8. **Scale-consistency:** step_delta does not redefine the semantics of the parent scale. `‖R(step_delta)‖ / (‖step_delta‖ + ε) < τ_rel`. Pair (R, Up) is fixed. Thresholds are data-driven.
9. **Cross-space validation:** any claim about "arbitrary spaces" must be validated on ≥4 space types (scalar grid, vector grid, irregular graph, tree hierarchy). A result on a single type is NOT sufficient.
10. **Seed determinism:** identical data + ρ + seed + budget = identical tree. Canonical traversal order (Z-order tie-break), deterministic probe (seed from coordinates), governor isolation from processing order. (Section 8A.)
11. **Cross-seed statistical stability:** metrics are stable across seeds (CV < τ_cv). Different trees are acceptable, different conclusions are not. (Section 8A.)
12. **Layout–topology correspondence:** memory layout format is determined by the space property vector [I(X), M(X,L), p], not by the type name. I = degree entropy (isotropy), M = Kendall rank correlation between topological and physical distance (metric gap), p = occupancy (dynamic density). The current policy table is a precomputed cache for known classes. (Section 2.1.)

---

# 12. Core Principle of Curiosity

Curiosity is a system that:

1. Computes only where there is signal.
2. Does so without creating artificial seams (halo).
3. Does not become blind to internal structure (probe / defense against false fixed points).
4. Manages budget consciously (EMA governor or StrictnessTracker + WasteBudget), not declaratively.
5. Adapts the informativeness function to conditions (two-stage gate), rather than relying on a single sensor.
9. **Decomposes ρ into reusable layers** (L0 Topology → L1 Presence → L2 Query), enabling FrozenTree reuse across queries.
6. **Does not destroy the semantics of the parent scale during refinement (scale-consistency).**
7. **Is reproducible under fixed conditions and statistically stable across seeds.**
8. **Selects memory layout by space properties (layout–topology correspondence), not by type name.**

Four conservation laws:
* budget conservation → budget governor (monolithic ρ) / StrictnessTracker + WasteBudget (three-layer)
* scale consistency → D_parent constraint + enforcement
* process determinism → canonical order + deterministic probe + governor isolation
* layout–topology correspondence → layout = f(I, M, p); policy table = precomputed cache

Formally: refinement must be boundary-aware, must include controlled exploration, must be budget-governed, must preserve scale-consistent representation, must be deterministic given fixed inputs, and must select memory layout by topological properties of the space.

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
| P2a sweep | PASS (ridge=100%, MANUAL_OK) | pass |

## Phase 1–3.5: Pipeline Validation and Decomposition (20–23 March 2026)

| Phase | Experiments | Key Result | Status |
|---|---|---|---|
| Phase 1 | exp1–exp4 (halo hardening, layout, DET-1, dirty signatures, segment compression) | Pipeline foundation, 240/240 DET-1 PASS | pass |
| Phase 2 | exp5–exp13 (pipeline assembly, e2e validation, SC-enforce, topo profiling, Enox infra) | 4 space types validated, pipeline production-ready | pass |
| Phase 3 | exp14 (anchors), exp15 (LCA), exp15b (bushes), exp16 (C-pre) | Anchors: grid PASS, graph/tree FAIL. LCA: FAIL (r<0.3). Bushes: FAIL (ARI<0.21). C-pre: PASS → UNFREEZE Track C | conditional |
| Phase 3.5 | exp17 (three-layer rho, 1080 configs) | Reusability 12/12 PASS, cascade quotas, streaming pipeline | pass |

## Open Questions

* ~~Is auto-tuning of gate thresholds needed? (currently manual instability/FSR thresholds)~~ → Closed: P2a sweep showed ridge=100%, manual thresholds sufficient (MANUAL_OK). P2b not needed.
* ~~Is a complex data structure needed? (Morton / dynamic list vs. fixed grid)~~ → Closed: Morton killed (exp09a, +12-15x overhead). Layout policy fixed per space type (exp10 series, 158K+ trials). D_direct for grid, hybrid D_direct/A_bitset for tree, D_blocked conditional for spatial graph.
* ~~Does the tree provide a semantic metric? (bushes, LCA-distance as feature)~~ → Closed: exp15 (LCA-distance Spearman r<0.3 all spaces — FAIL), exp15b (bushes: clusters exist, silhouette>0.4, but ARI<0.21 — unstable, FAIL). Tree is not semantic in the current architecture.
* How does the system behave with non-ideal refine? (step_delta ≠ GT − parent_coarse)
* Does phase schedule come alive with non-ideal refine?
* ~~**Scale-consistency baseline:** what is the separability of D_parent / D_hf on positive vs. negative cases? Is the chosen pair (R, Up) sufficiently discriminative?~~ → Closed: baseline completed, separability sufficient (AUC 0.806–1.000). D_parent formula updated to lf_frac, R updated to σ=3.0. coarse_shift generator fixed.
* **Fixed points:** are scale-stable fixed points a reliable local stopping criterion in practice, or is probe insufficient as the only safeguard?
* **(R, Up) sensitivity:** is the system's behavior qualitatively stable across different (R, Up) pairs? See section 8.10, Exp0.10.
* **Step_delta tolerance:** automatic mechanism for choosing τ_parent that balances feature loss vs. hierarchy drift. See section 8.9.
* **Track C spatial dependency:** transition to non-spatial domains (graphs, latent spaces) will require reworking spatially-dependent components: Halo (cosine feathering), R/Up (gaussian/bilinear), SeamScore (gradient energy), and parts of ρ (Laplacian-based HF energy). This is not incremental adaptation but partial redesign. Accepted as a known trade-off.
* ~~How to ensure "no broken features"?~~ → Closed: formalized via Scale-Consistency Invariant (section 8). Transitions to experimental queue as baseline experiment.

---

## 14. Enox Infrastructure: Observation Patterns (v1.8.4, 21 March 2026)

Four infrastructure patterns borrowed from the Enox framework (ideas, not code). All are pure observation/annotation: **they never modify pipeline state**. All defaults = off.

### 14.1 RegionURI

Deterministic unit address: `SHA256(parent_id | op_type | child_idx)`, truncated to 16 hex characters. Links a unit across the entire pipeline: init → dedup → gate → enforce → final. Analogous to Module A (canonicalization) for internal pipeline objects.

### 14.2 DecisionJournal

Append-only log of every decision: region_id, tick, gate_stage (stage1_healthy / stage2_utility / enforce), decision (pass / damped / rejected / d2_skip), metrics_snapshot, thresholds_used. No-op when disabled. Turns the pipeline from a black box into an auditable process.

### 14.3 MultiStageDedup

Three-level check before refinement: (a) exact hash — state unchanged, (b) metric distance — |Δρ| < ε, (c) external policy. In the current single-pass mode (ε=0.0) no level fires — this is a **scaffold** for multi-pass iterative refinement (Phase 3), where reprocessing an unchanged unit is pure budget waste.

### 14.4 PostStepSweep

After the refinement loop: tree traversal, searching for siblings with identical dirty-signatures. If > 5% of pairs are identical — signal for future merge. Works only on tree_hierarchy (grids and graphs lack parent-child structure).

### 14.5 Integration Principle

Observation infrastructure is cheapest to integrate into a simple pipeline. In Phase 3 (multi-pass, adaptive budgets) the same integration would require refactoring already complex code. Therefore the patterns are laid down now, activated later.
