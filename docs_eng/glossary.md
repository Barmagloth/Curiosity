# Curiosity Glossary

Project-specific terms without which the documentation reads like gibberish. Organized by conceptual groups.

---

## Core Concepts

**Adaptive refinement** — the core of the system. The idea is that the space is not refined uniformly, but only where the informativeness function ρ indicates it is justified. Opposite approaches: uniform refinement (refine everything) or random selection (choose randomly).

**Root coarse** — the very first coarse representation from which the entire refinement tree grows. The global anchor of the system. At L=0, this is the only coarse representation that exists.

**Parent coarse** — the coarse representation of the parent scale for a given refinement step. At level L, parent coarse is the output of level L−1. At L=1, parent coarse coincides with root coarse. This is the anchor in the scale-consistency invariant and in all refinement formulas: `refined = parent_coarse + step_delta`.

**Step delta** — the additive correction of a single refinement step. Initialized to zero, bounded in energy. Formula: `refined_L = parent_coarse_L + step_delta_L`. When the current refinement level is disabled, the system falls back to parent_coarse.

**Cumulative delta** — the accumulated correction relative to root coarse: `cumulative_delta_L = refined_L − root_coarse↓L`. Not used in local refinement mechanics; needed for drift diagnostics and analysis of accumulated effects across the tree.

**Coarse (deprecated shorthand)** — in earlier documentation, "coarse" was used without qualification to mean either root coarse or parent coarse depending on context. As of v1.6 terminology update, always use the explicit form. When "coarse" appears as an adjective (e.g., "coarse levels", "coarse-graining"), it retains its general meaning.

**Refinement** — the process: compute ρ → decide whether to split → refine selected regions → repeat. Each iteration increases resolution only in "interesting" areas.

**Split / Merge** — split divides a region into subregions (typically 4, as in a quadtree); merge is the reverse operation, collapsing children back into their parent. Hysteresis (different thresholds for split and merge) prevents oscillation.

---

## Space and Its Elements

**Cell** — an atomic region of state space X, represented by a tree node at some depth. The cell's scale is determined by the path depth from root, not by a fixed grid size. If a node is not refined further, it is a leaf of the current tree. In Curiosity's general model, dimensionality is not fixed; the concrete embodiment of a cell depends on the task (in image experiments it's a pixel block, but that's a special case, not the definition).

**Leaf** — a tree node that is not split further at the current step and is processed as a terminal region. Leaf status is a property of the current tree state, not of the level itself: at the same level L, some nodes may be leaves while others are internal (already split).

**Tile** — an operational unit of the pipeline: refine selection, caching, hashing, halo-blending. In a specific implementation, a tile may correspond to a node, a leaf, a batch of sibling nodes, or a spatial patch. A tile is **not** required to be rectangular or have a fixed size — this depends on the task. In image experiments, tile = square spatial block (e.g. 16×16 pixels). In the general case — a processing unit defined by the implementation.

**Level L** — a node's depth in the refinement tree. Increasing level = finer partitioning of space. Dimensionality in Curiosity is not a fixed number of axes, but the depth of refinement.

**Branching factor** — the number of child nodes produced by a split operation. A property of the refinement rule, not of the tile's geometric size. Not to be confused with `tile_size` (see below).

**tile_size** — a task-specific parameter whose meaning depends on the domain: in image experiments = spatial extent of a block (tile_size=16 → 16×16 pixels); in abstract space it may refer to the processing batch size. **Not a synonym of branching factor** — these are different entities that were historically labeled similarly.

**State space X** — the abstract state space that Curiosity operates on. Can be anything: image pixels, latent representations, neural network activations, any data. The system is not tied to any specific domain.

---

## Informativeness Function and Signals

**ρ (rho)** — the informativeness function. Determines where refinement makes sense. The semantics of the entire tree are determined by ρ. May include multiple components (residual, HF, variance, etc.).

**Interestingness** — informal name for ρ. An "interesting" region is one where ρ is high and refinement is justified. The interestingness policy determines how to balance different signals at different depth levels.

**Residual** — the error of the current approximation. The primary informativeness signal. On clean data, residual-only = ideal (oracle), but under noise/blur/alias it degrades (oracle correlation drops from 0.90 to 0.54).

**HF energy (high-frequency energy)** — structural energy computed via Laplacian or gradient. Catches sharp transitions but produces false positives on seams between tiles.

**Variance** — local variance or disagreement between sources. Sensitive to noise (bad) and to genuine uncertainty (good).

**Payoff** — expected gain from refinement minus cost. A split is allowed only if gain > cost.

**Oracle** — the ideal, practically unattainable signal (ground truth). Used for evaluation: how well our real signal correlates with the oracle. Oracle information is prohibited in production metrics — it is for validation only.

---

## Two-Stage Gate

**Two-stage gate** — the canonical architecture for combining ρ signals. Solves the problem: residual is good on clean data but breaks under noise; combination helps with noise but hurts on clean data.

**Stage 1** — binary check: "Is residual healthy?" using instability and FSR metrics. If healthy — residual-only is used (zero loss on clean data). If not — transition to Stage 2.

**Stage 2** — utility-weighted combination. Each expert (resid, var, hf) receives a weight: `U_i = median(gain) − λ·FSR − μ·instability`. Residual has a guaranteed minimum weight. Weights are smoothed by EMA with hysteresis.

**Instability** — a signal reliability metric. High instability means the residual "jumps around" and cannot be trusted. Threshold for switching Stage 1 → Stage 2.

**FSR (False Signal Rate)** — the fraction of false positives. High FSR means the signal frequently indicates "interesting here" where there is actually nothing.

---

## Boundaries and Seams

**Halo** — the overlap zone between tiles. A mandatory component: without halo, hard insertion of a refined tile creates a step at the boundary, which the Laplacian catches as a false HF signal. Minimum size: 3 cells. Implemented as cosine feathering relative to parent coarse. Applicability rule: boundary parallelism >= 3 AND no context leakage. Grid/graph: always applied. Tree/forest: never applied (context leakage makes halo harmful).

**Cosine feathering** — a method for smooth transition between refined and coarse tiles. Weight changes smoothly via cosine function from 1 (tile center) to 0 (boundary), instead of a sharp cutoff.

**Seam** — a boundary between tiles with different refinement levels. Artificial seams are an artifact that the system must not create. SeamScore measures their severity.

**SeamScore** — seam quality metric. Formula: `SeamScore = Jumpout / (Jumpin + eps)`. Jumpout — outward gradient from the boundary, Jumpin — inward gradient. Closer to 1 is better (seam is invisible). Validated on 4 space types: 2D scalar grids, vector grids, irregular graphs, tree hierarchies. Production-ready.

**Edge strips** — narrow strips of cells along the tile boundary used to compute SeamScore.

---

## Probe and Exploration

**Probe** — a dedicated portion of the budget (5–10%) for exploring unknown regions. Without probe, the system only sees what it has already found (exploitation-only = structural blindness). As of v1.6, probe is clarified as a mechanism for protection against false fixed points. Probe remains mandatory even when a scale-stable fixed point is reached.

**Exploration / Exploitation** — exploitation is refining already-found "interesting" areas; exploration is searching for new ones. 90–95% of budget goes to exploitation, 5–10% to exploration. Both are mandatory: exploration does not fix seams, halo does not replace exploration.

**Structural blindness** — a state where the system cannot see the internal structure of regions because it never examines them. Probe is the insurance against this.

**False fixed point** — a state of apparent stability that is not actually a true stopping point. Probe is the only safeguard, because a false fixed point is indistinguishable from a true one without external verification.

---

## Budget Governor

**Governor** — an EMA controller that manages strictness to keep spending within budget. Without the governor, the budget is a declaration, not a constraint.

**Strictness** — the quantile threshold for selecting candidate tiles for refinement. High strictness = only the most "interesting" tiles pass, spending is lower. Low = more tiles, spending is higher. The governor adjusts strictness so that cost/step stays within the corridor.

**EMA (Exponential Moving Average)** — exponentially weighted moving average. Used for smoothing: gate weights, feedback signal in the governor, etc. Reacts to recent data more strongly than to old data.

**Compliance (budget compliance)** — a metric of how well actual spending matches the target. Asymmetric: overbudget is penalized more heavily than underbudget.

**Warmup** — the initial N steps during which the governor does not adjust strictness but accumulates spending statistics. Needed for correct calibration.

**Hard cap** — a hard ceiling on spending per step (= 3× target). Safety fuse against spending explosion.

---

## Tree and Storage

**Refinement tree** — a log of split decision routes. Each root→leaf path = a sequence of decisions. Does not necessarily yield a "semantic" metric (this is tested in P3).

**Bush** — a set of paths in the tree leading to the same meaning. Formally: a cluster of leaf paths with small LCA distance.

**LCA (Lowest Common Ancestor)** — the nearest common ancestor of two nodes in the tree. LCA distance = a measure of similarity between two paths. The closer the LCA is to the root, the farther apart the paths are.

**Morton layout** — a data packing method using Z-curve (space-filling curve). Deferred: sort overhead consumes the benefit in current implementation.

**Block-sparse layout** — sparse block-based storage. Deferred: expansion ratio makes it inefficient in current configuration.

**Compact layout** — compact storage of only active cells. The sole surviving candidate (vs. grid). Exp0.9b0 — kill/go test.

**Dirty signature** — a 12-bit signature (seam_risk + uncert + mass) for quickly determining whether a region has changed. Debounce (2 consecutive hits) protects against false positives from noise.

**Temporal ramp** — scoring method for dirty signatures (exp11). For each step, compute the mean signature distance from baseline (hamming + component_diff) across all units. Score = mean(second_half) − mean(first_half) of the trajectory. Structural changes create a step function (low first half, high second half → positive score); noise creates a flat trajectory (score ≈ 0). Key distinction from step-to-step comparison: catches sustained level shift without reacting to random jumps.

---

## Scale-Consistency (v1.7)

**Scale-Consistency Invariant** — the requirement that step_delta does not redefine the semantics of the parent scale. Formally: `||R(step_delta)|| / (||step_delta|| + epsilon) < tau_rel`. Measures what fraction of step_delta energy is low-frequency (lf_frac). Closes the open question from v1.5 "how not to break features." See concept_v2.0.md, section 8, for details.

**R (coarse-graining operator)** — `gaussian blur (sigma=3.0) + decimation`. Projects the signal from fine to coarse scale. Fixed before experiments — an architectural choice.

**Up (restoration operator)** — `bilinear upsampling`. Projects the coarse component back to the original scale. This is **not** the inverse of R.

**Pair (R, Up)** — a fixed pair of operators for measuring scale-consistency. Different pairs yield different tree physics. Does not change during a single verification cycle.

**D_parent** — a metric for step_delta leakage into the parent scale: `D_parent = ||R(step_delta)|| / (||step_delta|| + epsilon)`. "lf_frac" normalization: measures what fraction of step_delta energy is low-frequency. R=gauss sigma=3.0. Higher = worse. The primary enforcement signal. Validated on 4 space types (AUC 0.824–1.000).

**D_hf** — a metric for high-frequency purity of step_delta: `D_hf = ‖step_delta - Up(R(step_delta))‖ / (‖step_delta‖ + ε)`. Higher = better (step_delta lives in the HF subspace). A diagnostic signal, not a hard constraint.

**τ_parent** — a data-driven threshold for D_parent, set by the baseline experiment. May depend on level L.

**Per-space thresholds** — approach from exp12a: thresholds τ_parent[L, space_type] are set independently per space type instead of a single global τ[L]. Reason: R/Up operators produce different D_parent dynamic ranges per space type (grids ~0.42–0.50, graph ~0.08, tree ~0.19), making a single threshold impossible. Selection method: youden_j. Aligns with layout selection policy — space_type is known statically.

**Step_delta tolerance** — the permissible fraction of parent_coarse that step_delta may alter when projected to the parent scale. Operationally determined by τ_parent. Two-sided risk: too tight → loss of legitimate features; too loose → hierarchy drift. Automatic choice mechanism is an open question. See concept_v2.0.md, section 8.9.

**Scale-stable fixed point** — a node where simultaneously: (1) gain < τ_gain, (2) D_parent < τ_parent, (3) stable for K steps. A local refinement stopping criterion. Probe remains mandatory as a safeguard.

---

## Determinism and Reproducibility (v1.8)

**DET-1 (Seed determinism)** — Hard Constraint: identical data + ρ + seed + budget = identical tree (bitwise match). Precondition for testability (Track A). Three components: canonical traversal order, deterministic probe, governor isolation.

**DET-2 (Cross-seed stability)** — Soft Constraint: across different seeds, metrics (PSNR, cost, compliance, SeamScore) are statistically stable. CV < τ_cv for each metric. Precondition for reproducibility (Track B).

**Canonical traversal order** — fixed tile traversal order for resolving tie-breaks when ρ values are equal. Implemented via Z-order/Morton index. Eliminates dependence on hash table iteration order and thread races.

**Deterministic probe** — probe with seed = f(tile_coordinates, depth_level, global_seed). Pseudo-random but reproducible given a fixed seed.

**Governor isolation** — requirement that EMA accumulation and governor decisions do not depend on the processing order of sibling branches. Processing in canonical order, EMA update strictly after the full step.

**CV (Coefficient of Variation)** — standard deviation / mean. A measure of relative metric variability across seeds.

---

## Metrics

**MSE (Mean Squared Error)** — mean squared error. The basic reconstruction quality metric.

**PSNR (Peak Signal-to-Noise Ratio)** — peak signal-to-noise ratio, in decibels. Derived from MSE. Higher = better. A 0.5 dB difference is noticeable, 1+ dB is significant.

**Winrate** — the fraction of scenarios in which method A beats method B. 98–100% = definitive superiority.

**StdCost** — standard deviation of spending. A measure of budget predictability. The governor reduces StdCost by ~50%.

**P95** — the 95th percentile of spending. A worst-case measure. The governor reduces P95 from 11.0 to ~6.5.

---

## Methodology

**Kill criteria** — conditions under which an experiment is closed as unsuccessful. Defined before launch. Two-sided: speed + memory. If it doesn't pass — the experiment is dead, not "improved upon."

**Cost-fair comparison** — a comparison in which computational costs are accounted for. You cannot compare a method with 10× overhead against a baseline and claim "we're better on PSNR" — cost must be included.

**Observable-only** — metrics use only what the system can observe on its own, without oracle information. No peeking at ground truth during evaluation.

**Holm-Bonferroni** — a statistical correction for multiple comparisons. When running N tests, the probability of a false positive increases. Holm-Bonferroni adjusts significance thresholds.

**Ablation** — removing one component of the system and measuring how much the result degrades. Shows the component's contribution.

---

## Priority Hierarchy

**P0–P4 and SC** — experiment priority levels:

- **P0**: GPU layout (foundation; without it, everything else is decorative)
- **P1**: Tree compression (dirty signatures, segment compression, anchors)
- **P2**: Auto-tuning of gate thresholds (instability/FSR)
- **P3**: Tree semantics (LCA distance, bushes, clustering)
- **SC**: Scale-consistency baseline + enforcement (parallel with P1)
- **P4**: "Don't break features" (downstream compatibility, partially formalized through Scale-Consistency Invariant in v1.7)

The order is strict: no jumping ahead without closing dependencies.

**Critical path**: P0 → P1 → P3 → P4. P2 and SC run in parallel with P1.

---

## Scenario Types (used in experiments)

**Clean** — clean data without distortions. Baseline scenario.

**Noise** — random noise added to data. Residual degrades.

**Blur** — blurring / defocus. Residual handles it (Stage 1 remains).

**Alias** — frequency aliasing artifacts (Nyquist). Edge case for the gate.

**Spatvar** — spatial variance. Non-uniform scene complexity.

**JPEG** — compression artifacts. The only scenario with a negative for the two-stage gate (−0.21 dB).

**Shift** — shift / scene change over time. Adaptability test.

---

## Goal Structure (v1.1)

**Track A — instrument** — Build the adaptive refinement pipeline that works correctly under budget. Active goal. All experiments P0–P4 and SC belong to Track A.

**Track B — research** — Use the built instrument to study the structure of refinement trees. Begins only after passing the Instrument Readiness Gate. Includes P3 (tree semantics) and open research questions.

**Track C — generalization** — Verify Curiosity's applicability to non-spatial domains: graphs, latent spaces, activations, feature hierarchies. Entry condition — successful Track B (semantic geometry confirmed). Long-term ambition, not a current goal.

**Instrument Readiness Gate** — Five criteria, all mandatory for transition to Track B: (1) invariant pass, (2) overhead profile, (3) stability pass, (4) one validated benchmark, (5) attribution diagnostics. Instrument readiness ≠ perfection; it is the minimum at which the instrument does not lie. Details: `target_problem_definition_v1.1.md`.

**Baselines** — Three baseline comparisons for Track A: dense baseline (full recomputation), same-budget random (random tile selection), same-budget uniform coarse (uniform coarse approximation).

---

## Frozen Concepts

**C (DAG + profiles)** — routing refinement via a directed acyclic graph instead of a tree. Frozen indefinitely. Entry contract for unfreezing: (1) at least 2 irreducible objectives, (2) a concrete downstream consumer, (3) an observable conflict between objectives.

**Matryoshka invariant** — the requirement that the representation at any "matryoshka" (nested refinement) level is a valid input for the downstream consumer. Not just visually smooth, but functionally correct. The Scale-Consistency Invariant (v1.7) is the formalization of this requirement at the level of individual nodes. Tested in P4.

**Depth-dependent representation (deferred)** — the hypothesis that different tree levels may require qualitatively different features (step_delta of different types at different scales). Current architecture uses a uniform representation at all levels. Deferred consciously: (1) requires a working instrument first (Track A); (2) target requirements for each level cannot be specified in advance — the approach is presumed iterative. Investigate after Track A.

---

## Topological Profiling (v2.0)

**Forman-Ricci curvature** — combinatorial curvature of a graph edge: F(e) = 4 - d_u - d_v + 3·|△(e)|, where d_u, d_v are endpoint degrees, |△(e)| is the number of triangles containing the edge. O(1) per edge with precomputed triangles. Positive = dense region (cliques), negative = bridge/bottleneck.

**Ollivier-Ricci curvature** — transport curvature of an edge: κ(u,v) = 1 - W₁(μ_u, μ_v)/d(u,v), where W₁ is the Wasserstein-1 distance between lazy random walk distributions. Computed exactly via EMD (transportation LP, scipy linprog HiGHS). Expensive: O(W³), W = d_u × d_v. Yields [-1, +1].

**Hybrid curvature engine** — three-phase budget-constrained approach: (1) Forman for ALL edges (cheap), (2) sort by |Forman| anomaly, (3) upgrade top-N anomalous to Ollivier, where N = floor(topo_budget / t_ollivier). Solves the problem: Swiss Roll 500 nodes — all 1800 edges eligible for Ollivier by κ_max, but budget allows only 19 exact calls.

**Synthetic Transport Probe** — hardware calibration at session start. Generates a synthetic EMD problem of size d_test=10, solves it n_trials times, measures median t_test. Extrapolates κ_max = W_test · ∛(τ_edge / t_test). One-time measurement (~52ms).

**κ_max** — maximum problem size d_u × d_v at which Ollivier-Ricci fits within the per-edge budget τ_edge. Determined by hardware calibration.

**σ_F (sigma Forman)** — standard deviation of Forman curvature across all graph edges. A measure of topological heterogeneity: high σ_F = mixture of bridges and cliques.

**η_F (eta Forman, topological entropy index)** — dimensionless index: η_F = σ_F / √(2⟨k⟩), where ⟨k⟩ = 2|E|/|N| is the mean degree. Normalization by √(2⟨k⟩) is the noise limit of Forman curvature variance for an Erdos-Renyi random graph with the same mean degree. η_F < 0.70 = structurally regular graph. η_F > 0.70 = structural chaos above random background. Threshold 0.70 chosen from the dead zone [0.60, 0.76] in the 35-graph corpus.

**Topo zone** — three-level stamp assigned to a graph at initialization: GREEN (κ_mean > 0, dense cliques, ECR < 5%), YELLOW (κ < 0, Gini < 0.12, η_F ≤ 0.70, regular lattices, ECR 10-25%), RED (structural chaos, ECR > 30%). Determines runtime policy: τ_eff, split budget, SC-enforce strictness.

**ECR (Edge Cut Ratio)** — fraction of graph edges crossing cluster boundaries after Leiden clustering: cut_edges / total_edges. A measure of clustering quality: low ECR = clean separation.

**Gini coefficient** — a measure of distribution inequality. In the profiling context: Gini(PageRank) = inequality of node influence. 0 = all nodes equal (lattice), >0.8 = one node dominates (hub-spoke). Threshold 0.12: below = homogeneous space, above = pronounced hubs.

---

## Cross-Space Validation and Topological Concepts

**Cross-space validation** — the principle that any claim about working in "arbitrary spaces" must be tested on >= 4 space types (scalar grid, vector grid, irregular graph, tree hierarchy). Without this, the claim is a declaration, not a fact.

**Context leakage** — when halo expansion bleeds into unrelated tiles. A critical problem for tree topology: in a tree, structural "neighbors" can be semantically distant, and extending halo to them is harmful rather than helpful.

**Boundary parallelism** — the number of independent cross-edges at a tile boundary. For halo to work correctly, boundary parallelism >= 3 is required. In grid/graph this is naturally satisfied; in tree/forest it is not.

---

## 16. Layout (GPU)

Terminology from the exp10 experiment series (P0 layout). Full methodology: `docs/layout_selection_policy.md`.

| Term | Definition |
|------|-----------|
| **D_direct** (packed tiles + direct tile_map) | Active tiles stored in compact array `tiles[k, ...]`. Lookup via `tile_map[tile_id] → slot` (int32, -1 = inactive). O(1) addressing without element-level reverse index. Production layout for scalar_grid and vector_grid. |
| **D_direct_per_level** | Same as D_direct but tile_map built independently per tree level. Used for tree_hierarchy with heavy compute and occupancy < 40%. |
| **A_bitset** (dense grid + bitset mask) | Full-size data tensor over entire universe. Activation tracked by bitset (1 bit per element). No indirection. Simple fallback for all space types. |
| **D_blocked** (graph block addressing) | Graph nodes partitioned into fixed-size blocks. `block_map[block_id] → slot`. Dense storage within blocks; cross-block edges handled separately. Only viable for spatial graphs with cbr ≤ 0.35. |
| **E_hash** (hash table lookup) | Archived fallback. Hash table for tile_id → slot. Dominated by D_direct at current scale. Resurrect only for very large or irregular tile universes. |
| **Contour A** (architectural viability) | Layout validation mode: manual stencil kernel, no framework temporaries. Tests pure layout economics: build + gather/scatter + addressing + resident memory. |
| **Contour B** (operational viability) | Layout validation mode: real operator (conv2d / matmul). Tests peak-step memory and GPU budget compatibility. |
| **cbr** (cross-block ratio) | Fraction of graph edges crossing block boundaries. Range [0, 1]. Lower = better partitioning. Threshold: cbr ≤ 0.35 for D_blocked. |
| **occupancy** (p) | Fraction of active tiles/nodes: k / N. For trees, per-level: p_l = k_l / N_l. |
| **resident memory** | Layout storage after build, before compute. Excludes operator temporary allocations. |
| **peak step memory** | Maximum GPU allocation during compute step. Includes operator workspace. |
| **workspace overhead** | peak_step − resident. Operator temporary allocations not attributable to layout. |
| **tile_map** | Array `int32[N_tiles]` where `tile_map[tile_id] = slot` (index into packed array) or -1 (inactive). Core of D_direct. |
| **padding waste** (pw) | Fraction of packed block storage occupied by padding (inactive slots within active blocks). Problem for graphs with fixed block_size. |

---

## Three-Layer Rho and Pipeline (v2.1, exp17)

**Three-Layer rho** — architectural decomposition of the monolithic informativeness function rho into three cascaded layers. Layer 0 (Topology) handles space structure (data-independent): clusters, bridges, hubs, density. Layer 1 (Presence) handles data existence (data-dependent, query-independent): binary map of "is there a non-trivial signal here". Layer 2 (Query) handles task-specific refinement: residual, HF energy, max absolute error. Each layer narrows the working set for the next. Reusability increases bottom-up: topology is immutable, the data map updates rarely, the query changes constantly. Introduced in exp17 (March 23, 2026).

**Cascade Quotas (Variant C)** — adaptive L1 pruning threshold mechanism tied to the L0 cluster structure. Instead of a fixed threshold (l1_threshold=0.01, which killed 97% of units on scalar_grid at scale 1000), each L0 cluster guarantees a minimum number of surviving units proportional to cluster size: quota = max(1, ceil(cluster_size * min_survival_ratio)). min_survival_ratio is tied to budget_fraction (typically 0.30). Information flows strictly top-down (L0->L1->L2): topology dictates survival quotas, data is filtered not by an abstract threshold but by the local context of its own geometry. Solves the "region extinction" problem — no L0 cluster can lose all its units. Implemented in layers.py, validated in exp17v2 (1080 configs, 12/12 PASS on reusability).

**Frozen Tree** — a serializable snapshot of Layers 0+1 (topological scores + presence scores + list of active units). A reusable spatial index: built once (expensive), then different Layer 2 queries run on top of it (cheap). Analogous to R-tree/quadtree in spatial databases. Contains: l0_scores, l1_scores, active_units, zone, memory footprint.

**Streaming Pipeline** — execution mode for the three-layer rho where L0 clusters are processed sequentially, each going through the full L0->L1->L2 cycle before moving to the next. Clusters are sorted by L0 priority score (most important first). A global budget cap ensures total refinements do not exceed budget_fraction * n_total. Two advantages: (1) first results appear after processing the first cluster, not after the full map; (2) L1 pruning genuinely reduces the number of refinements (budget per-cluster), not just scoring. Implemented in ThreeLayerPipeline.run_streaming().

**L0 Spatial Clusters** — cluster structure produced by Layer 0. For irregular_graph: Leiden communities (each unit = one cluster). For tree_hierarchy: depth-band grouping. For grids: spatial quadrant blocks (NQ*NQ). Used by Layer 1 for cascade quotas and by the streaming pipeline for traversal order.

**Industry Baselines** — standard industry approaches for spatial search and refinement, implemented in exp17 for comparison with the Curiosity pipeline: (1) cKDTree (scipy) — k-d tree + NN query; (2) Quadtree — quadrant splitting by rho (grid only); (3) Leiden + brute force — community detection + sort by rho within (graph only); (4) Wavelets — Haar DWT detail coefficients as a saliency map (scalar grid only).

**unit_rho** — the per-unit informativeness score computed by the rho function. In the three-layer architecture, unit_rho at Layer 2 is the task-specific signal (e.g., MSE residual) evaluated only on units that survived L0 and L1 filtering.

**anchor** — a node in the refinement tree designated as a fixed reference point during periodic rebuild. Anchors are retained across rebuild cycles to preserve structural continuity. Exp14 tested anchor + periodic rebuild strategies: grid divergence = 0 (PASS), graph/tree divergence > 0.20 (FAIL due to structural drift).

**RegionURI** — a deterministic SHA256 address for each unit in the tree: SHA256(parent_id|op_type|child_idx) truncated to 16 hex characters. Provides stable provenance tracking. Observation-only — never modifies pipeline state. Part of Enox infrastructure.

**Decision Journal** — an append-only log of gate and enforcement decisions with full metrics (region_id, tick, gate_stage, decision, thresholds). Used for debugging and post-hoc audit of pipeline behavior. Observation-only. Part of Enox infrastructure.

**topo profiling** — see Topological Profiling (v2.0) section. Multi-stage pre-runtime analysis of graph structure performed during IrregularGraphSpace initialization, before the first pipeline tick. Outputs per-node curvature, pagerank, clustering coefficients, and a topo_zone stamp (GREEN/YELLOW/RED).

**zone classification** — the process of assigning a topological zone (GREEN/YELLOW/RED) to a graph based on topo profiling results. Determines runtime policy: tau_eff, split budget, SC-enforce strictness. See Topo zone entry.

**MultiStageDedup** — three-level deduplication mechanism: (1) exact hash match, (2) metric-distance match within epsilon, (3) policy-level dedup. With epsilon=0.0 (default), never fires in single-pass mode. Scaffolding for future multi-pass scenarios. Observation-only. Part of Enox infrastructure.

**PostStepSweep** — post-step scan for identical siblings in tree_hierarchy based on dirty signature comparison. Identifies merge candidates (siblings whose signatures differ by less than the sweep threshold, default 5%). Observation-only. Part of Enox infrastructure.

---

## Trajectories and Bushes (exp15-exp16)

**Trajectory Profiles (C-pre)** — discrete clusters found in the feature space of refinement trajectories. Exp16 showed: trajectory features (how the pipeline refined each unit) form 2-7 natural clusters with Gap > 1.0 and Silhouette > 0.3 across all 4 space types. This means the pipeline does not behave uniformly everywhere — there are discrete "behavior types" during refinement. Served as the UNFREEZE signal for Track C. Potential applications: multi-objective optimization (different strategies for different profiles), anomaly detection in pipeline behavior.

**Bushes (exp15b)** — clusters of leaf-paths in the refinement tree. Hypothesis: if tree leaves form stable clusters (bushes), this indicates semantic structure. Result: Silhouette > 0.4 (clusters are visually real), but ARI < 0.6 (clusters are not stable across seeds). Kill criteria FAIL. However: (1) clusters exist within each seed; (2) their instability may reflect GT dependence rather than uselessness; (3) leaf-path similarity could potentially be used for cluster compaction (merge candidates), duplicate region detection, or as features for a downstream classifier. Revisit planned after Track C.
