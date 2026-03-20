# Curiosity Experiment Results

Series Exp0.1–Exp0.8 is complete. Below is a summary of each experiment with key numbers and conclusions.

---

## Exp0.1 — Does Adaptive Refinement Work?

**Question:** Does adaptive tile selection for refinement outperform random selection?

**Result:** Yes. Adaptive > random by MSE/PSNR at any coverage level.

**Conclusion:** Core hypothesis confirmed. Selective tile refinement works.

---

## Exp0.2 — Robustness on Real Data

**Question:** Are results robust on real images (not just synthetic)?

**Result:** Yes by MSE_rgb/PSNR (98–100% winrate). However, HF degradation (high-frequency artifacts) was discovered at low budgets.

**Conclusion:** Works, but a mechanism is needed to eliminate artifacts at tile boundaries. Motivation for halo.

---

## Exp0.3 — Is Halo Needed?

**Question:** Does overlap (halo) at tile boundaries eliminate HF artifacts?

**Result:** Yes. Halo ≥ 3 cells (pixels in image experiments) at tile_size=16 eliminates artifacts. Interior-only HF metric confirmed: the problem was specifically in seams, not inside tiles.

**Conclusion:** Boundary-aware blending (cosine feathering) with minimum overlap = 3 elements is a mandatory requirement.

---

## Exp0.4 — Is Combined ρ Needed?

**Question:** Should multiple informativeness signals be combined, or is residual sufficient?

**Result:** On clean data, residual-only = oracle. Combination provides no advantage.

**Conclusion:** Under ideal conditions, residual alone suffices. But what happens under noise?

---

## Exp0.5 — Does the Residual Oracle Break?

**Question:** Does the residual-only signal degrade under noise, blur, alias?

**Result:** Yes. Residual-to-oracle correlation drops from 0.90 to 0.54 under noise. Degradation at coarse level.

**Conclusion:** Residual-only is a fragile strategy. A fallback mechanism is needed.

---

## Exp0.6 — Binary Switch resid/combo

**Question:** Does a simple switch between residual and combined signal work?

**Result:** Partially. Clean/blur → residual, noise → combo. Alias is a borderline case; the switch doesn't always correctly identify the regime.

**Conclusion:** Binary logic is insufficient. A more flexible mechanism is needed.

---

## Exp0.7 / Exp0.7b — Soft Gating and Two-Stage Gate

**Question:** Can the binary switch be replaced with soft gating?

**Exp0.7:** Soft gating wins on noise/spatvar/jpeg (+0.08–1.10 dB), but loses on clean/blur (−1.6–2.5 dB) due to softmax "democracy".

**Exp0.7b:** Two-stage gate solves the problem. Stage 1 checks "is residual healthy?":
- If yes → residual-only (zero loss on clean/blur, Δ ≈ 0).
- If no → utility-weighted combination (+0.77–1.49 dB on noise/spatvar).
- JPEG: only downside (−0.21 dB), threshold tuning issue.

**Conclusion:** Two-stage gate is the correct architecture. Canonical solution.

---

## Exp0.8 — Schedule and Budget Governor

**Question:** Is a phase schedule (changing ρ weights by step) and budget governor needed?

**Budget governor (EMA):**
- StdCost halved (~5.15 → ~3.25).
- P95 from 11.0 to ~6.5.
- Compliance penalty from ~3.2 to ~0.5.
- PSNR slightly lower (−0.24 dB clean, −0.68 dB shift).
- **Governor is needed for budget predictability, not quality.**

**Phase schedule:** Showed no gain under current conditions. Deferred as optional extension.

**Conclusion:** EMA governor is a mandatory component. Phase schedule is not.

---

## Halo Cross-Space Validation (Phase 0)

**Question:** Does Halo (cosine feathering, overlap >= 3) work beyond 2D pixel grids?

**Results:**

| Space | Improvement | p-value | Verdict |
|---|---|---|---|
| T1 scalar grid | 2.02x | 3.81e-06 | Pass |
| T2 vector grid | 1.57x | 3.81e-06 | Pass |
| T3 irregular graph | 1.82x | 9.54e-06 | Pass |
| T4 tree hierarchy | 0.56x (WORSE) | 0.99 | Fail |

**Conclusion:** Halo works on grid/graph, fails on tree. Applicability rule derived: boundary parallelism >= 3 AND no context leakage. Grid/graph: always. Tree/forest: never.

---

## SC-baseline (Phase 0)

**Question:** Do D_parent and D_hf metrics separate positive and negative cases?

**Results (original formula):**

| Metric | AUC | Effect size (d) | Verdict |
|---|---|---|---|
| D_hf | 0.806 | 1.34 | Pass |
| D_parent (original) | 0.685 | 0.233 | Fail |

**Results (updated D_parent formula: R sigma=3.0 + lf_frac):**

| Metric | AUC | Effect size (d) | Verdict |
|---|---|---|---|
| D_parent (fixed) | 0.853 | 1.491 | Pass |

**Cross-space validation of D_parent (fixed):**

| Space | AUC |
|---|---|
| T1 scalar grid | 1.000 |
| T2 vector grid | 1.000 |
| T3 irregular graph | 1.000 |
| T4 tree hierarchy | 0.824 |

All spaces pass threshold (AUC >= 0.75).

**Fix:** coarse_shift generator corrected to spatially coherent sign fields.

**Conclusion:** D_parent with updated formula `||R(delta)|| / (||delta|| + epsilon)` (R=gauss sigma=3.0) is validated. The formula measures what fraction of delta energy is low-frequency (lf_frac).

---

## Summary Table

| Component | Status | Requirement Level |
|---|---|---|
| Adaptive refinement | Confirmed | System core |
| Halo (boundary blending) | Confirmed | Mandatory (>=3 px); grid/graph only |
| Probe (exploration) | Confirmed | Mandatory (5–10% budget) |
| Two-stage gate | Confirmed | Mandatory under noise/degradation |
| EMA budget governor | Confirmed | Mandatory |
| SeamScore metric | Validated | Stable within current validation scope |
| Halo cross-space | Validated | Grid/graph: yes. Tree: no. Rule derived |
| SC-baseline (D_hf) | Confirmed | AUC=0.806, d=1.34 |
| SC-baseline (D_parent fixed) | Confirmed | AUC=0.853, d=1.491; cross-space 0.824–1.000 |
| Phase schedule | Not confirmed | Deferred |
| Morton/block-sparse layout | Preliminarily unfavorable | Per 0.9a microbench; P0 open |

---

## Phase 1 — P0 Layout, Determinism, Sensitivity (March 2026)

---

## Exp10 (exp10_buffer_scaling) — Grid vs Compact Layout on GPU

**Question:** What is better for GPU — full-size grid or compact array with reverse index (reverse_map)?

**Kill criterion:** compact overhead >20% by time OR by VRAM → kill compact.

**Result:** KILL compact. Compute at O(k) is 18.5% faster, but reverse_map[M] on int32 yields +38.6% VRAM. 75/75 configs exceed VRAM threshold. Killed the specific implementation (compact with element-level reverse_map), not the sparse layout principle.

**Conclusion:** Grid locked in as baseline. But compute on compact is faster — meaning sparse with the right lookup is viable.

---

## Exp10d (exp10d_seed_determinism) — Bitwise Determinism (DET-1)

**Question:** Does the system produce bitwise identical results given the same seed?

**Kill criterion:** any divergence = FAIL.

**Result:** PASS 240/240. All 4 space types × 10 seeds × 3 budgets × CPU+CUDA — bitwise match.

**Components:** Canonical traversal order (Z-order tie-break), deterministic probe (SHA-256 seed), governor isolation (EMA commit after full step).

**Conclusion:** DET-1 passed. The system is deterministic.

---

## Exp10e (exp10e_tile_sparse) — Tile-Sparse Candidates

**Question:** Can a tile-sparse layout without global reverse_map beat grid?

**Candidates:**
- A (bitset): grid + bitset activation mask
- B (packed Morton): compact tiles + binary search by Morton keys
- C (paged): paged sparse scheme with macroblocks

**Result:** A alive (-20% time, +18% VRAM). B killed — binary search on GPU = +1700% time. C killed — +9000%.

**Conclusion:** Lookup is the bottleneck. Binary search and page machinery on GPU are dead. Need O(1) lookup.

---

## Exp10f (exp10f_packed_lookup) — Packed Tiles + Direct / Hash Lookup

**Question:** Does O(1) tile_map[id]→slot work instead of binary search?

**Candidates:**
- D_direct: packed tiles + tile_map (direct int32 index)
- E_hash: packed tiles + open-addressing hash table

**Result:** D_direct: 5× faster than grid, resident memory 5.5× smaller. E_hash: same speed, but build 10-30× slower. Peak VRAM inflated due to measurement artifact (conv2d temporaries).

**Conclusion:** Lookup solved. Hash not needed for bounded regular domains. E_hash archived as fallback.

---

## Exp10g (exp10g_dual_benchmark) — Dual-Contour Benchmark

**Question:** Does D_direct pass both contours — architectural (stencil) and operational (conv2d)?

**Modes:**
- Contour A: manual stencil kernel (pure layout check)
- Contour B: conv2d (real operator path)

**Result:** D_direct PASS both contours. Conv2d: -54% to -80% time, -36% to -86% peak VRAM. Stencil: +5-12% time (within threshold), resident ≤ grid.

**Conclusion:** D_direct (packed tiles + tile_map) is the winner for scalar grid. Exp10f artifact resolved.

---

## Exp10h (exp10h_cross_space) — Cross-Space Validation

**Question:** Does D_direct work on vector_grid and tree_hierarchy?

**Result:**
- Vector grid: 72/72 PASS both contours. Time -67% to -94%. Scales across channels.
- Tree hierarchy: 0/108 FAIL. Trees too small (15-585 nodes), tile_map overhead does not pay off. Resident ratio 1.16-1.33×.

**Conclusion:** Vector grid — production. Trees — need per-level analysis on larger configurations.

---

## Exp10i (exp10i_graph_blocks) — Block Addressing for Graphs

**Question:** Do fixed-size blocks work as a layout for irregular graphs?

**Graph types:** random_geometric, grid_graph, barabasi-albert.
**Partitioning strategies:** random, spatial (Morton), greedy (BFS).

**Result:**
- Compute fast everywhere (Contour B 100%). Diagnosis: compute healthy, representation sick.
- random_geometric: Contour A 58% PASS. Spatial partition best cbr=0.31.
- grid_graph: Contour A 67% PASS. Spatial partition best cbr=0.20.
- barabasi-albert: Contour A 0% PASS. Best cbr=0.66. Hub nodes break all blocks.
- Padding waste 50-97% (mean 0.77).

**Conclusion:** Graphs split into two classes. Spatial — blocks conditionally viable. Scale-free — blocks structurally incompatible. Fixed-size blocks are not a universal abstraction for graphs.

---

## Exp10j (exp10j_tree_perlevel) — Per-Level Break-Even for Trees

**Question:** At which tree levels does D_direct beat A_bitset?

**Sweep:** 158,000+ trials. Branching 2-32, depth up to 10, occupancy 0.01-0.70, payload 4-256 bytes, 3 activation patterns, 2 operators.

**Result:**
- matmul operator: D wins at occupancy < 37.5-40% at ANY level size (from 2 to 4096 nodes). Win rate 59%.
- stencil operator: D saves memory below the same threshold, but NEVER wins on time (3.3× slower).
- Activation pattern (random/clustered/frontier) — does not affect the threshold.

**Conclusion:** Trees require hybrid mode. Heavy compute + low occupancy → D_direct per-level. Light compute → A_bitset. Threshold p*≈0.40 is stable.

---

## Exp11 (exp11_dirty_signatures) — Dirty Signatures

**Question:** Can a 12-bit dirty signature + debounce replace full recompute for change detection?

**Result:** PASS. AUC: scalar_grid=0.925, vector_grid=1.000, irregular_graph=1.000, tree=0.910. All >0.8 kill criterion, all p_adj < 0.001 (Holm-Bonferroni).

**Bug history:** First run: FAIL (AUC 0.0 on scalar_grid). Root cause: debounce tracker compared step-to-step (caught derivative, not level shift). Noise produced constant jumps → trigger. Structural changes produced single impulse → debounce killed it. Second run: incorrectly replaced with oracle scoring (MSE vs GT) — AUC 1.000 but cheating. Third run: proper fix — baseline signature comparison + temporal ramp scoring. No ground truth.

**Conclusion:** 12-bit dirty signatures work. Key: compare against baseline, not step-to-step. Temporal ramp catches sustained level shift.

---

## DET-2 (exp11a_det2_stability) — Cross-Seed Stability

**Question:** Are pipeline metrics stable across different seeds?

**Sweep:** 20 seeds × 4 spaces × 2 budgets = 160 runs.

**Kill criterion (per-regime):**
- Regular spaces + low budget: CV < 0.10
- Irregular spaces + low budget: CV < 0.10
- Irregular spaces + high budget: CV < 0.25 (legitimate topological fluctuation)

**Result:** PASS 8/8. Metric mean_leaf_value removed (test harness artifact). 6 structural metrics stable.

**Conclusion:** Pipeline is reproducible. CV up to 0.25 at high budget on irregular topologies is governor behavior on chaotic rho landscapes, not a bug.

---

## Exp12a (exp12a_tau_parent) — Data-Driven τ_parent by Depth

**Question:** Can τ_parent[L] thresholds be found from data instead of manual tuning?

**Result:** PASS. Per-space thresholds τ[L, space_type] instead of global τ[L]. Best method: youden_j. Max accuracy drop 5.6pp (< 15pp kill criterion).

**Thresholds:** T1_scalar L1: τ=0.46, T3_graph L1: τ=0.08, T4_tree L1: τ=0.19. Specificity L1 = 1.000 (was 0.25 with global threshold).

**Conclusion:** Per-space thresholds solve the problem. R/Up operators produce different D_parent dynamic ranges — single threshold impossible, per-space mandatory.

---

## P2a (p2a_sensitivity) — Gate Threshold Sensitivity Sweep

**Question:** How sensitive is the system to manual thresholds of the two-stage gate?

**Result:** PASS. Ridge width 100% — thresholds are robust. P2b (adaptive tuning) not required.

**Conclusion:** Manual thresholds are ok. Gate is insensitive across a wide range.

---

## Exp10k (exp10k_cost_surface) — Cost Surface C(I, M, p)

**Question:** Is layout selection a smooth function of three space properties (isotropy I, metric gap M, density p), or a discrete classification?

**Result:** Boundary smoothness 0.496 — JAGGED. Surface is not smooth.

- Sparse vs dense: **confirmed** — A_bitset = 0 wins out of 810 trials. Sparse always wins.
- D_direct vs D_blocked: **unresolvable** — ~50% of adjacent grid points switch winner.

**Root cause:** three measurement flaws. (1) H(D) is blind to geometry — cannot distinguish "uniform order" from "uniform chaos". (2) lambda_2 is a global metric, cannot see local cache-line health. (3) Metrics computed on full X, not on active subgraph X_active.

**Conclusion:** Hypothesis v1.8.3 (layout = argmin C(I,M,p)) neither confirmed nor refuted — metrics insufficient. Need: I_active (entropy on X_active), M_local (mean cache miss on X_active edges). Deferred to Track C. Policy table works as empirical classification.

---

## Final Layout Policy (result of exp10 series)

Full methodology: `docs/layout_selection_policy.md`

| Space Type | Layout | Status |
|-----------------|--------|--------|
| scalar_grid | D_direct (packed tiles + tile_map) | Production |
| vector_grid | D_direct (packed tiles + tile_map) | Production |
| tree_hierarchy | Hybrid: D_direct per-level (p<0.40 + heavy compute), A_bitset otherwise | Validated |
| irregular_graph / spatial | D_blocked (block addressing) conditional | Conditional |
| irregular_graph / scale-free | A_bitset (dense + bitset) fallback | Fallback |
