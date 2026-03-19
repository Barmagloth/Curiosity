# Curiosity — Phase 1 Implementation Plan

## Context

Phase 0 COMPLETE (18 March 2026). Phase 1 starts now. Working directory: `R:\Projects\Curiosity`. GPU: RTX 2070, CUDA 12.8, venv at `.venv-gpu` (see `docs/environment_2.md`).

Critical path: **P0 (layout) -> DET-1 (determinism) -> Phase 2**. P0 layout CLOSED (19 March 2026).

---

## Phase 1 Results (19 March 2026)

| Stream | Result | Status |
|--------|--------|--------|
| S1 exp10 | KILL compact-with-reverse-map (VRAM +38.6%). Grid baseline. | ✅ Closed |
| S1b exp10d | DET-1 PASS (240/240 bitwise match CPU+CUDA) | ✅ Closed |
| S2 exp11 | FAIL 3/4 spaces (AUC 0.0-0.37). Architecture issue. | ❌ Needs redesign |
| S3 P2a | PASS — ridge 100%, MANUAL_OK. P2b NOT needed. | ✅ Closed |
| S4 exp12a | Thresholds found. L1 specificity low. | ⚠️ Partial |
| S5 deferred | Research note done. | ✅ Closed |

**Gate Phase 1 → Phase 2:** PASSED (grid fixed + DET-1 passed).
**But:** P0 reopened — exp10 killed a specific implementation, not the principle.

---

## Phase 1b: Layout Investigation — exp10e → exp10f → exp10g

**Motivation:** exp10 showed compute on O(k) is 18.5% faster than O(M). The failure was
in dense bookkeeping (element-level reverse_map[M] at int32). Tile-sparse without global
reverse_map may win both on time AND VRAM.

### exp10e — Tile-Sparse Layout Candidates (CLOSED)

**Folder:** `experiments/exp10e_tile_sparse/`

**Results:**
- A (bitset): **ALIVE** — time -27..31%, VRAM +18%. Execution layout, not memory layout.
- B (packed Morton + binary search): **KILLED** by time (+1700%). Storage idea lives on.
- C (paged): **KILLED** (+5000-30000%). Dead permanently.

### exp10f — Packed Tiles + Alternative Lookup (CLOSED)

**Folder:** `experiments/exp10f_packed_lookup/`

**Results:**
- D: passes Contour A, fails Contour B (peak VRAM from conv2d workspace).
- E: archived as contingency with resurrection triggers.

### exp10g — Dual-Mode Benchmark (CLOSED)

**Folder:** `experiments/exp10g_dual_benchmark/`

**Goal:** Manual stencil (Contour A) vs conv2d (Contour B). Separates layout overhead
from operator overhead. Resolves D's Contour B failure.

**Results:** D_direct PASS both contours. -54% to -80% time, -36% to -86% peak VRAM vs grid baseline.

### exp10h — Cross-Space D_direct (CLOSED)

**Folder:** `experiments/exp10h_cross_space/`

**Results:**
- vector_grid: 72/72 PASS both contours. D_direct confirmed for vector grids.
- tree_hierarchy: 0/108 FAIL. Trees too small to amortize per-level tile_map overhead.
- Diagnosis: NOT an architecture rejection. Break-even exists at large N_l + low occupancy.

### exp10i — Block Addressing for Graphs (CLOSED)

**Folder:** `experiments/exp10i_graph_blocks/`

**Results:**
- Spatial graphs (random_geometric, grid_graph): conditionally viable with spatial partition, cbr<0.30.
- Scale-free graphs (barabasi-albert): REJECTED, cbr=0.66. Fixed blocks do not work for scale-free topology.

### exp10j — Per-Level Break-Even for Trees (CLOSED)

**Folder:** `experiments/exp10j_tree_perlevel/`

**Results:**
- matmul operator: D_direct wins when occupancy < 37.5-40% at ANY level size.
- stencil operator: D_direct saves memory but NEVER wins on time.
- Contour B: 45% PASS.
- Policy: use D_direct per-level only when operator is compute-heavy (matmul-like) AND occupancy < 40%.

**Critical path:** P0 LAYOUT CLOSED -> Phase 2

---

## P0 LAYOUT — CLOSED (final summary)

### Layout policy by space type

| Space Type | Layout | Status | Evidence |
|------------|--------|--------|----------|
| scalar_grid | D_direct (packed tiles + tile_map) | Production | exp10g: both contours PASS |
| vector_grid | D_direct (packed tiles + tile_map) | Production | exp10h: 72/72 PASS |
| tree_hierarchy | Hybrid: D_direct per-level where occupancy < 40% + heavy compute; A_bitset otherwise | Validated | exp10j: break-even found |
| irregular_graph / spatial | D_blocked (block addressing) conditional | Conditional | exp10i: spatial partition, cbr<0.30 |
| irregular_graph / scale-free | A_bitset (dense grid + bitset mask) fallback | Fallback only | exp10i: blocks rejected, cbr=0.66 |

### Killed permanently
- Element-level reverse_map[M] (exp10: VRAM +38.6%)
- Binary search on GPU (exp10e-B: +1700%)
- Paged sparse tiles (exp10e-C: +9000%)
- Hash as primary lookup (exp10f-E: dominated by D_direct)
- Fixed blocks for scale-free graphs (exp10i: cbr 0.64-0.99)

### Possible follow-ups (NOT in current phase)
- Graphs: variable-size/adaptive blocks for spatial subclass; graph-native sparse (CSR/COO) for scale-free
- Trees: stencil path optimization (current manual stencil too slow) — low priority
- All: DET-2 cross-seed stability not yet verified

---

## Original Streams (for reference)

### S1: P0 — exp10_buffer_scaling (CLOSED)

**Result:** KILL compact-with-global-reverse-map. Grid is baseline.
Compact 18.5% faster (time), but +38.6% VRAM. 75/75 configs exceed VRAM kill threshold.
What was killed: dense bookkeeping, not sparse compute principle.

---

### S1b: DET-1 — exp10d_seed_determinism (after S1)

**Question:** Two runs, identical inputs + seed → bitwise identical tree?

**Folder:** `experiments/exp10d_seed_determinism/`

**Reuse:**
- `experiments/p2a_sensitivity/exp_p2a_sweep.py` — `run_single()` (adaptive pipeline with two-stage gate)
- `experiments/phase2_probe_seam/exp_seam_crossspace.py` — 4 space adapters

**Key implementation:**
- `CanonicalTraversal` — deterministic iteration order, Z-order tie-break
- `DeterministicProbe` — seed = f(coords, level, global_seed)
- `GovernorIsolation` — EMA update only after full step
- Test on CPU and GPU separately, all 4 space types
- `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8`

**Kill criteria:** ANY divergence = FAIL.

**Outputs:** `exp10d_results.json`, `exp10d_report.md`.

---

### S2: P1-B2 — exp11_dirty_signatures (CPU, parallel)

**Question:** 12-bit dirty signature + debounce → AUC > 0.8?

**Folder:** `experiments/exp11_dirty_signatures/`

**Reuse:**
- `experiments/phase2_probe_seam/exp_seam_crossspace.py` — SeamScore (seam_risk component)
- `experiments/p2a_sensitivity/exp_p2a_sweep.py` — `run_single()`, instability diagnostics (uncert component)
- `experiments/sc_baseline/metrics.py` — D_parent (mass component)

**Key implementation:**
- `DirtySignature` — 12-bit packed: 4 bits seam_risk + 4 bits uncert + 4 bits mass
- `DebounceTracker` — fires after 2 consecutive threshold crossings
- 3 scenarios: noise (transient, should NOT trigger), structural event (MUST trigger), drift (trigger with latency)
- Metrics: blast_radius, latency_to_trigger, burstiness

**Kill criteria:** AUC < 0.8 = FAIL. Must work on all 4 space types.

**Outputs:** `exp11_results.json`, `exp11_roc_curves.png`, `exp11_blast_radius.png`.

---

### S3: P2a — exp12 = p2a_sensitivity (CPU, CODE READY)

**Question:** How wide is the ridge of optimal thresholds?

**Folder:** `experiments/p2a_sensitivity/` (existing, code complete)

**Action:** Just RUN it.
```bash
.venv-gpu\Scripts\python experiments\p2a_sensitivity\exp_p2a_sweep.py
```

- 5 scenes x 4 spaces x 10x10 grid x 10 seeds = 20K configs
- Ridge >30% → MANUAL_OK, <10% → P2B_NEEDED
- Cross-space divergence >15pp → flag for architect

**Outputs:** `p2a_summary.json`, `p2a_full_results.json`, 5 heatmap PNGs, `p2a_ridge_comparison.png`.

---

### S4: SC-5 — exp12a_tau_parent (CPU, parallel)

**Question:** What are the data-driven thresholds tau_parent[L]?

**Folder:** `experiments/exp12a_tau_parent/`

**Reuse:**
- `experiments/sc_baseline/sc_baseline.py` — `sc1_prepare_baselines()`, `sc2_compute_metrics()`, `sc3_separability()`
- `experiments/sc_baseline/sc_baseline_crossspace.py` — cross-space data generators, `d_parent_lf_frac`
- `experiments/sc_baseline/operators.py`, `operators_v2.py` — R/Up for 4 spaces

**Key implementation:**
- `find_optimal_threshold(pos_vals, neg_vals, method)` — 3 methods: ROC optimal (Youden's J), F1-optimal, sensitivity@90%
- `compute_depth_specific_thresholds(records, method)` → `{L1: tau, L2: tau, L3: tau}`
- `check_space_specificity()` — if thresholds differ >2x across spaces → flag
- Leave-one-space-out cross-validation, 10 train seeds + 10 validation seeds

**Kill criteria:** accuracy drops >15pp on held-out space = problem.

**Outputs:** `exp12a_results.json`, `exp12a_thresholds.json`, `exp12a_roc_curves.png`.

---

### S5: Deferred Revisit — research note (lowest priority)

**Folder:** `experiments/exp_deferred_revisit/`

Research note (NOT experiment): Morton re-examination, block-sparse re-examination, phase schedule re-examination, new proposals.

**Output:** `deferred_revisit_note.md` for architect review.

---

## Worker Assignment

| Worker | Stream | Device | Blocks on | Est. Duration |
|--------|--------|--------|-----------|---------------|
| A | S1 (exp10) → S1b (exp10d) | GPU | — → S1 | 2-3d → 1-2d |
| B | S2 (exp11) | CPU | — | 2-3d |
| C | S3 (P2a run) | CPU | — | <1d |
| D | S4 (SC-5) | CPU | — | 1-2d |
| E | S5 (research note) | — | — | <1d |

Workers B, C, D, E start immediately. Worker A starts S1 (critical path), then S1b.

---

## Gate: Phase 1 -> Phase 2

- P0 layout CLOSED — full layout policy per space type (see table above)
- DET-1 PASSED (240/240 bitwise match)

## Architect Decisions (end of Phase 1) — resolved

- Layout: D_direct for grids (scalar + vector), hybrid for trees, D_blocked conditional for spatial graphs, A_bitset fallback for scale-free (S1 -> exp10 series)
- P2b needed? No — ridge 100% (S3)
- SC-5: thresholds found, L1 specificity low — needs architect decision (S4)
- Morton/block-sparse: Morton killed (binary search +1700%), block addressing viable only for spatial graphs (S5, exp10e, exp10i)

## Open questions for Phase 2

- exp11: redesign dirty signatures or accept limitation? (FAIL 3/4 spaces)
- exp12a: L1 without enforcement or rework? (specificity low)
- DET-2 cross-seed stability: needed before Phase 2?

## Shared Conventions

- Folders: `exp{N}_{short_name}/`, README.md in each
- Seeds: `np.random.default_rng(seed)`, 10-20 seeds min
- Device: `torch.device('cuda' if torch.cuda.is_available() else 'cpu')`
- Holm-Bonferroni for multiple comparisons
- Cross-space: 4 types (scalar grid, vector grid, irregular graph, tree hierarchy)
- Results appended to `docs/experiment_results.md`
