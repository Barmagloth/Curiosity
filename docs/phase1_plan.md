# Curiosity — Phase 1 Implementation Plan

## Context

Phase 0 COMPLETE (18 March 2026). Phase 1 starts now. Working directory: `R:\Projects\Curiosity`. GPU: RTX 2070, CUDA 12.8, venv at `.venv-gpu` (see `docs/environment_2.md`).

Critical path: **P0 (layout) → DET-1 (determinism) → Phase 2**.

---

## Streams (6 total, 4 parallel + 2 sequential)

### S1: P0 — exp10_buffer_scaling (GPU, CRITICAL PATH)

**Question:** Grid vs compact layout on GPU. Kill compact if overhead >20%.

**Folder:** `experiments/exp10_buffer_scaling/`

**Reuse:**
- `experiments/exp09a_layout_sandbox/exp09a_sandbox.py` — layout construction (build_fixed_grid, build_compact, make_mask). Port to CUDA tensors.

**Key implementation:**
- `SyntheticKernel` — PyTorch CUDA kernel: gather/compute/scatter on both grid (O(M) buffers) and compact (O(k) buffers)
- `measure_vram()` — via `torch.cuda.max_memory_allocated()`
- `run_buffer_probe(side, sparsity, pattern, n_seeds, device)` — run both layouts, measure wall-clock (median of 20 repeats, 5 warmup) and peak VRAM
- Cross-space validation: 4 types (scalar grid, vector grid, graph, tree)
- 10-20 seeds, Holm-Bonferroni

**Kill criteria:** compact overhead >20% (wall-clock OR VRAM) → kill compact, fix grid.

**Outputs:** `exp10_summary.json`, `exp10_buffer_scaling.png`, `exp10_vram_profile.png`, `README.md` with verdict.

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

## Gate: Phase 1 → Phase 2

- P0 layout FIXED (grid or compact)
- DET-1 PASSED (bitwise match)

## Architect Decisions (end of Phase 1)

- Grid or compact? (S1)
- P2b needed? Ridge same across spaces? (S3)
- SC-5 pass/fail? (S4)
- Unfreeze Morton/block-sparse/schedule? (S5)

## Shared Conventions

- Folders: `exp{N}_{short_name}/`, README.md in each
- Seeds: `np.random.default_rng(seed)`, 10-20 seeds min
- Device: `torch.device('cuda' if torch.cuda.is_available() else 'cpu')`
- Holm-Bonferroni for multiple comparisons
- Cross-space: 4 types (scalar grid, vector grid, irregular graph, tree hierarchy)
- Results appended to `docs/experiment_results.md`
