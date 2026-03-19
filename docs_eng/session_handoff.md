# Session Handoff — Curiosity Phase 1

Document for a new AI orchestrator session. Contains full context for immediately continuing work.

## Where We Are

Curiosity project. Phase 0 **complete** (March 18, 2026). Phase 1 **complete** (March 19, 2026). P0 layout **CLOSED**. Next step — Phase 2.

Workstation: **PC 2** (NVIDIA RTX 2070, 8 GB, CUDA 12.8). Working directory: `R:\Projects\Curiosity`.

## What to Read (in this order)

| # | File | Why |
|---|------|-----|
| 1 | `docs/phase1_plan.md` | **Phase 1 plan** — 6 streams, worker assignment, kill criteria, reuse map |
| 2 | `docs/concept_v1.8.md` | Canonical concept (current) |
| 3 | `docs/experiment_hierarchy.md` | Dependency graph, priorities, exp10+ numbering |
| 4 | `docs/teamplan.md` | Plan with Phase 0 mark, description of Phases 1-4 |
| 5 | `docs/environment_2.md` | How to activate .venv-gpu on PC 2 (CUDA) |

## What Was Done Previously (Phase 0)

### Experiments
1. **Environment**: PC 1 (AMD Radeon 780M, DirectML) + PC 2 (RTX 2070, CUDA 12.8)
2. **Halo cross-space**: grid/graph OK, tree FAIL (0.56x). Rule: parallelism >= 3 AND no leakage.
3. **P2a sweep**: code ready (20K configurations), NOT RUN
4. **SC-baseline**: D_parent = ||R(delta)|| / (||delta|| + eps), R = gauss sigma=3.0. AUC 0.824-1.000 across 4 spaces.

### Key Architectural Decisions
- Halo: NOT a universal invariant — rule depends on topology.
- D_parent: formula updated (lf_frac).
- Morton/block-sparse/phase schedule: DEFERRED, not rejected.

## Phase 1 — Main Stream Results (March 19, 2026)

| Stream | Result |
|--------|--------|
| S1 exp10 | KILL compact-with-reverse-map (VRAM +38.6%). Grid — baseline. The implementation was killed, not the sparse principle. |
| S1b exp10d | DET-1 PASS (240/240 bitwise match CPU+CUDA) |
| S2 exp11 | FAIL 3/4 spaces. Architectural problem. |
| S3 P2a | PASS — ridge 100%. Manual thresholds ok. P2b not needed. |
| S4 exp12a | Thresholds found. L1 specificity low. |
| S5 deferred | Research note done. |

Gate Phase 1 -> 2: PASSED (grid fixed + DET-1). P0 reopened for layout investigation.

---

## P0 LAYOUT — CLOSED (full exp10 series, March 19, 2026)

### Layout Glossary

- **D_direct** ("packed tiles + direct tile_map") — active tiles in a compact array, tile_map[tile_id] -> slot for O(1) lookup. No element-level reverse_map. Winner for grids.
- **A_bitset** ("dense grid + bitset mask") — full-size data tensor + activation bitmask. Simple fallback.
- **D_blocked** ("block addressing for graphs") — graph nodes split into fixed blocks, block_map[block_id] -> slot. Works only for spatial graphs.
- **E_hash** ("hash table lookup") — archival fallback, dominated by D_direct at current scale. Resurrection triggers documented.

### Full exp10 Series Chronology

| Experiment | What was tested | Result |
|------------|----------------|--------|
| exp10 | Grid layout vs compact layout with element-level reverse_map (scalar_grid) | KILL compact. reverse_map[M] on int32 = structural VRAM failure (+38.6%). Compute at O(k) was 18.5% faster. |
| exp10d | Bitwise determinism DET-1 (all 4 space types) | PASS 240/240 bitwise match CPU+CUDA. |
| exp10e | Three tile-sparse candidates: A=grid+bitset mask, B=packed tiles+Morton binary search, C=paged sparse (scalar_grid) | A alive (-20% time, +18% VRAM). B killed (binary search +1700%). C killed (+9000%). |
| exp10f | Packed tiles + direct tile_map O(1) vs hash table (scalar_grid) | D_direct: 5x faster, 5.5x less resident. Peak VRAM kill = measurement artifact. E_hash = same speed, build 10-30x slower. |
| exp10g | Two-mode benchmark: manual stencil (Contour A) + conv2d (Contour B) (scalar_grid) | D_direct PASS both contours. -54% to -80% time, -36% to -86% peak VRAM. |
| exp10h | Cross-space: vector_grid + tree_hierarchy | Vector: 72/72 PASS both contours. Tree: 0/108 FAIL (trees too small to amortize overhead). |
| exp10i | Block addressing for graphs with 3 partitioning strategies (3 graph types) | Spatial graphs (random_geometric, grid_graph): conditionally viable with spatial partition, cbr<0.30. Scale-free (barabasi-albert): REJECTED, cbr=0.66. |
| exp10j | Per-level break-even analysis of D_direct for trees (tree_hierarchy) | matmul: D wins at occupancy < 37.5-40% at ALL level sizes. stencil: D saves memory but NEVER wins on time. Contour B: 45% PASS. |

### Final Layout Policy

| Space type | Layout | Status | Evidence |
|------------|--------|--------|----------|
| scalar_grid | D_direct (packed tiles + tile_map) | Production | exp10g: both contours PASS |
| vector_grid | D_direct (packed tiles + tile_map) | Production | exp10h: 72/72 PASS |
| tree_hierarchy | Hybrid: D_direct per-level where occupancy < 40% + heavy compute; A_bitset otherwise | Validated | exp10j: break-even found |
| irregular_graph / spatial | D_blocked (block addressing) conditional | Conditional | exp10i: spatial partition, cbr<0.30 |
| irregular_graph / scale-free | A_bitset (dense grid + bitset mask) fallback | Fallback only | exp10i: blocks rejected, cbr=0.66 |

**Break-even for trees (exp10j):**
- matmul operator: D_direct wins at occupancy < 37.5-40% at ANY level size
- stencil operator: D_direct saves memory below the same threshold, but is always slower
- Policy: use D_direct per-level only when the operator is compute-heavy (matmul-like) AND occupancy < 40%
- Upper tree levels (small N_l, high occupancy) -> A_bitset
- Lower levels (large N_l, low occupancy, heavy compute) -> D_direct

### Killed Permanently

- Element-level reverse_map[M] (exp10: VRAM +38.6%)
- Binary search on GPU (exp10e-B: +1700%)
- Paged sparse tiles (exp10e-C: +9000%)
- Hash as primary lookup (exp10f-E: dominated by D_direct)
- Fixed blocks for scale-free graphs (exp10i: cbr 0.64-0.99)

---

## What to Do — Phase 2

P0 layout **CLOSED**. All space types have an assigned layout (see table above).

### Critical Path
```
P0 LAYOUT [CLOSED] -> Phase 2 (layout integration into runtime)
```

### Open Questions for the Architect (before starting Phase 2)
- exp11: redo dirty signatures or accept the limitation? (FAIL 3/4 spaces)
- exp12a: L1 without enforcement or rework? (specificity low)
- DET-2 cross-seed stability: needed before Phase 2 starts?

### Possible Follow-ups (NOT in the current phase)
- For graphs: variable-size/adaptive blocks for the spatial subclass; graph-native sparse (CSR/COO) for scale-free
- For trees: stencil path optimization (current manual stencil is too slow) — low priority
- For all: DET-2 cross-seed stability not yet verified

## Environment

```bash
# Activate venv (PC 2):
R:\Projects\Curiosity\.venv-gpu\Scripts\activate
# Python 3.12.11, PyTorch 2.10.0+cu128, CUDA 12.8

# Git auth:
gh auth setup-git
# Token: R:\Projects\.gh_tkn
```

## Principles

- **Cross-space validation** — 4 space types mandatory
- **Kill criteria before launch** — every experiment
- **Holm-Bonferroni** — for multiple comparisons
- **10-20 seeds** — for reproducibility
- **Barmagloth = architect** — makes decisions at forks, does not write code
