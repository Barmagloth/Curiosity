# exp10f_packed_lookup

**Question:** Can packed tile storage with direct tile_map lookup beat grid?
**Motivation:** exp10e-B showed packed storage saves VRAM (-30%) but binary search killed time (+1700%).
              Replace binary search with O(1) tile_map[tile_id] -> slot.
**Roadmap level:** P0 (0.9b2)
**Status:** closed — D passes Contour A, fails Contour B (peak VRAM). A_bitset operational default. D alive pending exp10g.

## Candidates
- D_direct_onfly: packed tiles + tile_map + on-the-fly halo lookup
- D_direct_prebuilt: packed tiles + tile_map + prebuilt neighbor_slots table
- E_hash_onfly: packed tiles + open-addressing hash table + on-the-fly
- E_hash_prebuilt: packed tiles + hash table + prebuilt neighbor_slots

## Results (54 configs: sides [64,128,256] x 6 sparsities x 3 patterns x 10 seeds)

| Candidate | Time vs grid | Resident memory | Peak VRAM vs grid |
|-----------|-------------|-----------------|-------------------|
| A_bitset (ref) | -20% | ~= grid (336KB) | +18% |
| D_direct_onfly | **-82%** | **5.5x smaller** (60KB) | +230% |
| D_direct_prebuilt | **-82%** | **5.5x smaller** | +196% |
| E_hash_onfly | -82% (= D) | ~= D | +228% |
| E_hash_prebuilt | -82% (= D) | ~= D | +197% |

## Dual-contour verdict

### Contour A — Architectural viability
Candidate alive if: runtime win + resident footprint < dense baseline + build cost reasonable + deterministic.

| Candidate | Runtime | Resident | Build | Deterministic | Contour A |
|-----------|---------|----------|-------|---------------|-----------|
| A_bitset | ✅ -20% | ❌ ~= grid | ✅ 0 | ✅ | **PASS** |
| D_direct | ✅ -82% | ✅ 5.5x smaller | ✅ 0.4ms | ✅ | **PASS** |
| E_hash | ✅ -82% | ✅ ~= D | ❌ 2-15ms (10-30x D) | ⚠️ harder | **DOMINATED by D** |

### Contour B — Operational viability (peak-step memory within GPU budget)
Candidate goes to default only if peak-step memory also within budget.

| Candidate | Peak VRAM vs grid | Contour B |
|-----------|-------------------|-----------|
| A_bitset | +18% | **PASS** → operational default |
| D_direct | +196-230% | **FAIL** (conv2d workspace overhead) |
| E_hash | +197-228% | **FAIL** (same + dominated by D) |

### Summary
- **A_bitset:** operational default (passes both contours)
- **D_direct:** architecturally sound, operationally blocked by workspace overhead.
  Status: ALIVE pending exp10g (dual-mode benchmark to separate layout vs operator cost)
- **E_hash:** archived as contingency fallback. Dominated by D at current scale.

## E resurrection triggers (DO NOT resurrect without at least one):
1. tile_map occupies >25-30% of packed tile data (universe too large for direct)
2. Tile universe becomes sparse/irregular enough that direct indexing is unnatural
3. Multi-level/global sparse addressing where dense array too large or empty
4. Hash build/runtime after optimization drops to ≤2x D build, ≤15% D runtime penalty

## Memory measurement note

Three numbers needed for fair comparison:
1. **Resident layout bytes** — persistent storage after build, before compute
2. **Peak step bytes** — max during compute
3. **Workspace overhead** — peak_step - resident (operator temporaries)

Clustered 256x256 sp=0.05 breakdown:
- Grid: resident=328KB, peak=886KB, workspace=558KB
- D: resident=60KB, peak=1020KB, workspace=960KB
- A: resident=336KB, peak=1138KB, workspace=802KB

D pays higher workspace tax because conv2d on packed [k,H,W,C] tensor allocates
temporaries proportional to input. Grid's workspace is also large but masked by
its already-large resident baseline.

## Follow-up: exp10g

Dual-mode benchmark to resolve D's Contour B status:
- **Mode 1 (layout benchmark):** manual stencil kernel, no conv2d temporaries.
  Tests pure layout economics: build + gather/scatter + addressing + resident.
- **Mode 2 (operator benchmark):** conv2d or real refinement operator.
  Tests practical peak-step cost and GPU envelope compatibility.
