# exp10e_tile_sparse

**Question:** Can tile-sparse layouts (without global reverse_map) beat grid on GPU?
**Kill criteria:** Per pattern class — overhead >20% vs grid in wall-clock OR VRAM → killed for that class
**Roadmap level:** P0 (0.9b1)
**Status:** closed

## Candidates

- A: Dense grid + bitset mask (improved baseline)
- B: Packed active tiles + sorted Morton keys + O(k) binary search lookup
- C: Paged sparse tiles (macroblocks, 2-level addressing)

## Results (54 configs: sides [64,128,256] × 6 sparsities × 3 patterns × 10 seeds)

| Candidate | Time vs grid | VRAM vs grid | Verdict |
|-----------|-------------|--------------|---------|
| A (bitset) | **-27% to -31%** | +5% to +28% (median +18%) | **ALIVE** — all patterns PASS |
| B (packed Morton) | +825% to +1503% | -30% to +243% | **KILLED** — all patterns FAIL (time) |
| C (paged) | +5352% to +9769% | -13% to +52% | **KILLED** — all patterns FAIL (time) |

## Analysis

**A is the only viable candidate, but it's not a true sparse-memory layout.**
It wins on compute (-27% to -31% faster) via more efficient masking, but still
carries full-grid data residency. VRAM overhead (+18% median) comes from bitset
metadata on top of full tensor. At high occupancy (70%), overhead drops to +5%.
A is a good execution layout, not a memory-saving layout.

**B's storage idea is valid, its lookup is catastrophic.**
- VRAM at low occupancy: **-30%** (genuinely sparse)
- VRAM at high occupancy: +237% (packed tiles expand)
- Time: +1700% everywhere — binary search on GPU is the killer

What B proves: packed-tile storage works for memory. Binary search addressing
does not work for hot-path lookup. The implementation is killed, the principle
of packed-tile sparse storage lives. See exp10f for alternative lookup strategies.

**C is dead.** Python-level per-page gather/scatter loops cannot compete with
tensor-level operations. +5000-30000% time overhead is not fixable without
CUDA kernel rewrite. Removed from consideration.

## Key principle

Tile-sparse, dense intra-tile. No element-level reverse_map.
Lookup only on O(k) support set.

## Follow-up

exp10f: packed active tiles (B's storage) with alternative lookup:
- hash table (cuckoo/robin-hood) instead of binary search
- direct index array on tile-level (not element-level)
- pre-built neighbour lists
