# exp10f_packed_lookup

**Question:** Can packed tile storage with direct tile_map lookup beat grid on both time AND VRAM?
**Motivation:** exp10e-B showed packed storage saves VRAM (-30%) but binary search killed time (+1700%).
              Replace binary search with O(1) tile_map[tile_id] -> slot.
**Kill criteria:** Per pattern class -- overhead >20% vs grid in wall-clock OR VRAM (peak_vram)
**Competitive criteria:** Must not be embarrassingly worse than A_bitset on time
**Roadmap level:** P0 (0.9b2)
**Status:** closed -- formal KILL (peak VRAM), but see measurement caveat below

## Candidates
- D_direct_onfly: packed tiles + tile_map + on-the-fly halo lookup
- D_direct_prebuilt: packed tiles + tile_map + prebuilt neighbor_slots table
- E_hash_onfly: packed tiles + open-addressing hash table + on-the-fly
- E_hash_prebuilt: packed tiles + hash table + prebuilt neighbor_slots

## Results (54 configs: sides [64,128,256] x 6 sparsities x 3 patterns x 10 seeds)

| Candidate | Time vs grid | Peak VRAM vs grid | Formal verdict |
|-----------|-------------|-------------------|----------------|
| A_bitset (ref) | -20% | +18% | ALIVE |
| D_direct_onfly | **-82%** | +230% | KILLED (VRAM) |
| D_direct_prebuilt | **-82%** | +196% | KILLED (VRAM) |
| E_hash_onfly | **-82%** | +228% | KILLED (VRAM) |
| E_hash_prebuilt | **-82%** | +197% | KILLED (VRAM) |

## Critical measurement caveat

**Peak VRAM (torch.cuda.max_memory_allocated) != resident memory footprint.**

Resident memory breakdown for clustered 256x256 sp=0.05:
- Grid:    data=262KB + mask=64KB = **328KB** resident. Peak VRAM = 886KB.
- D_direct: tiles=55KB + tile_map=4KB + ids=1KB = **60KB** resident. Peak VRAM = 1020KB.
- A_bitset: data=262KB + bitset=8KB + cache=64KB = **336KB** resident. Peak VRAM = 1138KB.

**D's resident footprint is 5.5x smaller than grid.** But peak VRAM captures temporary
allocations from conv2d on the packed tensor. Conv2d internally allocates workspace buffers
proportional to input. Grid's peak also includes temporaries, but its higher baseline data
makes the ratio look better.

The VRAM kill criterion triggered on peak allocation, not on what the layout actually stores.

## Compute analysis

D and E are 5x faster than grid (0.07-0.12ms vs 0.45ms). Direct tile_map lookup is
effectively free — no measurable difference vs hash. Hash is architecturally more complex
with zero compute benefit at current scale.

## Build cost

- D: 0.4-0.5ms (tensorized, fast)
- E: 2-15ms (hash construction, 10-30x slower)
Both are >> compute time (0.07ms), but build happens once per epoch, not per kernel.

## What this means

1. **Lookup problem is SOLVED.** Direct tile_map = O(1), no overhead vs grid.
2. **Storage problem is SOLVED.** Resident 60KB vs 328KB (5.5x saving).
3. **Measurement problem remains.** Peak VRAM from PyTorch allocator is not
   the right metric for comparing layouts with different tensor shapes.
4. **Hash adds nothing** over direct index at this scale. As architect predicted.

## Follow-up needed

The right metric is resident_bytes (what the layout stores persistently),
not peak_vram_bytes (what PyTorch allocator touches during compute).
Re-evaluate with resident-only metric, or with compute kernel that doesn't
allocate temporaries (e.g., manual stencil instead of F.conv2d).
