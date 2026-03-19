# exp10_buffer_scaling

**Question:** Grid vs compact layout — which has lower overhead on GPU?
**Kill criteria:** compact overhead >20% (wall-clock OR VRAM) → kill compact
**Roadmap level:** P0
**Status:** closed — KILL compact (VRAM +38.6%)

## Result

- **Time:** compact 18.5% FASTER (gather/scatter cheap on GPU, compute on O(k) faster than O(M))
- **VRAM:** compact +38.6% WORSE (75/75 configs exceed kill threshold)
- **Verdict:** KILL compact, fix grid

## What was killed (and what was NOT)

**Killed:** compact-with-global-reverse-map — the specific implementation where:
- `active_idx` = O(k) int32
- `reverse_map` = O(M) int32 (global element-level reverse index)
- compact data buffer = O(k)

The reverse_map[M] at int32 is structurally heavier than grid's bool mask[M] at 1 byte.
This is not a surprise — it's 4x the metadata per element.

**NOT killed:** the principle of sparse layout on GPU. Compute on O(k) was genuinely
faster. The failure is in dense bookkeeping (element-level reverse_map), not in
sparse computation.

**Implication:** tile-sparse layouts without global reverse_map remain viable candidates.
See exp10e for follow-up investigation.

## Scope limitation

Synthetic kernel was gather → 3x3 conv → scatter (local stencil). Conclusion is
strongest for this operational profile. Other compute patterns may shift the picture,
but kill criterion was already triggered for this one.

## Full exp10 series

This experiment (exp10) was the first in a series of 8 experiments (exp10 through exp10j)
that resolved the P0 layout question across all space types. For the complete series
results and final layout policy, see `exp10j_tree_perlevel/README.md`.

| Experiment | What was tested | Result |
|------------|----------------|--------|
| exp10 | Grid vs compact (reverse_map) | KILL compact (VRAM +38.6%) |
| exp10d | Seed determinism (DET-1) | PASS 240/240 bitwise match |
| exp10e | Three tile-sparse candidates | A_bitset alive, B/C killed |
| exp10f | Packed tiles: direct vs hash lookup | D_direct 5x faster, E_hash archived |
| exp10g | Dual-mode benchmark (stencil + matmul) | D_direct PASS both contours |
| exp10h | Cross-space (vector_grid + tree) | Vector 72/72 PASS, tree 0/108 FAIL |
| exp10i | Graph block-based addressing | Spatial graphs conditional, scale-free rejected |
| exp10j | Per-level tree break-even | matmul: D wins at p<0.40; stencil: D saves memory, never time |
