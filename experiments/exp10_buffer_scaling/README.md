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
