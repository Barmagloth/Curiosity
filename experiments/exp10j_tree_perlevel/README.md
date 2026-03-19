# exp10j_tree_perlevel

**Question:** At which tree level does D_direct (packed tiles + tile_map) break even with A_bitset, given sufficient N_l and low p_l?

**Motivation:** exp10h tested D_direct on whole trees and got FAIL 0/108, but diagnosis showed
configs were too small for per-level tile_map overhead to amortize. This experiment isolates
each tree level independently and sweeps much wider parameter ranges to find the actual
break-even thresholds.

**Kill criteria:** Dual contour per level (A: stencil/gather, B: matmul).
Pass = D_direct beats A_bitset on both resident memory AND wall-clock for at least one
(N_l, p_l) regime in each contour.

**Roadmap level:** P0 (0.9b3)
**Status:** done
**Predecessor:** exp10h (cross-space, tree FAIL 0/108)

## Design

Per-level independent benchmark: for each tree level, build A_bitset and D_direct layouts
FOR THAT LEVEL ONLY, measure memory (resident + peak) and time, compute break-even.

## Sweep parameters

| Parameter | Values |
|-----------|--------|
| branching | 2, 4, 8, 16, 32 |
| depth | max so bottom level >= 1024 nodes (br-dependent) |
| occupancy p_l | 0.01, 0.05, 0.1, 0.2, 0.4, 0.7 |
| activation pattern | random, clustered_subtree, frontier |
| payload (feat bytes) | 4, 16, 64, 256 (1 float to 64 floats) |
| operators | stencil (parent-child gather), matmul (small batched) |

### Depth per branching factor
- br=2: depth up to 10 (N_bottom=1024)
- br=4: depth up to 5 (N_bottom=1024)
- br=8: depth up to 4 (N_bottom=4096)
- br=16: depth up to 3 (N_bottom=4096)
- br=32: depth up to 3 (N_bottom=32768)

## Layouts (per level)

**A_bitset:** Full-size tensor (N_l, feat_dim) + bitset mask. Compute on full tensor, mask results.

**D_direct:** Packed tensor (k_l, feat_dim) + tile_map (N_l -> slot) + active_ids. Compute on packed tensor only.

## Dual contours

- **Contour A (stencil):** manual parent-child gather, no framework temporaries
- **Contour B (matmul):** small batched matmul on active nodes, realistic workspace

## Output per level

N_l, k_l, p_l, A_resident, D_resident, resident_ratio, A_peak, D_peak, peak_ratio,
A_time, D_time, time_ratio, A_build, D_build, verdict_A, verdict_B

## Break-even analysis

After sweep, fit:
- p*_mem(N_l): occupancy threshold where D_resident < A_resident
- p*_time(N_l): occupancy threshold where D_time < A_time
- Policy: use D if p_l < min(p*_mem, p*_time), else A

## Chunked execution

```
python exp10j_tree_perlevel.py --branching 2 --output results/chunk_br2.json
python exp10j_tree_perlevel.py --merge --output results/merged.json
```

## Timing protocol

10 seeds, 5 warmup, 20 repeats. Device: cuda (cpu fallback).

## Results

**128,160 total trials** across 5 branching factors, run on RTX 2070 (8.6 GB VRAM).

### Break-even thresholds

**p\*_mem (memory):** D_direct saves resident memory whenever p_l < ~0.40 across all N_l values.
The threshold is remarkably stable: p\*_mem converges to ~0.375-0.40 for N_l >= 8, and is 0.50 for
the smallest levels (N_l=2,4) where tile_map overhead is proportionally smaller.

**p\*_time (wall-clock):** Operator-dependent.
- **matmul:** D_direct wins time for p_l < 0.38-0.50 (varies by N_l). Median time_ratio ~1.04x
  means D is only marginally slower when it loses. Time win rate: 58.8%.
- **stencil:** D_direct **never** wins on time (0.0% win rate). The parent-child gather via
  tile_map lookup is ~3.3x slower than direct indexing on the full tensor, regardless of N_l or
  occupancy. This is the dominant bottleneck.

### At what N_l does D_direct start winning?

D_direct wins on **memory at all N_l values** tested (2 through 4096) whenever p_l is below the
threshold. For **time**, D_direct wins for matmul at all N_l values (no clear N_l threshold), but
never wins for stencil at any N_l.

### Which activation patterns favor D_direct?

Activation pattern has **negligible effect** on outcomes:
- random: both wins = 23.1%
- frontier: both wins = 23.0%
- clustered_subtree: both wins = 22.7%

The pattern (spatial locality) does not meaningfully affect whether D_direct wins. This suggests
the tile_map indirection cost dominates over cache/locality effects at these scales.

### Feature dimension impact

Larger feature dimensions favor D_direct on matmul:
- feat_dim=1 (4B payload): time win rate 47.4%
- feat_dim=4 (16B payload): time win rate 60.0%
- feat_dim=16 (64B payload): time win rate 61.9%
- feat_dim=64 (256B payload): time win rate 66.0%

This is expected: larger payloads increase the compute-to-overhead ratio, making the packed layout
more beneficial.

### Summary statistics

| Metric | Value |
|--------|-------|
| Total trials | 128,160 |
| Contour A (stencil) PASS | 3 / 64,080 (0.0%) |
| Contour B (matmul) PASS | 29,350 / 64,080 (45.8%) |
| Resident ratio (D/A) min | 0.024 |
| Resident ratio (D/A) median | 0.522 |
| Time ratio (D/A) min | 0.081 |
| Time ratio (D/A) median | 2.111 |

### Policy recommendation

**Use D_direct for matmul-like operators when p_l < 0.40.** This gives both memory savings
(~48% median resident reduction) and time wins (~59% win rate).

**Do NOT use D_direct for stencil/gather operators.** The tile_map indirection for parent-child
lookups is 3.3x slower than direct indexing, and this penalty persists across all N_l, occupancy
levels, and activation patterns. For gather-heavy operators, A_bitset is strictly better on time.

**Hybrid policy:** In a real tree where different levels run different operators, apply D_direct
per-level only to levels that use matmul-class operators AND have p_l < 0.40. All other levels
should use A_bitset.

### Kill criteria verdict

**Contour A (stencil): FAIL.** D never wins both memory and time simultaneously (only 3 edge-case
passes out of 64,080 trials).

**Contour B (matmul): PARTIAL PASS.** D wins both memory and time in 45.8% of trials, concentrated
at p_l < 0.40 and larger feature dimensions. The break-even threshold is clear and actionable.

## exp10 series: final layout policy (all space types)

exp10j closes the tree_hierarchy question opened by exp10h (FAIL 0/108). The failure was
NOT architectural rejection of D_direct for trees — it was configs too small for per-level
tile_map overhead to amortize. With wider sweeps (128K+ trials), the break-even is clear
and stable.

| Space type | Layout | Status | Evidence |
|------------|--------|--------|----------|
| scalar_grid | D_direct (packed tiles + direct tile_map) | Production | exp10g: both contours PASS |
| vector_grid | D_direct (packed tiles + direct tile_map) | Production | exp10h: 72/72 PASS |
| tree_hierarchy | Hybrid: D_direct per-level where p_l<0.40 + matmul op; A_bitset elsewhere | Validated | exp10j: break-even found |
| irregular_graph / spatial | D_blocked (graph block addressing) conditional | Conditional | exp10i: spatial partition, cbr<0.30 |
| irregular_graph / scale-free | A_bitset (dense grid + bitset mask) fallback | Fallback only | exp10i: blocks rejected, cbr=0.66 |

### Killed forever (full exp10 series)

- Element-level reverse_map[M] (exp10: VRAM +38.6%)
- Binary search lookup on GPU (exp10e-B: +1700%)
- Paged sparse tiles (exp10e-C: +9000%)
- Hash as primary lookup (exp10f-E: dominated by D_direct)
- Fixed-size blocks for scale-free graphs (exp10i: cbr 0.64-0.99)

## Environment

- Python 3.12.11, PyTorch 2.10.0+cu128
- GPU: NVIDIA GeForce RTX 2070 (8.6 GB VRAM)
- Venv: R:\Projects\Curiosity\.venv-gpu
