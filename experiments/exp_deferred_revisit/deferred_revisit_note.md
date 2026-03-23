# Deferred Revisit: Morton, Block-Sparse, Phase Schedule

Research note re-examining three architectural decisions deferred during Phase 0.
Intended audience: architect making go/no-go decisions.

---

## 1. Summary of Deferred Items

| Item | Original Result | Reason Deferred | Revisit Trigger |
|------|----------------|-----------------|-----------------|
| Morton layout | 12-15x sort overhead on CPU, zero compute benefit (exp09a) | No CPU advantage observed | GPU L2 cache locality may change the equation |
| Block-sparse layout | Expansion ratio problem: blocks include inactive elements (exp09a, B=8) | CPU overhead significant, memory waste at low sparsity | GPU warp-level execution naturally favors block-aligned access |
| Phase schedule | No benefit with ideal refine (Exp0.8v5); governor alone sufficient | Redundant under current conditions | Non-ideal refine (delta != GT - coarse) is the realistic case |

All three were correctly deferred: CPU sandbox results did not justify proceeding.
The question is whether GPU execution characteristics or non-ideal operating conditions change the verdict.

---

## 2. Morton Layout Re-examination

### 2.1 What exp09a showed

Morton sort on CPU adds 12-15x overhead to layout construction relative to compact (argsort on Morton codes dominates). Gather/compute/scatter benchmarks showed zero benefit -- CPU prefetching on sequential compact arrays is already effective. The sort overhead is pure cost with no measurable return.

### 2.2 GPU cache analysis: when does spatial locality matter?

On GPU, the relevant cache level is L2 (typically 4-6 MB on modern GPUs, shared across all SMs). L1/shared memory is per-SM and managed explicitly. The question is whether Morton-ordered access to global memory improves L2 hit rates.

**When Morton order helps:**
- Stencil/halo access patterns: each active tile reads neighbors in a 2D neighborhood. Row-major compact order scatters neighbors across cache lines when sparsity is moderate. Morton order clusters 2D-nearby tiles into contiguous memory, improving L2 spatial locality.
- Key metric: for a halo width w, each tile touches (2w+1)^2 neighbors. With w=3 (our default), that is 49 neighbors per tile. In row-major order on a 256x256 grid, neighbors can be up to 256 int32 positions apart in memory (~1 KB). Morton order reduces the expected distance to ~sqrt(block_area) positions.

**When Morton order does NOT help:**
- Pure gather/compute/scatter without neighbor access: each tile reads only its own value. Access is already coalesced if active_idx is sorted in any order. Morton adds nothing.
- Very high sparsity (>80% active): most tiles are active, so cache pressure is dominated by working set size, not access order. L2 will be thrashed regardless.
- Very low sparsity (<5% active): working set fits in L2 regardless of order. Both compact and Morton see near-100% hit rates.

### 2.3 Conditions for revival

Morton order provides >10% L2 cache benefit when ALL of the following hold:

1. **Halo/stencil access is present** (neighbor reads during refinement).
2. **Sparsity is in the mid-range:** 15-50% active tiles. Below 15%, working set fits in L2 anyway. Above 50%, the grid is dense enough that row-major order already has good locality.
3. **Grid side >= 128** (M >= 16K). Below this, the entire field fits in L2 even with random access.
4. **Clustered pattern**: Morton order's spatial grouping aligns with cluster structure. Random activation patterns gain less (neighbors are scattered regardless of memory order).

Estimated L2 hit rate improvement (based on cache line size 128B, L2 = 4MB):

| Grid side | Sparsity (active) | Pattern | Compact L2 hit rate | Morton L2 hit rate | Delta |
|-----------|--------------------|---------|--------------------|--------------------|-------|
| 128 | 30% | clustered | ~85% | ~90% | +5% |
| 256 | 30% | clustered | ~60% | ~78% | +18% |
| 256 | 15% | clustered | ~75% | ~88% | +13% |
| 512 | 30% | clustered | ~35% | ~55% | +20% |
| 256 | 30% | random | ~60% | ~65% | +5% |

Note: these are analytical estimates based on working set size vs. L2 capacity and spatial locality models. Actual numbers require GPU profiling (nsight compute, L2 hit rate counters).

### 2.4 Sort overhead on GPU

The critical difference: Morton sort is a ONE-TIME cost per layout rebuild, not per-step. On GPU, radix sort of k=20K uint32 Morton codes takes ~50-100 us (CUB DeviceRadixSort). This is negligible compared to refinement kernel time (typically ms-scale). The 12-15x CPU overhead that killed Morton in exp09a becomes irrelevant on GPU where:
- Sort is done by a dedicated GPU kernel (not Python argsort)
- Sort runs once per layout change, not per refinement step
- Sort cost is amortized over many refinement iterations

### 2.5 Recommendation: CONDITIONAL PROCEED

**Proceed with Morton layout evaluation on GPU IF the pipeline includes halo/stencil access.** The CPU sandbox killed Morton because (a) Python argsort is slow and (b) there was no neighbor access in the simple gather/compute/scatter benchmark. Both conditions are artifacts of the sandbox, not fundamental limitations.

**Concrete next step:** Add a Morton-vs-compact comparison to the P0 GPU pipeline (exp10 series), specifically in the halo/neighbor access kernel. Measure L2 hit rates with nsight compute. Kill criterion: if L2 hit rate improvement < 5% at grid side=256, sparsity=30%, clustered -- kill permanently.

**Estimated effort:** 2-3 hours to add Morton variant to existing GPU benchmark code.

---

## 3. Block-Sparse Layout Re-examination

### 3.1 What exp09a showed

Block-sparse with B=8 has an expansion ratio problem. The layout includes ALL tiles in any block that contains at least one active tile. At sparsity=30% with random pattern, expansion_ratio can reach 2-4x (processing 2-4x more tiles than necessary). At sparsity=5%, the expansion ratio can be as high as 10-20x (each active tile activates an entire 8x8=64-tile block).

Memory overhead is also significant: block_mask + active_blocks + expanded active_idx + reverse_map.

### 3.2 GPU warp utilization analysis

GPU executes in warps of 32 threads. If we process tiles in a compact list, warp utilization is always 100% (every thread has an active tile). With block-sparse, the warp processes a block of B*B tiles, some of which are inactive. Inactive threads execute the kernel but their results are discarded.

For B=8 (64 tiles per block, 2 warps per block):
- Warp utilization = (active tiles in block) / (B*B)
- At sparsity=30%, clustered: blocks near cluster centers are nearly full (~90% utilization). Blocks at cluster edges: ~30-50% utilization. Average: ~60-70%.
- At sparsity=30%, random: average block fill = 1 - (1 - 0.30)^64 probability of block being active, but average tiles per active block ~ 0.30 * 64 = 19.2 out of 64 = 30% utilization.

**Critical insight:** block-sparse is only competitive when blocks are well-filled. This requires clustered activation patterns AND appropriate block size.

### 3.3 Break-even sparsity calculation

Block-sparse beats compact when:
  block_overhead < scatter_overhead

Where:
- block_overhead = (expansion_ratio - 1) * compute_cost_per_tile (wasted computation on inactive tiles)
- scatter_overhead = cache_miss_penalty * miss_rate_compact (cache misses from scattered access in compact layout)

For block-sparse to break even:
  expansion_ratio < 1 + (cache_miss_penalty * miss_rate) / compute_cost

Typical values: cache_miss_penalty ~ 200-400 cycles (L2 miss to DRAM), compute_cost ~ 50-200 cycles per tile (depending on refinement complexity), miss_rate for compact ~ 10-40% (depending on grid size and sparsity).

Break-even expansion_ratio ~ 1.2 - 1.8x.

This means block-sparse is competitive when:
- **Sparsity >= 50% active** (most blocks are well-filled, expansion_ratio ~ 1.1-1.3x)
- **Clustered pattern** (blocks in cluster interiors are fully active)
- **B=4 instead of B=8** reduces worst-case expansion: 16 tiles per block instead of 64, so a single active tile wastes 15 instead of 63 slots

| Sparsity | Pattern | B=8 expansion | B=4 expansion | Competitive? |
|----------|---------|---------------|---------------|-------------|
| 5% | random | ~12x | ~4x | No |
| 5% | clustered | ~3x | ~1.8x | Marginal (B=4 only) |
| 15% | clustered | ~1.8x | ~1.3x | Yes (B=4) |
| 30% | clustered | ~1.3x | ~1.1x | Yes (both) |
| 50% | any | ~1.1x | ~1.05x | Yes |

### 3.4 Recommendation: CONDITIONAL PROCEED (B=4 only)

Block-sparse with B=8 is dead for sparsity < 30%. The expansion ratio is too punishing.

**B=4 is worth testing on GPU** for sparsity >= 15%, clustered patterns. At B=4, one block = 16 tiles = half a warp. Two blocks fill one warp exactly. The expansion ratio is manageable and the memory access pattern aligns with warp execution.

**Concrete next step:** If Morton proceeds to GPU testing (section 2), add a B=4 block-sparse variant to the same benchmark. The incremental cost is small. Kill criterion: if block-sparse B=4 is slower than compact at sparsity=30% clustered on GPU -- kill permanently.

**Do NOT test B=8.** The exp09a data is conclusive: expansion ratio at low sparsity is unacceptable and B=8 does not align with 32-thread warps (64 tiles = 2 warps, guaranteed under-utilization of the second warp at moderate sparsity).

---

## 4. Phase Schedule Re-examination

### 4.1 What Exp0.8v5 showed

With ideal refine (delta = GT - coarse), phase schedule (depth-dependent strictness) gave no benefit. The governor EMA controller alone handled budget management: StdCost -50%, P95 from 11.0 to ~6.5, compliance penalty from ~3.2 to ~0.5. Phase schedule was strictly redundant.

### 4.2 Non-ideal refine scenarios

In practice, delta is never GT - coarse. Realistic delta sources:
- **Noisy delta:** delta = (GT - coarse) + noise. The governor sees inflated cost because noisy deltas trigger more refinement (higher residual).
- **Partial convergence:** delta from an iterative solver that has not fully converged. Residual underestimates true error at depth, overestimates at surface.
- **Learned delta:** delta from a neural network. Systematic bias by depth (networks tend to produce larger deltas at coarse levels, smaller at fine levels).

In all three cases, the governor's EMA feedback loop adapts to the AVERAGE cost/step. It does not distinguish between depth levels. If depth-3 refinement is cheap and depth-7 is expensive, the governor finds an average strictness that under-budgets depth-7 and over-budgets depth-3.

### 4.3 Governor limitations

The governor fails (strictness hits maximum, candidates still exceed hard_cap) when:
- **Burst events:** a sudden change in input causes many regions to simultaneously need refinement. The EMA smooths this out, meaning the governor reacts slowly (by design, for stability). Phase schedule could pre-allocate budget to critical depths.
- **Depth imbalance:** if 80% of candidates are at depth 2 and 20% at depth 6, the governor's uniform strictness wastes budget on shallow splits that provide little gain, while starving deep splits that provide most of the quality improvement.
- **candidates >> hard_cap:** concept_v2.0 section 5.2 explicitly flags this case. When max strictness still yields more candidates than hard_cap, the governor is useless. Phase schedule with depth-dependent caps would bound each depth independently.

### 4.4 Quantifying the gap

At what point does governor-only fail? Estimated failure conditions:
- n_candidates > 3x hard_cap (governor cannot squeeze enough with strictness alone)
- Depth CV (coefficient of variation of candidate counts across depths) > 1.5 (extreme depth imbalance)
- Non-ideal refine with noise std > 0.3x signal std (noisy residual misleads the governor's EMA)

These are the realistic operating conditions for a production system. Exp0.8v5 tested none of them because it used ideal refine.

### 4.5 Recommendation: DEFER (but design the experiment)

Phase schedule should remain deferred until P0 layout is resolved and the GPU pipeline exists. However, the revisit experiment should be DESIGNED NOW so it can run immediately after P0:

**Proposed experiment (fits as exp10e or similar):**
1. Use the GPU pipeline from P0 with the finalized layout.
2. Replace ideal refine with three non-ideal variants: noisy (additive Gaussian), partial (early-stopped iterative), biased (depth-dependent scaling).
3. Compare governor-only vs. governor + phase schedule on each variant.
4. Metrics: PSNR, cost compliance, depth utilization balance, P95 cost.
5. Kill criterion: if phase schedule gives < 0.5 dB PSNR improvement AND < 10% compliance improvement on ALL three non-ideal variants -- kill permanently.

**Estimated effort:** 1-2 days once P0 pipeline exists. Most code is already in exp08_schedule.

---

## 5. New Proposals

Two ideas emerged from this analysis:

### 5.1 Morton + Block-Sparse Hybrid

Morton-sorted B=4 blocks: sort blocks by Morton code of their center coordinate, then within each block use linear order. This combines Morton's L2 locality benefits with block-sparse's warp-aligned execution. The sort is over blocks (k/16 elements), not tiles (k elements), so it is 16x cheaper.

Worth testing if both Morton and B=4 block-sparse individually show promise on GPU.

### 5.2 Depth-Aware Governor (instead of Phase Schedule)

Rather than a separate phase schedule, extend the governor itself with depth-aware EMA tracks. Each depth level maintains its own EMA cost tracker, and the governor adjusts per-depth strictness. This is lighter than a full phase schedule (no predefined depth budgets), but handles depth imbalance better than a single global EMA.

This could be tested as a variant in the phase schedule experiment (section 4.5).

---

## 6. Architect Decision Points

### Decision 1: Morton on GPU -- proceed to benchmark?
- **Cost:** 2-3 hours to add Morton variant to P0 GPU pipeline
- **Gate:** L2 hit rate improvement >= 5% at side=256, sp=0.30, clustered
- **Risk if skipped:** potential 10-20% cache benefit left on the table for halo-heavy workloads
- **Recommendation:** Yes, include in exp10 series

### Decision 2: Block-sparse B=4 on GPU -- proceed to benchmark?
- **Cost:** incremental (if Morton benchmark already exists)
- **Gate:** wall-clock faster than compact at sp=0.30, clustered
- **Risk if skipped:** low -- compact is the safe default
- **Recommendation:** Yes, but only B=4. Kill B=8 permanently.

### Decision 3: Phase schedule experiment -- design now or defer design?
- **Cost of design now:** 2-3 hours to write protocol
- **Cost of deferring design:** risk of forgetting the non-ideal refine scenarios
- **Recommendation:** Write the experiment protocol now, run it after P0

### Decision 4: Kill B=8 block-sparse permanently?
- **Evidence:** expansion ratio 3-12x at sparsity < 30%, does not align with warp size
- **Recommendation:** Yes, kill. No further consideration.

---

## 7. Layout Cost Surface C(I, M, p) — Deferred (exp10k)

### 7.1 What was attempted

exp10k tried to verify whether the Layout Selection Invariant (v1.8.3) produces a
smooth cost surface C(I, M, p) that can be interpolated for unknown space types.

Three axes:
- **I** = topological isotropy (degree entropy H(D))
- **M** = metric gap (attempted: Kendall tau, then spectral gap lambda_2)
- **p** = dynamic density (occupancy k/N)

810 trials across synthetic spaces with controlled (I, M, p). Benchmark: D_direct
vs D_blocked vs A_bitset, wall-clock as cost.

### 7.2 Result

**Boundary smoothness: 0.496.** Surface is JAGGED. ~50% of adjacent grid points
switch layout winner. Verdict: not a smooth deterministic law.

However: **A_bitset won 0 out of 810 trials.** The sparse-vs-dense boundary IS
smooth and absolute. Jaggedness is only in D_direct vs D_blocked comparison.

### 7.3 Why the surface was jagged — three measurement failures

**Failure 1: Isotropy blindness.** H(D) measures degree inequality, not geometry.
A random 4-regular graph has H(D)=0 (same as a grid) but 100% cache miss rate.
The metric cannot distinguish "uniform order" from "uniform chaos."

**Failure 2: Macroscopic metric gap.** Both Kendall tau and lambda_2 are global
metrics. Two perfect clusters connected by one bridge: lambda_2 collapses (signals
"terrible connectivity"), but GPU warps happily compute inside each cluster at 99.9%
L1 hit rate. Global topology scalers cannot evaluate local cache-line health.

**Failure 3: Ghost of the void.** I and M were computed on the FULL space X, but
compute runs on X_active (< 40% of nodes). The induced subgraph of active nodes has
different entropy, different spectral gap, different cache profile. We evaluated layout
cost using topology of a space that isn't even loaded into registers.

### 7.4 What needs to happen for a valid test

Two separate experiments, each with corrected metrics:

**Experiment A: Synthetic space classification surface.**

Goal: can we predict optimal layout for an ARBITRARY synthetic space from measurable
properties?

Corrected metrics (computed on X_active, not X):
- I_active = H(D | induced subgraph of active nodes)
- M_local = mean |addr(u) - addr(v)| / cache_line_size for edges within X_active.
  This is a direct proxy for L1/L2 cache miss rate. Cheap to compute: O(edges_active).
- p = occupancy (unchanged)

Why these are better:
- I_active sees the actual degree distribution the GPU processes
- M_local measures cache-line locality of actual memory accesses, not abstract topology
- Both are O(edges) to compute, not O(N^3) like eigendecomposition

**Experiment B: Production layout auto-selection.**

Goal: can the Curiosity runtime automatically choose between D_direct, D_blocked,
and A_bitset based on (I_active, M_local, p) without human classification?

This is only useful if Experiment A produces a smooth surface. If A is jagged,
auto-selection is impossible and policy table remains the correct approach.

Priority: B depends on A. A is not urgent (policy table works). Both are Track C.

### 7.5 What v1.8.3 should say

The Layout Selection Invariant (v1.8.3) should be labeled as:
- **Confirmed:** sparse always beats dense (A_bitset = 0 wins in 810 trials)
- **Hypothesis (unverified):** specific sparse variant selection follows a smooth
  C(I, M, p) law. Current metrics insufficient. Corrected metrics on X_active needed.
- **Operational:** policy table works as empirical classification. Does not require
  the continuous law to be true.

### 7.6 Resurrection triggers

Revisit exp10k when:
1. A new space type appears that doesn't fit existing policy table categories
2. Runtime auto-selection becomes a requirement (Track C)
3. Someone has time and curiosity to build proper X_active-based metrics

### 7.7 Estimated effort

- Experiment A with corrected metrics: 1-2 days (rewrite generator + metrics, rerun sweep)
- Experiment B: 1 day (add auto-selector to pipeline, test on real workloads)
- Total: 2-3 days, non-blocking, Track C priority
