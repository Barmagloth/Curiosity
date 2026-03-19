# exp10i_graph_blocks

**Question:** Can block-based addressing work for irregular graphs?
**Motivation:** D_direct needs regular tile grid. Graphs have no natural grid.
              Block partitioning + block_map is the graph analog of tile_map.
**Kill criteria:** Dual contour + cross_block_ratio < 0.5 + padding_waste < 50%
**Roadmap level:** P0 (0.9b4)
**Status:** closed — compute healthy, representation sick

## Results (19 March 2026)

**Contour A: 120/288 (42%) | Contour B: 288/288 (100%)**
Run on CPU (parallel chunked, 6 chunks x 48 configs).

### Per graph type

| graph_type | Contour A | Contour B | Best partition | Best cbr | Verdict |
|------------|-----------|-----------|----------------|----------|---------|
| random_geometric | 58% | 100% | spatial (Morton) | 0.31 | Conditionally viable |
| grid_graph | 67% | 100% | spatial | 0.20 | Conditionally viable |
| barabasi_albert | **0%** | 100% | greedy | 0.66 | **Rejected** |

### Cross-block ratio (mean, lower = better)

| graph_type | random | spatial | greedy |
|------------|--------|---------|--------|
| random_geometric | 0.928 | 0.310 | 0.180 |
| barabasi_albert | 0.929 | 0.909 | 0.663 |
| grid_graph | 0.928 | 0.196 | 0.250 |

## Diagnosis

**Compute-path is healthy everywhere.** -55% to -99% wall-clock vs baseline. Block
addressing and packed execution are not the bottleneck. Contour B = 100% means the
computational machinery works.

**The failure is in data representation.** Padding waste 50-97%. Fixed block_size for
graphs is the wrong base abstraction. For grids it worked because space is uniform.
For graphs, density and locality are jagged — one block size lies about the structure.

**Graphs split into two classes:**

1. **Spatial graphs** (random_geometric, grid_graph): locality exists, blocks capture it.
   Spatial partition (Morton) is the sweet spot — low cbr AND controlled padding.
   Block addressing is conditionally viable here.

2. **Scale-free / hub-heavy graphs** (barabási-albert): hub nodes blow edges across
   all block boundaries. cbr = 0.64-0.99 regardless of partition. Block addressing
   is structurally incompatible with power-law degree distributions.

## What was NOT disproven

- Sparse compute on graphs — that works fine (Contour B 100%)
- Block packing as concept — it fails only for scale-free topologies
- The principle: "compute healthy, representation sick"

## What was disproven

Fixed-size blocks are NOT a universal abstraction for graphs. They are acceptable
only for graphs with explicit spatial structure and low cbr. For hub-dominated graph
families, block addressing is a rejected variant, not a primary path.

## Continue-or-stop criteria for block-graph work

Continue blocked branch ONLY if simultaneously:
- padding < 40-50% starting point
- cbr < 0.25-0.30 starting point
- Contour B peak memory passes

If not met — change representation, don't "optimize a bit more."

## Layout policy (current)

- scalar_grid → D_direct production
- vector_grid → D_direct production
- tree_hierarchy → A_bitset fallback (exp10j investigating per-level D_direct)
- irregular_graph / spatial subclass → D_blocked conditional
- irregular_graph / scale-free subclass → blocked rejected, A_bitset fallback

## Next directions (NOT in current scope)

For spatial graphs: variable-size / adaptive blocks (not one fixed 8/16/32/64).
For non-spatial graphs: graph-native sparse (CSR/COO, active frontier, edge-centric).

## Files
- `exp10i_graph_blocks.py` -- main experiment (full grid, GPU)
- `exp10i_chunk.py` -- chunked runner (single graph_type x size, CPU)
- `exp10i_merge.py` -- merge chunk results into summary + report
- `results/` -- chunk JSONs, summary, report
