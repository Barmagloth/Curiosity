# exp10i_graph_blocks

**Question:** Can block-based addressing work for irregular graphs?
**Motivation:** D_direct needs regular tile grid. Graphs have no natural grid.
              Block partitioning + block_map is the graph analog of tile_map.
**Kill criteria:** Dual contour + cross_block_ratio < 0.5 + padding_waste < 50%
**Roadmap level:** P0 (0.9b4)
**Status:** PARTIAL FAIL (120/288 Contour A, 42%)

## Results (19 March 2026)

**Contour A: 120/288 (42%) | Contour B: 288/288 (100%)**
Run on CPU (parallel chunked, 6 chunks x 48 configs). Memory contours N/A (CPU).

### Per graph type

| graph_type | Contour A | Viable? |
|------------|-----------|---------|
| random_geometric | 56/96 (58%) | YES -- spatial/greedy partition keep cbr < 0.50 |
| barabasi_albert | 0/96 (0%) | NO -- scale-free structure resists all partitioning |
| grid_graph | 64/96 (67%) | YES -- spatial partition is best (cbr ~0.20) |

### Cross-block ratio (mean, lower = better)

| graph_type | random | spatial | greedy |
|------------|--------|---------|--------|
| random_geometric | 0.928 | 0.310 | 0.180 |
| barabasi_albert | 0.929 | 0.909 | 0.663 |
| grid_graph | 0.928 | 0.196 | 0.250 |

### Key findings

1. **Spatial graphs work.** random_geometric and grid_graph achieve low cbr
   with spatial (Morton order) or greedy (BFS growth) partitioning. Block-based
   addressing is viable for these topologies.

2. **Scale-free (BA) graphs fail.** Barabasi-Albert has hub nodes whose edges
   span many blocks regardless of partition strategy. Even greedy partition only
   gets cbr ~0.66, well above the 0.50 threshold. Block addressing is not viable
   for power-law degree distributions.

3. **Padding waste is high everywhere** (mean 0.77). At low sparsity (5-10%
   active), most block slots are empty. This is intrinsic to fixed-size blocks
   with sparse activation. Waste is worst at large block_size (0.80 at bs=64).

4. **Greedy partition is best for cbr** on spatial graphs (0.18 for RG, 0.25
   for grid) but causes MORE padding waste than spatial partition because
   BFS-growth creates unevenly sized effective blocks. Spatial partition
   (Morton order) is the sweet spot: low cbr AND same padding as random.

5. **D_graph_blocks is always faster** than graph_baseline on CPU (negative
   time overhead). This is because block-structured scatter/gather uses
   vectorized index_put_ vs the baseline's per-node Python loop. Contour B
   passes 100%. The speed advantage is from the implementation, not the layout.

6. **Block size has modest effect on cbr** (0.70 at bs=8 to 0.48 at bs=64).
   Larger blocks capture more local neighborhoods but waste more padding.

### Verdict

Block-based addressing works for spatial graphs (random_geometric, grid_graph)
with spatial or greedy partitioning. It fails for scale-free graphs (barabasi_albert).
Padding waste remains a concern at all configurations.

**Next steps:** For spatial graphs, proceed to full D_graph_blocks layout with
packed storage. For non-spatial irregular graphs, explore alternative approaches
(e.g., edge-centric storage, CSR-based packing, or hierarchical clustering).

## Approach
- Partition graph nodes into fixed-size blocks
- block_map[block_id] -> slot (same principle as tile_map)
- Test 3 partitioning strategies: random, spatial, greedy
- Test on 3 graph types: random_geometric, barabasi_albert, grid_graph

## Files
- `exp10i_graph_blocks.py` -- main experiment (full grid, GPU)
- `exp10i_chunk.py` -- chunked runner (single graph_type x size, CPU)
- `exp10i_merge.py` -- merge chunk results into summary + report
- `results/` -- chunk JSONs, summary, report
