# exp10k_cost_surface

**Question:** Is the cost surface C(I, M, p) smooth enough to define layout switching boundaries as a law, or is it a noisy classification?
**Motivation:** If C(I, M, p) is smooth, the Layout Selection Invariant holds: the optimal layout is a deterministic function of space characteristics, not an empirical lookup table.
**Kill criteria:** Boundary smoothness score > 0.85 confirms smooth invariant; < 0.70 means jagged (classification only).
**Roadmap level:** P0 (0.9b3)
**Status:** pending

## Method

Generate synthetic graph spaces with controlled (I, M, p) values:
- I = topological isotropy (degree entropy H(D))
- M = metric gap (Kendall tau between BFS distance and linear address)
- p = occupancy (fraction of active nodes)

Benchmark three layouts (A_bitset, D_direct, D_blocked) at each grid point.
Build the 3D scalar field C(I, M, p) per layout. The argmin surface defines the layout switching boundary.

## Sweep grid

- I: 8 values (0.0 to 3.5)
- M: 6 values (0.0 to 1.0)
- p: 7 values (0.01 to 0.7)
- N = 1024 nodes, feat_dim = 16, operator = matmul
- 10 seeds per grid point
- Total: 336 grid points x 10 seeds = 3360 trials

## Chunked execution

```
python exp10k_cost_surface.py --chunk 0 --n_chunks 10 --output results/chunk_0.json
python exp10k_cost_surface.py --merge --output results/merged.json
```

## Outputs

- results/exp10k_summary.json
- results/exp10k_report.md
- results/exp10k_cost_surface.png (cost heatmaps sliced at different p values)
- results/exp10k_winners.png (winner layout map sliced at different p values)
