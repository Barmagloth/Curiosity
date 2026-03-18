# S2 -- Halo Cross-Space Validation

## Motivation

Halo (cosine feathering with overlap >= 3 elements) was validated only on 2D pixel grids (CIFAR, 128x128, tile=16 in exp03_halo_diagnostic; 512x512, tile=16 in phase1_halo). SeamScore was validated across 4 space types in phase2_probe_seam/exp_seam_crossspace.py. This experiment closes the gap: validate that Halo works on all 4 space types.

## Space Types

| ID | Space | Size | Tile unit | Halo mechanism |
|----|-------|------|-----------|----------------|
| T1 | 2D scalar grid | 64x64 | 8x8 tiles | Overlap w=3 px, cosine ramp on extended patch |
| T2 | 2D vector grid (dim=32) | 64x64x32 | 8x8 tiles | Same as T1, applied per-channel |
| T3 | Irregular graph (k-NN) | 500 pts, k=8, 10 clusters | Cluster | 3-hop neighbors, cosine fade by hop distance |
| T4 | Tree hierarchy | depth=8, 255 nodes | Subtree at depth=3 (8 subtrees of 31 nodes) | 3-hop tree neighbors, cosine fade by hop distance |

## Protocol

For each space type, for each of 20 seeds (100..119):

1. Generate ground truth (GT) and coarse representation
2. Select ~30% of tiles with highest reconstruction error (budget=0.30)
3. **Hard condition (w=0):** Replace selected tiles with GT values. Unselected tiles remain at coarse. This creates sharp discontinuities at boundaries between refined and unrefined tiles.
4. **Halo condition (w=3):** Refine selected tiles using cosine-feathered blending. The overlap zone extends 3 elements/hops beyond each tile, with cosine-tapered weights that blend smoothly into the unrefined region.
5. Measure SeamScore at the active/inactive tile boundary only

## Seam Metric

SeamScore = Jump_boundary / (Jump_interior + eps)

- Jump_boundary: median of |x[p] - x[q]| for pairs (p,q) crossing active/inactive tile boundary
- Jump_interior: median of |x[p] - x[q]| for pairs one step inside the active tile
- Higher SeamScore = worse seam artifact

For T1/T2: pairs at grid-adjacent pixels across tile edges (only at active/inactive boundaries)
For T3: pairs at k-NN edges crossing active/inactive cluster boundary
For T4: pairs at tree edges (parent-child, sibling) crossing active/inactive subtree boundary

## Cosine Feathering

Grid spaces (T1, T2): Standard cosine ramp `0.5 * (1 - cos(pi * i / w))` applied in the overlap zone. Tiles contribute via weighted accumulation (sum of weight*delta / sum of weight).

Graph/Tree spaces (T3, T4): Core nodes get weight 1.0. Hop-d neighbors get `0.5 * (1 + cos(pi * d / (hops+1)))`, creating smooth decay from 1 to 0 across the halo zone.

## Statistical Analysis

- Paired Wilcoxon signed-rank test (one-sided: hard > halo), 20 seeds per space
- Holm-Bonferroni correction for 4 comparisons (family-wise alpha = 0.05)
- Effect size: median ratio of SeamScore_hard / SeamScore_halo

## Kill Criterion

**Halo must reduce seam artifacts by >= 1.5x (median ratio) on ALL 4 space types.**

Statistically: all 4 adjusted p-values < 0.05.

## Output

- `results/halo_crossspace.json` -- per-seed raw data + summary statistics
- `results/halo_crossspace.png` -- histogram of ratios per space type
