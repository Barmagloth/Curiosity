# S2 -- Halo Cross-Space Validation: Results

## Summary

| Space | Median Ratio | p_raw | p_adj (Holm-Bonf.) | Sig? | >= 1.5x? |
|-------|-------------|-------|-------------------|------|----------|
| T1: Scalar grid 64x64 | **2.020** | 9.54e-07 | 3.81e-06 | YES | YES |
| T2: Vector grid 64x64 d=32 | **1.573** | 9.54e-07 | 3.81e-06 | YES | YES |
| T3: Irregular graph 500 k=8 | **1.817** | 4.77e-06 | 9.54e-06 | YES | YES |
| T4: Tree depth=8, 255 nodes | **0.556** | 9.91e-01 | 9.91e-01 | NO | NO |

## Verdict: FAIL (T4)

Halo passes kill criterion on 3 of 4 space types but **fails on tree hierarchy**.

## Detailed Results

### T1: 2D Scalar Grid -- PASS
- SeamScore hard: median=1.416, mean=1.435
- SeamScore halo: median=0.713, mean=0.708
- Ratio: median=2.02x, min=1.79x
- All 20 seeds show ratio > 1.5x

### T2: 2D Vector Grid -- PASS
- SeamScore hard: median=1.583, mean=1.582
- SeamScore halo: median=1.002, mean=1.002
- Ratio: median=1.57x, min=1.45x
- 18/20 seeds show ratio > 1.5x (2 marginal at ~1.45x)

### T3: Irregular Graph -- PASS
- SeamScore hard: median=2.293, mean=2.201
- SeamScore halo: median=1.160, mean=1.196
- Ratio: median=1.82x, min=0.84x
- Some variance expected with stochastic k-NN topology
- One outlier seed has ratio < 1.0; median is robust

### T4: Tree Hierarchy -- FAIL
- SeamScore hard: median=0.364, mean=0.396
- SeamScore halo: median=0.639, mean=0.564
- Ratio: median=0.556x (halo is 1.8x WORSE than hard insert)

## Root Cause Analysis for T4 Failure

The tree topology has a fundamental structural difference from grids and graphs:

1. **Minimal boundary surface.** Each subtree (rooted at depth=3) connects to the outside world through exactly 1 edge: root-to-parent. This gives a single boundary pair for SeamScore measurement, making the metric extremely noisy.

2. **Halo bleeds into wrong context.** When the halo expands from an active subtree, it reaches the parent node (already correct at coarse level, delta=0) and the sibling subtree. The sibling subtree is inactive and has its own distinct coarse values. Applying a partial correction based on the active subtree's context to the sibling's root introduces error rather than reducing it.

3. **Asymmetric boundary.** In grids, the boundary zone has the same dimensionality as the interior (1D boundary in 2D space). In trees, the boundary is a single edge connecting subtrees of 31 nodes -- the boundary-to-interior ratio is ~1:31, far lower than grids (~8:64 per tile edge).

## Implications

- **Halo is validated for spaces with rich boundary topology** (grids, graphs with distributed boundaries). The 1.5x criterion is met or exceeded.
- **Halo in its current form does not generalize to tree hierarchies** with O(1) boundary edges per tile. The cosine feathering assumption (smooth transition across a zone) breaks down when the boundary is a single bottleneck edge.
- A tree-specific halo strategy would need to account for the tree's structural asymmetry -- e.g., only blending along the parent-child axis without spreading to siblings, or using a different blending weight schedule for tree edges.

## Recommendation

The concept_v1.7.md statement "Halo >= 3 pixels eliminates HF artifacts at tile boundaries" should be scoped to grid/graph topologies. Tree/hierarchical spaces need a separate boundary-smoothing mechanism, or the claim should note this limitation.
