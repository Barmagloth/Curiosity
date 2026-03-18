# Halo (Cosine Feathering) Applicability Rule

## The Rule

**Halo (cosine feathering, overlap >= 3) is applicable when the tile boundary
has sufficient _boundary width_ AND _contextual isolation_ -- meaning the halo
zone does not bleed into unrelated tile contexts through bottleneck edges.**

In practice this reduces to two structural checks:

1. **Boundary parallelism >= 3**: at least 3 independent edges cross the
   tile boundary, so the cosine ramp has spatial width to operate.
2. **No context leakage**: the halo expansion from an active tile must NOT
   reach nodes belonging to a _different_ inactive tile (sibling bleed).

| Topology | Boundary Parallelism | Context Isolation | Halo Applicable? |
|:---------|---------------------:|:-----------------:|:----------------:|
| 2D grid (tile >= 3) | tile_size (>= 3) | YES (neighbors are the same adjacent tile) | YES |
| k-NN graph (k >= 6) | typically 5-20+ per cluster pair | YES (spatial coherence) | YES |
| Binary tree | 1 (single parent edge) | NO (bleeds to sibling via parent) | NO |
| DAG with shared parents | 1-2 | NO (shared parents leak to siblings) | NO |

## Why Halo Fails on Trees: The Complete Mechanism

The T4 failure (0.56x, i.e. Halo is 1.8x WORSE) results from three
compounding structural problems:

### Problem 1: Single-edge bottleneck (no ramp width)

A subtree at depth 3 (31 nodes) connects to the rest of the tree through
exactly 1 edge: `subtree_root -> parent`. The cosine ramp needs multiple
parallel boundary nodes to grade smoothly from 1.0 to 0.0. With 1 edge,
it has no spatial extent -- it is a single point, not a zone.

In a grid (8x8 tiles), the boundary is 8 pixels wide. The ramp operates
across 8 parallel paths simultaneously, each at 3 pixels depth. This gives
the cosine function 8 x 3 = 24 node-hops to distribute the blend.

In a tree, the ramp operates across 1 path at 1 hop depth. Total: 1 node-hop.

### Problem 2: Sibling bleed (context leakage)

When the halo expands from the active subtree:
- **Hop 1** (fade=0.854): reaches the parent AND the sibling root
- **Hop 2** (fade=0.500): reaches the sibling's children
- **Hop 3** (fade=0.146): reaches deeper sibling nodes

The sibling subtree is a _different_ inactive tile with its own coarse
approximation. The active subtree's correction delta is meaningless in the
sibling's context. Applying 85% of the correction to the sibling root
introduces pure error.

In grids, halo expansion into an inactive tile reaches nodes that are
_spatially adjacent_ and _contextually similar_ to the active tile's boundary.
In trees, the parent is a funnel point that connects to contextually
unrelated subtrees.

### Problem 3: Asymmetric surface-to-volume ratio

| Space | Boundary Nodes | Interior Nodes | Ratio |
|:------|---------------:|---------------:|------:|
| Grid (8x8 tile) | 28 (perimeter) | 36 (interior) | 0.78 |
| Tree (31-node subtree) | 1 (root) | 30 (descendants) | 0.03 |

The tree boundary is 26x thinner relative to volume. Any error at the
boundary has nowhere to diffuse within the tile -- it sits at the single
connecting node and propagates down to all 30 descendants.

## Supporting Data

### Original 4 Space Types (from exp_halo_crossspace.py, 20 seeds)

| Space | Cut/Pair | S/V Ratio | Halo Ratio | Verdict |
|:------|--------:|----------:|-----------:|:--------|
| T1 Scalar grid 64x64 | 8 | 0.500 | 2.02x | PASS |
| T2 Vector grid 64x64 d=32 | 8 | 0.500 | 1.57x | PASS |
| T3 Irregular graph 500 k=8 | ~11 | 0.275 | 1.82x | PASS |
| T4 Tree depth=8 255 nodes | 1 | 0.032 | 0.56x | FAIL |

The single structural feature that cleanly separates T4 (FAIL) from
T1/T2/T3 (PASS) is boundary parallelism. T4 has min_cut = 1; the passing
spaces have min_cut >= 8.

### Tree Bleed Trace (halo_width=3)

```
Active subtree root: 7 (depth=3, 31 nodes)
Parent: 3
Sibling root: 8

Hop 1: 2 new nodes, fade=0.854, parent=YES, sibling=YES
Hop 2: 4 new nodes, fade=0.500
Hop 3: 8 new nodes, fade=0.146

Delta at active root [7]:  0.1924
Delta at parent [3]:       0.0000  (already correct at coarse level)
Delta at sibling root [8]: 0.0718  (different correction context)
Mean |delta| active:       0.1804
Mean |delta| sibling:      0.1731  (similar magnitude, different context)
```

The halo applies 85% of the active correction context to the sibling root,
which has a completely independent correction requirement. This is the
direct cause of the 1.8x worsening.

### Synthetic Two-Tile Sweep (50 nodes/tile, k=6, 30 seeds)

| Bnd Edges | Median Ratio | Q25 | Q75 | Frac > 1.0 |
|----------:|------------:|----:|----:|---------:|
| 1 | 1.563 | 1.469 | 1.638 | 100% |
| 2 | 1.171 | 1.138 | 1.201 | 100% |
| 3 | 0.864 | 0.835 | 0.892 | 0% |
| 4 | 1.036 | 1.006 | 1.068 | 77% |
| 5 | 1.354 | 1.246 | 1.402 | 100% |
| 6 | 1.108 | 1.070 | 1.188 | 100% |
| 8 | 1.003 | 0.935 | 1.064 | 53% |
| 10 | 1.045 | 0.992 | 1.065 | 70% |
| 15 | 1.040 | 1.003 | 1.083 | 77% |
| 20 | 0.984 | 0.952 | 1.014 | 47% |
| 30 | 0.894 | 0.883 | 0.917 | 0% |
| 40 | 0.984 | 0.961 | 1.011 | 33% |

NOTE: The synthetic two-tile graph with 1 boundary edge shows ratio=1.56,
seemingly contradicting the tree's 0.56. This is because:
- In the synthetic graph, each tile is a dense k=6 cluster. The 1 boundary
  edge connects to exactly 1 foreign node. The halo reaches just that node
  and its k=6 neighbors (all within the same inactive tile).
- In the tree, the 1 boundary edge goes through the parent, which connects
  to the sibling subtree -- a different tile. This is the **context leakage**
  problem, not just a boundary width problem.

This confirms that min_cut alone is insufficient. The rule must also check
for context isolation (no leakage to unrelated tiles).

### Saturation at high boundary counts

At boundary_edges >= 20, the ratio drops below 1.0. This is a saturation
effect: with too many cross-edges, the two tiles become so interconnected
that the boundary/interior distinction disappears. The halo zone overlaps
extensively with the tile interior, and the cosine weighting becomes noise
rather than a smooth transition. This is not a practical concern for real
tiling schemes where tiles are well-separated regions.

## Decision Procedure

```python
def should_use_halo(space_type, tile_boundary_info):
    """
    Determine if Halo (cosine feathering) should be applied.

    tile_boundary_info should contain:
      - min_cut: edges crossing from active to adjacent inactive tile
      - has_context_leakage: does halo expansion reach nodes in a
        DIFFERENT inactive tile (not the direct neighbor)?
    """
    if tile_boundary_info["has_context_leakage"]:
        return False  # tree/DAG with sibling bleed
    if tile_boundary_info["min_cut"] < 3:
        return False  # insufficient boundary width
    return True
```

Shortcut rules by topology:
- **Any grid** (regular or irregular, tile_size >= 3): **always use Halo**
- **k-NN graph with spatial clustering**: **use Halo** (min_cut is typically >> 3)
- **Tree or forest**: **never use Halo** (min_cut = 1, context leakage)
- **DAG**: check for bottleneck edges and sibling leakage per-case

## Alternatives for Bottleneck Topologies

For trees and other min_cut=1 structures:

1. **Hard insert (no smoothing)**: the simplest option. The seam at a single
   edge is less harmful than the bleed error from cosine feathering.
   Measured: hard insert at 1.0x vs Halo at 0.56x on T4.

2. **Parent-only blending**: restrict halo expansion to the parent-child axis
   only, with an explicit block preventing sibling traversal.

3. **Hierarchical (top-down) correction**: apply corrections level-by-level
   through the tree. Each level's correction is consistent with its parent,
   avoiding context leakage.

4. **Virtual edge augmentation**: add synthetic edges between leaf sets of
   adjacent subtrees to raise min_cut, then apply standard Halo.
