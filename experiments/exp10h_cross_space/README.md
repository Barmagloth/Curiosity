# exp10h_cross_space

**Question:** Does D_direct (packed tiles + tile_map) work on vector_grid and tree_hierarchy?
**Motivation:** exp10g validated D on scalar 2D grids. Need cross-space validation.
**Kill criteria:** Dual contour (A: architectural, B: operational) per space type
**Roadmap level:** P0 (0.9b3)
**Status:** tree FAIL (0/108), vector_grid pending

## Results — tree_hierarchy (19 March 2026)

**FAIL 0/108** — D_direct with per-level packed tiles never beats A_bitset.

### Sweep
- branching: 2, 4, 8
- depth: 3, 4
- features: 4, 8, 16
- sparsity (occupancy): 0.1–0.7

### Numbers
- resident ratio: 1.16–1.33x (D always heavier than grid)
- time overhead: +3% to +19%

### Diagnosis

Tree FAIL is **not** an architecture rejection. It is "current test configs too small
for D_direct to break even."

Per-level tile_map overhead is not amortized because N_l (nodes per level) is too small
in these configs. The tile_map costs O(N_l) int32 regardless of occupancy, so at small
N_l the bookkeeping dominates any packing savings.

**Break-even exists** when:
1. N_l is large enough (hundreds to thousands of nodes per level), AND
2. p_l (occupancy at that level) is low enough that packed storage << full allocation

Upper levels of a tree almost certainly stay A_bitset (small N_l, high occupancy).
The interesting regime is wide lower levels where N_l is large and occupancy can be low.

**Per-level analysis needed:** D_direct for trees requires per-level cost analysis,
not a global pass/fail. The correct question is "at which level does D break even?"
rather than "does D work for the whole tree?"

**Next:** exp10j — per-level independent benchmark with wider sweeps (branching up to 32,
bottom level >= 1024 nodes, occupancy down to 0.01).

**See also:** exp10i_graph_blocks tested block-based addressing for irregular graphs.
Result: spatial graphs (random_geometric, grid_graph) are viable with spatial/greedy
partitioning (cbr ~0.18-0.25), but scale-free graphs (barabasi_albert) fail (cbr ~0.66).
This confirms that D_direct's tile-grid assumption extends to spatial graph topologies
but not to arbitrary graph structures.

## Space types
- vector_grid: tiles[k, Ht, Wt, C] with C>1, same tile_map
- tree_hierarchy: per-level tile_map[L][node_id] -> slot, per-level packed_data[L]

## Expected outcome
- vector_grid: should work identically to scalar (just more channels)
- tree: per-level tile_map should save memory when tree is sparse at each level
