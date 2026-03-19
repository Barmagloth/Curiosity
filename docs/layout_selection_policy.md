# Layout Selection Policy v1.0

**Status:** APPROVED (exp10 series, 19 March 2026)
**Applies to:** all refinement operations in Curiosity
**Evidence:** exp10 through exp10j, 158 000+ trials, RTX 2070 8GB VRAM

---

## 1. Glossary

| Term | Full name | What it is |
|------|-----------|------------|
| **D_direct** | Packed tiles + direct tile_map | Active tiles stored in compact array `tiles[k, ...]`. Lookup via `tile_map[tile_id] → slot` (int32, -1 = inactive). O(1) addressing without element-level reverse index. |
| **D_direct_per_level** | Per-level packed tiles | Same as D_direct, but tile_map built independently for each tree level. Level l has its own `tile_map_l[N_l]` and `tiles_l[k_l, ...]`. |
| **D_blocked** | Graph block addressing | Graph nodes partitioned into fixed-size blocks. `block_map[block_id] → slot`. Nodes within a block stored densely; cross-block edges handled separately. |
| **A_bitset** | Dense grid + bitset mask | Full-size data tensor over entire universe. Activation tracked by bitset (1 bit per element). No indirection. Simple, no metadata overhead. |
| **E_hash** | Hash table lookup | Open-addressing hash table for tile_id → slot. Archived fallback; dominated by D_direct at bounded regular domains. |

## 2. Metrics

All layout decisions require these measurements:

| Metric | Symbol | Definition |
|--------|--------|------------|
| Occupancy | `p` | Fraction of active tiles: `k / N` where k = active, N = universe size |
| Per-level occupancy | `p_l` | Occupancy at tree level l: `k_l / N_l` |
| Cross-block ratio | `cbr` | Fraction of graph edges that cross block boundaries. Range [0, 1]. Lower = better partitioning. |
| Padding waste | `pw` | Fraction of packed block storage occupied by padding (inactive slots within active blocks). |
| Resident memory | | Persistent layout storage after build, before compute (bytes). |
| Peak step memory | | Maximum GPU allocation during compute step (bytes). |
| Workspace overhead | | `peak_step - resident`. Operator temporary allocations. |

## 3. Decision procedure

```
FUNCTION select_layout(space_type, operator_weight, geometry) → layout

INPUT:
  space_type      ∈ {scalar_grid, vector_grid, tree_hierarchy, irregular_graph}
  operator_weight ∈ {light, heavy}
    light = local gather/scatter, stencil, element-wise ops
    heavy = conv2d, matmul, multi-step fused kernels
  geometry:
    p or p_l          — occupancy (scalar or per-level)
    has_spatial_structure — boolean (for graphs)
    cbr               — cross-block ratio (for graphs, after partitioning)
    pw                — padding waste (for graphs)

OUTPUT:
  layout ∈ {D_direct, D_direct_per_level, D_blocked, A_bitset}
```

### Step 1 — Classify space

```
if space_type ∈ {scalar_grid, vector_grid}:
    goto RULE_GRID

if space_type == tree_hierarchy:
    goto RULE_TREE

if space_type == irregular_graph:
    goto RULE_GRAPH
```

### Step 2a — RULE_GRID (regular grids)

```
RULE_GRID:
    return D_direct
    [fallback: A_bitset if tile_map overhead > data at very small grids]
```

Rationale: D_direct passed both contours (architectural + operational) on scalar
and vector grids across all tested sizes and occupancies. No conditions.

### Step 2b — RULE_TREE (tree hierarchy)

Decision is **per-level**, not global.

```
RULE_TREE:
    for each level l in tree:
        compute p_l = k_l / N_l

        if operator_weight == heavy AND p_l < 0.40:
            level_layout[l] = D_direct_per_level
        else:
            level_layout[l] = A_bitset

    return level_layout
```

Rationale:
- Heavy operator + low occupancy: D_direct saves ~48% resident memory and wins
  on compute time 59% of trials. Break-even threshold p* ≈ 0.375–0.40, stable
  across all N_l from 2 to 4096 and all branching factors 2–32.
- Light operator (stencil): D_direct saves memory below same threshold but is
  always 3.3x SLOWER due to tile_map indirection. Never use D_direct for stencil.
- Activation pattern (random/clustered/frontier) has negligible effect on threshold.

### Step 2c — RULE_GRAPH (irregular graphs)

```
RULE_GRAPH:
    if NOT has_spatial_structure:
        return A_bitset                    # hub/scale-free: blocks rejected

    # Spatial graph — attempt block addressing
    partition = spatial_partition(graph)    # Morton order or BFS-locality
    block_size = 8                         # prefer smallest viable block
    compute cbr, pw from partition

    if cbr > 0.50:
        return A_bitset                    # hard reject: too many cross-edges

    if cbr <= 0.35 AND pw < 0.50:
        return D_blocked                   # acceptable block quality

    if 0.35 < cbr <= 0.50:
        return A_bitset                    # marginal zone: default to safe

    return A_bitset                        # fallback
```

Rationale:
- Spatial graphs (random_geometric, grid_graph): spatial partition achieves
  cbr ≈ 0.20–0.31. Block addressing viable with small blocks.
- Scale-free graphs (barabási-albert): best cbr = 0.66 with greedy partition.
  Hub nodes blow edges across all block boundaries. Structurally incompatible.
- Padding waste is universally high (~0.77 mean) with fixed-size blocks at
  low occupancy. This is an inherent limitation of fixed block_size.

## 4. Summary matrix

```
┌─────────────────────────────┬──────────────────────┬────────────┐
│ Space type                  │ Default layout       │ Conditions │
├─────────────────────────────┼──────────────────────┼────────────┤
│ scalar_grid                 │ D_direct             │ none       │
│ vector_grid                 │ D_direct             │ none       │
│ tree_hierarchy (heavy op)   │ D_direct_per_level   │ p_l < 0.40 │
│ tree_hierarchy (light op)   │ A_bitset             │ always     │
│ tree_hierarchy (high occ.)  │ A_bitset             │ p_l ≥ 0.40 │
│ irregular_graph (spatial)   │ D_blocked            │ cbr ≤ 0.35 │
│ irregular_graph (scale-free)│ A_bitset             │ always     │
└─────────────────────────────┴──────────────────────┴────────────┘

Universal fallback: A_bitset (always safe, never optimal)
```

## 5. Thresholds

| Parameter | Threshold | Action |
|-----------|-----------|--------|
| `p_l` (trees, heavy op) | < 0.40 | Use D_direct_per_level |
| `p_l` (trees, heavy op) | ≥ 0.40 | Use A_bitset |
| `p_l` (trees, light op) | any | Use A_bitset |
| `cbr` (spatial graphs) | ≤ 0.35 | D_blocked acceptable |
| `cbr` (spatial graphs) | 0.35–0.50 | Marginal; prefer A_bitset |
| `cbr` (any graph) | > 0.50 | Hard reject D_blocked |
| `pw` (graphs) | > 0.50 | Warning flag; reconsider block_size |

## 6. Permanently rejected approaches

| Approach | Reason | Evidence |
|----------|--------|----------|
| Element-level reverse_map[M] | int32 per element = VRAM +38.6% | exp10 |
| Binary search lookup on GPU | +1700% runtime; GPU hates branchy code | exp10e |
| Paged sparse tiles | Page machinery overhead +9000% | exp10e |
| Hash table as primary lookup | Same speed as direct, build 10-30x slower | exp10f |
| Fixed-size blocks for scale-free graphs | cbr 0.64–0.99; hub nodes break all partitions | exp10i |

## 7. Archived fallbacks (not active, resurrection triggers documented)

| Approach | Resurrect when |
|----------|---------------|
| E_hash (hash table lookup) | tile_map > 25–30% of resident; tile universe sparse/irregular; multi-level global addressing needed |
| Variable-size graph blocks | Spatial graphs need better packing than fixed blocks |
| Graph-native sparse (CSR/COO) | Scale-free graphs need non-fallback layout |

## 8. Hardware specificity

The **form** of this policy is portable: thresholds depend on p, cbr, operator weight.
The **coefficients** (0.40, 0.35) were calibrated on RTX 2070 / CUDA 12.8 / PyTorch 2.10.
On different hardware, recalibrate by running exp10g (grids) and exp10j (trees) sweeps.
The structural conclusions (which approaches work, which are killed) are hardware-independent.
