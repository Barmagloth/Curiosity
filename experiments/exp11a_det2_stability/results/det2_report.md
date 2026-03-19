# exp11a -- Cross-Seed Stability (DET-2) Report

**Verdict:** FAIL
**CV threshold:** 0.1
**Seeds:** 20 (0..19)
**Budgets:** [('low', 0.1), ('high', 0.3)]
**Spaces:** ['scalar_grid', 'vector_grid', 'irregular_graph', 'tree_hierarchy']
**Total runs:** 160
**Elapsed:** 1.2s

## Summary table

| Space | Budget | Pass | max CV | Failing metrics |
|-------|--------|------|--------|-----------------|
| scalar_grid | low | FAIL | 0.1729 | mean_leaf_value |
| scalar_grid | high | FAIL | 0.1736 | mean_leaf_value |
| vector_grid | low | PASS | 0.0908 | -- |
| vector_grid | high | PASS | 0.0908 | -- |
| irregular_graph | low | PASS | 0.0650 | -- |
| irregular_graph | high | FAIL | 0.2076 | total_cost, n_splits, max_depth, compliance, n_boundary_nodes |
| tree_hierarchy | low | PASS | 0.0398 | -- |
| tree_hierarchy | high | FAIL | 0.2474 | total_cost, n_splits, max_depth, compliance |

## Per-cell detail

### scalar_grid / low -- FAIL

| Metric | Mean | Std | CV |
|--------|------|-----|-----|
| total_cost | 2.0000 | 0.0000 | 0.0000 |
| n_splits | 2.0000 | 0.0000 | 0.0000 |
| max_depth | 2.0000 | 0.0000 | 0.0000 |
| compliance | 0.3333 | 0.0000 | 0.0000 |
| tree_size | 64.0000 | 0.0000 | 0.0000 |
| mean_leaf_value | 0.5680 | 0.0982 | 0.1729 **FAIL** |
| n_boundary_nodes | 1.0000 | 0.0000 | 0.0000 |

### scalar_grid / high -- FAIL

| Metric | Mean | Std | CV |
|--------|------|-----|-----|
| total_cost | 6.0000 | 0.0000 | 0.0000 |
| n_splits | 6.0000 | 0.0000 | 0.0000 |
| max_depth | 6.0000 | 0.0000 | 0.0000 |
| compliance | 0.3158 | 0.0000 | 0.0000 |
| tree_size | 64.0000 | 0.0000 | 0.0000 |
| mean_leaf_value | 0.5689 | 0.0988 | 0.1736 **FAIL** |
| n_boundary_nodes | 1.0000 | 0.0000 | 0.0000 |

### vector_grid / low -- PASS

| Metric | Mean | Std | CV |
|--------|------|-----|-----|
| total_cost | 1.0000 | 0.0000 | 0.0000 |
| n_splits | 1.0000 | 0.0000 | 0.0000 |
| max_depth | 1.0000 | 0.0000 | 0.0000 |
| compliance | 1.0000 | 0.0000 | 0.0000 |
| tree_size | 16.0000 | 0.0000 | 0.0000 |
| mean_leaf_value | 0.0226 | 0.0021 | 0.0908 |
| n_boundary_nodes | 1.0000 | 0.0000 | 0.0000 |

### vector_grid / high -- PASS

| Metric | Mean | Std | CV |
|--------|------|-----|-----|
| total_cost | 1.0000 | 0.0000 | 0.0000 |
| n_splits | 1.0000 | 0.0000 | 0.0000 |
| max_depth | 1.0000 | 0.0000 | 0.0000 |
| compliance | 0.2500 | 0.0000 | 0.0000 |
| tree_size | 16.0000 | 0.0000 | 0.0000 |
| mean_leaf_value | 0.0226 | 0.0021 | 0.0908 |
| n_boundary_nodes | 1.0000 | 0.0000 | 0.0000 |

### irregular_graph / low -- PASS

| Metric | Mean | Std | CV |
|--------|------|-----|-----|
| total_cost | 1.0000 | 0.0000 | 0.0000 |
| n_splits | 1.0000 | 0.0000 | 0.0000 |
| max_depth | 1.0000 | 0.0000 | 0.0000 |
| compliance | 1.0000 | 0.0000 | 0.0000 |
| tree_size | 10.0000 | 0.0000 | 0.0000 |
| mean_leaf_value | 0.4513 | 0.0293 | 0.0650 |
| n_boundary_nodes | 1.0000 | 0.0000 | 0.0000 |

### irregular_graph / high -- FAIL

| Metric | Mean | Std | CV |
|--------|------|-----|-----|
| total_cost | 2.1000 | 0.3000 | 0.1429 **FAIL** |
| n_splits | 2.1000 | 0.3000 | 0.1429 **FAIL** |
| max_depth | 2.1500 | 0.3571 | 0.1661 **FAIL** |
| compliance | 0.7000 | 0.1000 | 0.1429 **FAIL** |
| tree_size | 10.0000 | 0.0000 | 0.0000 |
| mean_leaf_value | 0.4641 | 0.0236 | 0.0509 |
| n_boundary_nodes | 1.0500 | 0.2179 | 0.2076 **FAIL** |

### tree_hierarchy / low -- PASS

| Metric | Mean | Std | CV |
|--------|------|-----|-----|
| total_cost | 1.0000 | 0.0000 | 0.0000 |
| n_splits | 1.0000 | 0.0000 | 0.0000 |
| max_depth | 1.0000 | 0.0000 | 0.0000 |
| compliance | 1.0000 | 0.0000 | 0.0000 |
| tree_size | 8.0000 | 0.0000 | 0.0000 |
| mean_leaf_value | 0.1662 | 0.0066 | 0.0398 |
| n_boundary_nodes | 1.0000 | 0.0000 | 0.0000 |

### tree_hierarchy / high -- FAIL

| Metric | Mean | Std | CV |
|--------|------|-----|-----|
| total_cost | 1.7500 | 0.4330 | 0.2474 **FAIL** |
| n_splits | 1.7500 | 0.4330 | 0.2474 **FAIL** |
| max_depth | 1.7500 | 0.4330 | 0.2474 **FAIL** |
| compliance | 0.8750 | 0.2165 | 0.2474 **FAIL** |
| tree_size | 8.0000 | 0.0000 | 0.0000 |
| mean_leaf_value | 0.1820 | 0.0112 | 0.0614 |
| n_boundary_nodes | 1.0000 | 0.0000 | 0.0000 |

## Kill criterion

CV > 0.1 for ANY metric in ANY (space, budget) cell = FAIL.

## Methodology

Reuses the AdaptivePipeline from exp10d (DET-1). Each of 160 runs uses
a unique seed to initialize both the space and the pipeline. Metrics are
collected per run and aggregated per (space, budget) cell. The coefficient
of variation (CV = std/mean) measures relative spread across seeds.
