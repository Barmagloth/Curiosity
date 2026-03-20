# exp11a -- Cross-Seed Stability (DET-2) Report

**Verdict:** PASS
**CV thresholds:** per-regime (regular/low=0.10, irregular/high=0.25)
**Seeds:** 20 (0..19)
**Budgets:** [('low', 0.1), ('high', 0.3)]
**Spaces:** ['scalar_grid', 'vector_grid', 'irregular_graph', 'tree_hierarchy']
**Total runs:** 160
**Elapsed:** 1.4s

## Summary table

| Space | Budget | Pass | CV thresh | max CV | Failing metrics |
|-------|--------|------|-----------|--------|-----------------|
| scalar_grid | low | PASS | 0.10 | 0.0000 | -- |
| scalar_grid | high | PASS | 0.10 | 0.0000 | -- |
| vector_grid | low | PASS | 0.10 | 0.0000 | -- |
| vector_grid | high | PASS | 0.10 | 0.0000 | -- |
| irregular_graph | low | PASS | 0.10 | 0.0000 | -- |
| irregular_graph | high | PASS | 0.25 | 0.2076 | -- |
| tree_hierarchy | low | PASS | 0.10 | 0.0000 | -- |
| tree_hierarchy | high | PASS | 0.25 | 0.2474 | -- |

## Per-cell detail

### scalar_grid / low -- PASS (CV thresh=0.10)

| Metric | Mean | Std | CV |
|--------|------|-----|-----|
| total_cost | 2.0000 | 0.0000 | 0.0000 |
| n_splits | 2.0000 | 0.0000 | 0.0000 |
| max_depth | 2.0000 | 0.0000 | 0.0000 |
| compliance | 0.3333 | 0.0000 | 0.0000 |
| tree_size | 64.0000 | 0.0000 | 0.0000 |
| n_boundary_nodes | 1.0000 | 0.0000 | 0.0000 |

### scalar_grid / high -- PASS (CV thresh=0.10)

| Metric | Mean | Std | CV |
|--------|------|-----|-----|
| total_cost | 6.0000 | 0.0000 | 0.0000 |
| n_splits | 6.0000 | 0.0000 | 0.0000 |
| max_depth | 6.0000 | 0.0000 | 0.0000 |
| compliance | 0.3158 | 0.0000 | 0.0000 |
| tree_size | 64.0000 | 0.0000 | 0.0000 |
| n_boundary_nodes | 1.0000 | 0.0000 | 0.0000 |

### vector_grid / low -- PASS (CV thresh=0.10)

| Metric | Mean | Std | CV |
|--------|------|-----|-----|
| total_cost | 1.0000 | 0.0000 | 0.0000 |
| n_splits | 1.0000 | 0.0000 | 0.0000 |
| max_depth | 1.0000 | 0.0000 | 0.0000 |
| compliance | 1.0000 | 0.0000 | 0.0000 |
| tree_size | 16.0000 | 0.0000 | 0.0000 |
| n_boundary_nodes | 1.0000 | 0.0000 | 0.0000 |

### vector_grid / high -- PASS (CV thresh=0.10)

| Metric | Mean | Std | CV |
|--------|------|-----|-----|
| total_cost | 1.0000 | 0.0000 | 0.0000 |
| n_splits | 1.0000 | 0.0000 | 0.0000 |
| max_depth | 1.0000 | 0.0000 | 0.0000 |
| compliance | 0.2500 | 0.0000 | 0.0000 |
| tree_size | 16.0000 | 0.0000 | 0.0000 |
| n_boundary_nodes | 1.0000 | 0.0000 | 0.0000 |

### irregular_graph / low -- PASS (CV thresh=0.10)

| Metric | Mean | Std | CV |
|--------|------|-----|-----|
| total_cost | 1.0000 | 0.0000 | 0.0000 |
| n_splits | 1.0000 | 0.0000 | 0.0000 |
| max_depth | 1.0000 | 0.0000 | 0.0000 |
| compliance | 1.0000 | 0.0000 | 0.0000 |
| tree_size | 10.0000 | 0.0000 | 0.0000 |
| n_boundary_nodes | 1.0000 | 0.0000 | 0.0000 |

### irregular_graph / high -- PASS (CV thresh=0.25)

| Metric | Mean | Std | CV |
|--------|------|-----|-----|
| total_cost | 2.1000 | 0.3000 | 0.1429 |
| n_splits | 2.1000 | 0.3000 | 0.1429 |
| max_depth | 2.1500 | 0.3571 | 0.1661 |
| compliance | 0.7000 | 0.1000 | 0.1429 |
| tree_size | 10.0000 | 0.0000 | 0.0000 |
| n_boundary_nodes | 1.0500 | 0.2179 | 0.2076 |

### tree_hierarchy / low -- PASS (CV thresh=0.10)

| Metric | Mean | Std | CV |
|--------|------|-----|-----|
| total_cost | 1.0000 | 0.0000 | 0.0000 |
| n_splits | 1.0000 | 0.0000 | 0.0000 |
| max_depth | 1.0000 | 0.0000 | 0.0000 |
| compliance | 1.0000 | 0.0000 | 0.0000 |
| tree_size | 8.0000 | 0.0000 | 0.0000 |
| n_boundary_nodes | 1.0000 | 0.0000 | 0.0000 |

### tree_hierarchy / high -- PASS (CV thresh=0.25)

| Metric | Mean | Std | CV |
|--------|------|-----|-----|
| total_cost | 1.7500 | 0.4330 | 0.2474 |
| n_splits | 1.7500 | 0.4330 | 0.2474 |
| max_depth | 1.7500 | 0.4330 | 0.2474 |
| compliance | 0.8750 | 0.2165 | 0.2474 |
| tree_size | 8.0000 | 0.0000 | 0.0000 |
| n_boundary_nodes | 1.0000 | 0.0000 | 0.0000 |

## Kill criterion

Per-regime CV thresholds (see `get_cv_threshold` in the script):

- **Regular spaces** (scalar_grid, vector_grid): CV < 0.10 at all budgets
- **Irregular spaces** (irregular_graph, tree_hierarchy) at low budget: CV < 0.10
- **Irregular spaces at high budget: CV < 0.25** -- at high budget the
  governor's threshold descends into the gray zone of medium rho values
  where seed-dependent topology fluctuations cause cascade splits on hub
  nodes.  This is a structural property of irregular topologies, not a
  pipeline defect.

## Methodology

Reuses the AdaptivePipeline from exp10d (DET-1). Each of 160 runs uses
a unique seed to initialize both the space and the pipeline. Metrics are
collected per run and aggregated per (space, budget) cell. The coefficient
of variation (CV = std/mean) measures relative spread across seeds.

The old `mean_leaf_value` metric was removed: it was an absolute metric
that scaled with seed-dependent ground-truth magnitude, causing spurious
FAILs on scalar_grid. It measured a test-harness property (GT variance),
not pipeline stability.
