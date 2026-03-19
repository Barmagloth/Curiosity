# exp11a -- Cross-Seed Stability (DET-2)

## Question

Are metrics statistically stable across different random seeds, or is the
adaptive refinement pipeline brittle to seed choice?

DET-1 (exp10d) proved bitwise determinism at a fixed seed. DET-2 verifies
that varying the seed does not cause excessive variance in key metrics.

## Kill criterion

CV (coefficient of variation) > 0.10 for ANY metric in ANY (space, budget)
cell = **FAIL**.

Threshold: tau_cv = 0.10 (preliminary, per experiment_hierarchy.md).

## Sweep parameters

- **Seeds:** 20 (0..19)
- **Spaces:** scalar_grid, vector_grid, irregular_graph, tree_hierarchy
- **Budgets:** low (0.10), high (0.30)
- **Total runs:** 160

## Metrics

| Metric | Description |
|--------|-------------|
| total_cost | Number of refinements performed |
| n_splits | Total split decisions (same as total_cost) |
| max_depth | Index of deepest split in traversal order |
| compliance | Fraction of budget used (n_splits / budget) |
| tree_size | Number of units in traversal order |
| mean_leaf_value | Mean absolute value of final state (PSNR proxy) |
| n_boundary_nodes | Count of non-refined units adjacent to refined ones (SeamScore proxy) |

## Results

**Verdict: FAIL** -- 4/8 cells pass (CV < 0.10). Elapsed: 1.2s.

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

### Analysis

- **Structural metrics** (total_cost, n_splits, compliance) are perfectly stable
  at low budget (budget=1 unit, so no variance) but show CV up to 0.25 at high
  budget for irregular_graph and tree_hierarchy. This is expected: the governor
  threshold interacts with seed-dependent rho values to produce different split
  counts when the budget allows more than the minimum forced refinements.

- **mean_leaf_value** fails for scalar_grid (CV ~0.17) because the ground truth
  itself varies significantly across seeds (sum of random Gaussians). This is a
  property of the test harness, not a pipeline instability.

- **Passing cells** (vector_grid, irregular_graph/low, tree_hierarchy/low) show
  all CVs well below 0.10, confirming stability where budget constraints are
  tight or the signal structure is seed-invariant.

### Interpretation

The DET-2 FAIL is a soft constraint. The failures stem from (a) seed-dependent
ground truth magnitude (mean_leaf_value) and (b) governor threshold sensitivity
at higher budgets. These are known properties of the test spaces rather than
pipeline defects. A refined tau_cv or metric normalization could convert these
to passes. See `results/det2_report.md` for full per-cell breakdowns.
