# exp12a_tau_parent

**Question:** What are the data-driven thresholds tau_parent[L, space_type]?
**Kill criteria:** accuracy drops >15pp on held-out seeds
**Roadmap level:** SC-5
**Status:** PASSED

## Background

SC-baseline (SC-0 through SC-4) has been completed and PASSED. D_parent (lf_frac
variant) achieves AUC 0.824-1.000 across all 4 space types. This experiment
determines the actual enforcement thresholds tau_parent[L, space_type] per depth
level L and per space type, using data-driven methods.

### Why per-space thresholds?

R/Up operators produce D_parent with different dynamic ranges per space type:
- Grids (T1, T2): D_parent values ~0.42-0.50
- Graph (T3): D_parent values as low as ~0.078
- Tree (T4): D_parent values ~0.19

A single global threshold trained on 3 spaces fails on the held-out 4th space
(e.g., L1 specificity=0.25 globally, T4_tree validation specificity=0.0).
Per-space thresholds align with the layout selection policy
(docs/layout_selection_policy.md): space_type is known statically, R/Up operators
are already different per space type, so thresholds should be per-space too.

## Design

### Threshold methods

Three methods are evaluated and the best is selected automatically:

| Method | Description |
|---|---|
| `youden_j` | ROC optimal point: maximizes sensitivity + specificity - 1 |
| `f1_optimal` | Threshold that maximizes F1 score |
| `sensitivity_at_90` | Lowest threshold achieving >=90% sensitivity |

### Cross-validation

Per-space, seed-based (train seeds vs held-out seeds):
- For each space type independently: compute thresholds on training seeds, validate on held-out seeds
- 10 training seeds + 10 validation seeds per space (configurable)
- Kill criterion: accuracy drops >15 percentage points on held-out seeds vs training

### Space types

| Code | Type | R operator | Up operator |
|---|---|---|---|
| T1 | Scalar grid 64x64 | Gaussian blur sigma=3.0 + decimation | Bilinear upsampling |
| T2 | Vector grid 64x64 dim=32 | Per-channel Gaussian + decimation | Per-channel bilinear |
| T3 | Irregular graph 500 pts k=8 | Cluster-mean pooling | Scatter-back |
| T4 | Tree hierarchy depth=8 | Subtree-mean | Broadcast |

### Depth levels

Depth is simulated differently per space type:
- Grids: tile size variation (L1=16, L2=8, L3=4)
- Graph: cluster count variation (L1=10, L2=25, L3=50)
- Tree: coarse_depth variation (L1=2, L2=4, L3=6)

### Statistical testing

- Mann-Whitney U per level per space (neg > pos)
- Holm-Bonferroni correction for multiple comparisons (alpha=0.05)

## Results

**Best method:** youden_j (mean CV accuracy: 0.9139)
**Kill criterion:** PASSED (no space exceeds 15pp drop; max drop = 5.6pp)

### Recommended thresholds (per-space, per-level)

| Key | tau_parent | Accuracy | Sensitivity | Specificity | F1 |
|---|---|---|---|---|---|
| T1_scalar_L1 | 0.4595 | 1.000 | 1.000 | 1.000 | 1.000 |
| T1_scalar_L2 | 0.3697 | 1.000 | 1.000 | 1.000 | 1.000 |
| T1_scalar_L3 | 0.3673 | 1.000 | 1.000 | 1.000 | 1.000 |
| T2_vector_L1 | 0.4192 | 0.822 | 0.733 | 1.000 | 0.846 |
| T2_vector_L2 | 0.2487 | 1.000 | 1.000 | 1.000 | 1.000 |
| T2_vector_L3 | 0.1848 | 1.000 | 1.000 | 1.000 | 1.000 |
| T3_graph_L1 | 0.0784 | 1.000 | 1.000 | 1.000 | 1.000 |
| T3_graph_L2 | 0.1431 | 1.000 | 1.000 | 1.000 | 1.000 |
| T3_graph_L3 | 0.2535 | 1.000 | 1.000 | 1.000 | 1.000 |
| T4_tree_L1 | 0.1884 | 0.767 | 0.650 | 1.000 | 0.788 |
| T4_tree_L2 | 0.3326 | 0.789 | 0.733 | 0.900 | 0.822 |
| T4_tree_L3 | 0.5635 | 0.800 | 0.700 | 1.000 | 0.824 |

### Cross-validation (per-space seed CV, youden_j)

| Space | L1 drop | L2 drop | L3 drop | Kill? |
|---|---|---|---|---|
| T1_scalar | 2.2pp | 2.2pp | 1.1pp | NO |
| T2_vector | 0.0pp | 3.3pp | 1.1pp | NO |
| T3_graph | 1.1pp | 0.0pp | 2.2pp | NO |
| T4_tree | 1.1pp | 2.2pp | 4.4pp | NO |

### Statistical significance

All 12 tests (4 spaces x 3 levels) reject H0 at alpha=0.05 after Holm-Bonferroni
correction (all p < 3.3e-06).

## Outputs

| File | Description |
|---|---|
| `exp12a_results.json` | Full results: thresholds per method per level per space, CV, significance |
| `exp12a_thresholds.json` | Recommended thresholds (best method, per space per level) |
| `exp12a_roc_curves.png` | ROC curves per level, overlay all spaces |

## Usage

```bash
cd experiments/exp12a_tau_parent
python exp12a_tau_parent.py [--output-dir DIR] [--n-train-seeds 10] [--n-val-seeds 10] [--base-seed 42]
```

## Dependencies

- numpy, scipy, scikit-learn, matplotlib
- `experiments/sc_baseline/` (operators, baselines_v2)
