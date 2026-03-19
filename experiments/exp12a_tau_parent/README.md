# exp12a_tau_parent

**Question:** What are the data-driven thresholds τ_parent[L] per depth level?
**Kill criteria:** accuracy drops >15pp on held-out space
**Roadmap level:** SC-5
**Status:** open

## Background

SC-baseline (SC-0 through SC-4) has been completed and PASSED. D_parent (lf_frac
variant) achieves AUC 0.824-1.000 across all 4 space types. This experiment
determines the actual enforcement thresholds τ_parent[L] per depth level L,
using data-driven methods.

## Design

### Threshold methods

Three methods are evaluated and the best is selected automatically:

| Method | Description |
|---|---|
| `youden_j` | ROC optimal point: maximizes sensitivity + specificity - 1 |
| `f1_optimal` | Threshold that maximizes F1 score |
| `sensitivity_at_90` | Lowest threshold achieving >=90% sensitivity |

### Cross-validation

Leave-one-space-out (4 folds):
- Train on 3 space types, validate on held-out space
- 10 training seeds + 10 validation seeds per fold (configurable)
- Kill criterion: accuracy drops >15 percentage points on held-out vs training

### Space types

| Code | Type | R operator | Up operator |
|---|---|---|---|
| T1 | Scalar grid 64x64 | Gaussian blur σ=3.0 + decimation | Bilinear upsampling |
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

## Outputs

| File | Description |
|---|---|
| `exp12a_results.json` | Full results: thresholds per method per level per space, CV, significance |
| `exp12a_thresholds.json` | Recommended thresholds (best method, per level) |
| `exp12a_roc_curves.png` | ROC curves per level, overlay all spaces |

## Usage

```bash
cd experiments/exp12a_tau_parent
python exp12a_tau_parent.py [--output-dir DIR] [--n-train-seeds 10] [--n-val-seeds 10] [--base-seed 42]
```

## Dependencies

- numpy, scipy, scikit-learn, matplotlib
- `experiments/sc_baseline/` (operators, baselines_v2)
