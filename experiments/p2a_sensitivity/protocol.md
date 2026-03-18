# P2a — Sensitivity Sweep: Two-Stage Gate Thresholds

Roadmap level: P2 (experiment_hierarchy.md, step 4, future exp12)

## Question

Are the manually-set thresholds of the two-stage gate (instability_threshold=0.25, FSR_threshold=0.20) robust across scenes and space types, or do they require auto-tuning?

## Hypothesis

If the performance ridge is wide (>30% of parameter space within 5% of optimal), manual thresholds are sufficient and P2b (auto-tuning) is unnecessary.

## Design

### Independent variables

- **instability_threshold**: 10 values, 0.1x to 10x of default (0.25), log-spaced
- **FSR_threshold**: 10 values, 0.1x to 10x of default (0.20), log-spaced
- **Scene type** (5): clean, noise, blur, spatial_variation, jpeg_artifacts
- **Space type** (4): scalar_grid, vector_grid, irregular_graph, tree_hierarchy
- **Seed** (10): for statistical power

Total configurations: 10 x 10 x 5 x 4 x 10 = 20,000 runs.

### Dependent variables

Per configuration (averaged over seeds):
- **Quality**: PSNR (scalar/vector grid) or analogous MSE-based metric (graph/tree)
- **SeamScore**: average seam quality at refined boundaries
- **Cost**: number of units refined

### Procedure

For each (scene, space, instab_thresh, fsr_thresh, seed):
1. Generate scene field (64x64 base)
2. Initialize space adapter, compute coarse approximation
3. Run probe diagnostics: compute FSR and instability on 15% probe subset
4. Apply two-stage gate decision with given thresholds
5. Select and refine top-error units up to budget
6. Measure quality, seam score, cost

### Ridge width computation

- **1D ridge**: for each threshold axis, find the contiguous range where quality stays within 5% of optimal. Report as fraction of log-swept range.
- **2D ridge**: fraction of (instab x fsr) grid cells within 5% of optimal.

## Kill criteria

| Ridge width (2D) | Verdict | Action |
|---|---|---|
| > 30% of cells | PASS (wide ridge) | Manual thresholds ok. P2b not needed. |
| 10%--30% | INCONCLUSIVE | Consider targeted P2b for narrow cases. |
| < 10% | FAIL (narrow ridge) | P2b (auto-tuning) IS needed. |

Applied per (scene x space) pair. Overall verdict by majority:
- >= 60% PASS and 0 FAIL -> MANUAL_OK
- >= 30% FAIL -> P2B_NEEDED
- Otherwise -> INCONCLUSIVE

### Cross-space divergence check

If ridge width differs by >15 percentage points between space types for the same scene, this is flagged as an **architectural signal**: thresholds may need to be space-aware, not just scene-aware.

## Expected outputs

1. `results/p2a_summary.json` — ridge widths, verdicts, divergence flags
2. `results/p2a_full_results.json` — full quality/seam/cost grids
3. `results/p2a_heatmap_{scene}.png` — quality heatmaps per scene (4 spaces each)
4. `results/p2a_ridge_comparison.png` — bar chart comparing ridge widths

## Controls

- 10 seeds per configuration (Wilcoxon-level statistical power)
- Log-spaced sweep covers 2 orders of magnitude (0.1x to 10x)
- Default thresholds at grid center for direct comparison
- SeamScore measured independently of quality to detect threshold regimes where quality is ok but seam artifacts appear

## Dependencies

- No GPU required (CPU-only numpy/scipy/matplotlib)
- Self-contained: does not import from exp07b (reimplements gate logic to avoid path issues)
- Reuses space type concepts from phase2_probe_seam but with fresh adapters

## Status

Open.
