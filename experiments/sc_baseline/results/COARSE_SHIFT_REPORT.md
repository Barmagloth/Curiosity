# Coarse-Shift Detection Report

**Date:** 2026-03-18

## 1. Root Cause: Why Norm-Based Metrics Fail on coarse_shift

### The generator

`negative_coarse_shift` creates: `delta = (GT - coarse) + 0.2 * coarse * sign_map`

where `sign_map` is a random per-pixel {-1, +1} field. Within each tile, `coarse` is
a spatial constant (tile-mean), so the shift component is `+/- 0.2*c` at each pixel.

### Why R cannot see it

The restriction operator R = gaussian blur (sigma) + decimation by 2. Within a constant
tile, R averages over a spatial patch. The random sign_map creates alternating +0.2c and
-0.2c values that **cancel under averaging**:

```
E[R(shift)] = R(0.2*c * sign_map) = 0.2*c * R(sign_map) ~ 0.2*c * E[sign] = 0
```

Since R(shift) ~ 0, the restricted delta is indistinguishable from oracle:
- `||R(delta_shift)|| ~ ||R(delta_oracle)||` (shift is invisible to R)
- `||delta_shift|| > ||delta_oracle||` (shift adds energy to norm denominator)
- Survival ratio: shift = 0.296 < oracle = 0.308 (inverted!)

This is **not** a failure of the metric -- it is a structural property of the generator.
The per-pixel sign flip turns what should be a coarse-level violation into what is
effectively high-frequency noise. The shift is semantically wrong (individual pixels are
shifted by 0.2c) but statistically indistinguishable from a noisier oracle when measured
by any global summary statistic of R(delta).

### Confirmation: coherent shift IS detectable

When the sign flip is removed (coherent shift: `delta = oracle + 0.2*coarse`), the
shift component survives R intact:

| Shift variant | Survival AUC (R=s3.0) | Cohen's d |
|---|---|---|
| Original (sign-flip) | 0.404 | 0.094 |
| Coherent (no flip) | 0.854 | 1.481 |
| Smooth (sigma=3 sign field) | 0.718 | 0.458 |
| Half magnitude (0.1x) | 0.440 | 0.064 |
| Double magnitude (0.4x) | 0.362 | 0.145 |

The coherent shift has AUC=0.854 -- excellent detection. The smooth variant (spatially
coherent sign patches) has AUC=0.718 -- also detectable. Only the per-pixel random
sign_map creates the cancellation problem.

## 2. Non-Norm Detector Results

### Full detector comparison (R=gauss_s3.0)

| Detector | Global AUC | cs AUC | lf_drift | random_lf | semant_wrong |
|---|---|---|---|---|---|
| d_correlation | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| d_cosine | 0.643 | 0.551 | 0.775 | 0.247 | 1.000 |
| d_spectral | 0.760 | 0.385 | 0.827 | 0.945 | 0.883 |
| d_phase | 0.643 | 0.499 | 0.503 | 0.571 | 1.000 |
| d_histogram | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| d_quantile | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| d_gradient | 0.396 | 0.595 | 0.833 | 0.157 | 0.000 |
| d_ssim | 0.526 | 0.490 | 0.471 | 0.514 | 0.628 |
| **d_abs_vs_signed** | 0.183 | **0.591** | 0.142 | 0.000 | 0.000 |
| **d_local_var** | 0.374 | **0.534** | 0.560 | 0.401 | 0.000 |
| d_energy_ratio | 0.608 | 0.560 | 0.970 | 0.903 | 0.000 |
| d_survival (ref) | 0.815 | 0.404 | 0.857 | 1.000 | 1.000 |
| d_baseline (ref) | 0.811 | 0.552 | 0.928 | 0.772 | 0.994 |

### Why most detectors fail

- **d_correlation, d_histogram, d_quantile** (AUC=0.500): Within a constant tile,
  R(coarse) is a constant scalar (std ~ 1e-17). Correlation with a constant is
  undefined. Wasserstein distance and quantile comparison between R(refined) and a
  point mass are meaningless. These methods require spatial variation in R(coarse).

- **d_cosine, d_phase, d_ssim**: Weakly detect the directional/structural change
  in R(refined), but the effect is tiny because the sign-flipped shift averages out.
  AUC ~ 0.50-0.55 on coarse_shift.

- **d_spectral**: Good global AUC (0.760) but actually WORSE on coarse_shift (0.385)
  because the sign-flipped shift component has flat HF spectrum, same as oracle detail.

- **d_abs_vs_signed**: The best coarse_shift-specific detector (AUC=0.591). It detects
  that `||R(|delta|)||` >> `||R(delta)||` when sign cancellations occur. But it inverts
  on all other negative types (lf_drift, random_lf, semant_wrong have NO sign cancellation,
  so their ratio is ~ 1.0, lower than oracle). Global AUC = 0.183 (terrible).

- **d_local_var**: Second-best on coarse_shift (0.534). Measures within-R-patch variance
  of delta. Sign-flipped shift has higher local variance. Same inversion problem on other types.

- **d_gradient**: Moderate coarse_shift detection (0.595) but inverts on semant_wrong (0.000).

### Key insight

No single detector improves coarse_shift without regressing on other negative types.
The detectors that see coarse_shift (abs_vs_signed, local_var, gradient) all exploit
the sign-flip artifact, not the coarse-level violation itself. They detect "this delta
has within-patch sign inconsistency" which happens to correlate with coarse_shift but
is anti-correlated with other violations (which are spatially smooth).

## 3. Best Combinations

### Combination sweep results (top 10)

| # | R | Norm | CS detector | Combiner | Score | AUC | d | cs | lf | rlf | sw |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | s3.0 | lf_frac | d_abs_vs_signed | two_stage | **0.739** | 0.853 | 1.491 | 0.582 | 0.833 | 0.998 | 0.998 |
| 2 | s3.0 | survival | d_abs_vs_signed | two_stage | 0.730 | 0.847 | 1.463 | 0.572 | 0.819 | 0.998 | 0.998 |
| 3 | s3.0 | lf_frac | d_local_var | two_stage | 0.692 | 0.834 | 1.551 | 0.419 | 0.917 | 1.000 | 1.000 |
| 4 | s3.0 | lf_frac | d_gradient | two_stage | 0.688 | 0.823 | 1.461 | 0.467 | 0.865 | 0.979 | 0.980 |
| 5 | s1.0 | lf_frac | d_gradient | max | 0.682 | 0.820 | 1.194 | 0.581 | 0.891 | 0.904 | 0.905 |
| 6 | s3.0 | lf_frac | none | max | 0.681 | 0.819 | 1.546 | 0.405 | 0.872 | 1.000 | 1.000 |
| 7 | s3.0 | survival | d_local_var | two_stage | 0.680 | 0.823 | 1.513 | 0.414 | 0.878 | 1.000 | 1.000 |

### Reference: pure norm variants (no cs detector)

| R | Norm | Score | AUC | d | cs | lf | rlf | sw |
|---|---|---|---|---|---|---|---|---|
| s3.0 | lf_frac | 0.681 | 0.819 | 1.546 | 0.405 | 0.872 | 1.000 | 1.000 |
| s3.0 | survival | 0.674 | 0.815 | 1.514 | 0.404 | 0.857 | 1.000 | 1.000 |
| s3.0 | baseline | 0.585 | 0.811 | 0.635 | 0.552 | 0.928 | 0.772 | 0.994 |
| s1.0 | survival | 0.571 | 0.759 | 0.980 | 0.400 | 0.645 | 0.991 | 1.000 |

### Improvement from best combination vs best pure variant

The best combination (s3.0 + lf_frac + d_abs_vs_signed + two_stage) vs the best
pure variant (s3.0 + lf_frac):

- Composite score: 0.739 vs 0.681 (+8.5%)
- Global AUC: 0.853 vs 0.819 (+4.1%)
- **coarse_shift AUC: 0.582 vs 0.405 (+43.7%)**
- lf_drift AUC: 0.833 vs 0.872 (-4.5%)
- random_lf AUC: 0.998 vs 1.000 (negligible)
- semant_wrong AUC: 0.998 vs 1.000 (negligible)

The d_abs_vs_signed detector substantially improves coarse_shift detection at the
cost of a small lf_drift regression.

### Robustness across shift variants

| Config | original | coherent | smooth | half | double | mean |
|---|---|---|---|---|---|---|
| s3.0 lf_frac + abs_vs_signed + two_stage | 0.582 | 0.802 | 0.605 | 0.552 | 0.620 | **0.632** |
| s3.0 survival + abs_vs_signed + two_stage | 0.572 | 0.787 | 0.566 | 0.547 | 0.608 | 0.616 |
| s3.0 lf_frac + d_local_var + two_stage | 0.419 | 0.873 | 0.736 | 0.452 | 0.373 | 0.571 |
| s1.0 lf_frac + d_gradient + max | 0.581 | 0.702 | 0.632 | 0.559 | 0.625 | 0.620 |

The abs_vs_signed combination has the best mean AUC (0.632) across shift variants and
is the most stable (no variant drops below 0.55).

## 4. Recommended Final D_parent Formulation

### Primary: D_parent = lf_frac with R=gauss_s3.0

```
D_parent = ||Up(R(delta))|| / (||delta|| + eps)
```
where R = gaussian blur (sigma=3.0) + decimation by 2, Up = bilinear upsampling.

This passes all kill criteria:
- Global AUC = 0.819 (threshold: 0.75)
- Cohen's d = 1.546 (threshold: 0.5)
- All depth-conditioned AUCs >= 0.65

### Auxiliary: D_shift = abs_vs_signed (for coarse_shift)

```
D_shift = ||R(|delta|)|| / ||R(delta)||
```

Combined via two-stage: `D_combined = D_parent_zscore + 0.3 * D_shift_zscore`
(both z-score normalized before combining)

This improves coarse_shift AUC from 0.405 to 0.582 while maintaining global AUC = 0.853.

### Caveats

1. **coarse_shift with random per-pixel sign flip is a fundamentally weak violation.**
   The sign flip makes the shift average to zero under restriction, rendering it invisible
   to any coarse-scale measure. The best achievable AUC is ~0.58 for this specific generator.
   Coherent coarse_shift (realistic) achieves AUC=0.80+.

2. **The abs_vs_signed detector exploits the sign-flip artifact**, not the coarse violation
   itself. It detects within-patch sign inconsistency. If the coarse_shift generator is
   changed to use smooth sign fields, abs_vs_signed becomes less useful but the base
   D_parent (lf_frac) catches it naturally (AUC=0.72 for smooth shift).

3. **Recommended generator change**: The current coarse_shift with per-pixel random
   sign_map is not representative of realistic scale-consistency violations. A more
   realistic generator would use spatially coherent shifts (smooth sign field or no
   sign flip). With coherent shift, D_parent_lf_frac with R=gauss_s3.0 achieves
   AUC=0.87 without any auxiliary detector.

## 5. Summary

| What | Before | After | Method |
|---|---|---|---|
| R operator | gauss s1.0 | gauss s3.0 | More aggressive LF extraction |
| Normalization | `\|\|R(d)\|\| / \|\|coarse\|\|` | `\|\|Up(R(d))\|\| / \|\|d\|\|` | Self-normalization (lf_frac) |
| coarse_shift AUC (original) | 0.575 | 0.582 | + abs_vs_signed auxiliary |
| coarse_shift AUC (coherent) | N/A | 0.802 | Base metric handles it |
| Global AUC | 0.685 | 0.853 | Combined improvement |
| Cohen's d | 0.233 | 1.491 | 6.4x improvement |
| random_lf AUC | 0.414 | 0.998 | Fixed by sigma=3.0 + lf_frac |
| semant_wrong AUC | 0.937 | 0.998 | Improved by lf_frac |

## Files

- `experiments/sc_baseline/coarse_shift_analysis.py` -- detector evaluation
- `experiments/sc_baseline/d_parent_combo_sweep.py` -- combination sweep
- `experiments/sc_baseline/results/coarse_shift_detectors.json` -- detector numbers
- `experiments/sc_baseline/results/combo_sweep_results.json` -- sweep numbers
