# R Operator Variant Experiment Results

**Date:** 2026-03-18
**Objective:** Find an R operator that makes D_parent pass kill criteria (AUC >= 0.75, Cohen's d >= 0.5).

## Background

The baseline R operator (gaussian blur sigma=1.0 + decimation by 2 / bilinear upsample) fails D_parent kill criteria:
- Global AUC = 0.685 (need >= 0.75)
- Cohen's d = 0.233 (need >= 0.5)
- Weak spots: `coarse_shift` AUC=0.575, `random_lf` AUC=0.414

D_hf passes fine with all R variants (AUC ~0.80, d ~1.3).

## Variants Tested

| Variant | R operator | Up operator |
|---|---|---|
| baseline | Gaussian blur (sigma=1.0) + decimation | Bilinear upsample |
| lanczos | Lanczos-3 downsample | Cubic upsample |
| area | Block mean (2x2 area average) | Bilinear upsample |
| wavelet | Haar wavelet (= block mean for approx.) | Piecewise constant (nearest) |
| gauss_s0.5 | Gaussian blur (sigma=0.5) + decimation | Bilinear upsample |
| gauss_s2.0 | Gaussian blur (sigma=2.0) + decimation | Bilinear upsample |
| gauss_s3.0 | Gaussian blur (sigma=3.0) + decimation | Bilinear upsample |
| wavelet_bl | Haar wavelet (block mean) | Bilinear upsample |

## Summary Table (D_parent)

| Variant | AUC | Cohen's d | Depth AUC | QSep | Result |
|---|---|---|---|---|---|
| baseline | 0.685 | 0.233 | OK | 3/3 | FAIL |
| lanczos | 0.669 | 0.192 | FAIL | 3/3 | FAIL |
| area | 0.669 | 0.187 | FAIL | 3/3 | FAIL |
| wavelet | 0.669 | 0.187 | FAIL | 3/3 | FAIL |
| gauss_s0.5 | 0.659 | 0.167 | FAIL | 3/3 | FAIL |
| gauss_s2.0 | 0.741 | 0.502 | OK | 3/3 | FAIL (AUC < 0.75) |
| **gauss_s3.0** | **0.811** | **0.635** | **OK** | **3/3** | **PASS** |
| wavelet_bl | 0.669 | 0.187 | FAIL | 3/3 | FAIL |

## Per-Negative-Type Breakdown (D_parent AUC)

| Variant | coarse_shift | lf_drift | random_lf | semant_wrong |
|---|---|---|---|---|
| baseline | 0.575 | 0.815 | 0.414 | 0.937 |
| lanczos | 0.574 | 0.804 | 0.387 | 0.910 |
| area | 0.575 | 0.809 | 0.389 | 0.903 |
| wavelet | 0.575 | 0.809 | 0.389 | 0.903 |
| gauss_s0.5 | 0.586 | 0.800 | 0.356 | 0.896 |
| gauss_s2.0 | 0.558 | 0.849 | 0.577 | 0.981 |
| **gauss_s3.0** | **0.552** | **0.928** | **0.772** | **0.994** |
| wavelet_bl | 0.575 | 0.809 | 0.389 | 0.903 |

## Depth-Conditioned AUC (D_parent)

| Variant | L=1 | L=2 | L=3 |
|---|---|---|---|
| baseline | 0.670 | 0.737 | 0.677 |
| lanczos | 0.637 | 0.699 | 0.664 |
| area | 0.640 | 0.700 | 0.665 |
| gauss_s2.0 | 0.699 | 0.782 | 0.749 |
| **gauss_s3.0** | **0.708** | **0.819** | **0.846** |

## SC-0 Idempotence

All variants pass (tolerance < 0.30):
- baseline: 8.24e-02
- lanczos: 4.90e-02
- area: 6.78e-02
- wavelet: 0.00e+00 (exact)
- gauss_s0.5: 9.22e-02
- gauss_s2.0: 7.00e-02
- gauss_s3.0: 7.27e-02

## D_hf (unaffected)

All variants maintain D_hf performance (AUC 0.80-0.82, d 1.16-1.37). The R operator change does not degrade D_hf.

## Analysis

1. **Downsampling kernel type (lanczos, area, wavelet) does not help.** All non-gaussian variants perform worse than the gaussian baseline. The sharper cutoff of Lanczos and the exact energy preservation of area/wavelet actually hurt D_parent because they preserve more of the positive delta's structure in R(delta), raising the D_parent floor for positives.

2. **Gaussian sigma is the dominant parameter.** Increasing sigma from 1.0 to 3.0 monotonically improves D_parent AUC (0.685 -> 0.741 -> 0.811) and Cohen's d (0.233 -> 0.502 -> 0.635). Higher sigma means more aggressive low-pass filtering before decimation, which suppresses R(delta) for high-frequency positive deltas while retaining it for truly low-frequency negative violations.

3. **The `random_lf` problem is solved by sigma=3.0.** AUC jumps from 0.414 (baseline) to 0.772 (gauss_s3.0). With sigma=1.0, the gaussian does not sufficiently attenuate the mid-frequency content in positive deltas, making them look similar to random LF noise after R. With sigma=3.0, positive deltas are almost entirely suppressed by R, while random LF noise retains measurable energy.

4. **`coarse_shift` remains the hardest negative type** (AUC=0.552 across all variants). This is expected: coarse_shift adds a DC offset proportional to the local coarse value, and the positive oracle delta (GT - coarse) also has non-zero projection onto coarse. The distinction between "legitimate refinement toward GT" and "shifting the coarse mean" is intrinsically subtle when measured only by R(delta) norm. This may require a different metric approach (e.g., correlation-based rather than norm-based) rather than R operator tuning.

5. **D_hf is robust to R operator choice.** All variants yield D_hf AUC in [0.80, 0.82], confirming D_hf's independence from the specific R/Up pair.

## Recommendation

**Use `gauss_s3.0`** (gaussian blur sigma=3.0 + decimation by 2 / bilinear upsample) as the R operator.

- D_parent: AUC=0.811, d=0.635 -- passes all kill criteria
- D_hf: AUC=0.810, d=1.330 -- unchanged from baseline
- SC-0 idempotence: 7.27e-02 -- well within tolerance
- Implementation: single-line change in `operators.py` (sigma=1.0 -> sigma=3.0)

The `coarse_shift` weakness (AUC=0.552) is a known limitation. It does not prevent the overall pass because the other negative types provide strong separation. If coarse_shift detection needs improvement in the future, it should be addressed at the metric level (e.g., normalized correlation with coarse), not at the R operator level.

## Files

- `experiments/sc_baseline/operators_v2.py` -- alternative R/Up implementations
- `experiments/sc_baseline/sc_baseline_r_variants.py` -- variant comparison experiment
- `experiments/sc_baseline/results/r_variants_results.json` -- full numeric results
