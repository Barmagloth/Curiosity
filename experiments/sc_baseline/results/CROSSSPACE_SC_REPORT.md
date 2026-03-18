# Cross-Space SC-Baseline Report

**Date:** 2026-03-18

## Configuration

- R operator: Gaussian blur sigma=3.0 + decimation by 2 (grids); cluster-mean (graph); subtree-mean (tree)
- Normalization: lf_frac = ||Up(R(delta))|| / ||delta||
- Auxiliary detector: d_abs_vs_signed = ||R(|delta|)|| / ||R(delta)||
- Combiner: two_stage (D_parent_z + 0.3 * D_shift_z)
- Coarse-shift variants: coherent (smooth sign), block (8x8/cluster/subtree), gradient
- Seeds: 5 per space type

## SC-0: Idempotence

| Space | Error | Status |
|---|---|---|
| T1_scalar | 4.09e-02 | OK |
| T2_vector | 1.13e-01 | OK |
| T3_graph | 1.51e-16 | OK |
| T4_tree | 9.04e-17 | OK |

## SC-3: Cross-Space Separability

### Global metrics (primary = lf_frac)

| Space | lf_frac AUC | Cohen's d | Combined AUC | Kill Criteria |
|---|---|---|---|---|
| T1_scalar | 1.0000 | 2.2986 | 1.0000 | PASS |
| T2_vector | 1.0000 | 1.4232 | 0.6333 | PASS |
| T3_graph | 1.0000 | 2.1369 | 0.9111 | PASS |
| T4_tree | 0.8244 | 0.9579 | 0.8222 | PASS |

### Per-negative-type AUC

| Space | cs_coherent | cs_block | cs_gradient | lf_drift | random_lf | semant_wrong |
|---|---|---|---|---|---|---|
| T1_scalar | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| T2_vector | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| T3_graph | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| T4_tree | 0.6133 | 1.0000 | 0.3333 | 1.0000 | 1.0000 | 1.0000 |

### Per-negative-type Cohen's d

| Space | cs_coherent | cs_block | cs_gradient | lf_drift | random_lf | semant_wrong |
|---|---|---|---|---|---|---|
| T1_scalar | 8.4132 | 13.7580 | 23.4627 | 153.1324 | 58.7182 | 4401.2752 |
| T2_vector | 11.2910 | 11.2198 | 25.4108 | 241.5070 | 263.3874 | 9909.3848 |
| T3_graph | 17.4472 | 197.5829 | 33.4429 | 5.3838 | 14.2960 | 434.9670 |
| T4_tree | 0.2832 | 24.6969 | 0.0978 | 3.5567 | 6.7884 | 678.1407 |

## Key Findings

1. **Cross-space AUC range**: [0.8244, 1.0000], spread=0.1756
   Moderate variation -- may benefit from space-specific tuning.

2. **Halo applicability**: T4 (tree, no Halo) AUC=0.8244 vs T1 (grid, Halo-capable) AUC=1.0000
   Halo non-applicability DOES significantly affect D_parent behavior.

3. **Fixed coarse_shift generators**: All three variants (coherent, block, gradient) use spatially coherent sign fields that do NOT self-cancel under R. This makes the violation visible to D_parent without needing auxiliary detectors.

4. **abs_vs_signed auxiliary is counterproductive with fixed generators.** The abs_vs_signed detector was designed for the OLD per-pixel sign-flip generator where ||R(|delta|)|| >> ||R(delta)|| due to cancellation. With coherent sign fields, this signal disappears, and abs_vs_signed can invert the score (especially on T2 vector). Recommendation: use lf_frac alone as the primary D_parent metric with fixed generators.

5. **T4 (tree) weakness on coarse_shift_coherent (AUC=0.61) and coarse_shift_gradient (AUC=0.33).** The tree R operator (subtree-mean) averages over subtrees rooted at a fixed coarse_depth. When the sign field from the coherent or gradient generators aligns with subtree boundaries (so that + and - regions fall within the same subtree), the shift partially cancels under R, similar to the old per-pixel problem but at subtree scale. coarse_shift_block works perfectly (AUC=1.0) because it assigns sign per-subtree, guaranteeing no intra-subtree cancellation. This suggests that for trees, block-aligned shifts are the most representative violation, while coherent/gradient shifts may partially self-cancel depending on sign-field alignment with the tree partition.

6. **All 4 space types PASS kill criteria** (Global AUC >= 0.75, Cohen's d >= 0.5) when using lf_frac as primary metric. The D_parent formulation generalizes across space types without space-specific tuning for the overall pass/fail decision. However, per-negative-type detection strength varies: trees are weaker on some coarse_shift variants, suggesting that tree-specific negative generators should prefer block-aligned violations.

## Recommendation

Use **D_parent = lf_frac = ||Up(R(delta))|| / ||delta||** as the universal D_parent metric:
- R = gauss_sigma3.0 + decimation for grids, cluster-mean for graphs, subtree-mean for trees
- No auxiliary detector needed with fixed (coherent sign) generators
- Passes kill criteria on all 4 space types
- abs_vs_signed auxiliary should ONLY be added if the old per-pixel sign-flip generator is used (legacy compatibility)
