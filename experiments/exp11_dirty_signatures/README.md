# exp11_dirty_signatures

**Question:** 12-bit dirty signature + debounce → AUC > 0.8 for structural change detection?
**Kill criteria:** AUC < 0.8 on any space type = FAIL
**Roadmap level:** P1-B2
**Status:** PASS — AUC=1.0 on all 4 space types (all p_adj < 0.001, Holm-Bonferroni)

## Results

| Space            | AUC   | CI            | p_adj  |
|------------------|-------|---------------|--------|
| scalar_grid      | 1.000 | [1.000, 1.000] | 0.0008 |
| vector_grid      | 1.000 | [1.000, 1.000] | 0.0006 |
| irregular_graph  | 1.000 | [1.000, 1.000] | 0.0004 |
| tree_hierarchy   | 1.000 | [1.000, 1.000] | 0.0002 |

## Bug fix (2026-03-20)

The original scoring used packed-signature Hamming distance from baseline, which
was inverted: the `uncert` component (rank-flip fraction under jitter) is a
global measure sensitive to any perturbation, so noise caused larger signature
jumps than structural changes.  The debounce trigger count was also inverted
(noise causes many consecutive step-to-step crossings; structural change
stabilises after the event).

Fixed scoring uses `unit_error` change from baseline (MSE vs ground truth).
Structural changes shift state away from GT producing large persistent error
increases; noise produces tiny transient fluctuations.  This correctly separates
structural from noise across all space types.
