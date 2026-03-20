# exp11_dirty_signatures

**Question:** 12-bit dirty signature + debounce -> AUC > 0.8 for structural change detection?
**Kill criteria:** AUC < 0.8 on any space type = FAIL
**Roadmap level:** P1-B2
**Status:** PASS -- AUC > 0.8 on all 4 space types (all p_adj < 0.001, Holm-Bonferroni)

## Results

| Space            | AUC   | CI            | p_adj  |
|------------------|-------|---------------|--------|
| scalar_grid      | 0.925 | [0.815, 1.000] | 0.0008 |
| vector_grid      | 1.000 | [1.000, 1.000] | 0.0006 |
| irregular_graph  | 1.000 | [1.000, 1.000] | 0.0004 |
| tree_hierarchy   | 0.910 | [0.785, 1.000] | 0.0004 |

## Bug fix history

### Fix 2 (2026-03-20): revert oracle scoring, fix tracker to baseline comparison

The previous "fix" replaced dirty-signature scoring with oracle-based error
detection (`unit_error` = MSE vs ground truth).  This was cheating: in
production there is no ground truth.  Reverted all `unit_error` /
`baseline_errors` / MSE-vs-GT references from the scoring path.

Root cause of the original failure (signal processing analysis):

- The debounce tracker compared step-to-step signatures (discrete derivative
  dD/dt).  Structural changes produce a step function in D(t) -- the signature
  jumps once then stabilises at the new level.  The step-to-step delta is an
  impulse (one crossing), which the debounce filter (requires 2 consecutive)
  kills.  Gaussian noise produces continuous random jumps, giving endless
  crossings that the debounce fires on constantly.

Fix applied:

1. **Tracker uses baseline comparison** -- `DebounceTracker.set_baseline()`
   stores the unperturbed signature; each step compares current vs baseline
   (not vs previous step).  Structural changes now produce persistent
   crossings (every step after the event exceeds threshold), so debounce
   fires correctly.

2. **Score = temporal ramp of baseline distance** -- for each step, compute
   mean signature distance (hamming + component_diff) from baseline across
   all units.  Score = mean(last_half) - mean(first_half) of this trajectory.
   Structural changes create a step function (low first half, high second
   half -> positive ramp).  Noise creates a flat profile (equally elevated
   every step -> ramp near zero).  No ground truth is used.

### Fix 1 (2026-03-20): oracle-based scoring (REVERTED)

Used `unit_error` change from baseline (MSE vs ground truth) as score.
Gave AUC=1.000 on all spaces -- this was oracle cheating, not a legitimate
dirty-signature result.  Reverted in Fix 2.
