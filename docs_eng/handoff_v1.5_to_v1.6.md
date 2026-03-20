> **Historical changelog.** See session_handoff.md for current status.

# Handoff: v1.5 → v1.6

What changed and why. For those adding files to the repository.

---

## What to Add to the Repository

1. `curiosity_concept_v1_6.md` — updated concept
2. `scale_consistency_verification_protocol_v1_0.md` — new document, verification protocol

---

## Essence of Changes

### Open Question from v1.5 is Closed

In v1.5, the "Open Questions" section contained this question:

> "How to ensure 'don't break features'?"

It has been formalized and closed through **Scale-Consistency Invariant** (new section 8 in v1.6).

In short: delta (refinement correction) must not push low-frequency content onto the parent scale. Formally: `‖R(delta)‖ / (α·‖coarse‖ + β) < τ_rel`. The operator pair (R, Up) is fixed before experiments. Thresholds are data-driven from baseline.

### What Has NOT Changed

Everything from v1.5 is preserved unchanged: halo, probe, governor, two-stage gate, results Exp0.1–Exp0.8. The new invariant is an extension, not a replacement.

---

## Key Conceptual Points (for context)

**Why asymmetry, not `R(refined) ≈ coarse`:**
A symmetric requirement breaks the matryoshka — coarse ceases to be an anchor and becomes a function of refined. That is a different architecture. Correct: refined is subordinate to coarse, not vice versa.

**Why two metrics, not one:**
D_parent and D_hf measure different things. D_parent measures how much delta leaks into the parent (`‖R(delta)‖` in the numerator). D_hf measures how much delta is detail-like, i.e., lives in the HF subspace (`‖delta - Up(R(delta))‖` in the numerator). The numerators differ, but both metrics are built through the same operator pair (R, Up) — complete statistical independence is not guaranteed. These are two diagnostic axes, not two independent detectors.

**Why baseline before enforcement:**
Without measuring distributions on known-good and known-bad cases, τ is just a number from thin air. The kill criterion is fixed in advance: if separability is insufficient, metrics change, not thresholds.

**Connection to probe:**
Probe is reformulated more precisely: it is not just exploration, but protection against false fixed points. Scale-stable fixed point + probe = proper refinement termination.

---

## Next Step in the Experimental Queue

Baseline experiment for scale-consistency (described in `scale_consistency_verification_protocol_v1_0.md`). Must precede any enforcement.
