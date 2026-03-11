# Curiosity Experiment Results

Series Exp0.1–Exp0.8 is complete. Below is a summary of each experiment with key numbers and conclusions.

---

## Exp0.1 — Does Adaptive Refinement Work?

**Question:** Does adaptive tile selection for refinement outperform random selection?

**Result:** Yes. Adaptive > random by MSE/PSNR at any coverage level.

**Conclusion:** Core hypothesis confirmed. Selective tile refinement works.

---

## Exp0.2 — Robustness on Real Data

**Question:** Are results robust on real images (not just synthetic)?

**Result:** Yes by MSE_rgb/PSNR (98–100% winrate). However, HF degradation (high-frequency artifacts) was discovered at low budgets.

**Conclusion:** Works, but a mechanism is needed to eliminate artifacts at tile boundaries. Motivation for halo.

---

## Exp0.3 — Is Halo Needed?

**Question:** Does overlap (halo) at tile boundaries eliminate HF artifacts?

**Result:** Yes. Halo ≥ 3 cells (pixels in image experiments) at tile_size=16 eliminates artifacts. Interior-only HF metric confirmed: the problem was specifically in seams, not inside tiles.

**Conclusion:** Boundary-aware blending (cosine feathering) with minimum overlap = 3 elements is a mandatory requirement.

---

## Exp0.4 — Is Combined ρ Needed?

**Question:** Should multiple informativeness signals be combined, or is residual sufficient?

**Result:** On clean data, residual-only = oracle. Combination provides no advantage.

**Conclusion:** Under ideal conditions, residual alone suffices. But what happens under noise?

---

## Exp0.5 — Does the Residual Oracle Break?

**Question:** Does the residual-only signal degrade under noise, blur, alias?

**Result:** Yes. Residual-to-oracle correlation drops from 0.90 to 0.54 under noise. Degradation at coarse level.

**Conclusion:** Residual-only is a fragile strategy. A fallback mechanism is needed.

---

## Exp0.6 — Binary Switch resid/combo

**Question:** Does a simple switch between residual and combined signal work?

**Result:** Partially. Clean/blur → residual, noise → combo. Alias is a borderline case; the switch doesn't always correctly identify the regime.

**Conclusion:** Binary logic is insufficient. A more flexible mechanism is needed.

---

## Exp0.7 / Exp0.7b — Soft Gating and Two-Stage Gate

**Question:** Can the binary switch be replaced with soft gating?

**Exp0.7:** Soft gating wins on noise/spatvar/jpeg (+0.08–1.10 dB), but loses on clean/blur (−1.6–2.5 dB) due to softmax "democracy".

**Exp0.7b:** Two-stage gate solves the problem. Stage 1 checks "is residual healthy?":
- If yes → residual-only (zero loss on clean/blur, Δ ≈ 0).
- If no → utility-weighted combination (+0.77–1.49 dB on noise/spatvar).
- JPEG: only downside (−0.21 dB), threshold tuning issue.

**Conclusion:** Two-stage gate is the correct architecture. Canonical solution.

---

## Exp0.8 — Schedule and Budget Governor

**Question:** Is a phase schedule (changing ρ weights by step) and budget governor needed?

**Budget governor (EMA):**
- StdCost halved (~5.15 → ~3.25).
- P95 from 11.0 to ~6.5.
- Compliance penalty from ~3.2 to ~0.5.
- PSNR slightly lower (−0.24 dB clean, −0.68 dB shift).
- **Governor is needed for budget predictability, not quality.**

**Phase schedule:** Showed no gain under current conditions. Deferred as optional extension.

**Conclusion:** EMA governor is a mandatory component. Phase schedule is not.

---

## Summary Table

| Component | Status | Requirement Level |
|---|---|---|
| Adaptive refinement | Confirmed | System core |
| Halo (boundary blending) | Confirmed | Mandatory (≥3 px) |
| Probe (exploration) | Confirmed | Mandatory (5–10% budget) |
| Two-stage gate | Confirmed | Mandatory under noise/degradation |
| EMA budget governor | Confirmed | Mandatory |
| SeamScore metric | Validated | Stable within current validation scope |
| Phase schedule | Not confirmed | Deferred |
| Morton/block-sparse layout | Preliminarily unfavorable | Per 0.9a microbench; P0 open |
