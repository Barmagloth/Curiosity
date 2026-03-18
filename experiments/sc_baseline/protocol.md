# SC-Baseline Experiment Protocol

Links to canonical protocol: `docs/scale_consistency_verification_protocol_v1.0.md`
Conceptual basis: `docs/concept_v1.6.md` section 8 (Scale-Consistency Invariant)

---

## Purpose

Verify that D_parent and D_hf metrics carry discriminative information
before introducing enforcement. This is steps SC-0 through SC-3 of the
canonical protocol.

---

## Fixed Parameters (Step 0)

| Parameter | Value | File |
|---|---|---|
| R | gaussian blur (sigma=1.0) + decimation (factor 2) | `operators.py` |
| Up | bilinear upsampling (scipy.ndimage.zoom, order=1) | `operators.py` |
| alpha | 1.0 | `metrics.py` |
| beta | 1e-4 * mean(\|coarse\|) | `metrics.py` |
| eps | 1e-4 * mean(\|delta\|) | `metrics.py` |

The pair (R, Up) does not change within one verification cycle.

---

## Space Types

| Space | R implementation | Up implementation |
|---|---|---|
| T1: 2D scalar grid | gaussian_filter + [::2, ::2] | scipy.ndimage.zoom (bilinear) |
| T2: 2D vector grid | per-channel gaussian_filter + decimation | per-channel zoom |
| T3: irregular graph | cluster-mean pooling | scatter cluster means to nodes |
| T4: tree hierarchy | subtree-mean at coarse_depth | broadcast ancestor value to subtree |

---

## Baselines (Step 1)

### Positive (scale-consistency guaranteed by construction)

- **oracle**: delta = GT - coarse
- **scaled**: delta = 0.5 * (GT - coarse)
- **noisy**: delta = (GT - coarse) + N(0, 0.005)

### Negative (scale-consistency intentionally violated)

- **lf_drift**: correct delta + LF sinusoid (amplitude=0.3, freq=0.5)
- **coarse_shift**: delta shifts coarse-mean by 20% with random signs
- **random_lf**: gaussian-smoothed random noise (sigma=4.0, amplitude=0.5)
- **semant_wrong**: delta = -2 * coarse (flips sign, extreme case)

---

## Depth Simulation

Three levels simulated via tile size:
- L=1: tile_size=16 (coarsest)
- L=2: tile_size=8
- L=3: tile_size=4 (finest)

---

## Structure Types

Grid is divided into three vertical regions:
- **smooth**: low-frequency sinusoidal
- **boundary**: sharp edge + low-frequency
- **texture**: high-frequency sinusoidal

---

## Metrics (Step 2)

```
R_delta = R(delta)
P_LF_delta = Up(R_delta)
delta_HF = delta - P_LF_delta

D_parent = ||R_delta|| / (alpha * ||coarse|| + beta)
D_hf     = ||delta_HF|| / (||delta|| + eps)
```

Orientation:
- D_parent: higher = worse (negative cases should be higher)
- D_hf: higher = better (positive cases should be higher)

---

## Separability Analysis (Step 3)

### 3.1 Global
- ROC-AUC (with correct orientation)
- PR-AUC
- Cohen's d
- Quantile separation

### 3.2 Depth-conditioned
Same as 3.1, per level L.

### 3.3 Regime-conditioned
Same as 3.1, per structure type (smooth, boundary, texture).

### Negative-type breakdown
Each negative type analyzed separately vs all positives.

---

## Kill Criteria (Step 4)

| Criterion | Threshold |
|---|---|
| Global ROC-AUC | >= 0.75 |
| Depth-conditioned AUC (each level) | >= 0.65 |
| Effect size (Cohen's d) | >= 0.5 |
| Quantile separation | on >= 2/3 of levels |

If any metric fails: do NOT tune tau, do NOT introduce enforcement.
Diagnose which negative type separates poorly and revisit R/Up or metric design.

---

## Output

Results saved to `results/sc_baseline_results.json` containing:
- SC-0 idempotence errors
- SC-2 per-case metric values
- SC-3 separability analysis with kill criteria evaluation

---

## What comes next (not implemented here)

- **SC-4**: Threshold setting (data-driven tau_parent, tau_hf) — only if SC-3 passes
- **SC-5**: Enforcement integration into refinement loop
