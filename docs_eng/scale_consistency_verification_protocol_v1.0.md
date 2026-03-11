# Scale-Consistency Verification Protocol (v1.0)

Scale-Consistency Invariant verification protocol for the Curiosity project.

This document is procedural. Conceptual justification is in `curiosity_concept_v1.6.md`, section 8.

---

## Context

Before introducing enforcement of scale-consistency, it is necessary to verify that the metrics D_parent and D_hf carry discriminative information. This protocol defines how to check this and when to stop.

---

## Step 0. Preliminary Fixation

Before any experiments, fix and commit:

| Parameter | Value | Justification |
|---|---|---|
| R | gaussian blur + decimation | Linear, GPU-cheap, aligned with parent_coarse + step_delta |
| Up | bilinear upsampling | Stable, interpretable, without introducing foreign semantics |
| α | 1.0 (initial) | Normalization to parent_coarse |
| β | small constant (e.g., 1e-4 · mean(‖parent_coarse‖)) | Protection against division by zero |
| ε | small constant (analogous to β) | Stabilizer in D_hf |

**The pair (R, Up) does not change during one verification cycle.** If a different pair needs to be checked — this is a separate cycle.

Verify idempotency of R: `‖R(parent_coarse) − parent_coarse_downsampled‖` should be negligibly small.

---

## Step 1. Data Preparation

### 1.1 Positive baseline (correct cases)

Two confidence levels — store separately:

**Strong positive** (prioritized):
- Synthetic data with known GT: `step_delta = GT − parent_coarse` (near-oracle)
- Confidence level high: scale-consistency is guaranteed by construction

**Empirical positive** (auxiliary):
- Cases from Exp0.1–Exp0.8 with high PSNR and confirmed gains
- Confidence level lower: good PSNR does not guarantee the absence of semantic drift, only external result
- Use for verification: do distributions of D_parent/D_hf match strong positive? If not — interesting diagnostic signal

Both types are needed; do not mix into one sample until distribution compatibility is verified.

### 1.2 Negative baseline (intentionally poor cases)

Cases where the invariant is intentionally violated:

- **LF drift:** add a low-frequency sinusoid (scale > tile_size) to correct step_delta
- **Parent_coarse shift:** step_delta intentionally shifts the parent_coarse-mean of the region by 10–30%
- **Random LF step_delta:** step_delta = random low-frequency noise unrelated to GT
- **Semant-wrong:** step_delta flips the sign of parent_coarse in the region (extreme case)

For each negative case, save the violation type as a label.

---

## Step 2. Metric Computation

For each (node, level, type) compute:

```python
# R = gaussian_blur_then_downsample
# Up = bilinear_upsample_to_original_size

R_delta = R(step_delta)                   # coarse-projection of step_delta
P_LF_delta = Up(R_delta)                  # LF-component of step_delta in original scale
delta_HF = step_delta - P_LF_delta        # HF-remainder

D_parent = norm(R_delta) / (alpha * norm(parent_coarse) + beta)
D_hf     = norm(delta_HF) / (norm(step_delta) + eps)
```

Save: `(D_parent, D_hf, level, structure_type, case_type: pos/neg, neg_type)`

---

## Step 3. Separability Analysis

**Metric orientation (fix before computation):**

| Metric | Direction | Meaning |
|---|---|---|
| D_parent | higher = worse | negative cases should give larger values |
| D_hf | higher = better | positive cases should give larger values |

AUC is computed with orientation in mind. For D_hf, "positive = larger D_hf" means that when computing ROC-AUC, the positive and negative labels are assigned accordingly. If you get AUC < 0.5 without orientation — this is not metric death, but a flipped sign.

### 3.1 Global separability

For each metric (D_parent, D_hf):
- ROC-AUC: positive vs. negative
- PR-AUC
- Effect size (Cohen's d or rank-biserial correlation)
- Quantile separation (with orientation): D_parent: `median(neg) > q75(pos)`; D_hf: `median(pos) > q75(neg)`

### 3.2 Depth-conditioned separability

Repeat 3.1 separately for each level L.

Reason: a metric may work well at L=1 and fail at L=3.

### 3.3 Regime-conditioned separability

Repeat 3.1 separately for each structure type (smooth, boundary, texture).

---

## Step 4. Kill Criterion (fixed before launch)

### Acceptance Threshold

A metric is considered architecturally viable if **all** conditions are met:

| Criterion | Threshold |
|---|---|
| Global ROC-AUC | ≥ 0.75 |
| Depth-conditioned AUC (each level) | ≥ 0.65 |
| Effect size | ≥ medium (d ≥ 0.5) |
| Quantile separation | D_parent: `median(neg) > q75(pos)` on at least 2/3 of levels; D_hf: `median(pos) > q75(neg)` on at least 2/3 of levels |

Thresholds are conservative for the first cycle. May be revised after observing real distributions.

### Kill Criterion

If at least one metric does not pass the acceptance threshold:

1. **DO NOT tune τ**
2. **DO NOT introduce enforcement**
3. Diagnose: which negative case type separates poorly?
4. Depending on diagnosis:
   - If D_parent separates poorly → check choice of R or normalization
   - If D_hf separates poorly → check pair (R, Up) or redefine positive/negative cases
   - If both → reconsider metric construction; a different feature may be needed

### What kill criterion forbids

Tuning τ to accommodate poor separability. This masks the problem instead of solving it.

---

## Step 5. Threshold Setting (if acceptance passed)

### 5.1 Data-driven τ_parent

```python
# From positive baseline:
τ_parent = quantile(D_parent[positive], q=0.95)

# Or with breakdown by levels:
τ_parent[L] = quantile(D_parent[positive, level=L], q=0.95)
```

### 5.2 Data-driven τ_hf (diagnostic threshold)

Analogous to τ_parent, but D_hf is used as a signal, not a hard constraint.

### 5.3 Energy dependence (optional, second pass)

If systematic dependence of D_parent on ‖step_delta‖ is found in baseline:
- On small step_delta many false positives → consider adaptive τ_rel(E)
- On large step_delta τ is too soft → τ_rel(E) = decreasing function of energy

This is a second pass, not the first.

---

## Step 6. Enforcement

After validation and threshold setting:

```python
def check_scale_consistency(step_delta, parent_coarse, level):
    R_delta = R(step_delta)
    D_parent = norm(R_delta) / (alpha * norm(parent_coarse) + beta)

    if D_parent > tau_parent[level]:
        return "REJECT"  # damp step_delta, reject split, increase local strictness
    return "OK"
```

D_parent as signal in ρ (contextual, not self-sufficient):

```python
# Increase probe priority if:
if D_parent > tau_warn AND residual > tau_residual AND gain > tau_gain_min:
    increase_probe_priority(node)

# DO NOT encourage split if:
if D_parent > tau_warn AND gain < tau_gain_min:
    # Problem in refinement mechanics, not in data structure
    pass
```

---

## Relationship to Other Mechanisms

| Mechanism | What it protects | Orthogonality |
|---|---|---|
| Halo | Local tile boundaries (geometry) | Does not intersect with scale-consistency |
| SeamScore | Boundary artifacts within level | Does not intersect with D_parent |
| D_parent / D_hf | Cross-scale semantic drift | Does not intersect with halo/seam |
| Probe | Protection against false fixed points | Complements scale-stable stop criterion |
| Budget governor | Global spending | Orthogonal to scale-consistency |

Scale-stable fixed point (local stopping):
```
gain_from_split < τ_gain
AND D_parent < τ_parent
AND stable for K steps in a row
```

Probe remains mandatory even when a fixed point is reached.

---

## What NOT to Do

- Do not use learned Up at the verification stage (introduces its own semantics)
- Do not change the pair (R, Up) during one cycle
- Do not interpret high D_parent as an unconditional signal "split here"
- Do not replace D_parent + D_hf with one "grand unified index" without evidence
- Do not tune τ to accommodate poor separability
