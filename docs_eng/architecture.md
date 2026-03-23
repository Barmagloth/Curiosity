# Curiosity Architecture

This document describes the system's key components and architectural decisions. Each decision is backed by experimental results.

---

## High-Level Pipeline

```
State space X (arbitrary nature)
       |
       v
[Root coarse] — initial coarse approximation (see concept_v2.0.md §1.2 for terminology)
       |
       v
[Informativeness function ρ(x)] — determines where to refine
       |
       v
[Split decision] — region subdivision decision
       |
       v
[Adaptive refinement] — refine selected regions
       |
       v
[Scale-Consistency check] — D_parent < τ_parent? (v1.7)
       |
       v
[Halo blending] — cosine feathering at boundaries
       (if applicable per topology rule)
       |
       v
[Budget governor] — EMA budget control
       |
       v
[Probe allocation] — 5–10% budget for exploration
```

---

## Component 1: Informativeness Function ρ(x)

### Signals

| Signal | Purpose | Weakness |
|---|---|---|
| Residual | Primary: current approximation error | Degrades under noise (corr 0.90→0.54) |
| HF energy | Laplacian/gradient, structural energy | False positives at seams |
| Variance | Local dispersion / disagreement | Loves noise |
| Payoff | Expected gain vs. cost | — |

### Two-Stage Gate (Canonical Solution)

**Stage 1:** Residual health check
- Metrics: instability, FSR (False Signal Rate)
- If healthy: ρ = residual-only (zero loss on clean data)
- If unhealthy: transition to Stage 2

**Stage 2:** Utility-weighted combination
- Each expert weighted by: U_i = median(gain) − λ·FSR − μ·instability
- Residual has minimum guaranteed weight (prior)
- Weights EMA-smoothed with hysteresis
- Normalization: quantile (rank-based), not absolute

### Three-Layer Rho Decomposition (Phase 3.5, exp17)

rho is decomposed into three architectural layers:

| Layer | Purpose | Recomputation frequency |
|-------|---------|------------------------|
| L0 (topology) | Structural properties of the space (curvature, connectivity) | Rarely (on tree rebuild) |
| L1 (presence) | Cascade quotas, dirty signatures | Every step (incremental) |
| L2 (query) | Specific query (residual, HF, utility-weighted combination) | Every query |

Cascade quotas (Variant C) manage budget across layers. Streaming pipeline enables L0/L1 reuse between queries. Reusability validated 12/12 PASS (min 0.838) across 1080 configurations.

---

## Component 2: Halo (Boundary-Aware Blending)

**Why:** Hard insertion of a refined tile creates a step discontinuity at the boundary. The Laplacian flags this as a false HF signal. The smarter the adaptive selection, the stronger the artifact.

**Implementation:**
- Overlap ≥ 3 elements (not "pixels" — the system is not tied to images)
- Cosine feathering relative to parent coarse

**Applicability rule (Phase 0, concept v1.7 §6):**

Halo is applicable ONLY when BOTH conditions are met:
1. **Boundary parallelism ≥ 3** — at least 3 independent cross-edges at the tile boundary
2. **No context leakage** — halo expansion does not leak into unrelated tiles

| Topology | Halo? | Reason |
|---|---|---|
| Grid (tile ≥ 3) | ✅ Always | boundary = wide strip, no leakage |
| k-NN graph | ✅ Apply | min_cut typically >> 3 |
| Tree / Forest | ❌ Never | min_cut=1 (bottleneck), context leakage into sibling subtrees |
| DAG | ⚠️ Per-case | Check boundary parallelism and leakage |

Root cause tree failure: single-edge bottleneck + sibling bleed (85% fade at hop 1 reaches foreign subtree) + extreme S/V asymmetry (0.032 vs 0.5 for grid).

**Note:** Properties "zero initialization", "energy boundedness", "valid rollback on level disable" belong to step_delta / refinement level, not to halo. Halo is a boundary reconciliation mechanism, not a residual carrier.

---

## Component 3: Probe (Exploration)

**Why:** Without exploration the system becomes structurally blind. Exploitation-only misses shifts, rare patterns, and new regions of interest.

**Budget:** 5–10% of total budget per step.

**Probe priorities:**
1. Coarse residual / variance
2. Uncertainty
3. Time since last check

**Trade-off:** On stationary scenes, probe may slightly reduce PSNR, but it is insurance against blindness, not a quality optimization.

---

## Component 4: Budget Governor (EMA Controller)

**Why:** Without the governor, the budget is a declaration, not a constraint.

**Control variable:** strictness — quantile threshold for candidate selection.

**Parameters:**
- Δstrictness ≤ clamp (anti-oscillation)
- hard_cap_per_step = 3× target (safety fuse)
- Warmup: N steps with frozen strictness
- Compliance: asymmetric (overbudget penalty > underbudget)

**Numbers (Exp0.8v5):**
- StdCost: −50% (~5.15 → ~3.25)
- P95: 11.0 → ~6.5
- Compliance penalty: −85% (~3.2 → ~0.5)
- PSNR: −0.24 dB (clean), −0.68 dB (shift) — negligible cost

---

## Component 5: SeamScore — Seam Quality Metric

**Formula:** `SeamScore = Jumpout / (Jumpin + eps)`

Computed over edge strips (bands at tile boundaries).

**Validation:** 2D scalar grids, vector-valued grids, irregular graphs, tree hierarchies.

**Status:** Validated and stable within current validation scope (4 space types). Final production-readiness depends on P0–P4 outcomes.

---

## Component 6: Scale-Consistency Invariant (v1.7)

**Why:** Halo ensures local geometric correctness at tile boundaries. But a refined level can be smooth at seams while semantically contradicting the parent coarse — step_delta smuggles low-frequency meaning upward. The Scale-Consistency Invariant guarantees this does not happen.

**Principle:** parent coarse is the anchor, step_delta is the subordinate correction. Step_delta must be invisible from the level above.

**Formal requirement:**
```
‖R(step_delta)‖ / (‖step_delta‖ + ε) < τ_rel
```

**Operator pair (R, Up):**
- **R** (coarse-graining): `gaussian blur (σ=3.0) + decimation`. Projects fine → coarse.
- **Up** (restoration): `bilinear upsampling`. Projects back to fine. Not the inverse of R.
- The pair is fixed before experiments. Different pairs yield different tree physics.

**Metrics:**

| Metric | Formula | Interpretation |
|---|---|---|
| D_parent | `‖R(step_delta)‖ / (‖step_delta‖ + ε)` | Fraction of LF energy in step_delta. Lower = better. Enforcement signal. |
| D_hf | `‖step_delta - Up(R(step_delta))‖ / (‖step_delta‖ + ε)` | HF purity of step_delta. Higher = better. Diagnostic, not hard constraint. |

**Enforcement (after baseline validation):**
- `D_parent > τ_parent` → damp step_delta / reject split / increase local strictness
- D_parent is also used as a contextual signal in ρ (not self-sufficient)

**Thresholds:** τ_parent is data-driven from the baseline experiment, may depend on level L.

**Note on step_delta tolerance:** τ_parent effectively defines step_delta tolerance — the permissible fraction of parent_coarse alterable by step_delta. This balances two risks: too tight → loss of legitimate features; too loose → hierarchy drift. Optimal trade-off is data-dependent. See concept_v2.0.md §8.9.

**Cross-space validation (Phase 0):**

| Space | D_parent AUC | D_hf AUC |
|---|---|---|
| T1 Scalar grid | 1.000 | 0.806 |
| T2 Vector grid | 1.000 | 0.810 |
| T3 Irregular graph | 1.000 | — |
| T4 Tree hierarchy | 0.824 | — |

**Status:** SC-baseline (SC-0..SC-4) ✅ COMPLETE. D_parent validated across 4 spaces. SC-5 (τ_parent) and SC-enforce — open. Protocol: `scale_consistency_verification_protocol_v1.0.md`.

---

## Component 7: Refinement Tree

The tree is a log of split decisions. Each root-to-leaf path = a sequence of decisions.

**Requirements:**
- GPU-friendly structure (flat packing, no pointer chasing)
- Morton and block-sparse layouts: **deferred** per Exp0.9a microbench (sort overhead / expansion ratio). Final decision after P0 (0.9b0+).
- Compact layouts: preliminarily promising under low sparsity + large grid. Kill/go in Exp0.9b0.

**Bush** = a set of paths leading to the same meaning. Distance metric via LCA / common prefix.

---

## Technology Stack

- **Language:** Python
- **ML framework:** PyTorch
- **GPU:** CUDA (environment_2) / DirectML (environment_1, AMD GPU)
- **Environments:** `.venv` (CPU, Python 3.13) + `.venv-gpu` (DirectML, Python 3.12). See `docs/environment_1.md`
- **Experiments:** Jupyter Notebooks
- **Documentation:** Versioned markdown
