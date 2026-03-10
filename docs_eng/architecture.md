# Curiosity Architecture

This document describes the system's key components and architectural decisions. Each decision is backed by experimental results.

---

## High-Level Pipeline

```
State space X (arbitrary nature)
       |
       v
[Coarse representation] — initial coarse approximation
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
[Halo blending] — cosine feathering at boundaries
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

---

## Component 2: Halo (Boundary-Aware Blending)

**Why:** Hard insertion of a refined tile creates a step discontinuity at the boundary. The Laplacian flags this as a false HF signal. The smarter the adaptive selection, the stronger the artifact.

**Implementation:**
- Overlap ≥ 3 discretization elements (at tile_size=16: ≥3–4 pixels)
- Cosine feathering relative to coarse level
- Halo initialized to zero, energy-bounded
- Disabling current level → valid rollback to previous

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

**Status:** Production-ready.

---

## Component 6: Refinement Tree

The tree is a log of split decisions. Each root-to-leaf path = a sequence of decisions.

**Requirements:**
- GPU-friendly structure (flat packing, no pointer chasing)
- Morton and block-sparse layouts: **inefficient** (overhead exceeds gain)
- Compact layouts: promising under low sparsity + large grid

**Bush** = a set of paths leading to the same meaning. Distance metric via LCA / common prefix.

---

## Technology Stack

- **Language:** Python
- **ML framework:** PyTorch
- **GPU:** CUDA
- **Experiments:** Jupyter Notebooks
- **Documentation:** Versioned markdown
