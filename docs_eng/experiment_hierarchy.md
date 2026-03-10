# Curiosity — Experiment Hierarchy (v2.0)

This document captures the current status, dependencies, and execution order of experiments.

Updated after Phase 1–2 (halo, probe, seam metric) and Exp0.9a sandbox (layout).

---

# Closed (No Further Experiments Needed)

| # | Question | Status | Source |
|---|----------|--------|--------|
| 1 | Does adaptive refinement work? | **Yes** | PoC, Exp0.1–0.2 |
| — | Is halo mandatory? | **Yes**, w ∈ [2,4], cosine feather | Exp0.2–0.3, Phase 1 |
| — | Is probe mandatory? | **Yes**, uncert, 5–10% budget | Exp0.8, Phase 2 |
| — | SeamScore as production metric | **Yes**, dual check works in 4 spaces | Phase 2 |
| — | Governor (EMA) for budget | **Yes**, StdCost −50%, penalty −85% | Exp0.8 |
| 2 | Is combined interestingness needed? | **Yes, under signal degradation.** Two-stage gate | Exp0.4–0.7b |
| 3 | Phase schedule by depth? | **No** under current conditions | Exp0.8v5 |
| — | Morton layout | **Dead** (sort overhead, zero compute benefit) | 0.9a sandbox |
| — | Block-sparse layout | **Dead** (expansion ratio) | 0.9a sandbox |
| — | Non-overlapping writes determinism | **Clean** (bitwise match) | 0.9a sandbox |

---

# Open — Current Hierarchy

## Level 0: Infrastructure Prerequisites

Without answers to these questions, everything above is decorative.

### P0. GPU Layout: grid vs compact

The only surviving candidate. No upward dependencies — this is the foundation.

```
P0. Layout (grid vs compact on GPU)
├── 0.9b0: buffer-scaling probe — O(k) vs O(M) buffers, synthetic kernel
│         kill/go for compact
├── 0.9b:  end-to-end pipeline (if compact survives)
│         grid vs compact, 4 spaces × 2 budgets
│         metrics: VRAM, wall-clock, #kernels, SeamScore identity
├── 0.9c:  scaling (if compact survives)
│         sweep over M, clustered + random
└── 0.9h:  halo overlap determinism on GPU
          overlapping writes → accumulation order
```

**P0 output:** fixed layout for everything downstream. Either grid (most likely) or compact (if O(k) buffers save it).

---

## Level 1: Route Representation

Depends on P0 (layout determines how active_idx enters the pipeline). Does not depend on tree "meaning" — pure storage engineering.

### P1. Compression and Structure Maintenance

```
P1. Tree / route compression
├── B2: dirty signatures (12 bits: seam_risk + uncert + mass)
│       debounce (2 consecutive hits)
│       scenarios: noise / structural event / drift
│       metrics: blast radius, latency-to-trigger, burstiness
│       ↑ this is the foundation — without it B1 and B3 don't know when to fire
│
├── B1: segment compression (degree-2 + signature-stable + length cap)
│       depends on B2 (merge criterion = signature stability)
│       metrics: memory vs node-per-node, local update cost
│
└── B3: anchors + periodic rebuild
        depends on B1 + B2
        scenario: frequent local updates
        comparison: (a) local only (b) local + periodic rebuild
        metrics: total cost over N steps, "dirt" accumulation
```

**P1 output:** tree storage format (flat nodes vs segments), dirty detection mechanism, rebuild strategy.

---

## Level 2: Combined Signal Reliability

Depends on P0 (pipeline works), does not depend on P1 (compression is orthogonal).

### P2. Auto-Tuning and ρ Robustness

Two-stage gate confirmed (Exp0.7b), but thresholds (instability, FSR) are manual.

```
P2. ρ-gate auto-tuning
├── P2a: sensitivity analysis — sweep instability/FSR thresholds
│        on existing scenes (clean/noise/blur/spatvar/jpeg)
│        question: how flat is the "optimality ridge"?
│        if wide → manual thresholds are fine, auto-tuning not needed
│        if narrow → adaptive threshold needed
│
└── P2b: adaptive threshold (only if P2a shows narrow ridge)
         online estimation of instability/FSR percentiles
         metrics: PSNR stability across scenes, overhead
```

**P2 output:** either "manual thresholds ± 30% — no difference" (and the question is closed), or a concrete auto-tuning mechanism.

---

## Level 3: Tree Semantics

Depends on P0 (layout) + P1 (storage format).

### P3. Does the Tree Provide a Semantic Metric?

```
P3. Tree semantics
├── P3a: LCA-distance as feature
│        on a real tree from the pipeline: does LCA-distance
│        correlate with "semantic similarity" (‖feature_i − feature_j‖)?
│        if no → tree = just a log, not a metric
│
├── P3b: bushes — path clusters
│        are there natural clusters among leaf paths?
│        metrics: silhouette, stability across runs
│
└── C-pre: profile discreteness check
           trajectory features (EMA quantiles, split signatures, stability)
           question: is there cluster structure?
           if yes → C unfreezes
           if no → C is dead
```

**P3 output:** either "tree is purely a log, provides no semantic metric" (ok, not a problem), or a concrete method for extracting semantics.

---

## Level 4: Global Coherence ("Don't Break Features")

Depends on **everything above**. Meta-question from Concept v1.5.

### P4. Representation Coherence Under Non-Uniform Depth

```
P4. "Don't break features"
├── P4a: downstream consumer test
│        feed adaptive-refined representation into a simple downstream
│        (classifier / autoencoder)
│        compare with dense-refined and coarse-only
│        question: does downstream break under non-uniform depth?
│
├── P4b: matryoshka invariant
│        verify that the representation at any "matryoshka" level
│        is valid as consumer input
│        (not just visually smooth, but functionally correct)
│
└── P4c: guarantee mechanism (if P4a/b show a problem)
         options: padding/projection layer, consistency loss,
         depth-aware normalization
```

**P4 output:** either "non-uniform depth doesn't break downstream" (and the question is closed), or a concrete protection mechanism.

---

# Frozen

## C. DAG + Profiles

**Entry contract (all three simultaneously):**

1. At least two irreducible objectives (cannot be collapsed into a scalar without semantic loss)
2. A concrete downstream consumer that actually relies on these objectives
3. An observable conflict: different optimal solutions under different objectives on the same data

Pre-experiment C-pre (in P3) may signal unfreezing, but is not sufficient by itself.

If the contract is not met — freeze is indefinite.

---

# Dependency Graph

```
P0 (GPU layout)
 ├──→ P1 (tree compression)  ──→ P3 (tree semantics)
 │                                       │
 └──→ P2 (ρ auto-tuning)                ├──→ C-pre
                                          │
                              P4 ("don't break features")
                              depends on P0 + P1 + P2 + P3
```

**Critical path:** P0 → P1 → P3 → P4.

**Parallel branch:** P2 (auto-tuning) runs in parallel with P1; both are needed before P4.

---

# Working Order

1. **P0: 0.9b0** — buffer-scaling probe, kill/go for compact
2. **P0: 0.9b/0.9c/0.9h** — if compact survives; otherwise lock in grid
3. **P1-B2** — dirty signatures (parallel with P0 GPU part, runs on CPU)
4. **P2a** — sensitivity sweep of gate thresholds (parallel with P1)
5. **P1-B1** — segment compression (after B2)
6. **P1-B3** — anchors + rebuild (after B1+B2)
7. **P3a/P3b** — tree semantics (after P1)
8. **C-pre** — profile cluster check (after P3, cheap)
9. **P4** — "don't break features" (after everything)

---

# Principles

* Judge numbers first, then ambitions.
* No "next stage" is locked in advance.
* Kill criteria are two-sided (speed + memory).
* Forensic-grade protocol: controls, Holm-Bonferroni, cost-fair comparisons.
