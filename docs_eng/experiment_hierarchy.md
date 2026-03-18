# Curiosity — Experiment Hierarchy (v2.0)

This document captures the current status, dependencies, and execution order of experiments.

Updated after Phase 0 (parallel validation: halo cross-space, SC-baseline, D_parent fix).

---

# Mapping: folders → questions → validation plan

| Folder | Question | Validation plan | Status |
|--------|----------|-----------------|--------|
| `exp01_poc/` | Does adaptive refinement work? | — | ✅ Yes (PoC) |
| `exp02_cifar_poc/` | (same, CIFAR) | — | ✅ Yes |
| `exp03_halo_diagnostic/` | Is halo required? | — | ✅ Yes |
| `phase1_halo/` | Halo: r_min, blending, hardened | §A (A1+A2+A3) | ✅ Closed |
| `exp04_combined_interest/` | Is combined interest needed? | — | ✅ Yes |
| `exp05_break_oracle/` | Oracle-free verification | — | ✅ Yes |
| `exp06_adaptive_switch/` | Auto-switch ρ | — | ✅ Yes |
| `exp07_gate/` | Two-stage gate | — | ✅ Yes |
| `exp08_schedule/` | Schedule + governor + probe | — | ✅ Closed |
| `phase2_probe_seam/` | Probe + SeamScore validation | §B (B1+B2) | ✅ Closed |
| `exp09a_layout_sandbox/` | Layout: grid vs compact (microbench) | §C (C3/Exp0.9a) | ✅ Partial |
| `halo_crossspace/` | Halo applicability across space types | Phase 0 | ✅ Closed (rule derived) |
| `sc_baseline/` | Scale-consistency D_parent/D_hf verification | Phase 0 (SC-0..SC-4) | ✅ Closed (SC-5 open) |
| `p2a_sensitivity/` | Sensitivity sweep of gate thresholds | P2a | 🔓 Code ready, not run |
| *(future)* | Layout GPU end-to-end | §C → P0 (0.9b0+) | 🔓 Open |

**Note:** §A/B/C are sections of the validation plan written between Exp0.3 and Phase 1.
In §B, "B1/B2" = probe scenes. In P1 below, "B1/B2/B3" = tree compression. Different contexts.

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
| — | Halo cross-space applicability | **Rule derived** (grid/graph: yes, tree: no). boundary parallelism >= 3 AND no context leakage | Phase 0 |
| — | Morton layout | **Deferred** (sort overhead, zero compute benefit) | 0.9a sandbox |
| — | Block-sparse layout | **Deferred** (expansion ratio) | 0.9a sandbox |
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

## Level SC: Scale-Consistency Invariant (v1.7)

Partially formalizes the meta-question from v1.5 "how not to break features." Depends on P0 (pipeline), independent of P1/P2/P3. Can run in parallel.

### SC-baseline. Verification of Metrics D_parent / D_hf

```
SC-baseline. Scale-Consistency Verification
├── SC-0: fix pair (R, Up), verify R idempotence                          ✅ COMPLETE
├── SC-1: prepare positive (strong + empirical) and negative baselines    ✅ COMPLETE
├── SC-2: compute D_parent, D_hf across all cases                        ✅ COMPLETE
├── SC-3: analyze separability (AUC, effect size, quantile separation)    ✅ COMPLETE
│         globally + by depth + by structure type
├── SC-4: kill criterion — PASSED with updated D_parent formula
│         (R=gauss sigma=3.0, lf_frac normalization, AUC=0.853, d=1.491) ✅ COMPLETE
└── SC-5: set data-driven tau_parent[L] — needs data-driven threshold setting
```

**Kill criterion:** Global ROC-AUC >= 0.75, Depth-conditioned AUC >= 0.65, Effect size >= medium (d >= 0.5). If not met — change metrics, **do not** tweak thresholds.

**SC-4 result:** PASSED. Updated D_parent formula: `||R(delta)|| / (||delta|| + epsilon)`, R=gauss sigma=3.0. AUC=0.853, d=1.491. Cross-space: T1=1.000, T2=1.000, T3=1.000, T4=0.824 (all >= 0.75).

**SC-5 status:** tau_parent needs data-driven threshold setting.

**SC-baseline output:** validated thresholds tau_parent[L] or decision to revise metric construction.

Full protocol: `docs/scale_consistency_verification_protocol_v1.0.md`.

### SC-enforce. Enforcement (after SC-baseline)

```
SC-enforce. Scale-Consistency Enforcement
├── damp step_delta / reject split / increase local strictness when D_parent > τ_parent
└── D_parent as contextual signal in ρ (not self-sufficient)
```

### Exp0.10. (R, Up) Sensitivity Probe (after SC-baseline)

```
Exp0.10. (R, Up) Sensitivity Probe
├── Dependency: SC-baseline (need validated metrics for one pair first)
├── Pairs to test:
│   ├── gaussian + bilinear (current default)
│   ├── box + nearest (coarsest variant)
│   ├── Lanczos + bicubic (more precise)
│   └── haar wavelet decomposition (fundamentally different decomposition)
├── Measured:
│   ├── D_parent / D_hf distributions — do thresholds shift?
│   ├── Tree topology divergence — same data → different trees?
│   ├── PSNR ceiling — does R affect quality at equal budget?
│   └── SC-baseline separability — does ROC-AUC change?
└── Kill criterion:
    if topology + D_parent stable (±20%) across pairs → default justified
    if divergence > 50% → need pair selection mechanism (new open question)
```

**Exp0.10 output:** either "choice of R is not critical, default stands" or "pair selection mechanism needed."

---

## Level 4: Global Coherence ("Don't Break Features")

Depends on **everything above** + SC-baseline. Meta-question from Concept v1.5, partially formalized through Scale-Consistency Invariant (Concept v1.7, section 8).

### P4. Representation Coherence Under Non-Uniform Depth

```
P4. "Don't break features"
├── P4a: downstream consumer test
│        feed adaptive-refined representation into a simple downstream
│        (classifier / autoencoder)
│        compare with dense-refined and coarse-only
│        question: does downstream break under non-uniform depth?
│        (with scale-consistency enforcement vs. without)
│
├── P4b: matryoshka invariant
│        verify that the representation at any "matryoshka" level
│        is valid as consumer input
│        (not just visually smooth, but functionally correct)
│
└── P4c: guarantee mechanism (if P4a/b show a problem)
         options: padding/projection layer, consistency loss,
         depth-aware normalization, stricter τ_parent
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
 ├──→ P2 (ρ auto-tuning)                ├──→ C-pre
 │                                        │
 └──→ SC-baseline (✅ SC-0..SC-4) ──→ SC-5 ──→ SC-enforce ──→ P4 ("don't break features")
                                              depends on P0 + P1 + P2 + P3 + SC
```

**Critical path:** P0 → P1 → P3 → P4.

**Parallel branches:** P2 and SC-baseline — both run in parallel with P1, all are needed before P4.

---

# Working Order

1. **P0: 0.9b0** — buffer-scaling probe, kill/go for compact
2. **P0: 0.9b/0.9c/0.9h** — if compact survives; otherwise lock in grid
3. **P1-B2** — dirty signatures (parallel with P0 GPU part, runs on CPU)
4. **P2a** — sensitivity sweep of gate thresholds, **5 scenes × 4 space types** (code ready, parallel with P1)
5. **SC-5** — set data-driven τ_parent[L] (SC-0..SC-4 ✅ complete; parallel with P1)
6. **P1-B1** — segment compression (after B2)
7. **P1-B3** — anchors + rebuild (after B1+B2)
8. **SC-enforce** — enforcement of scale-consistency (after SC-5)
9. **P3a/P3b** — tree semantics (after P1)
10. **C-pre** — profile cluster check (after P3, cheap)
11. **P4** — "don't break features" (after everything + SC)

---

# Naming Convention (v3+)

Historically, numbering grew organically: chronological IDs (0.1–0.9a),
validation plan sections (§A/B/C), roadmap levels (P0–P4),
sub-experiments within levels (B1–B3 in P1). Result: confusion.

**Rules for new experiments:**

1. **Single sequential numbering.** Next experiment = `exp10`.
   Integer numbering, no dots (dots confused with sub-versions).
   Number = creation order. Never reused.

2. **Sub-experiments use lowercase letter.** `exp10a`, `exp10b`, `exp10c`.
   One series = one numeric root.

3. **Folder = `exp{N}{suffix}_{short_name}/`.** Examples:
   `exp10_buffer_scaling/`, `exp10a_synthetic_kernel/`, `exp11_dirty_signatures/`.

4. **Roadmap mapping — only in this document**, not in folder names.
   Folders don't contain "P0" or "B2" in their names.

5. **Each folder contains README.md** (short, 5–15 lines):
   - Question/hypothesis (one sentence)
   - Kill criteria
   - Link to roadmap level (P0/P1/P2/...)
   - Status (open / closed / killed)

6. **Legacy names are not renamed.** `phase1_halo/`, `phase2_probe_seam/`,
   `exp09a_layout_sandbox/` — historical legacy, linked to new numbering
   via the mapping table above.

**Working order → experiment number mapping:**

| Step | Roadmap | Description | Future exp# |
|------|---------|-------------|-------------|
| 1 | P0 | buffer-scaling probe (kill/go compact) | exp10 |
| 2 | P0 | end-to-end pipeline grid vs compact | exp10a/b/c |
| 3 | P1-B2 | dirty signatures | exp11 |
| 4 | P2a | sensitivity sweep of gate thresholds (5 scenes × 4 spaces) | exp12 |
| 5 | SC-5 | set data-driven τ_parent[L] (SC-0..SC-4 ✅) | exp12a |
| 6 | P1-B1 | segment compression | exp13 |
| 7 | P1-B3 | anchors + rebuild | exp14 |
| 8 | SC-enforce | enforcement of scale-consistency | exp14a |
| 9 | P3a/b | tree semantics | exp15 |
| 10 | C-pre | profile cluster check | exp16 |
| 11 | P4 | "don't break features" | exp17 |

Numbers are provisional. If an unplanned experiment arises between steps,
it gets the next free number.

---

# Instrument Readiness Gate

All experiments P0–P4 + SC belong to **Track A** (building the instrument). Transition to **Track B** (researching tree structure) requires passing the Instrument Readiness Gate:

1. **Invariant pass** — all mandatory invariants hold
2. **Overhead profile** — overhead does not consume the gain
3. **Stability pass** — system is stable across runs
4. **One validated benchmark** — adaptive > random > coarse with confirmed numbers
5. **Attribution diagnostics** — each module's contribution measured (ablation)

Details: `docs/target_problem_definition_v1.1.md`.

After successful Track B, **Track C** opens (generalization to non-spatial domains: graphs, latent spaces, activations). Long-term ambition, not a current goal.

---

# Principles

* Judge numbers first, then ambitions.
* No "next stage" is locked in advance.
* Kill criteria are two-sided (speed + memory).
* Forensic-grade protocol: controls, Holm-Bonferroni, cost-fair comparisons.
