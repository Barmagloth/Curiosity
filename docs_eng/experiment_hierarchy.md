# Curiosity Experiment Hierarchy (v2.0)

This document describes the experiment dependency graph, statuses, and execution order.

---

## Completed Series: Exp0.1–Exp0.8

All experiments below are complete; results are incorporated into concept_v1.5.md.

```
Exp0.1 (adaptive vs random)
  |
  v
Exp0.2 (real images)
  |
  v
Exp0.3 (halo / boundary blending)
  |
  v
Exp0.4 (residual-only on clean data)
  |
  v
Exp0.5 (residual degradation under noise)
  |
  v
Exp0.6 (binary switch)
  |
  v
Exp0.7 -> Exp0.7b (soft gating -> two-stage gate)
  |
  v
Exp0.8 (budget governor + phase schedule)
```

---

## Active Frontier: Exp0.9

### Exp0.9a — Complete
Status: incorporated into v1.5 documentation.

### Exp0.9b0 — Buffer-scaling probe (P0)
Status: **next in queue**.
Priority: P0 (highest).
Kill criteria: defined.

---

## Next Priority Hierarchy (P0–P4)

```
P0: GPU layout
  — Morton and block-sparse layouts: inefficient due to overhead
  — Compact layouts: promising under low sparsity + large grid
  — Status: partially explored, negative result for Morton

P1: Tree compression
  — Branch compression (segment compression)
  — Depends on: stable tree structure

P2: Auto-tuning
  — Auto-tuning of gate thresholds (instability/FSR thresholds)
  — Currently manual

P3: Tree semantics
  — Bushes, LCA-distance as feature
  — Does the tree provide a semantic metric?

P4: Downstream compatibility
  — "Don't break features": compatibility guarantee with consumers
  — Non-uniform refinement depth vs. feature vector coherence
```

---

## Longer Horizon (No Concrete Dates)

These directions are flagged as **premature** without concrete multi-objective use cases:

- **DAG-based routing** — refinement routing via directed acyclic graph instead of tree
- **Wasserstein distance routing** — route selection by Wasserstein distance
- **Segment compression** — tree branch compression
- **Discrete signature systems** — discrete signatures for dirty detection

---

## Planning Principle

Experiments are structured with **explicit kill criteria** before execution. If an experiment fails its kill criteria, it is closed — not patched.

Execution order: strictly P0→P1→P2→P3→P4. No skipping ahead without closing dependencies.
