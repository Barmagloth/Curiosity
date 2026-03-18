# Curiosity — Adaptive Refinement System

## What Is This

Curiosity is a research ML project building a system of **adaptive refinement** for abstract computational spaces.

The system is not tied to images or any specific domain. It operates on an arbitrary state space X, where an informativeness function ρ(x) determines where refinement is justified.

## Key Idea

> Dimensionality is not a fixed number but a depth of refinement.
> Refinement should occur only where it is justified informationally and budget-wise.

The goal is to reshape the search space so that:
- it can be explored **adaptively**
- computational resources are spent only where they yield information
- feature interdependencies are preserved (no "broken features")

## Project Status

The experiment series Exp0.1–Exp0.8 is **complete**. Results are consolidated into v1.8 documentation. The system is validated: adaptive refinement works and outperforms random selection under budget constraints. The Scale-Consistency Invariant has been formalized. v1.8 adds determinism invariants (DET-1, DET-2).

Next frontier — Exp0.9b0 (buffer-scaling probe, P0) and SC-baseline (Scale-Consistency baseline validation).

## Repository Structure

```
README.md                          — project overview (Russian)
README_ENG.md                      — project overview (English, this file)
docs/                              — documentation (Russian)
docs_eng/                          — documentation (English)
  target_problem_definition_v1.1.md — project goals, success criteria, Track A→B→C
  concept_v1.8.md                  — canonical concept document
  concept_v1.7_historical.md       — previous version (after Phase 0)
  concept_v1.5_historical.md       — previous version (after Exp0.1–Exp0.8)
  concept_v1.4_historical.md       — early version (after Exp0.2–0.3)
  scale_consistency_verification_protocol_v1.0.md — SC baseline experiment protocol
  handoff_v1.5_to_v1.6.md          — changelog v1.5→v1.6
  experiment_results.md            — all experiment results Exp0.1–Exp0.8
  experiment_hierarchy.md          — experiment dependency graph and roadmap
  architecture.md                  — system architecture and key decisions
  workplan.md                      — implementation plan (modules A–F, mini-roadmap)
  handoff.md                       — project transfer document
  glossary.md                      — project glossary
experiments/
  ARTIFACT_INVENTORY.md            — artifact inventory from Claude/ChatGPT dialogs
  exp01_poc/                       — Exp0.1: PoC adaptive refinement (IPYNB)
  exp02_cifar_poc/                 — Exp0.2: CIFAR PoC (IPYNB)
  exp03_halo_diagnostic/           — Exp0.3: halo diagnostic (IPYNB)
  exp04_combined_interest/         — Exp0.4: combined interest (code+protocol+data)
  exp05_break_oracle/              — Exp0.5: break oracle (code+data)
  exp06_adaptive_switch/           — Exp0.6: adaptive ρ switch (code)
  exp07_gate/                      — Exp0.7/0.7b: soft gate + two-stage (code+protocol+data)
  exp08_schedule/                  — Exp0.8: dynamic schedule (code+design+data)
  exp09a_layout_sandbox/           — Exp0.9a: layout microbench (validation plan §C / C3)
  phase1_halo/                     — Validation plan §A (A1+A2+A3): halo/overlap hardening
  phase2_probe_seam/               — Validation plan §B (B1+B2): probe + seam metric
```

## Recommended Reading Order

For a new project participant:

1. **`docs_eng/target_problem_definition_v1.1.md`** — why the project exists, what counts as success
2. **`docs_eng/concept_v1.8.md`** — canonical concept (all validated decisions)
3. **`docs_eng/glossary.md`** — project terminology
4. **`docs_eng/architecture.md`** — architecture and components
5. **`docs_eng/experiment_results.md`** — Exp0.1–Exp0.8 results with numbers
6. **`docs_eng/experiment_hierarchy.md`** — dependency graph and roadmap
7. **`docs_eng/handoff.md`** — project transfer (status + first task)
8. **`docs_eng/workplan.md`** — implementation plan

## Technologies

- Python, PyTorch, CUDA (GPU)
- Jupyter Notebooks for experiments
- Versioned markdown for documentation

## Independently Valuable Components

The system is built from modules that survive the architecture scrutiny not because they are "useful in the overall picture," but because they provide standalone value:

- **Content-addressable cache** — memoization by input hash; reduces redundant computation
- **Incremental recomputation** — differential update on tree changes; avoids full re-evaluation
- **ROI attention mask (ρ function)** — informativeness-weighted candidate selection; works across domains
- **Adaptive complexity mapping (quadtree/octree + hysteresis)** — progressive space partitioning; generalizes to any dimensionality
- **SeamScore** — seam quality metric; measures smoothness at boundaries independent of the refinement system
- **Selection principle:** only modules that provide standalone value survive the kill criteria. All components above pass this test independently.

## Methodology

- Every claim is backed by data ("judge numbers, then ambitions")
- Forensic-grade protocols: explicit controls, Holm-Bonferroni corrections, observable-only diagnostics
- Cost-fair comparisons: computational overhead accounted for
- Kill criteria defined before each experiment
- No oracle information in metrics

## Authorship

Project led by Barmagloth. Research conducted with Claude (Anthropic) and ChatGPT (OpenAI) as assistants.
