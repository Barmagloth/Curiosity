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

Phase 0 (Exp0.1–Exp0.8) complete. Phase 1 (P0 layout, DET-1, sensitivity, scale-consistency) complete. P0 Layout **closed** — final layout selection policy across all space types documented in `docs/layout_selection_policy.md`. The exp10 series (8 sub-experiments, 158,000+ trials) determined the optimal layout for each space type.

Next frontier — **Phase 2** (end-to-end pipeline validation).

## Repository Structure

```
README.md                          — project overview (Russian)
README_ENG.md                      — project overview (English, this file)
docs/                              — documentation (Russian)
docs_eng/                          — documentation (English)
  concept_v1.8.md                  — canonical concept document
  layout_selection_policy.md       — layout selection methodology (P0 result)
  experiment_hierarchy.md          — experiment dependency graph and roadmap
  session_handoff.md               — session transfer document
  phase1_plan.md                   — Phase 1 plan (completed)
  target_problem_definition_v1.1.md — project goals, success criteria, Track A→B→C
  scale_consistency_verification_protocol_v1.0.md — SC baseline protocol
  architecture.md                  — system architecture
  workplan.md                      — implementation plan (modules A–F)
  glossary.md                      — project glossary
  teamplan.md                      — team plan
  handoff.md                       — project transfer document (legacy)
  experiment_results.md            — Exp0.1–Exp0.8 results
experiments/
  exp01_poc/                       — Exp0.1: PoC adaptive refinement
  exp02_cifar_poc/                 — Exp0.2: CIFAR PoC
  exp03_halo_diagnostic/           — Exp0.3: halo diagnostic
  exp04_combined_interest/         — Exp0.4: combined interest
  exp05_break_oracle/              — Exp0.5: break oracle
  exp06_adaptive_switch/           — Exp0.6: adaptive ρ switch
  exp07_gate/                      — Exp0.7/0.7b: soft gate + two-stage
  exp08_schedule/                  — Exp0.8: dynamic schedule + governor + probe
  exp09a_layout_sandbox/           — Exp0.9a: layout microbench (CPU sandbox)
  exp10_buffer_scaling/            — P0: grid vs compact on GPU (exp10 series)
  exp10d_seed_determinism/         — DET-1: bitwise determinism validation
  exp10e_tile_sparse/              — P0: tile-sparse candidates (A/B/C)
  exp10f_packed_lookup/            — P0: packed tiles + direct/hash lookup
  exp10g_dual_benchmark/           — P0: dual-mode benchmark (stencil + conv2d)
  exp10h_cross_space/              — P0: cross-space (vector_grid + tree)
  exp10i_graph_blocks/             — P0: block-based addressing for graphs
  exp10j_tree_perlevel/            — P0: per-level tree break-even analysis
  exp11_dirty_signatures/          — dirty signature compression
  exp12a_tau_parent/               — data-driven τ_parent per depth
  exp_deferred_revisit/            — research note: deferred questions
  halo_crossspace/                 — halo applicability across space types
  p2a_sensitivity/                 — sensitivity sweep of gate thresholds
  phase1_halo/                     — Phase 1: halo/overlap hardening
  phase2_probe_seam/               — Phase 2: probe + seam metric
  sc_baseline/                     — SC-baseline: scale-consistency verification
```

## Recommended Reading Order

For a new project participant:

1. **`docs_eng/target_problem_definition_v1.1.md`** — why the project exists, what counts as success
2. **`docs_eng/concept_v1.8.md`** — canonical concept (all validated decisions)
3. **`docs_eng/glossary.md`** — project terminology
4. **`docs_eng/architecture.md`** — architecture and components
5. **`docs_eng/layout_selection_policy.md`** — layout selection methodology by space type
6. **`docs_eng/experiment_results.md`** — Exp0.1–Exp0.8 results with numbers
7. **`docs_eng/experiment_hierarchy.md`** — dependency graph and roadmap
8. **`docs_eng/handoff.md`** — project transfer (status + first task)
9. **`docs_eng/workplan.md`** — implementation plan

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
