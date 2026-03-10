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

The experiment series Exp0.1–Exp0.8 is **complete**. Results are consolidated into v1.5 documentation. The system is validated: adaptive refinement works and outperforms random selection under budget constraints.

Next frontier — Exp0.9b0 (buffer-scaling probe, P0).

## Repository Structure

```
README.md                          — project overview (Russian)
README_ENG.md                      — project overview (English, this file)
docs/                              — documentation (Russian)
docs_eng/                          — documentation (English)
  concept_v1.5.md                  — canonical concept document
  concept_v1.4_historical.md       — early concept version (after Exp0.2–0.3)
  experiment_results.md            — all experiment results Exp0.1–Exp0.8
  experiment_hierarchy.md          — experiment dependency graph and roadmap
  architecture.md                  — system architecture and key decisions
  workplan.md                      — implementation plan (modules A–F, mini-roadmap)
  handoff.md                       — project handoff document
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

## Technologies

- Python, PyTorch, CUDA (GPU)
- Jupyter Notebooks for experiments
- Versioned markdown for documentation

## Methodology

- Every claim is backed by data ("judge numbers, then ambitions")
- Forensic-grade protocols: explicit controls, Holm-Bonferroni corrections, observable-only diagnostics
- Cost-fair comparisons: computational overhead accounted for
- Kill criteria defined before each experiment
- No oracle information in metrics

## Authorship

Project led by Barmagloth. Research conducted with Claude (Anthropic) and ChatGPT (OpenAI) as assistants.
