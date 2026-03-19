# exp10k Cost Surface Report

Device: cuda
N = 1024, feat_dim = 16, seeds = 10
Total trials: 810 valid, 0 errors

## Layout Win Counts

- D_blocked: 32/81 (39.5%)
- D_direct: 49/81 (60.5%)

## Smoothness Analysis

Boundary smoothness score: **0.4964**
Adjacent pairs agreeing: 69/139
Boundary transitions: 70

**JAGGED -- The cost surface is noisy. Layout selection is a classification problem, not a deterministic law.**

## Decision Boundaries

Transitions between layouts at adjacent grid points:

| From (I, M, p) | To (I, M, p) | From Layout | To Layout |
|-----------------|--------------|-------------|-----------|
| (0.0, 0.0, 0.01) | (0.5, 0.0, 0.01) | D_direct | D_blocked |
| (0.0, 0.0, 0.01) | (0.0, 0.2, 0.01) | D_direct | D_blocked |
| (0.0, 0.0, 0.01) | (0.0, 0.0, 0.05) | D_direct | D_blocked |
| (0.0, 0.0, 0.05) | (0.5, 0.0, 0.05) | D_blocked | D_direct |
| (0.0, 0.0, 0.05) | (0.0, 0.2, 0.05) | D_blocked | D_direct |
| (0.0, 0.0, 0.05) | (0.0, 0.0, 0.1) | D_blocked | D_direct |
| (0.0, 0.0, 0.1) | (0.5, 0.0, 0.1) | D_direct | D_blocked |
| (0.0, 0.0, 0.1) | (0.0, 0.2, 0.1) | D_direct | D_blocked |
| (0.0, 0.0, 0.2) | (0.5, 0.0, 0.2) | D_direct | D_blocked |
| (0.0, 0.0, 0.3) | (0.0, 0.0, 0.5) | D_direct | D_blocked |
| (0.0, 0.0, 0.5) | (0.5, 0.0, 0.5) | D_blocked | D_direct |
| (0.0, 0.0, 0.5) | (0.0, 0.0, 0.7) | D_blocked | D_direct |
| (0.0, 0.2, 0.01) | (0.0, 0.2, 0.05) | D_blocked | D_direct |
| (0.0, 0.2, 0.05) | (0.0, 0.2, 0.1) | D_direct | D_blocked |
| (0.0, 0.2, 0.1) | (0.5, 0.2, 0.1) | D_blocked | D_direct |
| (0.0, 0.2, 0.1) | (0.0, 0.2, 0.2) | D_blocked | D_direct |
| (0.0, 0.8, 0.7) | (0.0, 1.0, 0.7) | D_direct | D_blocked |
| (0.0, 1.0, 0.05) | (0.5, 1.0, 0.05) | D_direct | D_blocked |
| (0.0, 1.0, 0.05) | (0.0, 1.0, 0.1) | D_direct | D_blocked |
| (0.0, 1.0, 0.1) | (0.0, 1.0, 0.2) | D_blocked | D_direct |
| (0.0, 1.0, 0.5) | (0.0, 1.0, 0.7) | D_direct | D_blocked |
| (0.0, 1.0, 0.7) | (0.5, 1.0, 0.7) | D_blocked | D_direct |
| (0.5, 0.0, 0.01) | (1.0, 0.0, 0.01) | D_blocked | D_direct |
| (0.5, 0.0, 0.01) | (0.5, 0.0, 0.05) | D_blocked | D_direct |
| (0.5, 0.0, 0.05) | (0.5, 0.0, 0.1) | D_direct | D_blocked |
| (0.5, 0.0, 0.1) | (0.5, 0.2, 0.1) | D_blocked | D_direct |
| (0.5, 0.0, 0.2) | (0.5, 0.0, 0.3) | D_blocked | D_direct |
| (0.5, 0.2, 0.01) | (0.5, 0.2, 0.05) | D_blocked | D_direct |
| (0.5, 0.6, 0.5) | (1.0, 0.6, 0.5) | D_blocked | D_direct |
| (0.5, 0.6, 0.5) | (0.5, 0.8, 0.5) | D_blocked | D_direct |
| (0.5, 0.6, 0.5) | (0.5, 0.6, 0.7) | D_blocked | D_direct |
| (0.5, 0.6, 0.7) | (1.0, 0.6, 0.7) | D_direct | D_blocked |
| (0.5, 0.8, 0.05) | (1.0, 0.8, 0.05) | D_direct | D_blocked |
| (0.5, 0.8, 0.05) | (0.5, 1.0, 0.05) | D_direct | D_blocked |
| (0.5, 0.8, 0.1) | (1.0, 0.8, 0.1) | D_direct | D_blocked |
| (0.5, 0.8, 0.1) | (0.5, 1.0, 0.1) | D_direct | D_blocked |
| (0.5, 0.8, 0.1) | (0.5, 0.8, 0.2) | D_direct | D_blocked |
| (0.5, 0.8, 0.2) | (0.5, 1.0, 0.2) | D_blocked | D_direct |
| (0.5, 0.8, 0.2) | (0.5, 0.8, 0.3) | D_blocked | D_direct |
| (0.5, 0.8, 0.3) | (1.0, 0.8, 0.3) | D_direct | D_blocked |
| (0.5, 1.0, 0.01) | (0.5, 1.0, 0.05) | D_direct | D_blocked |
| (0.5, 1.0, 0.1) | (0.5, 1.0, 0.2) | D_blocked | D_direct |
| (1.0, 0.4, 0.3) | (1.5, 0.4, 0.3) | D_blocked | D_direct |
| (1.0, 0.4, 0.3) | (1.0, 0.6, 0.3) | D_blocked | D_direct |
| (1.0, 0.4, 0.3) | (1.0, 0.4, 0.5) | D_blocked | D_direct |
| (1.0, 0.4, 0.5) | (1.0, 0.4, 0.7) | D_direct | D_blocked |
| (1.0, 0.6, 0.01) | (1.0, 0.6, 0.05) | D_direct | D_blocked |
| (1.0, 0.6, 0.05) | (1.0, 0.6, 0.1) | D_blocked | D_direct |
| (1.0, 0.6, 0.1) | (1.5, 0.6, 0.1) | D_direct | D_blocked |
| (1.0, 0.6, 0.1) | (1.0, 0.8, 0.1) | D_direct | D_blocked |
| ... | ... | ... | ... |
| (70 total transitions) | | | |

## Interpretation

If smoothness > 0.85: the layout switching boundary is a smooth surface in (I, M, p) space. The Layout Selection Invariant holds as a deterministic law: given (I, M, p), the optimal layout is predictable.

If smoothness 0.70-0.85: boundaries exist but are noisy. The invariant is a useful heuristic with some stochastic overlap zones.

If smoothness < 0.70: the cost surface is chaotic. Layout selection is a classification problem requiring per-instance benchmarking, not a closed-form law.

