# exp10k_cost_surface

**Question:** Is the cost surface C(I, M, p) smooth enough to predict optimal layout from space properties alone?
**Roadmap level:** Track C (deferred)
**Status:** CLOSED — hypothesis not confirmed, deferred

## What we tried

Verify the Layout Selection Invariant (v1.8.3): can we define layout = argmin C(I, M, p)
as a smooth, interpolatable law? If yes — new space types get layout prediction without
full benchmark series. If no — policy table stays as empirical classification.

## Method

Generated synthetic graph spaces with controlled (I, M, p):
- **I** = topological isotropy (degree entropy H(D)) — controls degree inequality
- **M** = metric gap — v1: Kendall tau (broken), v2: spectral gap lambda_2 of Laplacian
- **p** = occupancy (fraction of active nodes)

Four quadrants of (I, M) covered via topology:
- Low I + high M: regular grid
- Low I + low M: Erdos-Renyi
- High I + low M: Barabasi-Albert (global hubs)
- High I + high M: clustered local hubs (feudal lords)

Benchmarked A_bitset, D_direct, D_blocked at each point. 810 trials total.

## Result

**Boundary smoothness: 0.496** (69/139 adjacent pairs agree). JAGGED.

| Layout | Wins |
|--------|------|
| D_direct | 49 |
| D_blocked | 32 |
| A_bitset | **0** |

**Sparse vs dense: SMOOTH and absolute.** A_bitset never won. Any sparse layout beats dense.

**D_direct vs D_blocked: JAGGED.** ~50% of adjacent grid points switch winner.
Not measurement noise — systematic measurement error.

## Why it failed — three measurement flaws

**1. Isotropy blindness (I).** H(D) measures degree inequality, not geometry.
A random 4-regular graph has H(D) = 0 (same as grid) but 100% cache miss rate.
Metric cannot distinguish "uniform order" from "uniform chaos."

**2. Macroscopic metric gap (M).** Both Kendall tau and lambda_2 are global metrics.
Two perfect clusters + one bridge: lambda_2 collapses (signals "terrible connectivity"),
but GPU warps compute inside clusters at 99.9% L1 hit rate. Global topology scalpers
cannot evaluate local cache-line health.

**3. Ghost of the void.** I and M computed on FULL space X, but compute runs on
X_active (< 40% of nodes). The induced subgraph of active nodes has different entropy,
different spectral gap, different cache profile. We evaluated layout cost using
topology of a space that isn't even loaded into registers.

## What would fix it

Corrected metrics (computed on X_active, not X):
- **I_active** = H(D | induced subgraph of active nodes)
- **M_local** = mean |addr(u) - addr(v)| / cache_line_size for edges within X_active
  (direct proxy for L1/L2 cache miss rate, O(edges_active) to compute)
- **p** = unchanged

Two follow-up experiments documented in `exp_deferred_revisit/deferred_revisit_note.md` sec 7:
- **Experiment A:** rescan surface with corrected metrics on synthetic spaces
- **Experiment B:** auto-selection in production pipeline (depends on A being smooth)

## What this means for the project

- **Policy table works.** Empirical classification by space type is correct and sufficient.
  Does not need a continuous law to function.
- **v1.8.3** downgraded from "law" to "hypothesis" in concept_v1.8.md.
- **Confirmed:** sparse always beats dense (useful in itself).
- **Not confirmed:** which sparse variant is optimal as a function of (I, M, p).

## Files
- `exp10k_cost_surface.py` — generator + benchmark + analysis (supports --chunk, --merge)
- `results/chunk_*.json` — raw data (5 chunks, 810 trials)
- `results/exp10k_summary.json` — aggregated results + verdicts
- `results/exp10k_report.md` — auto-generated report
- `results/exp10k_*.png` — cost heatmaps and winner maps
