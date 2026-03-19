# Exp10g: Dual-Mode Benchmark (Layout vs Operator)

## Motivation

Exp10f showed that packed tiles + direct tile_map (candidate D) achieves:
- 5x faster compute than grid baseline
- 5.5x smaller resident memory

BUT fails the peak VRAM criterion because `F.conv2d` allocates temporary
workspace buffers that dominate peak memory at small grid sizes.

This conflates two separate questions:
1. Is the packed tile **layout** better? (addressing + memory access pattern)
2. Is the packed tile layout compatible with **PyTorch operators**?

Exp10g separates these with two benchmark modes.

## Benchmark Modes

### Mode 1: Manual Stencil (Layout Benchmark)

- NO `F.conv2d`, NO PyTorch operator temporaries
- Manual 3x3 stencil: for each active element, read 8 neighbors + self,
  compute weighted average
- Grid/A_bitset: uses `torch.roll` shifts (no temporaries)
- D_direct: uses tile_map neighbor lookup + padded tile assembly
- Isolates **pure layout cost**: addressing overhead + memory access pattern

### Mode 2: Conv2d (Operator Benchmark)

- `F.conv2d` with 3x3 kernel, same as exp10f
- Shows real operational cost including workspace allocations
- Reveals whether the operator implementation (not the layout) is the
  bottleneck

## Memory Measurement

Three numbers per configuration:

| Metric | Definition |
|--------|-----------|
| `resident_bytes` | GPU memory after layout build, before compute |
| `peak_bytes` | `max_memory_allocated` during compute |
| `workspace_bytes` | `peak - resident` (temporary overhead) |

## Dual Kill Criteria

### Contour A (Architectural Viability) -- Mode 1

- `resident_bytes`: D must be < grid
- `wall_clock`: D must not be >20% slower than grid
- `build_cost`: reported separately (not a kill criterion)

### Contour B (Operational Viability) -- Mode 2

- `peak_bytes`: overhead >20% vs grid -> FAIL
- `wall_clock`: overhead >20% vs grid -> FAIL

## Candidates

| Name | Description | Source |
|------|------------|--------|
| `grid_baseline` | Full data tensor [side*side] + bool mask | Inline |
| `A_bitset` | Full grid + packed bitset mask | exp10e |
| `D_direct` | Packed tiles + tile_map + neighbor_slots | exp10f |

## Sweep Parameters

- sides: [64, 128, 256]
- sparsities: [0.05, 0.10, 0.20, 0.30, 0.50, 0.70]
- patterns: ['random', 'clustered', 'checkerboard']
- tile_size: 8
- n_seeds: 10
- n_warmup: 5, n_repeat: 20

## Outputs

All saved to `results/` subdirectory:

- `exp10g_summary.json` -- full data + verdicts
- `exp10g_report.md` -- human-readable report with tables
- `exp10g_resident_comparison.png` -- resident memory (Mode 1)
- `exp10g_peak_comparison.png` -- peak memory (Mode 2)
- `exp10g_workspace_comparison.png` -- workspace overhead (both modes)
- `exp10g_time_comparison.png` -- wall-clock (both modes)

## Running

```bash
cd R:\Projects\Curiosity
.venv-gpu\Scripts\python experiments\exp10g_dual_benchmark\exp10g_dual_benchmark.py
```

## Expected Outcome

If Contour A passes and Contour B fails, we know the packed tile layout
is architecturally sound but needs a custom stencil operator (not F.conv2d)
to avoid workspace overhead. This would justify investing in a fused
tile-stencil kernel for production use.
