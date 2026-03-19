# exp10i: Block-Based Addressing for Irregular Graphs -- Report

## Summary

- **Total configs**: 288
- **Device**: CPU (parallel chunked run)
- **Contour A** (time overhead < 20% AND cross_block_ratio < 0.50): **120/288** PASS (42%)
- **Contour B** (time overhead < 20%): **288/288** PASS (100%)
- **Padding warnings** (waste > 50%): 240/288

> Note: Run on CPU; memory contours (resident/peak VRAM) are not
> evaluated. Verdicts focus on wall-clock overhead and blocking quality.

## Results by graph type

### random_geometric

Contour A: 56/96 | Contour B: 96/96 | Padding warnings: 80

| N | sp | bs | partition | A | B | time_oh | cbr | pw |
|---|----|----|----------|---|---|---------|-----|-----|
| 256 | 0.05 | 8 | random | FAIL | PASS | -81.0% | 0.97 | 0.85 !! |
| 256 | 0.05 | 8 | spatial | FAIL | PASS | -75.5% | 0.52 | 0.85 !! |
| 256 | 0.05 | 8 | greedy | PASS | PASS | -81.3% | 0.34 | 0.86 !! |
| 256 | 0.05 | 16 | random | FAIL | PASS | -70.5% | 0.94 | 0.91 !! |
| 256 | 0.05 | 16 | spatial | PASS | PASS | -80.4% | 0.34 | 0.91 !! |
| 256 | 0.05 | 16 | greedy | PASS | PASS | -79.7% | 0.19 | 0.92 !! |
| 256 | 0.05 | 32 | random | FAIL | PASS | -78.3% | 0.88 | 0.94 !! |
| 256 | 0.05 | 32 | spatial | PASS | PASS | -76.5% | 0.21 | 0.94 !! |
| 256 | 0.05 | 32 | greedy | PASS | PASS | -77.3% | 0.10 | 0.95 !! |
| 256 | 0.05 | 64 | random | FAIL | PASS | -73.1% | 0.76 | 0.95 !! |
| 256 | 0.05 | 64 | spatial | PASS | PASS | -78.1% | 0.11 | 0.95 !! |
| 256 | 0.05 | 64 | greedy | PASS | PASS | -78.4% | 0.06 | 0.96 !! |
| 256 | 0.1 | 8 | random | FAIL | PASS | -89.6% | 0.97 | 0.83 !! |
| 256 | 0.1 | 8 | spatial | FAIL | PASS | -87.5% | 0.52 | 0.81 !! |
| 256 | 0.1 | 8 | greedy | PASS | PASS | -87.5% | 0.35 | 0.82 !! |
| 256 | 0.1 | 16 | random | FAIL | PASS | -89.0% | 0.94 | 0.88 !! |
| 256 | 0.1 | 16 | spatial | PASS | PASS | -86.7% | 0.34 | 0.86 !! |
| 256 | 0.1 | 16 | greedy | PASS | PASS | -86.2% | 0.18 | 0.88 !! |
| 256 | 0.1 | 32 | random | FAIL | PASS | -89.8% | 0.88 | 0.90 !! |
| 256 | 0.1 | 32 | spatial | PASS | PASS | -87.3% | 0.21 | 0.89 !! |
| 256 | 0.1 | 32 | greedy | PASS | PASS | -88.0% | 0.10 | 0.91 !! |
| 256 | 0.1 | 64 | random | FAIL | PASS | -86.6% | 0.76 | 0.90 !! |
| 256 | 0.1 | 64 | spatial | PASS | PASS | -88.8% | 0.11 | 0.90 !! |
| 256 | 0.1 | 64 | greedy | PASS | PASS | -88.7% | 0.06 | 0.93 !! |
| 256 | 0.3 | 8 | random | FAIL | PASS | -95.9% | 0.97 | 0.68 !! |
| 256 | 0.3 | 8 | spatial | FAIL | PASS | -95.9% | 0.52 | 0.68 !! |
| 256 | 0.3 | 8 | greedy | PASS | PASS | -95.8% | 0.35 | 0.74 !! |
| 256 | 0.3 | 16 | random | FAIL | PASS | -95.9% | 0.94 | 0.70 !! |
| 256 | 0.3 | 16 | spatial | PASS | PASS | -95.9% | 0.34 | 0.70 !! |
| 256 | 0.3 | 16 | greedy | PASS | PASS | -95.8% | 0.19 | 0.79 !! |
| 256 | 0.3 | 32 | random | FAIL | PASS | -95.9% | 0.87 | 0.70 !! |
| 256 | 0.3 | 32 | spatial | PASS | PASS | -95.8% | 0.21 | 0.70 !! |
| 256 | 0.3 | 32 | greedy | PASS | PASS | -95.8% | 0.12 | 0.83 !! |
| 256 | 0.3 | 64 | random | FAIL | PASS | -95.9% | 0.74 | 0.70 !! |
| 256 | 0.3 | 64 | spatial | PASS | PASS | -95.7% | 0.11 | 0.70 !! |
| 256 | 0.3 | 64 | greedy | PASS | PASS | -95.7% | 0.06 | 0.88 !! |
| 256 | 0.5 | 8 | random | FAIL | PASS | -97.4% | 0.97 | 0.50 |
| 256 | 0.5 | 8 | spatial | FAIL | PASS | -97.4% | 0.52 | 0.50 |
| 256 | 0.5 | 8 | greedy | PASS | PASS | -97.4% | 0.34 | 0.61 !! |
| 256 | 0.5 | 16 | random | FAIL | PASS | -97.5% | 0.94 | 0.50 |
| 256 | 0.5 | 16 | spatial | PASS | PASS | -97.6% | 0.34 | 0.50 |
| 256 | 0.5 | 16 | greedy | PASS | PASS | -97.4% | 0.17 | 0.67 !! |
| 256 | 0.5 | 32 | random | FAIL | PASS | -97.5% | 0.88 | 0.50 |
| 256 | 0.5 | 32 | spatial | PASS | PASS | -97.6% | 0.21 | 0.50 |
| 256 | 0.5 | 32 | greedy | PASS | PASS | -97.5% | 0.11 | 0.74 !! |
| 256 | 0.5 | 64 | random | FAIL | PASS | -97.4% | 0.74 | 0.50 |
| 256 | 0.5 | 64 | spatial | PASS | PASS | -97.4% | 0.11 | 0.50 |
| 256 | 0.5 | 64 | greedy | PASS | PASS | -97.3% | 0.06 | 0.78 !! |
| 1024 | 0.05 | 8 | random | FAIL | PASS | -94.2% | 0.99 | 0.85 !! |
| 1024 | 0.05 | 8 | spatial | FAIL | PASS | -92.4% | 0.53 | 0.85 !! |
| 1024 | 0.05 | 8 | greedy | PASS | PASS | -92.1% | 0.35 | 0.86 !! |
| 1024 | 0.05 | 16 | random | FAIL | PASS | -92.9% | 0.98 | 0.91 !! |
| 1024 | 0.05 | 16 | spatial | PASS | PASS | -91.2% | 0.38 | 0.91 !! |
| 1024 | 0.05 | 16 | greedy | PASS | PASS | -91.7% | 0.20 | 0.92 !! |
| 1024 | 0.05 | 32 | random | FAIL | PASS | -92.3% | 0.97 | 0.94 !! |
| 1024 | 0.05 | 32 | spatial | PASS | PASS | -92.1% | 0.25 | 0.94 !! |
| 1024 | 0.05 | 32 | greedy | PASS | PASS | -91.9% | 0.12 | 0.95 !! |
| 1024 | 0.05 | 64 | random | FAIL | PASS | -91.2% | 0.94 | 0.95 !! |
| 1024 | 0.05 | 64 | spatial | PASS | PASS | -90.8% | 0.15 | 0.95 !! |
| 1024 | 0.05 | 64 | greedy | PASS | PASS | -91.0% | 0.07 | 0.96 !! |
| 1024 | 0.1 | 8 | random | FAIL | PASS | -96.5% | 0.99 | 0.82 !! |
| 1024 | 0.1 | 8 | spatial | FAIL | PASS | -96.1% | 0.53 | 0.82 !! |
| 1024 | 0.1 | 8 | greedy | PASS | PASS | -96.0% | 0.35 | 0.83 !! |
| 1024 | 0.1 | 16 | random | FAIL | PASS | -96.4% | 0.98 | 0.88 !! |
| 1024 | 0.1 | 16 | spatial | PASS | PASS | -95.8% | 0.38 | 0.88 !! |
| 1024 | 0.1 | 16 | greedy | PASS | PASS | -95.6% | 0.20 | 0.89 !! |
| 1024 | 0.1 | 32 | random | FAIL | PASS | -95.9% | 0.97 | 0.90 !! |
| 1024 | 0.1 | 32 | spatial | PASS | PASS | -95.8% | 0.25 | 0.90 !! |
| 1024 | 0.1 | 32 | greedy | PASS | PASS | -95.3% | 0.12 | 0.92 !! |
| 1024 | 0.1 | 64 | random | FAIL | PASS | -95.8% | 0.94 | 0.90 !! |
| 1024 | 0.1 | 64 | spatial | PASS | PASS | -95.7% | 0.15 | 0.90 !! |
| 1024 | 0.1 | 64 | greedy | PASS | PASS | -95.3% | 0.07 | 0.94 !! |
| 1024 | 0.3 | 8 | random | FAIL | PASS | -97.2% | 0.99 | 0.68 !! |
| 1024 | 0.3 | 8 | spatial | FAIL | PASS | -98.6% | 0.53 | 0.68 !! |
| 1024 | 0.3 | 8 | greedy | PASS | PASS | -98.5% | 0.35 | 0.73 !! |
| 1024 | 0.3 | 16 | random | FAIL | PASS | -97.9% | 0.99 | 0.70 !! |
| 1024 | 0.3 | 16 | spatial | PASS | PASS | -98.5% | 0.38 | 0.70 !! |
| 1024 | 0.3 | 16 | greedy | PASS | PASS | -97.9% | 0.20 | 0.78 !! |
| 1024 | 0.3 | 32 | random | FAIL | PASS | -98.0% | 0.97 | 0.70 !! |
| 1024 | 0.3 | 32 | spatial | PASS | PASS | -98.2% | 0.25 | 0.70 !! |
| 1024 | 0.3 | 32 | greedy | PASS | PASS | -98.0% | 0.12 | 0.82 !! |
| 1024 | 0.3 | 64 | random | FAIL | PASS | -98.1% | 0.94 | 0.70 !! |
| 1024 | 0.3 | 64 | spatial | PASS | PASS | -97.9% | 0.15 | 0.70 !! |
| 1024 | 0.3 | 64 | greedy | PASS | PASS | -97.7% | 0.07 | 0.86 !! |
| 1024 | 0.5 | 8 | random | FAIL | PASS | -98.2% | 0.99 | 0.50 |
| 1024 | 0.5 | 8 | spatial | FAIL | PASS | -98.3% | 0.53 | 0.50 |
| 1024 | 0.5 | 8 | greedy | PASS | PASS | -97.8% | 0.35 | 0.60 !! |
| 1024 | 0.5 | 16 | random | FAIL | PASS | -98.2% | 0.99 | 0.50 |
| 1024 | 0.5 | 16 | spatial | PASS | PASS | -98.2% | 0.38 | 0.50 |
| 1024 | 0.5 | 16 | greedy | PASS | PASS | -98.3% | 0.21 | 0.66 !! |
| 1024 | 0.5 | 32 | random | FAIL | PASS | -98.2% | 0.97 | 0.50 |
| 1024 | 0.5 | 32 | spatial | PASS | PASS | -98.4% | 0.25 | 0.50 |
| 1024 | 0.5 | 32 | greedy | PASS | PASS | -98.4% | 0.12 | 0.72 !! |
| 1024 | 0.5 | 64 | random | FAIL | PASS | -98.3% | 0.94 | 0.50 |
| 1024 | 0.5 | 64 | spatial | PASS | PASS | -98.3% | 0.15 | 0.50 |
| 1024 | 0.5 | 64 | greedy | PASS | PASS | -98.3% | 0.07 | 0.78 !! |

### barabasi_albert

Contour A: 0/96 | Contour B: 96/96 | Padding warnings: 80

| N | sp | bs | partition | A | B | time_oh | cbr | pw |
|---|----|----|----------|---|---|---------|-----|-----|
| 256 | 0.05 | 8 | random | FAIL | PASS | -80.1% | 0.98 | 0.84 !! |
| 256 | 0.05 | 8 | spatial | FAIL | PASS | -72.5% | 0.96 | 0.85 !! |
| 256 | 0.05 | 8 | greedy | FAIL | PASS | -78.2% | 0.73 | 0.86 !! |
| 256 | 0.05 | 16 | random | FAIL | PASS | -75.8% | 0.94 | 0.91 !! |
| 256 | 0.05 | 16 | spatial | FAIL | PASS | -81.4% | 0.93 | 0.91 !! |
| 256 | 0.05 | 16 | greedy | FAIL | PASS | -74.0% | 0.67 | 0.92 !! |
| 256 | 0.05 | 32 | random | FAIL | PASS | -73.0% | 0.88 | 0.94 !! |
| 256 | 0.05 | 32 | spatial | FAIL | PASS | -77.0% | 0.85 | 0.94 !! |
| 256 | 0.05 | 32 | greedy | FAIL | PASS | -74.4% | 0.61 | 0.95 !! |
| 256 | 0.05 | 64 | random | FAIL | PASS | -76.5% | 0.75 | 0.95 !! |
| 256 | 0.05 | 64 | spatial | FAIL | PASS | -71.9% | 0.70 | 0.95 !! |
| 256 | 0.05 | 64 | greedy | FAIL | PASS | -74.1% | 0.51 | 0.97 !! |
| 256 | 0.1 | 8 | random | FAIL | PASS | -87.7% | 0.97 | 0.83 !! |
| 256 | 0.1 | 8 | spatial | FAIL | PASS | -88.4% | 0.96 | 0.82 !! |
| 256 | 0.1 | 8 | greedy | FAIL | PASS | -86.1% | 0.73 | 0.84 !! |
| 256 | 0.1 | 16 | random | FAIL | PASS | -87.3% | 0.94 | 0.88 !! |
| 256 | 0.1 | 16 | spatial | FAIL | PASS | -88.6% | 0.92 | 0.88 !! |
| 256 | 0.1 | 16 | greedy | FAIL | PASS | -88.7% | 0.67 | 0.90 !! |
| 256 | 0.1 | 32 | random | FAIL | PASS | -86.5% | 0.87 | 0.90 !! |
| 256 | 0.1 | 32 | spatial | FAIL | PASS | -87.1% | 0.84 | 0.90 !! |
| 256 | 0.1 | 32 | greedy | FAIL | PASS | -84.9% | 0.61 | 0.93 !! |
| 256 | 0.1 | 64 | random | FAIL | PASS | -85.6% | 0.75 | 0.90 !! |
| 256 | 0.1 | 64 | spatial | FAIL | PASS | -86.2% | 0.69 | 0.90 !! |
| 256 | 0.1 | 64 | greedy | FAIL | PASS | -84.1% | 0.51 | 0.95 !! |
| 256 | 0.3 | 8 | random | FAIL | PASS | -96.0% | 0.97 | 0.68 !! |
| 256 | 0.3 | 8 | spatial | FAIL | PASS | -95.7% | 0.96 | 0.68 !! |
| 256 | 0.3 | 8 | greedy | FAIL | PASS | -95.9% | 0.73 | 0.76 !! |
| 256 | 0.3 | 16 | random | FAIL | PASS | -95.9% | 0.94 | 0.70 !! |
| 256 | 0.3 | 16 | spatial | FAIL | PASS | -95.7% | 0.92 | 0.70 !! |
| 256 | 0.3 | 16 | greedy | FAIL | PASS | -95.6% | 0.67 | 0.84 !! |
| 256 | 0.3 | 32 | random | FAIL | PASS | -95.8% | 0.88 | 0.70 !! |
| 256 | 0.3 | 32 | spatial | FAIL | PASS | -95.9% | 0.85 | 0.70 !! |
| 256 | 0.3 | 32 | greedy | FAIL | PASS | -95.4% | 0.60 | 0.88 !! |
| 256 | 0.3 | 64 | random | FAIL | PASS | -95.8% | 0.75 | 0.70 !! |
| 256 | 0.3 | 64 | spatial | FAIL | PASS | -95.8% | 0.70 | 0.70 !! |
| 256 | 0.3 | 64 | greedy | FAIL | PASS | -95.0% | 0.51 | 0.94 !! |
| 256 | 0.5 | 8 | random | FAIL | PASS | -97.2% | 0.97 | 0.50 |
| 256 | 0.5 | 8 | spatial | FAIL | PASS | -97.0% | 0.96 | 0.50 |
| 256 | 0.5 | 8 | greedy | FAIL | PASS | -97.4% | 0.73 | 0.70 !! |
| 256 | 0.5 | 16 | random | FAIL | PASS | -97.6% | 0.94 | 0.50 |
| 256 | 0.5 | 16 | spatial | FAIL | PASS | -97.5% | 0.92 | 0.50 |
| 256 | 0.5 | 16 | greedy | FAIL | PASS | -97.3% | 0.67 | 0.79 !! |
| 256 | 0.5 | 32 | random | FAIL | PASS | -97.5% | 0.88 | 0.50 |
| 256 | 0.5 | 32 | spatial | FAIL | PASS | -97.5% | 0.84 | 0.50 |
| 256 | 0.5 | 32 | greedy | FAIL | PASS | -96.9% | 0.61 | 0.87 !! |
| 256 | 0.5 | 64 | random | FAIL | PASS | -97.4% | 0.75 | 0.50 |
| 256 | 0.5 | 64 | spatial | FAIL | PASS | -97.4% | 0.69 | 0.50 |
| 256 | 0.5 | 64 | greedy | FAIL | PASS | -97.0% | 0.51 | 0.91 !! |
| 1024 | 0.05 | 8 | random | FAIL | PASS | -93.0% | 0.99 | 0.86 !! |
| 1024 | 0.05 | 8 | spatial | FAIL | PASS | -93.1% | 0.99 | 0.85 !! |
| 1024 | 0.05 | 8 | greedy | FAIL | PASS | -91.9% | 0.75 | 0.86 !! |
| 1024 | 0.05 | 16 | random | FAIL | PASS | -93.0% | 0.98 | 0.91 !! |
| 1024 | 0.05 | 16 | spatial | FAIL | PASS | -92.8% | 0.98 | 0.91 !! |
| 1024 | 0.05 | 16 | greedy | FAIL | PASS | -91.0% | 0.72 | 0.92 !! |
| 1024 | 0.05 | 32 | random | FAIL | PASS | -92.3% | 0.97 | 0.94 !! |
| 1024 | 0.05 | 32 | spatial | FAIL | PASS | -92.2% | 0.96 | 0.94 !! |
| 1024 | 0.05 | 32 | greedy | FAIL | PASS | -90.4% | 0.68 | 0.95 !! |
| 1024 | 0.05 | 64 | random | FAIL | PASS | -91.8% | 0.94 | 0.95 !! |
| 1024 | 0.05 | 64 | spatial | FAIL | PASS | -91.9% | 0.92 | 0.95 !! |
| 1024 | 0.05 | 64 | greedy | FAIL | PASS | -89.4% | 0.64 | 0.97 !! |
| 1024 | 0.1 | 8 | random | FAIL | PASS | -96.4% | 0.99 | 0.82 !! |
| 1024 | 0.1 | 8 | spatial | FAIL | PASS | -96.5% | 0.99 | 0.82 !! |
| 1024 | 0.1 | 8 | greedy | FAIL | PASS | -95.9% | 0.75 | 0.84 !! |
| 1024 | 0.1 | 16 | random | FAIL | PASS | -96.2% | 0.99 | 0.88 !! |
| 1024 | 0.1 | 16 | spatial | FAIL | PASS | -96.4% | 0.98 | 0.88 !! |
| 1024 | 0.1 | 16 | greedy | FAIL | PASS | -95.5% | 0.72 | 0.90 !! |
| 1024 | 0.1 | 32 | random | FAIL | PASS | -96.0% | 0.97 | 0.90 !! |
| 1024 | 0.1 | 32 | spatial | FAIL | PASS | -95.6% | 0.96 | 0.90 !! |
| 1024 | 0.1 | 32 | greedy | FAIL | PASS | -94.9% | 0.68 | 0.94 !! |
| 1024 | 0.1 | 64 | random | FAIL | PASS | -96.0% | 0.94 | 0.90 !! |
| 1024 | 0.1 | 64 | spatial | FAIL | PASS | -95.3% | 0.92 | 0.90 !! |
| 1024 | 0.1 | 64 | greedy | FAIL | PASS | -94.3% | 0.64 | 0.95 !! |
| 1024 | 0.3 | 8 | random | FAIL | PASS | -98.0% | 0.99 | 0.69 !! |
| 1024 | 0.3 | 8 | spatial | FAIL | PASS | -97.7% | 0.99 | 0.68 !! |
| 1024 | 0.3 | 8 | greedy | FAIL | PASS | -98.0% | 0.75 | 0.77 !! |
| 1024 | 0.3 | 16 | random | FAIL | PASS | -98.0% | 0.98 | 0.70 !! |
| 1024 | 0.3 | 16 | spatial | FAIL | PASS | -98.0% | 0.98 | 0.70 !! |
| 1024 | 0.3 | 16 | greedy | FAIL | PASS | -97.9% | 0.71 | 0.84 !! |
| 1024 | 0.3 | 32 | random | FAIL | PASS | -98.1% | 0.97 | 0.70 !! |
| 1024 | 0.3 | 32 | spatial | FAIL | PASS | -98.0% | 0.96 | 0.70 !! |
| 1024 | 0.3 | 32 | greedy | FAIL | PASS | -97.7% | 0.68 | 0.90 !! |
| 1024 | 0.3 | 64 | random | FAIL | PASS | -98.0% | 0.94 | 0.70 !! |
| 1024 | 0.3 | 64 | spatial | FAIL | PASS | -98.1% | 0.92 | 0.70 !! |
| 1024 | 0.3 | 64 | greedy | FAIL | PASS | -97.1% | 0.64 | 0.93 !! |
| 1024 | 0.5 | 8 | random | FAIL | PASS | -98.4% | 0.99 | 0.50 |
| 1024 | 0.5 | 8 | spatial | FAIL | PASS | -98.3% | 0.99 | 0.50 |
| 1024 | 0.5 | 8 | greedy | FAIL | PASS | -98.1% | 0.75 | 0.70 !! |
| 1024 | 0.5 | 16 | random | FAIL | PASS | -98.7% | 0.99 | 0.50 |
| 1024 | 0.5 | 16 | spatial | FAIL | PASS | -98.3% | 0.98 | 0.50 |
| 1024 | 0.5 | 16 | greedy | FAIL | PASS | -97.7% | 0.71 | 0.79 !! |
| 1024 | 0.5 | 32 | random | FAIL | PASS | -98.4% | 0.97 | 0.50 |
| 1024 | 0.5 | 32 | spatial | FAIL | PASS | -98.6% | 0.96 | 0.50 |
| 1024 | 0.5 | 32 | greedy | FAIL | PASS | -97.9% | 0.68 | 0.86 !! |
| 1024 | 0.5 | 64 | random | FAIL | PASS | -98.7% | 0.94 | 0.50 |
| 1024 | 0.5 | 64 | spatial | FAIL | PASS | -98.6% | 0.92 | 0.50 |
| 1024 | 0.5 | 64 | greedy | FAIL | PASS | -97.8% | 0.64 | 0.92 !! |

### grid_graph

Contour A: 64/96 | Contour B: 96/96 | Padding warnings: 80

| N | sp | bs | partition | A | B | time_oh | cbr | pw |
|---|----|----|----------|---|---|---------|-----|-----|
| 256 | 0.05 | 8 | random | FAIL | PASS | -78.7% | 0.97 | 0.84 !! |
| 256 | 0.05 | 8 | spatial | PASS | PASS | -75.9% | 0.33 | 0.85 !! |
| 256 | 0.05 | 8 | greedy | PASS | PASS | -81.8% | 0.39 | 0.86 !! |
| 256 | 0.05 | 16 | random | FAIL | PASS | -77.2% | 0.94 | 0.91 !! |
| 256 | 0.05 | 16 | spatial | PASS | PASS | -75.3% | 0.20 | 0.91 !! |
| 256 | 0.05 | 16 | greedy | PASS | PASS | -80.9% | 0.27 | 0.91 !! |
| 256 | 0.05 | 32 | random | FAIL | PASS | -80.8% | 0.88 | 0.94 !! |
| 256 | 0.05 | 32 | spatial | PASS | PASS | -79.5% | 0.13 | 0.93 !! |
| 256 | 0.05 | 32 | greedy | PASS | PASS | -73.4% | 0.18 | 0.94 !! |
| 256 | 0.05 | 64 | random | FAIL | PASS | -78.8% | 0.75 | 0.95 !! |
| 256 | 0.05 | 64 | spatial | PASS | PASS | -76.4% | 0.07 | 0.95 !! |
| 256 | 0.05 | 64 | greedy | PASS | PASS | -77.0% | 0.10 | 0.95 !! |
| 256 | 0.1 | 8 | random | FAIL | PASS | -89.0% | 0.97 | 0.82 !! |
| 256 | 0.1 | 8 | spatial | PASS | PASS | -85.5% | 0.33 | 0.83 !! |
| 256 | 0.1 | 8 | greedy | PASS | PASS | -86.1% | 0.39 | 0.83 !! |
| 256 | 0.1 | 16 | random | FAIL | PASS | -87.9% | 0.94 | 0.87 !! |
| 256 | 0.1 | 16 | spatial | PASS | PASS | -85.2% | 0.20 | 0.88 !! |
| 256 | 0.1 | 16 | greedy | PASS | PASS | -86.6% | 0.28 | 0.89 !! |
| 256 | 0.1 | 32 | random | FAIL | PASS | -88.6% | 0.88 | 0.90 !! |
| 256 | 0.1 | 32 | spatial | PASS | PASS | -87.3% | 0.13 | 0.90 !! |
| 256 | 0.1 | 32 | greedy | PASS | PASS | -85.8% | 0.18 | 0.90 !! |
| 256 | 0.1 | 64 | random | FAIL | PASS | -86.6% | 0.75 | 0.90 !! |
| 256 | 0.1 | 64 | spatial | PASS | PASS | -88.7% | 0.07 | 0.90 !! |
| 256 | 0.1 | 64 | greedy | PASS | PASS | -88.1% | 0.10 | 0.90 !! |
| 256 | 0.3 | 8 | random | FAIL | PASS | -96.1% | 0.97 | 0.68 !! |
| 256 | 0.3 | 8 | spatial | PASS | PASS | -96.0% | 0.33 | 0.67 !! |
| 256 | 0.3 | 8 | greedy | PASS | PASS | -95.8% | 0.40 | 0.71 !! |
| 256 | 0.3 | 16 | random | FAIL | PASS | -95.7% | 0.94 | 0.70 !! |
| 256 | 0.3 | 16 | spatial | PASS | PASS | -95.9% | 0.20 | 0.70 !! |
| 256 | 0.3 | 16 | greedy | PASS | PASS | -95.9% | 0.27 | 0.73 !! |
| 256 | 0.3 | 32 | random | FAIL | PASS | -96.0% | 0.88 | 0.70 !! |
| 256 | 0.3 | 32 | spatial | PASS | PASS | -96.1% | 0.13 | 0.70 !! |
| 256 | 0.3 | 32 | greedy | PASS | PASS | -95.9% | 0.17 | 0.73 !! |
| 256 | 0.3 | 64 | random | FAIL | PASS | -95.9% | 0.75 | 0.70 !! |
| 256 | 0.3 | 64 | spatial | PASS | PASS | -96.1% | 0.07 | 0.70 !! |
| 256 | 0.3 | 64 | greedy | PASS | PASS | -95.6% | 0.11 | 0.76 !! |
| 256 | 0.5 | 8 | random | FAIL | PASS | -97.6% | 0.97 | 0.50 |
| 256 | 0.5 | 8 | spatial | PASS | PASS | -97.6% | 0.33 | 0.50 |
| 256 | 0.5 | 8 | greedy | PASS | PASS | -97.5% | 0.40 | 0.56 !! |
| 256 | 0.5 | 16 | random | FAIL | PASS | -97.7% | 0.94 | 0.50 |
| 256 | 0.5 | 16 | spatial | PASS | PASS | -97.6% | 0.20 | 0.50 |
| 256 | 0.5 | 16 | greedy | PASS | PASS | -97.6% | 0.27 | 0.58 !! |
| 256 | 0.5 | 32 | random | FAIL | PASS | -97.5% | 0.87 | 0.50 |
| 256 | 0.5 | 32 | spatial | PASS | PASS | -97.6% | 0.13 | 0.50 |
| 256 | 0.5 | 32 | greedy | PASS | PASS | -97.5% | 0.18 | 0.60 !! |
| 256 | 0.5 | 64 | random | FAIL | PASS | -97.6% | 0.75 | 0.50 |
| 256 | 0.5 | 64 | spatial | PASS | PASS | -97.6% | 0.07 | 0.50 |
| 256 | 0.5 | 64 | greedy | PASS | PASS | -97.6% | 0.11 | 0.55 !! |
| 1024 | 0.05 | 8 | random | FAIL | PASS | -93.0% | 0.99 | 0.85 !! |
| 1024 | 0.05 | 8 | spatial | PASS | PASS | -91.5% | 0.35 | 0.85 !! |
| 1024 | 0.05 | 8 | greedy | PASS | PASS | -91.8% | 0.42 | 0.86 !! |
| 1024 | 0.05 | 16 | random | FAIL | PASS | -92.8% | 0.99 | 0.91 !! |
| 1024 | 0.05 | 16 | spatial | PASS | PASS | -91.0% | 0.23 | 0.91 !! |
| 1024 | 0.05 | 16 | greedy | PASS | PASS | -91.3% | 0.29 | 0.91 !! |
| 1024 | 0.05 | 32 | random | FAIL | PASS | -92.5% | 0.97 | 0.94 !! |
| 1024 | 0.05 | 32 | spatial | PASS | PASS | -91.9% | 0.16 | 0.94 !! |
| 1024 | 0.05 | 32 | greedy | PASS | PASS | -90.5% | 0.20 | 0.94 !! |
| 1024 | 0.05 | 64 | random | FAIL | PASS | -91.9% | 0.94 | 0.95 !! |
| 1024 | 0.05 | 64 | spatial | PASS | PASS | -91.3% | 0.10 | 0.95 !! |
| 1024 | 0.05 | 64 | greedy | PASS | PASS | -90.0% | 0.14 | 0.95 !! |
| 1024 | 0.1 | 8 | random | FAIL | PASS | -96.7% | 0.99 | 0.82 !! |
| 1024 | 0.1 | 8 | spatial | PASS | PASS | -95.6% | 0.35 | 0.83 !! |
| 1024 | 0.1 | 8 | greedy | PASS | PASS | -95.8% | 0.42 | 0.83 !! |
| 1024 | 0.1 | 16 | random | FAIL | PASS | -96.2% | 0.98 | 0.87 !! |
| 1024 | 0.1 | 16 | spatial | PASS | PASS | -95.4% | 0.23 | 0.88 !! |
| 1024 | 0.1 | 16 | greedy | PASS | PASS | -95.3% | 0.29 | 0.88 !! |
| 1024 | 0.1 | 32 | random | FAIL | PASS | -96.1% | 0.97 | 0.90 !! |
| 1024 | 0.1 | 32 | spatial | PASS | PASS | -95.3% | 0.16 | 0.90 !! |
| 1024 | 0.1 | 32 | greedy | PASS | PASS | -95.2% | 0.21 | 0.90 !! |
| 1024 | 0.1 | 64 | random | FAIL | PASS | -95.6% | 0.94 | 0.90 !! |
| 1024 | 0.1 | 64 | spatial | PASS | PASS | -95.2% | 0.10 | 0.90 !! |
| 1024 | 0.1 | 64 | greedy | PASS | PASS | -95.2% | 0.14 | 0.91 !! |
| 1024 | 0.3 | 8 | random | FAIL | PASS | -98.6% | 0.99 | 0.68 !! |
| 1024 | 0.3 | 8 | spatial | PASS | PASS | -98.6% | 0.35 | 0.68 !! |
| 1024 | 0.3 | 8 | greedy | PASS | PASS | -98.6% | 0.42 | 0.70 !! |
| 1024 | 0.3 | 16 | random | FAIL | PASS | -98.6% | 0.98 | 0.70 !! |
| 1024 | 0.3 | 16 | spatial | PASS | PASS | -98.5% | 0.23 | 0.70 !! |
| 1024 | 0.3 | 16 | greedy | PASS | PASS | -98.4% | 0.29 | 0.74 !! |
| 1024 | 0.3 | 32 | random | FAIL | PASS | -98.5% | 0.97 | 0.70 !! |
| 1024 | 0.3 | 32 | spatial | PASS | PASS | -98.4% | 0.16 | 0.70 !! |
| 1024 | 0.3 | 32 | greedy | PASS | PASS | -98.5% | 0.21 | 0.75 !! |
| 1024 | 0.3 | 64 | random | FAIL | PASS | -98.5% | 0.94 | 0.70 !! |
| 1024 | 0.3 | 64 | spatial | PASS | PASS | -98.6% | 0.10 | 0.70 !! |
| 1024 | 0.3 | 64 | greedy | PASS | PASS | -98.3% | 0.14 | 0.75 !! |
| 1024 | 0.5 | 8 | random | FAIL | PASS | -98.4% | 0.99 | 0.50 |
| 1024 | 0.5 | 8 | spatial | PASS | PASS | -98.3% | 0.35 | 0.50 |
| 1024 | 0.5 | 8 | greedy | PASS | PASS | -98.3% | 0.42 | 0.56 !! |
| 1024 | 0.5 | 16 | random | FAIL | PASS | -98.1% | 0.98 | 0.50 |
| 1024 | 0.5 | 16 | spatial | PASS | PASS | -98.6% | 0.23 | 0.50 |
| 1024 | 0.5 | 16 | greedy | PASS | PASS | -98.2% | 0.29 | 0.58 !! |
| 1024 | 0.5 | 32 | random | FAIL | PASS | -98.3% | 0.97 | 0.50 |
| 1024 | 0.5 | 32 | spatial | PASS | PASS | -98.4% | 0.16 | 0.50 |
| 1024 | 0.5 | 32 | greedy | PASS | PASS | -98.5% | 0.21 | 0.59 !! |
| 1024 | 0.5 | 64 | random | FAIL | PASS | -98.3% | 0.94 | 0.50 |
| 1024 | 0.5 | 64 | spatial | PASS | PASS | -98.5% | 0.10 | 0.50 |
| 1024 | 0.5 | 64 | greedy | PASS | PASS | -98.6% | 0.13 | 0.60 !! |

## Partition method comparison

- **random_partition**: mean cbr=0.928, mean pw=0.745, mean time_oh=-92.9%, Contour A: 0/96
- **spatial_partition**: mean cbr=0.472, mean pw=0.745, mean time_oh=-92.7%, Contour A: 56/96
- **greedy_partition**: mean cbr=0.365, mean pw=0.827, mean time_oh=-92.5%, Contour A: 64/96

## Block size analysis

- **block_size=8**: mean cbr=0.698, mean pw=0.730, Contour A: 24/72
- **block_size=16**: mean cbr=0.620, mean pw=0.771, Contour A: 32/72
- **block_size=32**: mean cbr=0.555, mean pw=0.790, Contour A: 32/72
- **block_size=64**: mean cbr=0.479, mean pw=0.800, Contour A: 32/72

## Cross-block ratio heatmap (mean cbr by graph_type x partition)

| graph_type | random | spatial | greedy |
|------------|--------|---------|--------|
| random_geometric | 0.928 | 0.310 | 0.180 |
| barabasi_albert | 0.929 | 0.909 | 0.663 |
| grid_graph | 0.928 | 0.196 | 0.250 |

## Key Findings

- **random_geometric**: VIABLE (56/96 = 58% Contour A)
- **barabasi_albert**: NOT VIABLE (0/96 = 0% Contour A)
- **grid_graph**: VIABLE (64/96 = 67% Contour A)

### Does spatial/greedy help for non-spatial graphs (barabasi_albert)?

  - random_partition: mean cbr = 0.929
  - spatial_partition: mean cbr = 0.909
  - greedy_partition: mean cbr = 0.663

### Padding waste vs block size

  - bs=8: mean pw = 0.730, warnings: 60/72
  - bs=16: mean pw = 0.771, warnings: 60/72
  - bs=32: mean pw = 0.790, warnings: 60/72
  - bs=64: mean pw = 0.800, warnings: 60/72

## Verdict

**OVERALL: FAIL** -- Block-based addressing only works for 120/288 (42%) configurations. Graph structure resists blocking in most cases.
