# Exp10g: Dual-Mode Benchmark Report

## Experiment Design

Two benchmark modes separate layout cost from operator cost:
- **Mode 1 (Stencil):** Manual 3x3 stencil, no F.conv2d. Isolates pure layout cost.
- **Mode 2 (Conv2d):** F.conv2d with 3x3 kernel. Shows real operator cost including workspace.

- sides: [64, 128, 256]
- sparsities: [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
- patterns: ['random', 'clustered', 'checkerboard']
- tile_size: 8
- n_seeds: 10, n_warmup: 5, n_repeat: 20
- kill threshold: 20%

## Contour A (Architectural Viability -- Stencil Mode)

Overall: **FAIL** (40/54 configs pass)

| Config | D Time vs grid | D Resident vs grid | Contour A |
|--------|---------------:|-------------------:|-----------|
| side=128_sp=0.05_pat=checkerboard | +7.7% | 0.96x | PASS |
| side=128_sp=0.05_pat=clustered | +7.3% | 0.30x | PASS |
| side=128_sp=0.05_pat=random | +7.4% | 0.96x | PASS |
| side=128_sp=0.1_pat=checkerboard | +6.6% | 0.99x | PASS |
| side=128_sp=0.1_pat=clustered | +7.6% | 0.48x | PASS |
| side=128_sp=0.1_pat=random | +6.8% | 0.99x | PASS |
| side=128_sp=0.2_pat=checkerboard | +6.5% | 0.99x | PASS |
| side=128_sp=0.2_pat=clustered | +7.5% | 0.64x | PASS |
| side=128_sp=0.2_pat=random | +7.9% | 0.99x | PASS |
| side=128_sp=0.3_pat=checkerboard | +7.3% | 0.99x | PASS |
| side=128_sp=0.3_pat=clustered | +6.2% | 0.81x | PASS |
| side=128_sp=0.3_pat=random | +6.9% | 0.99x | PASS |
| side=128_sp=0.5_pat=checkerboard | +7.5% | 0.99x | PASS |
| side=128_sp=0.5_pat=clustered | +7.2% | 0.92x | PASS |
| side=128_sp=0.5_pat=random | +7.7% | 0.99x | PASS |
| side=128_sp=0.7_pat=checkerboard | +7.2% | 0.99x | PASS |
| side=128_sp=0.7_pat=clustered | +6.7% | 0.98x | PASS |
| side=128_sp=0.7_pat=random | +7.0% | 0.99x | PASS |
| side=256_sp=0.05_pat=checkerboard | +6.0% | 0.96x | PASS |
| side=256_sp=0.05_pat=clustered | +5.8% | 0.22x | PASS |
| side=256_sp=0.05_pat=random | +7.3% | 0.96x | PASS |
| side=256_sp=0.1_pat=checkerboard | +5.6% | 0.99x | PASS |
| side=256_sp=0.1_pat=clustered | +5.1% | 0.33x | PASS |
| side=256_sp=0.1_pat=random | +6.3% | 0.99x | PASS |
| side=256_sp=0.2_pat=checkerboard | +6.4% | 0.99x | PASS |
| side=256_sp=0.2_pat=clustered | +6.5% | 0.49x | PASS |
| side=256_sp=0.2_pat=random | +6.8% | 0.99x | PASS |
| side=256_sp=0.3_pat=checkerboard | +5.9% | 0.99x | PASS |
| side=256_sp=0.3_pat=clustered | +4.6% | 0.65x | PASS |
| side=256_sp=0.3_pat=random | +6.7% | 0.99x | PASS |
| side=256_sp=0.5_pat=checkerboard | +5.7% | 0.99x | PASS |
| side=256_sp=0.5_pat=clustered | +5.6% | 0.84x | PASS |
| side=256_sp=0.5_pat=random | +7.5% | 0.99x | PASS |
| side=256_sp=0.7_pat=checkerboard | +5.1% | 0.99x | PASS |
| side=256_sp=0.7_pat=clustered | +3.1% | 0.94x | PASS |
| side=256_sp=0.7_pat=random | +6.5% | 0.99x | PASS |
| side=64_sp=0.05_pat=checkerboard | +12.0% | 1.01x | FAIL |
| side=64_sp=0.05_pat=clustered | +11.7% | 0.50x | PASS |
| side=64_sp=0.05_pat=random | +11.1% | 1.01x | FAIL |
| side=64_sp=0.1_pat=checkerboard | +8.1% | 1.02x | FAIL |
| side=64_sp=0.1_pat=clustered | +8.6% | 0.69x | PASS |
| side=64_sp=0.1_pat=random | +11.7% | 1.02x | FAIL |
| side=64_sp=0.2_pat=checkerboard | +9.0% | 1.02x | FAIL |
| side=64_sp=0.2_pat=clustered | +8.5% | 0.79x | PASS |
| side=64_sp=0.2_pat=random | +12.2% | 1.02x | FAIL |
| side=64_sp=0.3_pat=checkerboard | +9.5% | 1.02x | FAIL |
| side=64_sp=0.3_pat=clustered | +7.7% | 0.90x | PASS |
| side=64_sp=0.3_pat=random | +8.2% | 1.02x | FAIL |
| side=64_sp=0.5_pat=checkerboard | +8.5% | 1.02x | FAIL |
| side=64_sp=0.5_pat=clustered | +9.6% | 1.01x | FAIL |
| side=64_sp=0.5_pat=random | +6.3% | 1.02x | FAIL |
| side=64_sp=0.7_pat=checkerboard | +8.2% | 1.02x | FAIL |
| side=64_sp=0.7_pat=clustered | +8.2% | 1.02x | FAIL |
| side=64_sp=0.7_pat=random | +8.4% | 1.02x | FAIL |

## Contour B (Operational Viability -- Conv2d Mode)

Overall: **PASS** (54/54 configs pass)

| Config | D Time vs grid | D Peak vs grid | D Workspace vs grid | Contour B |
|--------|---------------:|---------------:|-------------------:|-----------|
| side=128_sp=0.05_pat=checkerboard | -74.9% | -36.5% | 0.44x | PASS |
| side=128_sp=0.05_pat=clustered | -79.9% | -81.2% | 0.12x | PASS |
| side=128_sp=0.05_pat=random | -74.8% | -36.5% | 0.44x | PASS |
| side=128_sp=0.1_pat=checkerboard | -73.8% | -37.4% | 0.43x | PASS |
| side=128_sp=0.1_pat=clustered | -78.3% | -70.0% | 0.20x | PASS |
| side=128_sp=0.1_pat=random | -74.7% | -37.4% | 0.43x | PASS |
| side=128_sp=0.2_pat=checkerboard | -75.0% | -42.3% | 0.38x | PASS |
| side=128_sp=0.2_pat=clustered | -77.2% | -63.1% | 0.24x | PASS |
| side=128_sp=0.2_pat=random | -75.0% | -42.3% | 0.38x | PASS |
| side=128_sp=0.3_pat=checkerboard | -74.8% | -46.4% | 0.34x | PASS |
| side=128_sp=0.3_pat=clustered | -76.2% | -56.6% | 0.27x | PASS |
| side=128_sp=0.3_pat=random | -74.9% | -46.4% | 0.34x | PASS |
| side=128_sp=0.5_pat=checkerboard | -75.0% | -53.0% | 0.28x | PASS |
| side=128_sp=0.5_pat=clustered | -75.2% | -56.3% | 0.26x | PASS |
| side=128_sp=0.5_pat=random | -74.5% | -53.0% | 0.28x | PASS |
| side=128_sp=0.7_pat=checkerboard | -75.5% | -58.3% | 0.24x | PASS |
| side=128_sp=0.7_pat=clustered | -74.3% | -58.9% | 0.24x | PASS |
| side=128_sp=0.7_pat=random | -75.0% | -58.3% | 0.24x | PASS |
| side=256_sp=0.05_pat=checkerboard | -54.3% | -36.5% | 0.45x | PASS |
| side=256_sp=0.05_pat=clustered | -76.0% | -85.7% | 0.10x | PASS |
| side=256_sp=0.05_pat=random | -54.7% | -36.6% | 0.45x | PASS |
| side=256_sp=0.1_pat=checkerboard | -53.9% | -37.2% | 0.43x | PASS |
| side=256_sp=0.1_pat=clustered | -73.2% | -79.7% | 0.14x | PASS |
| side=256_sp=0.1_pat=random | -53.9% | -37.2% | 0.43x | PASS |
| side=256_sp=0.2_pat=checkerboard | -56.0% | -42.1% | 0.38x | PASS |
| side=256_sp=0.2_pat=clustered | -67.9% | -71.3% | 0.19x | PASS |
| side=256_sp=0.2_pat=random | -55.3% | -42.1% | 0.38x | PASS |
| side=256_sp=0.3_pat=checkerboard | -53.7% | -46.2% | 0.34x | PASS |
| side=256_sp=0.3_pat=clustered | -63.6% | -64.7% | 0.22x | PASS |
| side=256_sp=0.3_pat=random | -54.6% | -46.2% | 0.34x | PASS |
| side=256_sp=0.5_pat=checkerboard | -53.3% | -53.0% | 0.29x | PASS |
| side=256_sp=0.5_pat=clustered | -58.6% | -60.1% | 0.24x | PASS |
| side=256_sp=0.5_pat=random | -56.0% | -53.0% | 0.29x | PASS |
| side=256_sp=0.7_pat=checkerboard | -65.8% | -58.3% | 0.24x | PASS |
| side=256_sp=0.7_pat=clustered | -64.4% | -60.5% | 0.23x | PASS |
| side=256_sp=0.7_pat=random | -54.5% | -58.3% | 0.24x | PASS |
| side=64_sp=0.05_pat=checkerboard | -77.3% | -35.7% | 0.43x | PASS |
| side=64_sp=0.05_pat=clustered | -76.6% | -70.4% | 0.18x | PASS |
| side=64_sp=0.05_pat=random | -76.5% | -35.7% | 0.43x | PASS |
| side=64_sp=0.1_pat=checkerboard | -78.3% | -37.5% | 0.41x | PASS |
| side=64_sp=0.1_pat=clustered | -79.0% | -59.2% | 0.26x | PASS |
| side=64_sp=0.1_pat=random | -78.6% | -37.5% | 0.41x | PASS |
| side=64_sp=0.2_pat=checkerboard | -80.3% | -41.9% | 0.37x | PASS |
| side=64_sp=0.2_pat=clustered | -78.7% | -56.6% | 0.26x | PASS |
| side=64_sp=0.2_pat=random | -76.9% | -41.9% | 0.37x | PASS |
| side=64_sp=0.3_pat=checkerboard | -80.4% | -46.0% | 0.33x | PASS |
| side=64_sp=0.3_pat=clustered | -78.8% | -52.5% | 0.29x | PASS |
| side=64_sp=0.3_pat=random | -78.8% | -46.0% | 0.33x | PASS |
| side=64_sp=0.5_pat=checkerboard | -78.8% | -52.2% | 0.28x | PASS |
| side=64_sp=0.5_pat=clustered | -79.2% | -52.9% | 0.27x | PASS |
| side=64_sp=0.5_pat=random | -80.3% | -52.2% | 0.28x | PASS |
| side=64_sp=0.7_pat=checkerboard | -77.7% | -57.6% | 0.24x | PASS |
| side=64_sp=0.7_pat=clustered | -78.0% | -57.6% | 0.24x | PASS |
| side=64_sp=0.7_pat=random | -78.6% | -57.6% | 0.24x | PASS |

## Detailed Comparison (All Candidates)

### Mode 1 (Stencil -- Layout Benchmark)

| Candidate | Time vs grid | Resident vs grid | Workspace vs grid | Contour A |
|-----------|------------:|----------------:|-----------------:|-----------|
| A_bitset | +0.2% | 1.82x | 1.00x | N/A |
| D_direct | +7.3% | 0.99x | 3.28x | FAIL |

### Mode 2 (Conv2d -- Operator Benchmark)

| Candidate | Time vs grid | Peak vs grid | Workspace vs grid | Contour B |
|-----------|------------:|------------:|-----------------:|-----------|
| A_bitset | +0.1% | 1.26x | 1.00x | N/A |
| D_direct | -75.0% | 0.47x | 0.28x | PASS |

## Build Cost

| Config | grid | A_bitset | D_direct |
|--------|-----:|---------:|---------:|
| side=128_sp=0.05_pat=checkerboard | 0.00ms | 0.55ms | 1.18ms |
| side=128_sp=0.05_pat=clustered | 0.00ms | 0.55ms | 1.19ms |
| side=128_sp=0.05_pat=random | 0.00ms | 0.55ms | 1.21ms |
| side=128_sp=0.1_pat=checkerboard | 0.00ms | 0.55ms | 1.19ms |
| side=128_sp=0.1_pat=clustered | 0.00ms | 0.56ms | 1.19ms |
| side=128_sp=0.1_pat=random | 0.00ms | 0.55ms | 1.19ms |
| side=128_sp=0.2_pat=checkerboard | 0.00ms | 0.55ms | 1.19ms |
| side=128_sp=0.2_pat=clustered | 0.00ms | 0.55ms | 1.19ms |
| side=128_sp=0.2_pat=random | 0.00ms | 0.55ms | 1.18ms |
| side=128_sp=0.3_pat=checkerboard | 0.00ms | 0.56ms | 1.23ms |
| side=128_sp=0.3_pat=clustered | 0.00ms | 0.56ms | 1.19ms |
| side=128_sp=0.3_pat=random | 0.00ms | 0.55ms | 1.19ms |
| side=128_sp=0.5_pat=checkerboard | 0.00ms | 0.55ms | 1.20ms |
| side=128_sp=0.5_pat=clustered | 0.00ms | 0.55ms | 1.21ms |
| side=128_sp=0.5_pat=random | 0.00ms | 0.55ms | 1.21ms |
| side=128_sp=0.7_pat=checkerboard | 0.00ms | 0.55ms | 1.18ms |
| side=128_sp=0.7_pat=clustered | 0.00ms | 0.56ms | 1.19ms |
| side=128_sp=0.7_pat=random | 0.00ms | 0.54ms | 1.18ms |
| side=256_sp=0.05_pat=checkerboard | 0.00ms | 0.56ms | 1.25ms |
| side=256_sp=0.05_pat=clustered | 0.00ms | 0.57ms | 1.23ms |
| side=256_sp=0.05_pat=random | 0.00ms | 0.57ms | 1.26ms |
| side=256_sp=0.1_pat=checkerboard | 0.00ms | 0.56ms | 1.25ms |
| side=256_sp=0.1_pat=clustered | 0.00ms | 0.57ms | 1.23ms |
| side=256_sp=0.1_pat=random | 0.00ms | 0.57ms | 1.26ms |
| side=256_sp=0.2_pat=checkerboard | 0.00ms | 0.57ms | 1.26ms |
| side=256_sp=0.2_pat=clustered | 0.00ms | 0.57ms | 1.23ms |
| side=256_sp=0.2_pat=random | 0.00ms | 0.57ms | 1.26ms |
| side=256_sp=0.3_pat=checkerboard | 0.00ms | 0.57ms | 1.25ms |
| side=256_sp=0.3_pat=clustered | 0.00ms | 0.57ms | 1.31ms |
| side=256_sp=0.3_pat=random | 0.00ms | 0.57ms | 1.26ms |
| side=256_sp=0.5_pat=checkerboard | 0.00ms | 0.57ms | 1.26ms |
| side=256_sp=0.5_pat=clustered | 0.00ms | 0.58ms | 1.27ms |
| side=256_sp=0.5_pat=random | 0.00ms | 0.57ms | 1.26ms |
| side=256_sp=0.7_pat=checkerboard | 0.00ms | 0.56ms | 1.30ms |
| side=256_sp=0.7_pat=clustered | 0.00ms | 0.59ms | 1.70ms |
| side=256_sp=0.7_pat=random | 0.00ms | 0.57ms | 1.30ms |
| side=64_sp=0.05_pat=checkerboard | 0.00ms | 0.52ms | 1.13ms |
| side=64_sp=0.05_pat=clustered | 0.00ms | 0.53ms | 1.14ms |
| side=64_sp=0.05_pat=random | 0.00ms | 0.55ms | 1.15ms |
| side=64_sp=0.1_pat=checkerboard | 0.00ms | 0.55ms | 1.20ms |
| side=64_sp=0.1_pat=clustered | 0.00ms | 0.54ms | 1.21ms |
| side=64_sp=0.1_pat=random | 0.00ms | 0.52ms | 1.12ms |
| side=64_sp=0.2_pat=checkerboard | 0.00ms | 0.55ms | 1.22ms |
| side=64_sp=0.2_pat=clustered | 0.00ms | 0.56ms | 1.19ms |
| side=64_sp=0.2_pat=random | 0.00ms | 0.54ms | 1.15ms |
| side=64_sp=0.3_pat=checkerboard | 0.00ms | 0.55ms | 1.21ms |
| side=64_sp=0.3_pat=clustered | 0.00ms | 0.54ms | 1.16ms |
| side=64_sp=0.3_pat=random | 0.00ms | 0.55ms | 1.24ms |
| side=64_sp=0.5_pat=checkerboard | 0.00ms | 0.54ms | 1.17ms |
| side=64_sp=0.5_pat=clustered | 0.00ms | 0.56ms | 1.21ms |
| side=64_sp=0.5_pat=random | 0.00ms | 0.56ms | 1.24ms |
| side=64_sp=0.7_pat=checkerboard | 0.00ms | 0.55ms | 1.19ms |
| side=64_sp=0.7_pat=clustered | 0.00ms | 0.54ms | 1.19ms |
| side=64_sp=0.7_pat=random | 0.00ms | 0.54ms | 1.20ms |
