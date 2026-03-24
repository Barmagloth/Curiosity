# exp19 — Multi-Tick Parameter Sweep + Benchmark

Phase 4 validation: find optimal parameters for the multi-tick pipeline
and benchmark against single-tick and streaming modes.

## Sub-experiments

### exp19a: max_ticks scaling law
- **Question:** How should max_ticks scale with n_total?
- **Sweep:** n_total in {64, 256, 1000} x max_ticks in {1, 2, 3, 5, 8, 13, 20} x 4 spaces x 10 seeds
- **Kill:** multi-tick PSNR >= 95% of single-tick PSNR at same budget
- **Output:** optimal max_ticks = f(n_total) approximation

### exp19b: WeightedRhoGate stress test
- **Question:** Do EMA weights converge under perturbations?
- **Sweep:** ema_weight_alpha in {0.1, 0.2, 0.3, 0.5} x instability profiles (stable, ramp, step, oscillating) x 4 spaces x 10 seeds
- **Kill:** weight variance over last 3 ticks < 0.05
- **Output:** stable alpha range, convergence time

### exp19c: parameter sweep
- **Sweep:** convergence_window in {2, 3, 4} x pilot_ticks in {2, 3, 5} x pilot_thresh_factor in {0.5, 0.7, 0.9} x min_roi_fraction in {0.05, 0.10, 0.15, 0.25} x 4 spaces x 10 seeds
- **Kill:** PSNR >= 90% of single-tick at same budget AND n_ticks < max_ticks (convergence used)
- **Output:** recommended defaults per parameter

### exp19d: pipeline benchmark (real data)
- **Planned:** single-tick vs multi-tick vs streaming on CIFAR/ImageNet/real graphs
- **Deferred** until exp19a-c determine optimal parameters
