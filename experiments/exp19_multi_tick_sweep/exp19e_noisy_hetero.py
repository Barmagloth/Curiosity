#!/usr/bin/env python3
"""
exp19e — Multi-tick validation on noisy and heterogeneous data.

Multi-tick gate should shine here: residual degrades under noise,
gate shifts EMA weights to combo; ROI filters out noise-dominated units;
convergence stops wasting budget on noise.

Three data regimes:
  A) Noisy: GT + gaussian noise (σ = 0.05, 0.10, 0.20, 0.40)
  B) Heterogeneous: half space smooth, half sharp features
  C) Mixed: heterogeneous + noise

Each regime tested with mt=1 (baseline) vs mt=3 vs mt=5.
Medium + large scale, 10 seeds per config.
"""

import sys
import json
import time
import numpy as np
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENTS_DIR / "exp_phase2_pipeline"))
sys.path.insert(0, str(EXPERIMENTS_DIR / "exp10d_seed_determinism"))

from exp10d_seed_determinism import ScalarGridSpace, VectorGridSpace
from pipeline import CuriosityPipeline, PipelineConfig
import space_registry as _sr


# =====================================================================
# Noisy + heterogeneous space wrappers
# =====================================================================

class NoisyScalarGrid(ScalarGridSpace):
    """ScalarGridSpace with observation noise added to GT."""
    name = "scalar_grid"

    def __init__(self, noise_sigma=0.1, N=128, tile=8):
        super().__init__(N=N, tile=tile)
        self._noise_sigma = noise_sigma

    def setup(self, seed: int):
        super().setup(seed)
        # Store clean GT for quality measurement
        self._clean_gt = self.gt.copy()
        # Add noise to GT (what the system observes)
        rng = np.random.default_rng(seed + 10000)
        self.gt = self._clean_gt + rng.normal(0, self._noise_sigma, self.gt.shape)


class HeteroScalarGrid(ScalarGridSpace):
    """ScalarGridSpace with heterogeneous complexity.

    Left half: smooth (single low-freq gaussian)
    Right half: sharp features (many narrow gaussians + step edges)
    """
    name = "scalar_grid"

    def __init__(self, N=128, tile=8):
        super().__init__(N=N, tile=tile)

    def setup(self, seed: int):
        rng = np.random.default_rng(seed)
        x = np.linspace(0, 1, self.N, endpoint=False)
        xx, yy = np.meshgrid(x, x)

        gt = np.zeros((self.N, self.N))

        # Left half: smooth (one broad gaussian)
        cx, cy = 0.25, 0.5
        gt += 0.8 * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * 0.15**2))

        # Right half: sharp features
        for _ in range(15):
            cx = rng.uniform(0.55, 0.95)
            cy = rng.uniform(0.05, 0.95)
            sigma = rng.uniform(0.01, 0.04)  # narrow
            amp = rng.uniform(0.3, 1.0)
            gt += amp * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))

        # Step edges in right half
        gt += 0.5 * (xx > 0.6) * (yy > 0.3) * (yy < 0.7)
        gt += 0.3 * (xx > 0.75) * (yy > 0.5)

        self.gt = gt

        # Coarse
        self.coarse = np.zeros_like(self.gt)
        for ti in range(self.NT):
            for tj in range(self.NT):
                s = slice(ti * self.T, (ti + 1) * self.T)
                cs = slice(tj * self.T, (tj + 1) * self.T)
                self.coarse[s, cs] = self.gt[s, cs].mean()


class MixedScalarGrid(HeteroScalarGrid):
    """Heterogeneous + noise."""
    name = "scalar_grid"

    def __init__(self, noise_sigma=0.1, N=128, tile=8):
        super().__init__(N=N, tile=tile)
        self._noise_sigma = noise_sigma

    def setup(self, seed: int):
        super().setup(seed)
        self._clean_gt = self.gt.copy()
        rng = np.random.default_rng(seed + 20000)
        self.gt = self._clean_gt + rng.normal(0, self._noise_sigma, self.gt.shape)


# =====================================================================
# Experiment configs
# =====================================================================

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


N_SEEDS = 10
BUDGET = 0.30
MT_VALUES = [1, 3, 5]

SCENARIOS = []

# A) Noisy at different σ levels
for sigma in [0.05, 0.10, 0.20, 0.40]:
    SCENARIOS.append({
        "name": f"noisy_s{sigma:.2f}_medium",
        "factory": lambda s=sigma: NoisyScalarGrid(noise_sigma=s, N=128, tile=8),
        "regime": "noisy",
        "sigma": sigma,
        "scale": "medium",
        "has_clean_gt": True,
    })
    SCENARIOS.append({
        "name": f"noisy_s{sigma:.2f}_large",
        "factory": lambda s=sigma: NoisyScalarGrid(noise_sigma=s, N=256, tile=8),
        "regime": "noisy",
        "sigma": sigma,
        "scale": "large",
        "has_clean_gt": True,
    })

# B) Heterogeneous
SCENARIOS.append({
    "name": "hetero_medium",
    "factory": lambda: HeteroScalarGrid(N=128, tile=8),
    "regime": "hetero",
    "sigma": 0.0,
    "scale": "medium",
    "has_clean_gt": False,
})
SCENARIOS.append({
    "name": "hetero_large",
    "factory": lambda: HeteroScalarGrid(N=256, tile=8),
    "regime": "hetero",
    "sigma": 0.0,
    "scale": "large",
    "has_clean_gt": False,
})

# C) Mixed (hetero + noise)
for sigma in [0.05, 0.10, 0.20]:
    SCENARIOS.append({
        "name": f"mixed_s{sigma:.2f}_medium",
        "factory": lambda s=sigma: MixedScalarGrid(noise_sigma=s, N=128, tile=8),
        "regime": "mixed",
        "sigma": sigma,
        "scale": "medium",
        "has_clean_gt": True,
    })
    SCENARIOS.append({
        "name": f"mixed_s{sigma:.2f}_large",
        "factory": lambda s=sigma: MixedScalarGrid(noise_sigma=s, N=256, tile=8),
        "regime": "mixed",
        "sigma": sigma,
        "scale": "large",
        "has_clean_gt": True,
    })


# =====================================================================
# Runner
# =====================================================================

def run_single(scenario, max_ticks, seed):
    """Run one (scenario, max_ticks, seed) config."""
    cfg = PipelineConfig(
        max_ticks=max_ticks,
        budget_fraction=BUDGET,
        topo_profiling_enabled=False,
    )
    pipe = CuriosityPipeline(cfg)

    factory = scenario["factory"]
    old = _sr.SPACE_FACTORIES.get("scalar_grid")
    _sr.SPACE_FACTORIES["scalar_grid"] = factory

    try:
        t0 = time.time()
        r = pipe.run("scalar_grid", seed=seed, budget_fraction=BUDGET)
        wall = time.time() - t0
    finally:
        if old is not None:
            _sr.SPACE_FACTORIES["scalar_grid"] = old

    result = {
        "name": scenario["name"],
        "regime": scenario["regime"],
        "sigma": scenario["sigma"],
        "scale": scenario["scale"],
        "max_ticks": max_ticks,
        "seed": seed,
        "psnr_gain": r.quality_psnr - r.coarse_psnr,
        "psnr_final": r.quality_psnr,
        "n_refined": r.n_refined,
        "n_total": r.n_total,
        "n_ticks_executed": r.n_ticks_executed,
        "convergence_reason": r.convergence_reason,
        "wall_time": wall,
        "gate_stage": r.gate_stage,
        "final_w_resid": r.final_ema_weights.get("w_resid", 1.0),
    }

    # If has clean GT: compute PSNR vs clean (not noisy) GT
    if scenario["has_clean_gt"]:
        space = factory()
        space.setup(seed)
        if hasattr(space, '_clean_gt'):
            clean_gt = space._clean_gt
            mse_vs_clean = float(np.mean((clean_gt - r.final_state) ** 2))
            data_range = float(clean_gt.max() - clean_gt.min())
            if mse_vs_clean > 1e-15 and data_range > 1e-15:
                psnr_vs_clean = float(10.0 * np.log10(data_range**2 / mse_vs_clean))
            else:
                psnr_vs_clean = 100.0
            mse_coarse_clean = float(np.mean((clean_gt - r.initial_state) ** 2))
            if mse_coarse_clean > 1e-15:
                psnr_coarse_clean = float(10.0 * np.log10(data_range**2 / mse_coarse_clean))
            else:
                psnr_coarse_clean = 100.0
            result["psnr_vs_clean"] = psnr_vs_clean
            result["psnr_gain_vs_clean"] = psnr_vs_clean - psnr_coarse_clean

    return result


def main():
    total = len(SCENARIOS) * len(MT_VALUES) * N_SEEDS
    print("=" * 70)
    print("exp19e — Noisy + heterogeneous data stress test")
    print(f"  scenarios: {len(SCENARIOS)}")
    print(f"  max_ticks: {MT_VALUES}")
    print(f"  seeds: {N_SEEDS}")
    print(f"  total: {total}")
    print("=" * 70)

    all_results = []
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    for scenario in SCENARIOS:
        print(f"\n--- {scenario['name']} ---")
        for mt in MT_VALUES:
            results = []
            for seed in range(N_SEEDS):
                try:
                    r = run_single(scenario, mt, seed)
                    results.append(r)
                    all_results.append(r)
                except Exception as e:
                    all_results.append({
                        "name": scenario["name"], "max_ticks": mt,
                        "seed": seed, "error": str(e),
                    })

            valid = [r for r in results if "error" not in r]
            if valid:
                med_psnr = np.median([r["psnr_gain"] for r in valid])
                med_w_resid = np.median([r["final_w_resid"] for r in valid])
                ticks = np.median([r["n_ticks_executed"] for r in valid])
                extra = ""
                if "psnr_vs_clean" in valid[0]:
                    med_clean = np.median([r["psnr_gain_vs_clean"] for r in valid])
                    extra = f"  vs_clean={med_clean:+.2f}"
                print(f"  mt={mt}: PSNR {med_psnr:+.2f} dB  w_resid={med_w_resid:.3f}  "
                      f"ticks={ticks:.0f}{extra}")

        # Save incrementally per scenario
        scenario_results = [r for r in all_results if r.get("name") == scenario["name"]]
        out_path = out_dir / f"exp19e_{scenario['name']}.json"
        with open(out_path, "w") as f:
            json.dump(scenario_results, f, indent=2, cls=NumpyEncoder)

    # Summary
    with open(out_dir / "exp19e_combined.json", "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)

    print("\n" + "=" * 70)
    print("  SUMMARY: multi-tick advantage under noise/heterogeneity")
    print("=" * 70)

    valid_all = [r for r in all_results if "error" not in r]

    # Group by regime
    for regime in ["noisy", "hetero", "mixed"]:
        regime_results = [r for r in valid_all if r.get("regime") == regime]
        if not regime_results:
            continue
        print(f"\n  [{regime.upper()}]")

        scenarios_in_regime = sorted(set(r["name"] for r in regime_results))
        for sc_name in scenarios_in_regime:
            single = [r for r in regime_results if r["name"] == sc_name and r["max_ticks"] == 1]
            if not single:
                continue
            single_psnr = np.median([r["psnr_gain"] for r in single])
            print(f"    {sc_name}:")
            for mt in MT_VALUES:
                multi = [r for r in regime_results if r["name"] == sc_name and r["max_ticks"] == mt]
                if multi:
                    mt_psnr = np.median([r["psnr_gain"] for r in multi])
                    w_resid = np.median([r["final_w_resid"] for r in multi])
                    ratio = mt_psnr / max(abs(single_psnr), 0.01) * 100
                    extra = ""
                    if "psnr_gain_vs_clean" in multi[0]:
                        clean_gain = np.median([r["psnr_gain_vs_clean"] for r in multi])
                        extra = f"  clean_gain={clean_gain:+.2f}"
                    print(f"      mt={mt}: {mt_psnr:+.2f} dB ({ratio:.0f}%)  "
                          f"w_resid={w_resid:.3f}{extra}")

    # Key question: does multi-tick maintain quality vs CLEAN GT better than single-tick?
    print("\n  KEY METRIC: PSNR vs clean GT (noisy scenarios only)")
    noisy = [r for r in valid_all if r.get("regime") in ("noisy", "mixed")
             and "psnr_gain_vs_clean" in r]
    if noisy:
        for mt in MT_VALUES:
            mt_clean = [r["psnr_gain_vs_clean"] for r in noisy if r["max_ticks"] == mt]
            if mt_clean:
                print(f"    mt={mt}: median PSNR gain vs clean = {np.median(mt_clean):+.3f} dB")

    print("\n" + "=" * 70)
    print("  exp19e COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
