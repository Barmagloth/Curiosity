#!/usr/bin/env python3
"""
exp19d — Pipeline benchmark on real data.

Tests single-tick vs multi-tick on:
  - CIFAR-10 images (scalar_grid: grayscale, vector_grid: RGB)
  - Real-world graphs (networkx: karate, les_miserables, florentine)

Compares mt=1 (single-tick) vs mt=3 (exp19a sweet spot) vs mt=5.
"""

import sys
import json
import time
import numpy as np
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENTS_DIR / "exp_phase2_pipeline"))
sys.path.insert(0, str(EXPERIMENTS_DIR / "exp10d_seed_determinism"))

from exp10d_seed_determinism import ScalarGridSpace, VectorGridSpace, IrregularGraphSpace
from pipeline import CuriosityPipeline, PipelineConfig
import space_registry as _sr


# =====================================================================
# Real-data space adapters
# =====================================================================

class CIFARScalarGridSpace(ScalarGridSpace):
    """scalar_grid from a real CIFAR-10 image (grayscale, upscaled)."""
    name = "scalar_grid"

    def __init__(self, image_idx=0, N=128, tile=8):
        super().__init__(N=N, tile=tile)
        self._image_idx = image_idx

    def setup(self, seed: int):
        from PIL import Image
        import torchvision
        ds = torchvision.datasets.CIFAR10(
            root='R:/Projects/Curiosity/data/cifar10', download=False, train=True)
        # Use seed to pick image index (deterministic)
        idx = (self._image_idx + seed * 7) % len(ds)
        img, _ = ds[idx]
        # Convert to grayscale numpy, resize to N×N
        gray = np.array(img.convert('L'), dtype=np.float64) / 255.0
        # Resize from 32×32 to N×N using bilinear
        from scipy.ndimage import zoom
        factor = self.N / 32
        self.gt = zoom(gray, factor, order=1)

        # Coarse = tile-averaged version
        self.coarse = np.zeros_like(self.gt)
        for ti in range(self.NT):
            for tj in range(self.NT):
                s = slice(ti * self.T, (ti + 1) * self.T)
                cs = slice(tj * self.T, (tj + 1) * self.T)
                self.coarse[s, cs] = self.gt[s, cs].mean()


class CIFARVectorGridSpace(VectorGridSpace):
    """vector_grid from CIFAR-10 RGB channels."""
    name = "vector_grid"

    def __init__(self, image_idx=0, N=64, tile=8, D=3):
        super().__init__(N=N, tile=tile, D=D)
        self._image_idx = image_idx

    def setup(self, seed: int):
        import torchvision
        ds = torchvision.datasets.CIFAR10(
            root='R:/Projects/Curiosity/data/cifar10', download=False, train=True)
        idx = (self._image_idx + seed * 7) % len(ds)
        img, _ = ds[idx]
        rgb = np.array(img, dtype=np.float64) / 255.0  # (32, 32, 3)
        from scipy.ndimage import zoom
        factor_h = self.N / 32
        factor_w = self.N / 32
        self.gt = zoom(rgb, (factor_h, factor_w, 1), order=1)

        self.coarse = np.zeros_like(self.gt)
        for ti in range(self.NT):
            for tj in range(self.NT):
                s = slice(ti * self.T, (ti + 1) * self.T)
                cs = slice(tj * self.T, (tj + 1) * self.T)
                for d in range(self.D):
                    self.coarse[s, cs, d] = self.gt[s, cs, d].mean()


class RealGraphSpace(IrregularGraphSpace):
    """irregular_graph from a real-world networkx graph."""
    name = "irregular_graph"

    def __init__(self, graph_name="karate", k=6):
        # Initialize with dummy params, override in setup
        self._graph_name = graph_name
        self._k = k
        super().__init__(n_points=10, k=k)  # dummy, will be overridden

    def setup(self, seed: int):
        import networkx as nx

        # Load real graph
        if self._graph_name == "karate":
            G = nx.karate_club_graph()
        elif self._graph_name == "les_miserables":
            G = nx.les_miserables_graph()
        elif self._graph_name == "florentine":
            G = nx.florentine_families_graph()
        else:
            raise ValueError(f"Unknown graph: {self._graph_name}")

        # Build adjacency and positions
        nodes = sorted(G.nodes())
        n = len(nodes)
        node_to_idx = {nd: i for i, nd in enumerate(nodes)}

        # Build neighbor dict
        self.neighbors = {}
        for nd in nodes:
            i = node_to_idx[nd]
            self.neighbors[i] = [node_to_idx[nb] for nb in G.neighbors(nd)]

        # Generate positions (spring layout with seed)
        pos = nx.spring_layout(G, seed=seed)
        self.pos = np.array([pos[nd] for nd in nodes])
        self.n_pts = n

        # Cluster via Leiden (community detection)
        try:
            import leidenalg
            import igraph as ig
            edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]
            ig_g = ig.Graph(n=n, edges=edges)
            partition = leidenalg.find_partition(ig_g, leidenalg.ModularityVertexPartition,
                                                 seed=seed)
            self.labels = np.zeros(n, dtype=int)
            for cid, members in enumerate(partition):
                for m in members:
                    self.labels[m] = cid
            self.n_clusters = len(partition)
        except ImportError:
            # Fallback: greedy modularity
            from networkx.algorithms.community import greedy_modularity_communities
            comms = list(greedy_modularity_communities(G))
            self.labels = np.zeros(n, dtype=int)
            for cid, members in enumerate(comms):
                for m in members:
                    self.labels[node_to_idx[m]] = cid
            self.n_clusters = len(comms)

        # Generate ground truth: node values based on cluster + noise
        rng = np.random.default_rng(seed)
        cluster_means = rng.uniform(0.2, 0.8, self.n_clusters)
        self.gt = np.array([cluster_means[self.labels[i]] + rng.normal(0, 0.1)
                            for i in range(n)])

        # Coarse: cluster mean
        self.coarse = np.zeros(n, dtype=np.float64)
        for c in range(self.n_clusters):
            mask = self.labels == c
            if mask.sum() > 0:
                self.coarse[mask] = self.gt[mask].mean()

    def get_initial_state(self):
        return self.coarse.copy()


# =====================================================================
# Runner
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


CONFIGS = [
    # (name, space_type, factory)
    ("cifar_scalar_128", "scalar_grid", lambda: CIFARScalarGridSpace(N=128, tile=8)),
    ("cifar_vector_64", "vector_grid", lambda: CIFARVectorGridSpace(N=64, tile=8, D=3)),
    ("karate_graph", "irregular_graph", lambda: RealGraphSpace("karate")),
    ("les_miserables_graph", "irregular_graph", lambda: RealGraphSpace("les_miserables")),
    ("florentine_graph", "irregular_graph", lambda: RealGraphSpace("florentine")),
]

MT_VALUES = [1, 3, 5]
N_SEEDS = 10
BUDGET = 0.30


def run_single(name, space_type, factory, max_ticks, seed):
    cfg = PipelineConfig(
        max_ticks=max_ticks,
        budget_fraction=BUDGET,
        topo_profiling_enabled=(space_type == "irregular_graph"),
    )
    pipe = CuriosityPipeline(cfg)

    old = _sr.SPACE_FACTORIES.get(space_type)
    _sr.SPACE_FACTORIES[space_type] = factory

    try:
        t0 = time.time()
        r = pipe.run(space_type, seed=seed, budget_fraction=BUDGET)
        wall = time.time() - t0
    finally:
        if old is not None:
            _sr.SPACE_FACTORIES[space_type] = old

    return {
        "name": name,
        "space_type": space_type,
        "max_ticks": max_ticks,
        "seed": seed,
        "psnr_gain": r.quality_psnr - r.coarse_psnr,
        "psnr_final": r.quality_psnr,
        "n_refined": r.n_refined,
        "n_total": r.n_total,
        "n_ticks_executed": r.n_ticks_executed,
        "convergence_reason": r.convergence_reason,
        "wall_time": wall,
    }


def main():
    total = len(CONFIGS) * len(MT_VALUES) * N_SEEDS
    print("=" * 70)
    print("exp19d — Real data benchmark")
    print(f"  configs: {[c[0] for c in CONFIGS]}")
    print(f"  max_ticks: {MT_VALUES}")
    print(f"  seeds: {N_SEEDS}, budget: {BUDGET}")
    print(f"  total: {total}")
    print("=" * 70)

    all_results = []
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, space_type, factory in CONFIGS:
        print(f"\n--- {name} ({space_type}) ---")
        for mt in MT_VALUES:
            results = []
            for seed in range(N_SEEDS):
                try:
                    r = run_single(name, space_type, factory, mt, seed)
                    results.append(r)
                    all_results.append(r)
                except Exception as e:
                    all_results.append({
                        "name": name, "space_type": space_type,
                        "max_ticks": mt, "seed": seed, "error": str(e),
                    })

            valid = [r for r in results if "error" not in r]
            if valid:
                med_psnr = np.median([r["psnr_gain"] for r in valid])
                med_wall = np.median([r["wall_time"] for r in valid])
                med_ticks = np.median([r["n_ticks_executed"] for r in valid])
                n_units = valid[0]["n_total"]
                print(f"  mt={mt:2d}: PSNR gain {med_psnr:+.2f} dB  "
                      f"wall {med_wall:.2f}s  ticks_used={med_ticks:.0f}  "
                      f"n_units={n_units}")

    # Save
    with open(out_dir / "exp19d_real_data.json", "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)

    # Summary table
    print("\n" + "=" * 70)
    print("  SUMMARY: multi-tick vs single-tick on real data")
    print("=" * 70)
    valid_all = [r for r in all_results if "error" not in r]
    for name, _, _ in CONFIGS:
        single = [r for r in valid_all if r["name"] == name and r["max_ticks"] == 1]
        if not single:
            continue
        single_psnr = np.median([r["psnr_gain"] for r in single])
        print(f"\n  {name} (single-tick: {single_psnr:+.2f} dB):")
        for mt in MT_VALUES[1:]:
            multi = [r for r in valid_all if r["name"] == name and r["max_ticks"] == mt]
            if multi:
                mt_psnr = np.median([r["psnr_gain"] for r in multi])
                ratio = mt_psnr / max(abs(single_psnr), 0.01) * 100
                print(f"    mt={mt}: {mt_psnr:+.2f} dB ({ratio:.0f}% of single)")

    print("\n" + "=" * 70)
    print("  exp19d COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
