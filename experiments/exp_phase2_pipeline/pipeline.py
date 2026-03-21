#!/usr/bin/env python3
"""
Curiosity -- Phase 2 Unified Pipeline.

Assembles all validated components:
  - Layout selection (per space type)
  - Halo (cosine feathering, topological rule)
  - Two-stage gate (residual-first + utility fallback)
  - Budget governor (EMA controller, step-isolated)
  - Deterministic probe (5-10% exploration)
  - SC-enforce (damp/reject with waste budget)
  - Quality metrics (MSE/PSNR tracking)

CPU-only. No PyTorch dependency for core logic.
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

# Local imports
from config import PipelineConfig
from space_registry import (
    SPACE_FACTORIES, HALO_POLICY, LAYOUT_POLICY,
    make_sc_operators, THRESHOLD_PATH,
    CanonicalTraversal, DeterministicProbe, GovernorIsolation,
    SCEnforcer, load_thresholds, StrictnessTracker, WasteBudget,
    RefinementTree, should_compress, N_CRITICAL_D2,
)
from topo_features import (
    extract_topo_features, topo_adjusted_rho,
    calibrate_transport_probe, reset_calibration,
)


# ===================================================================
# Pipeline result
# ===================================================================

@dataclass
class PipelineResult:
    """Full result of a unified pipeline run."""
    final_state: np.ndarray
    initial_state: np.ndarray
    quality_mse: float
    quality_psnr: float
    coarse_mse: float
    coarse_psnr: float
    n_refined: int
    n_total: int
    n_passed: int
    n_damped: int
    n_rejected: int
    reject_rate: float
    governor_ema_final: float
    traversal_order: list
    probe_set: list
    gate_stage: str
    gate_weights: dict
    enforce_log: list
    waste_budget_exhausted: bool
    compression_viable: bool
    compression_reason: str
    compression_d2_skipped: int
    topo_zone: str                  # GREEN/YELLOW/RED or "n/a"
    topo_eta_f: float               # eta_F entropy index (0.0 if n/a)
    topo_computation_ms: float      # pre-runtime profiling time
    topo_tau_factor: float          # zone-based tau multiplier applied
    wall_time_seconds: float
    config: dict


# ===================================================================
# Two-stage gate
# ===================================================================

class TwoStageGate:
    """Two-stage expert gate: residual-healthy collapse vs utility softmax."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    def evaluate(self, probe_diag: dict, instabilities: dict) -> tuple:
        """Evaluate the gate and return (weights, dominant, stage).

        Args:
            probe_diag: dict mapping expert name -> {'median_gain': float, 'fsr': float}
            instabilities: dict mapping expert name -> float instability score

        Returns:
            (weights: dict, dominant: str, stage: str)
        """
        cfg = self.cfg

        # Stage 1: Is residual healthy?
        resid_fsr = probe_diag.get("resid", {}).get("fsr", 0.0)
        resid_instab = instabilities.get("resid", 0.0)

        if (resid_fsr <= cfg.health_fsr_threshold
                and resid_instab <= cfg.health_instab_threshold):
            # Collapse to residual
            weights = {"resid": cfg.healthy_resid_weight}
            return weights, "resid", "stage1_healthy"

        # Stage 2: Utility-based softmax
        utilities = {}
        for name, d in probe_diag.items():
            inst = instabilities.get(name, 0.0)
            u = d["median_gain"] - cfg.lambda_fsr * d["fsr"] - cfg.mu_instab * inst
            utilities[name] = u

        names = list(utilities.keys())
        u_arr = np.array([utilities[n] for n in names])
        u_arr = u_arr / (cfg.temperature + 1e-12)
        exp_u = np.exp(u_arr - u_arr.max())
        soft = exp_u / (exp_u.sum() + 1e-12)
        weights = {n: float(v) for n, v in zip(names, soft)}

        # Floor/ceiling enforcement
        weights["resid"] = max(weights.get("resid", 0), cfg.resid_min_weight)
        for n in names:
            if n != "resid":
                weights[n] = min(weights[n], cfg.other_max_weight)

        # Renormalize
        total = sum(weights.values())
        weights = {n: v / total for n, v in weights.items()}
        dominant = max(weights, key=weights.get)
        return weights, dominant, "stage2_utility"


# ===================================================================
# Utility: MSE -> PSNR
# ===================================================================

def _mse_to_psnr(mse: float, data_range: float) -> float:
    """Convert MSE to PSNR given the data range."""
    if mse < 1e-15:
        return 100.0
    return float(10.0 * np.log10(data_range ** 2 / mse))


# ===================================================================
# CuriosityPipeline -- main class
# ===================================================================

class CuriosityPipeline:
    """Unified Phase 2 pipeline assembling all validated components.

    Usage::

        pipe = CuriosityPipeline()
        result = pipe.run("scalar_grid", seed=42)
        print(result.quality_psnr, result.reject_rate)
    """

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.gate = TwoStageGate(self.config)

    def run(self, space_type: str, seed: int,
            budget_fraction: float = None) -> PipelineResult:
        """Run the full pipeline on one configuration.

        Args:
            space_type:       One of scalar_grid, vector_grid, irregular_graph,
                              tree_hierarchy.
            seed:             Global seed for determinism.
            budget_fraction:  Override config budget_fraction if given.

        Returns:
            PipelineResult with quality metrics, enforcement stats, and config.
        """
        cfg = self.config
        budget = budget_fraction or cfg.budget_fraction
        t0 = time.time()

        # 1. Init space
        space = SPACE_FACTORIES[space_type]()
        space.setup(seed)
        state = space.get_initial_state()
        initial_state = state.copy()
        units = space.get_units()
        coords = space.get_coords(units)
        n_total = len(units)
        n_budget = max(1, int(budget * n_total))

        # 1b. Compression guard (tree_hierarchy only)
        compression_viable = False
        compression_reason = "not_tree"
        d2_skip_set = set()

        if space_type == "tree_hierarchy":
            parent_map = {}
            children_map = {}
            for i in range(n_total):
                parent_map[i] = (i - 1) // 2 if i > 0 else None
                children_map[i] = [c for c in (2*i+1, 2*i+2) if c < n_total]
            rtree = RefinementTree(parent_map, children_map)
            stats = rtree.chain_statistics()
            n_d2 = stats["degree2_nodes"]

            viable, reason = should_compress(
                n_active=n_total, budget_step=n_budget, n_stable_d2=n_d2)
            compression_viable = viable
            compression_reason = reason if not viable else "viable"

            if viable:
                d2_skip_set = {n for n in range(n_total) if rtree.degree(n) == 2}

        # 1c. Topological profiling (irregular_graph only, one-time pre-runtime)
        topo = None
        topo_zone = "n/a"
        topo_eta_f = 0.0
        topo_computation_ms = 0.0
        topo_tau_factor = 1.0

        if space_type == "irregular_graph" and cfg.topo_profiling_enabled:
            import networkx as nx
            cal = calibrate_transport_probe()

            # Build networkx graph from adapter's neighbor dict
            G_nx = nx.Graph()
            G_nx.add_nodes_from(range(space.n_pts))
            for i, nbrs in space.neighbors.items():
                for j in nbrs:
                    G_nx.add_edge(i, j)

            labels = space.labels
            node_list = sorted(G_nx.nodes())
            topo = extract_topo_features(
                G_nx, labels, node_list,
                kappa_max=cal.kappa_max,
                topo_budget_ms=cfg.topo_budget_ms)
            topo_zone = topo.topo_zone
            topo_eta_f = topo.eta_F
            topo_computation_ms = topo.computation_ms

            # Zone-based tau modulation factor
            if topo_zone == "GREEN":
                topo_tau_factor = cfg.tau_zone_factor_green
            elif topo_zone == "RED":
                topo_tau_factor = cfg.tau_zone_factor_red
            else:
                topo_tau_factor = cfg.tau_zone_factor_yellow

        # 2. Probe selection
        probe_set = DeterministicProbe.select_probe_units(
            units, cfg.probe_fraction, coords, level=0, global_seed=seed)

        # 3. Compute rho + canonical traversal
        rho_values = np.array([space.unit_rho(state, u) for u in units])

        # 3b. Topo-adjusted rho: boost clusters at bridges/hubs
        if topo is not None:
            rho_values = topo_adjusted_rho(rho_values, topo)

        ordered = CanonicalTraversal.sort_units(units, rho_values, space.name)

        # 4. Gate: compute probe diagnostics
        # Simplified: use rho as the "residual expert" diagnostic
        probe_rho = {u: space.unit_rho(state, u) for u in probe_set}
        median_gain = float(np.median(list(probe_rho.values()))) if probe_rho else 0.0
        probe_diag = {"resid": {"median_gain": median_gain, "fsr": 0.0}}
        instabilities = {"resid": 0.0}
        gate_weights, gate_dominant, gate_stage = self.gate.evaluate(
            probe_diag, instabilities)

        # 5. Setup SC-enforce
        enforce_log: List[dict] = []
        n_passed = n_damped = n_rejected = 0
        waste_exhausted = False

        if cfg.enforce_enabled:
            R_fn, Up_fn = make_sc_operators(space_type, space)
            # Map space_type to threshold key prefix
            SPACE_TO_THRESH = {
                "scalar_grid": "T1_scalar",
                "vector_grid": "T2_vector",
                "irregular_graph": "T3_graph",
                "tree_hierarchy": "T4_tree",
            }
            thresh_prefix = SPACE_TO_THRESH[space_type]
            all_thresholds = load_thresholds(str(THRESHOLD_PATH))
            # Apply topo zone modulation to thresholds for irregular_graph
            effective_thresholds = dict(all_thresholds)
            if topo_tau_factor != 1.0:
                for key in list(effective_thresholds.keys()):
                    if isinstance(effective_thresholds[key], (int, float)):
                        effective_thresholds[key] = effective_thresholds[key] * topo_tau_factor
                    elif isinstance(effective_thresholds[key], dict):
                        effective_thresholds[key] = {
                            k: v * topo_tau_factor
                            for k, v in effective_thresholds[key].items()
                        }

            enforcer = SCEnforcer(
                tau_parent=effective_thresholds, R_fn=R_fn, Up_fn=Up_fn,
                space_type=thresh_prefix,
                damp_factor=cfg.damp_factor,
                max_damp_iterations=cfg.max_damp_iterations,
                n_active=n_total)
            strictness = StrictnessTracker(
                escalation_factor=cfg.strictness_escalation,
                decay_factor=cfg.strictness_decay)
            enforcer.strictness_tracker = strictness
            waste = WasteBudget(n_budget, omega=cfg.waste_omega)

        # 6. Governor
        governor = GovernorIsolation(alpha=cfg.ema_alpha)

        # 7. Refinement loop
        refined_count = 0
        n_d2_skipped = 0
        n_active_remaining = n_total
        for unit in ordered:
            if refined_count >= n_budget:
                break
            if cfg.enforce_enabled and waste_exhausted:
                break

            # Compression: skip degree-2 transit nodes
            if unit in d2_skip_set:
                n_d2_skipped += 1
                continue

            # Get halo parameter per space type
            if space_type == "tree_hierarchy":
                halo_param = 0  # disabled
            elif space_type == "irregular_graph":
                halo_param = cfg.halo_hops
            else:
                halo_param = cfg.halo_width

            # Compute refinement candidate
            old_state = state.copy()
            if space_type in ("irregular_graph", "tree_hierarchy"):
                state_candidate = space.refine_unit(state, unit, halo_hops=halo_param)
            else:
                state_candidate = space.refine_unit(state, unit, halo=halo_param)

            delta = state_candidate - old_state

            # SC-enforce check
            if cfg.enforce_enabled:
                unit_id = str(unit)
                result = enforcer.check_and_enforce(
                    delta, space.coarse, level=1, unit_id=unit_id)

                enforce_log.append({
                    "unit": str(unit),
                    "action": result.action,
                    "d_parent": result.d_parent_value,
                    "original_d_parent": result.original_d_parent,
                    "damp_iterations": result.damp_iterations,
                })

                if result.action == "pass":
                    state = state_candidate
                    n_passed += 1
                elif result.action == "damped":
                    state = old_state + result.enforced_delta
                    n_damped += 1
                elif result.action == "rejected":
                    state = old_state  # revert
                    n_rejected += 1
                    s = strictness.get(unit_id)
                    if waste.record_reject(unit_id, s):
                        waste_exhausted = True
                        break
                    continue  # don't count toward budget

                if result.action in ("damped", "rejected"):
                    strictness.escalate(unit_id)
            else:
                state = state_candidate
                n_passed += 1

            # -- Segment compression guard (re-evaluate inside loop) --
            if space_type == "tree_hierarchy" and d2_skip_set:
                n_active_remaining -= 1
                budget_left = n_budget - refined_count
                viable_now, reason_now = should_compress(
                    n_active=n_active_remaining,
                    budget_step=budget_left,
                    n_stable_d2=len(d2_skip_set),
                )
                if not viable_now:
                    # Guards no longer met -- stop skipping d2 nodes
                    d2_skip_set.clear()
                    compression_viable = False
                    compression_reason = reason_now

            governor.accumulate(1.0)
            governor.commit_step()
            refined_count += 1

        # Strictness decay at end of step
        if cfg.enforce_enabled:
            strictness.decay_all()

        # 8. Compute quality metrics
        gt = space.gt
        mse_final = float(np.mean((gt - state) ** 2))
        mse_coarse = float(np.mean((gt - initial_state) ** 2))
        data_range = float(gt.max() - gt.min())

        psnr_final = _mse_to_psnr(mse_final, data_range)
        psnr_coarse = _mse_to_psnr(mse_coarse, data_range)

        reject_rate = n_rejected / max(1, n_passed + n_damped + n_rejected)

        wall_time = time.time() - t0

        return PipelineResult(
            final_state=state,
            initial_state=initial_state,
            quality_mse=mse_final,
            quality_psnr=psnr_final,
            coarse_mse=mse_coarse,
            coarse_psnr=psnr_coarse,
            n_refined=refined_count,
            n_total=n_total,
            n_passed=n_passed,
            n_damped=n_damped,
            n_rejected=n_rejected,
            reject_rate=reject_rate,
            governor_ema_final=governor.get_ema(),
            traversal_order=[str(u) for u in ordered],
            probe_set=[str(u) for u in probe_set],
            gate_stage=gate_stage,
            gate_weights=gate_weights,
            enforce_log=enforce_log,
            waste_budget_exhausted=waste_exhausted,
            compression_viable=compression_viable,
            compression_reason=compression_reason,
            compression_d2_skipped=n_d2_skipped,
            topo_zone=topo_zone,
            topo_eta_f=topo_eta_f,
            topo_computation_ms=topo_computation_ms,
            topo_tau_factor=topo_tau_factor,
            wall_time_seconds=wall_time,
            config=cfg.__dict__,
        )


# ===================================================================
# Smoke test
# ===================================================================

def _run_smoke_test():
    """Quick smoke test: run all 4 space types, print summary, save JSON."""
    print("=" * 70)
    print("Phase 2 Unified Pipeline -- Smoke Test")
    print("=" * 70)

    pipe = CuriosityPipeline()
    space_types = ["scalar_grid", "vector_grid", "irregular_graph", "tree_hierarchy"]
    results_all = {}

    for st in space_types:
        print(f"\n  Running {st} ...", end=" ", flush=True)
        r = pipe.run(st, seed=42, budget_fraction=0.30)
        print(f"done ({r.wall_time_seconds:.2f}s)")

        print(f"    PSNR coarse: {r.coarse_psnr:7.2f}  |  PSNR final: {r.quality_psnr:7.2f}")
        print(f"    n_refined: {r.n_refined:4d} / {r.n_total:4d}")
        print(f"    passed: {r.n_passed:4d}  damped: {r.n_damped:4d}  rejected: {r.n_rejected:4d}"
              f"  reject_rate: {r.reject_rate:.3f}")
        print(f"    gate: {r.gate_stage}  weights: {r.gate_weights}")
        print(f"    waste_exhausted: {r.waste_budget_exhausted}")
        print(f"    compression: viable={r.compression_viable} "
              f"reason={r.compression_reason} d2_skipped={r.compression_d2_skipped}")
        if r.topo_zone != "n/a":
            print(f"    topo: zone={r.topo_zone}  eta_F={r.topo_eta_f:.2f}  "
                  f"tau_factor={r.topo_tau_factor:.2f}  profiling={r.topo_computation_ms:.0f}ms")

        if r.reject_rate > pipe.config.reject_rate_alert:
            print(f"    ** ALERT: reject_rate {r.reject_rate:.3f} > "
                  f"threshold {pipe.config.reject_rate_alert:.3f}")

        results_all[st] = {
            "quality_mse": r.quality_mse,
            "quality_psnr": r.quality_psnr,
            "coarse_mse": r.coarse_mse,
            "coarse_psnr": r.coarse_psnr,
            "n_refined": r.n_refined,
            "n_total": r.n_total,
            "n_passed": r.n_passed,
            "n_damped": r.n_damped,
            "n_rejected": r.n_rejected,
            "reject_rate": r.reject_rate,
            "governor_ema_final": r.governor_ema_final,
            "gate_stage": r.gate_stage,
            "gate_weights": r.gate_weights,
            "waste_budget_exhausted": r.waste_budget_exhausted,
            "compression_viable": r.compression_viable,
            "compression_reason": r.compression_reason,
            "compression_d2_skipped": r.compression_d2_skipped,
            "topo_zone": r.topo_zone,
            "topo_eta_f": r.topo_eta_f,
            "topo_computation_ms": r.topo_computation_ms,
            "topo_tau_factor": r.topo_tau_factor,
            "wall_time_seconds": r.wall_time_seconds,
        }

    # Save results
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "smoke_test.json"
    with open(out_path, "w") as f:
        json.dump(results_all, f, indent=2)

    print(f"\n  Results saved to {out_path}")
    print("=" * 70)
    print("  SMOKE TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    _run_smoke_test()
