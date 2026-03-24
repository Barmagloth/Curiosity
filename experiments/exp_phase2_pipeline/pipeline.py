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

Phase 4 additions:
  - Multi-tick loop with convergence detection
  - WeightedRhoGate (EMA weight transitions)
  - Cold-start + pilot threshold calibration
  - ROI gating per unit
  - GovernorEMA (smooth strictness control)

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
from enox_infra import (
    region_uri, DecisionJournal, JournalEntry,
    MultiStageDedup, PostStepSweep,
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
    # Enox infrastructure
    enox_journal_summary: dict      # {total, by_decision: {...}}
    enox_dedup_stats: dict          # DedupStats as dict
    enox_sweep_stats: dict          # SweepStats as dict
    enox_uri_map: dict              # {unit_str: final_uri} (empty if disabled)
    wall_time_seconds: float
    config: dict
    # Phase 4: Multi-tick
    tick_log: list = field(default_factory=list)
    n_ticks_executed: int = 1
    convergence_reason: str = "max_ticks"
    pilot_thresholds: dict = field(default_factory=dict)
    governor_strictness_history: list = field(default_factory=list)
    final_ema_weights: dict = field(default_factory=dict)


# ===================================================================
# Phase 4: TickState
# ===================================================================

@dataclass
class TickState:
    """Mutable state evolving across ticks in multi-tick mode."""
    refined_set: set = field(default_factory=set)
    evaluated_set: set = field(default_factory=set)
    rejected_set: set = field(default_factory=set)
    prev_rho_signs: dict = field(default_factory=dict)  # unit -> sign(rho)
    accepted_history: list = field(default_factory=list)  # n_accepted per tick
    pilot_instab: list = field(default_factory=list)
    pilot_fsr: list = field(default_factory=list)
    calibrated_instab_thresh: float = None
    calibrated_fsr_thresh: float = None
    reference_median_gain: float = 0.0
    budget_remaining: int = 0
    # EMA weights for rho signals
    w_resid: float = 1.0
    w_hf: float = 0.0
    w_var: float = 0.0


# ===================================================================
# Phase 4: Helper functions
# ===================================================================

def _compute_fsr(units, rho_values, tick_state: TickState) -> float:
    """FSR = sign-flip rate among NON-refined units vs previous tick.

    First tick: stores signs, returns 0.0 (no previous data).
    """
    if not tick_state.prev_rho_signs:
        # First tick: store signs, return 0.0
        for i, u in enumerate(units):
            tick_state.prev_rho_signs[u] = 1 if rho_values[i] >= 0 else -1
        return 0.0

    flips = 0
    count = 0
    for i, u in enumerate(units):
        if u in tick_state.refined_set:
            continue  # exclude refined (issue 3)
        count += 1
        prev_sign = tick_state.prev_rho_signs.get(u, 0)
        curr_sign = 1 if rho_values[i] >= 0 else -1
        if prev_sign != curr_sign and prev_sign != 0:
            flips += 1

    # Update stored signs for next tick
    for i, u in enumerate(units):
        tick_state.prev_rho_signs[u] = 1 if rho_values[i] >= 0 else -1

    return flips / max(count, 1)


def _compute_instability(rho_values, tick_state: TickState, units) -> float:
    """Instability = CV(|rho|) among non-refined units."""
    vals = []
    for i, u in enumerate(units):
        if u not in tick_state.refined_set:
            vals.append(abs(rho_values[i]))
    if not vals:
        return 0.0
    arr = np.array(vals)
    mean_abs = arr.mean()
    if mean_abs < 1e-10:
        return 0.0
    return float(arr.std() / mean_abs)


def _cold_start_thresholds(rho_values, tick_state: TickState,
                            topo_zone: str, cfg: PipelineConfig):
    """Set initial gate thresholds from L0/topo zone and initial rho distribution.

    Instead of blind pilot (K ticks of equal weights), use pre-computed info:
    - topo zone gives structural estimate (GREEN=stable, RED=chaotic)
    - CV of initial rho gives empirical instability estimate
    """
    # Empirical instability from initial rho
    abs_rho = np.abs(rho_values)
    mean_rho = abs_rho.mean()
    cv_rho = float(abs_rho.std() / max(mean_rho, 1e-10))

    # Zone-based prior
    if topo_zone == "GREEN":
        instab_prior = 0.15  # resid likely stable
    elif topo_zone == "RED":
        instab_prior = 0.40  # resid likely unstable
    else:
        instab_prior = 0.25  # default / YELLOW / n/a

    # Blend prior with empirical CV
    instab_thresh = 0.5 * instab_prior + 0.5 * cv_rho
    instab_thresh *= cfg.pilot_thresh_factor

    fsr_thresh = max(cfg.pilot_fsr_floor, instab_thresh * 0.5)

    tick_state.calibrated_instab_thresh = instab_thresh
    tick_state.calibrated_fsr_thresh = fsr_thresh


def _compute_unit_roi(space, state_before: np.ndarray,
                       state_after: np.ndarray) -> float:
    """ROI = global MSE reduction from refining this unit."""
    gt = space.gt
    mse_before = float(np.mean((gt - state_before) ** 2))
    mse_after = float(np.mean((gt - state_after) ** 2))
    gain = max(0.0, mse_before - mse_after)
    return gain  # cost = 1.0 for now


def _check_convergence(tick_state: TickState, cfg: PipelineConfig,
                        probe_count: int) -> Optional[str]:
    """Check if pipeline has converged (no progress for K ticks)."""
    history = tick_state.accepted_history
    K = cfg.convergence_window
    if len(history) >= K and all(h == 0 for h in history[-K:]):
        return "zero_accepted"
    return None


def _select_tick_probes(units, coords, tick_state: TickState,
                         cfg: PipelineConfig, seed: int, tick: int) -> list:
    """Select probes from unevaluated, unrefined units (issue 7)."""
    eligible = []
    eligible_coords = []
    for i, u in enumerate(units):
        if u not in tick_state.evaluated_set and u not in tick_state.refined_set:
            eligible.append(u)
            eligible_coords.append(coords[i])

    if not eligible:
        return []

    return DeterministicProbe.select_probe_units(
        eligible, cfg.probe_fraction, eligible_coords,
        level=tick, global_seed=seed)


def _compute_weighted_rho(space, state, units, tick_state: TickState) -> np.ndarray:
    """Compute rho = sum(w_i * signal_i), skipping zero-weight signals.

    Currently only resid is implemented. HF and variance signals
    will be added when multi-signal pipeline is built.
    """
    n = len(units)

    # Always compute resid (unit_rho = MSE)
    resid_values = np.array([space.unit_rho(state, u) for u in units])

    rho = tick_state.w_resid * resid_values

    # HF signal (when weight > 0 and signal is available)
    # TODO: implement HF computation per unit
    # if tick_state.w_hf > 1e-6:
    #     hf_values = np.array([_unit_hf(space, state, u) for u in units])
    #     rho += tick_state.w_hf * hf_values

    # Variance signal (when weight > 0)
    # TODO: implement variance computation per unit
    # if tick_state.w_var > 1e-6:
    #     var_values = np.array([_unit_var(space, state, u) for u in units])
    #     rho += tick_state.w_var * var_values

    return rho


# ===================================================================
# WeightedRhoGate (replaces TwoStageGate)
# ===================================================================

class WeightedRhoGate:
    """Weighted rho gate with dynamic EMA weights.

    Replaces the discrete Stage 1/2 switching with smooth EMA weight transitions:
    - When resid is stable: w_resid ≈ 1.0, others ≈ 0.0
    - When resid is unstable: EMA shifts weights toward combo

    SC-enforce is unaffected because delta = refine_unit() doesn't depend on rho.
    rho determines WHO is refined, not HOW.
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    def evaluate(self, fsr: float, instability: float,
                 tick_state: TickState, tick: int,
                 probe_median_gain: float) -> dict:
        """Update EMA weights based on resid health metrics.

        Returns dict with gate info for logging.
        """
        cfg = self.cfg
        single_tick = (cfg.max_ticks <= 1)

        if single_tick:
            # Backward compat: keep current behavior (resid-only)
            return {
                "stage": "single_tick",
                "w_resid": 1.0, "w_hf": 0.0, "w_var": 0.0,
                "instability": 0.0, "fsr": 0.0,
            }

        # Cold-start thresholds should be set by _cold_start_thresholds before first tick
        instab_thresh = tick_state.calibrated_instab_thresh or cfg.health_instab_threshold
        fsr_thresh = tick_state.calibrated_fsr_thresh or cfg.health_fsr_threshold

        # Pilot fine-tuning: collect stats and refine thresholds
        in_pilot = (tick < cfg.pilot_ticks)
        if in_pilot:
            tick_state.pilot_instab.append(instability)
            tick_state.pilot_fsr.append(fsr)
            # On last pilot tick: refine thresholds
            if tick == cfg.pilot_ticks - 1 and tick_state.pilot_instab:
                observed_instab = sorted(tick_state.pilot_instab)
                observed_fsr = sorted(tick_state.pilot_fsr)
                median_instab = observed_instab[len(observed_instab) // 2]
                median_fsr = observed_fsr[len(observed_fsr) // 2]
                # Fine-tune: blend cold-start with observed
                tick_state.calibrated_instab_thresh = min(
                    instab_thresh,
                    median_instab * cfg.pilot_thresh_factor
                )
                tick_state.calibrated_fsr_thresh = max(
                    cfg.pilot_fsr_floor,
                    min(fsr_thresh, median_fsr * cfg.pilot_thresh_factor)
                )

        # Determine target weights based on health
        resid_healthy = (instability <= instab_thresh and fsr <= fsr_thresh)

        if resid_healthy:
            target_w_resid = 1.0
            target_w_hf = 0.0
            target_w_var = 0.0
            stage = "healthy"
        else:
            # Shift toward combo -- proportional to how unhealthy
            # More instability -> more weight to HF/variance
            instab_excess = max(0, instability - instab_thresh) / max(instab_thresh, 1e-6)
            combo_strength = min(1.0, instab_excess)  # 0..1
            target_w_resid = max(cfg.resid_min_weight, 1.0 - 0.5 * combo_strength)
            target_w_hf = (1.0 - target_w_resid) * 0.6  # 60% of combo to HF
            target_w_var = (1.0 - target_w_resid) * 0.4  # 40% to variance
            stage = "combo"

        # EMA update (smooth transition, no discrete jumps)
        alpha = cfg.ema_weight_alpha
        tick_state.w_resid += alpha * (target_w_resid - tick_state.w_resid)
        tick_state.w_hf += alpha * (target_w_hf - tick_state.w_hf)
        tick_state.w_var += alpha * (target_w_var - tick_state.w_var)

        # Renormalize to sum=1
        total = tick_state.w_resid + tick_state.w_hf + tick_state.w_var
        if total > 1e-10:
            tick_state.w_resid /= total
            tick_state.w_hf /= total
            tick_state.w_var /= total

        return {
            "stage": stage,
            "w_resid": tick_state.w_resid,
            "w_hf": tick_state.w_hf,
            "w_var": tick_state.w_var,
            "instability": instability,
            "fsr": fsr,
            "instab_thresh": instab_thresh,
            "fsr_thresh": fsr_thresh,
            "in_pilot": in_pilot,
        }


# ===================================================================
# GovernorEMA (restored from exp08)
# ===================================================================

class GovernorEMA:
    """EMA-based governor for smooth strictness control.

    Restored from exp08v5_schedule.py. Two-layer architecture:
    - Hardware parameter sets RANGE (leash length)
    - EMA feedback moves WITHIN range based on runtime cost

    Orthogonal to StrictnessTracker (per-unit reputation)
    and WasteBudget (kill switch).

    NOT applicable in streaming mode (cross-cluster bleed).
    """

    def __init__(self, cfg: PipelineConfig, target_per_tick: float):
        self.enabled = cfg.governor_ema_enabled
        self.alpha = cfg.ema_alpha
        self.corridor_hi = target_per_tick * cfg.governor_corridor_hi
        self.corridor_lo = target_per_tick * cfg.governor_corridor_lo
        self.clamp = cfg.governor_strictness_clamp
        self.warmup = cfg.governor_warmup_ticks
        self.ema_cost = target_per_tick  # initialize at target
        self.strictness = 0.5  # start in the middle
        self.tick_count = 0
        self.history = []

    def update(self, cost_this_tick: float) -> float:
        """Update EMA and adjust strictness. Returns new strictness."""
        if not self.enabled:
            return self.strictness

        self.tick_count += 1
        self.ema_cost = self.alpha * cost_this_tick + (1.0 - self.alpha) * self.ema_cost

        if self.tick_count > self.warmup:
            if self.ema_cost > self.corridor_hi:
                self.strictness = min(0.995, self.strictness + self.clamp)
            elif self.ema_cost < self.corridor_lo:
                self.strictness = max(0.5, self.strictness - self.clamp)

        self.history.append({
            "tick": self.tick_count,
            "ema_cost": self.ema_cost,
            "strictness": self.strictness,
        })
        return self.strictness


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
        self.gate = WeightedRhoGate(self.config)

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

        # ============================================================
        # Phase A: One-time setup (unchanged)
        # ============================================================

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

        # 1d. Enox infrastructure init
        enox_any = (cfg.enox_journal_enabled or cfg.enox_dedup_enabled
                    or cfg.enox_sweep_enabled)
        journal = DecisionJournal(enabled=cfg.enox_journal_enabled)
        dedup = MultiStageDedup(
            enabled=cfg.enox_dedup_enabled, epsilon=cfg.enox_dedup_epsilon)
        uri_map: Dict[int, str] = {}
        if enox_any:
            for i, unit in enumerate(units):
                uri_map[unit] = region_uri("root", "init", i)
        enox_tick = 0

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

        # 6. Governor (per-unit telemetry, DET-1 compat)
        governor = GovernorIsolation(alpha=cfg.ema_alpha)

        # ============================================================
        # Phase B: Multi-tick loop
        # ============================================================

        single_tick_mode = (cfg.max_ticks <= 1)

        tick_state = TickState(budget_remaining=n_budget)
        tick_log = []
        convergence_reason = "max_ticks"

        # Cold-start thresholds (skip in single-tick for backward compat)
        if not single_tick_mode:
            initial_rho = np.array([space.unit_rho(state, u) for u in units])
            if topo is not None:
                initial_rho = topo_adjusted_rho(initial_rho, topo)
            _cold_start_thresholds(initial_rho, tick_state, topo_zone, cfg)

        # Governor EMA
        target_per_tick = max(1, n_budget // max(1, cfg.max_ticks))
        gov_ema = GovernorEMA(cfg, target_per_tick)

        # Tracking for backward compat result fields
        last_gate_info = None
        last_ordered = None
        last_probe_set = []

        refined_count = 0
        n_d2_skipped = 0
        n_active_remaining = n_total

        for tick in range(cfg.max_ticks):
            tick_budget = min(target_per_tick, tick_state.budget_remaining)
            if single_tick_mode:
                tick_budget = n_budget  # single-tick uses full budget
            if tick_budget <= 0:
                convergence_reason = "budget_exhausted"
                break

            # B2: Recompute rho
            if single_tick_mode and tick == 0:
                # Backward compat: use original rho computation
                rho_values = np.array([space.unit_rho(state, u) for u in units])
                if topo is not None:
                    rho_values = topo_adjusted_rho(rho_values, topo)
            else:
                rho_values = _compute_weighted_rho(space, state, units, tick_state)
                if topo is not None:
                    rho_values = topo_adjusted_rho(rho_values, topo)

            # B3: FSR + instability
            if single_tick_mode:
                fsr = 0.0
                instability = 0.0
            else:
                fsr = _compute_fsr(units, rho_values, tick_state)
                instability = _compute_instability(rho_values, tick_state, units)

            # B4: Probes
            if single_tick_mode:
                probe_set = DeterministicProbe.select_probe_units(
                    units, cfg.probe_fraction, coords, level=0, global_seed=seed)
            else:
                probe_set = _select_tick_probes(units, coords, tick_state, cfg, seed, tick)
            last_probe_set = probe_set
            probe_rho = {u: space.unit_rho(state, u) for u in probe_set} if probe_set else {}
            median_gain = float(np.median(list(probe_rho.values()))) if probe_rho else 0.0

            # B5: Gate
            gate_info = self.gate.evaluate(fsr, instability, tick_state, tick, median_gain)
            last_gate_info = gate_info

            # B6: Sort
            ordered = CanonicalTraversal.sort_units(units, rho_values, space.name)
            last_ordered = ordered

            # B7: Inner loop (nearly identical to current)
            tick_accepted = 0
            tick_rejected = 0
            tick_damped = 0
            tick_gains = []

            for unit in ordered:
                if tick_accepted >= tick_budget:
                    break
                if cfg.enforce_enabled and waste_exhausted:
                    break
                if unit in tick_state.refined_set:
                    continue  # skip already refined

                # Compression: skip degree-2 transit nodes
                if unit in d2_skip_set:
                    n_d2_skipped += 1
                    if enox_any:
                        journal.append(JournalEntry(
                            uri_map.get(unit, ""), enox_tick, gate_info.get("stage", ""),
                            "skip_d2", {}, {}))
                        enox_tick += 1
                    continue

                # Enox: dedup check (before computing refinement candidate)
                if cfg.enox_dedup_enabled:
                    unit_idx = units.index(unit) if unit in units else 0
                    local_bytes = state.ravel()[unit_idx:unit_idx+1].tobytes()
                    rho_val = float(rho_values[unit_idx]) if unit_idx < len(rho_values) else 0.0
                    dedup_action = dedup.check(
                        uri_map.get(unit, ""), local_bytes, rho_val)
                    if dedup_action != "process":
                        journal.append(JournalEntry(
                            uri_map.get(unit, ""), enox_tick, gate_info.get("stage", ""),
                            dedup_action, {"rho": rho_val}, {}))
                        enox_tick += 1
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

                # ROI check (issue 6) - only after tick 0 in multi-tick mode
                if not single_tick_mode and tick > 0 and tick_state.reference_median_gain > 0:
                    roi = _compute_unit_roi(space, old_state, state_candidate)
                    min_roi = tick_state.reference_median_gain * cfg.min_roi_fraction
                    if roi < min_roi:
                        tick_state.evaluated_set.add(unit)
                        state = old_state  # ensure state not modified
                        continue  # below ROI threshold

                # SC-enforce check
                unit_accepted = False
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
                        unit_accepted = True
                    elif result.action == "damped":
                        state = old_state + result.enforced_delta
                        n_damped += 1
                        tick_damped += 1
                        unit_accepted = True
                    elif result.action == "rejected":
                        state = old_state  # revert
                        n_rejected += 1
                        tick_rejected += 1
                        # Enox journal: rejected
                        if enox_any:
                            journal.append(JournalEntry(
                                uri_map.get(unit, ""), enox_tick, gate_info.get("stage", ""),
                                "rejected",
                                {"rho": float(rho_values[units.index(unit)] if unit in units else 0),
                                 "d_parent": result.d_parent_value,
                                 "damp_iterations": result.damp_iterations},
                                {"tau_effective": result.original_d_parent}))
                            enox_tick += 1
                        s = strictness.get(unit_id)
                        if waste.record_reject(unit_id, s):
                            waste_exhausted = True
                            break
                        tick_state.rejected_set.add(unit)
                        tick_state.evaluated_set.add(unit)
                        continue  # don't count toward budget

                    # Enox journal: pass or damped
                    if enox_any:
                        journal.append(JournalEntry(
                            uri_map.get(unit, ""), enox_tick, gate_info.get("stage", ""),
                            result.action,
                            {"rho": float(rho_values[units.index(unit)] if unit in units else 0),
                             "d_parent": result.d_parent_value,
                             "damp_iterations": result.damp_iterations},
                            {"tau_effective": result.original_d_parent}))
                        enox_tick += 1

                    if result.action in ("damped", "rejected"):
                        strictness.escalate(unit_id)
                else:
                    state = state_candidate
                    n_passed += 1
                    unit_accepted = True
                    # Enox journal: pass (no enforce)
                    if enox_any:
                        journal.append(JournalEntry(
                            uri_map.get(unit, ""), enox_tick, gate_info.get("stage", ""),
                            "pass", {}, {}))
                        enox_tick += 1

                # Enox: update URI after refinement applied
                if enox_any:
                    uri_map[unit] = region_uri(
                        uri_map.get(unit, ""), "refine", enox_tick)

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

                if unit_accepted:
                    tick_state.refined_set.add(unit)
                    tick_state.evaluated_set.add(unit)
                    tick_accepted += 1
                    if not single_tick_mode:
                        if tick > 0 and tick_state.reference_median_gain > 0:
                            tick_gains.append(_compute_unit_roi(space, old_state, state))
                        else:
                            tick_gains.append(float(space.unit_rho(old_state, unit)))

            # B8: Reference gain after tick 0
            if tick == 0 and tick_gains:
                tick_state.reference_median_gain = float(np.median(tick_gains))

            # B9: Governor EMA
            gov_strictness = gov_ema.update(float(tick_accepted))

            # Record tick
            tick_state.accepted_history.append(tick_accepted)
            tick_state.budget_remaining -= tick_accepted

            tick_log.append({
                "tick": tick,
                "accepted": tick_accepted,
                "rejected": tick_rejected,
                "damped": tick_damped,
                "gate": gate_info,
                "roi_median": float(np.median(tick_gains)) if tick_gains else 0.0,
                "governor_strictness": gov_strictness,
            })

            # B10: Convergence check (skip in single-tick mode)
            if not single_tick_mode:
                reason = _check_convergence(tick_state, cfg, len(probe_set))
                if reason:
                    convergence_reason = reason
                    break

            # B11: Strictness decay
            if cfg.enforce_enabled:
                strictness.decay_all()

        n_ticks_executed = len(tick_log)

        # ============================================================
        # Phase C: Post-loop (unchanged)
        # ============================================================

        # 8. Enox: post-step sweep
        sweep = PostStepSweep(
            enabled=cfg.enox_sweep_enabled,
            sibling_threshold=cfg.enox_sweep_threshold)
        sweep_stats = sweep.run(space_type, state, units, n_total)

        # 9. Compute quality metrics
        gt = space.gt
        mse_final = float(np.mean((gt - state) ** 2))
        mse_coarse = float(np.mean((gt - initial_state) ** 2))
        data_range = float(gt.max() - gt.min())

        psnr_final = _mse_to_psnr(mse_final, data_range)
        psnr_coarse = _mse_to_psnr(mse_coarse, data_range)

        reject_rate = n_rejected / max(1, n_passed + n_damped + n_rejected)

        wall_time = time.time() - t0

        # Backward compat: populate gate_stage/gate_weights from last gate eval
        if last_gate_info:
            gate_stage = last_gate_info.get("stage", "single_tick")
            gate_weights = {
                "resid": last_gate_info.get("w_resid", 1.0),
                "hf": last_gate_info.get("w_hf", 0.0),
                "var": last_gate_info.get("w_var", 0.0),
            }
        else:
            gate_stage = "single_tick"
            gate_weights = {"resid": 1.0}

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
            traversal_order=[str(u) for u in (last_ordered or [])],
            probe_set=[str(u) for u in last_probe_set],
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
            enox_journal_summary=journal.summary() if enox_any else {"total": 0, "by_decision": {}},
            enox_dedup_stats=dedup.stats.to_dict() if cfg.enox_dedup_enabled else {},
            enox_sweep_stats=sweep_stats.to_dict() if cfg.enox_sweep_enabled else {},
            enox_uri_map={str(k): v for k, v in uri_map.items()} if cfg.enox_include_uri_map else {},
            wall_time_seconds=wall_time,
            config=cfg.__dict__,
            # Phase 4 fields
            tick_log=tick_log,
            n_ticks_executed=n_ticks_executed,
            convergence_reason=convergence_reason,
            pilot_thresholds={
                "instab_thresh": tick_state.calibrated_instab_thresh,
                "fsr_thresh": tick_state.calibrated_fsr_thresh,
            },
            governor_strictness_history=gov_ema.history,
            final_ema_weights={
                "w_resid": tick_state.w_resid,
                "w_hf": tick_state.w_hf,
                "w_var": tick_state.w_var,
            },
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
        if r.enox_journal_summary.get("total", 0) > 0:
            print(f"    enox journal: {r.enox_journal_summary}")
        if r.enox_dedup_stats:
            print(f"    enox dedup: {r.enox_dedup_stats}")
        if r.enox_sweep_stats:
            print(f"    enox sweep: {r.enox_sweep_stats}")
        # Phase 4 info
        print(f"    ticks: {r.n_ticks_executed}  convergence: {r.convergence_reason}")

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
            "enox_journal_summary": r.enox_journal_summary,
            "enox_dedup_stats": r.enox_dedup_stats,
            "enox_sweep_stats": r.enox_sweep_stats,
            "wall_time_seconds": r.wall_time_seconds,
            "n_ticks_executed": r.n_ticks_executed,
            "convergence_reason": r.convergence_reason,
            "tick_log": r.tick_log,
        }

    # Multi-tick smoke test (all 4 spaces, max_ticks=5)
    print("\n" + "-" * 70)
    print("  Multi-tick smoke test (all spaces, max_ticks=5)")
    print("-" * 70)
    mt_cfg = PipelineConfig(max_ticks=5)
    mt_pipe = CuriosityPipeline(mt_cfg)
    for st in space_types:
        r = mt_pipe.run(st, seed=42, budget_fraction=0.30)
        print(f"\n  [{st}]")
        print(f"    PSNR coarse: {r.coarse_psnr:7.2f}  |  PSNR final: {r.quality_psnr:7.2f}")
        print(f"    n_refined: {r.n_refined:4d} / {r.n_total:4d}")
        print(f"    ticks: {r.n_ticks_executed}  convergence: {r.convergence_reason}")
        print(f"    pilot_thresholds: {r.pilot_thresholds}")
        print(f"    final_ema_weights: {r.final_ema_weights}")
        for tl in r.tick_log:
            print(f"      tick {tl['tick']}: accepted={tl['accepted']} rejected={tl['rejected']} "
                  f"damped={tl['damped']} roi_median={tl['roi_median']:.6f}")
        results_all[f"{st}_mt5"] = {
            "quality_mse": r.quality_mse,
            "quality_psnr": r.quality_psnr,
            "n_refined": r.n_refined,
            "n_total": r.n_total,
            "n_ticks_executed": r.n_ticks_executed,
            "convergence_reason": r.convergence_reason,
            "tick_log": r.tick_log,
            "pilot_thresholds": r.pilot_thresholds,
            "final_ema_weights": r.final_ema_weights,
            "governor_strictness_history": r.governor_strictness_history,
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
