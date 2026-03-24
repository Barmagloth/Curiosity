"""
Curiosity -- Phase 2 Pipeline Configuration.

All tuneable knobs for the unified pipeline live here as a single dataclass.
Defaults correspond to the values validated across P2a experiments.
"""

from dataclasses import dataclass


@dataclass
class PipelineConfig:
    # Budget
    budget_fraction: float = 0.30
    probe_fraction: float = 0.10

    # Gate thresholds (from P2a: ridge 100%, manual ok)
    health_fsr_threshold: float = 0.20
    health_instab_threshold: float = 0.25
    healthy_resid_weight: float = 0.95
    healthy_other_weight: float = 0.025
    lambda_fsr: float = 2.0
    mu_instab: float = 1.0
    resid_min_weight: float = 0.20
    other_max_weight: float = 0.55
    temperature: float = 0.2

    # Governor
    ema_alpha: float = 0.1

    # Halo
    halo_width: int = 2       # grid spaces overlap width
    halo_hops: int = 1        # graph spaces hop count
    # Tree: halo disabled (boundary parallelism < 3)

    # SC-enforce
    enforce_enabled: bool = True
    damp_factor: float = 0.5
    max_damp_iterations: int = 3
    waste_omega: float = 0.2    # waste budget = 20% of step budget
    strictness_escalation: float = 1.5
    strictness_decay: float = 0.9

    # Topological profiling (irregular_graph only)
    topo_profiling_enabled: bool = True
    topo_budget_ms: float = 50.0        # budget for curvature computation
    tau_zone_factor_green: float = 1.3   # relax tau_eff for GREEN graphs
    tau_zone_factor_yellow: float = 1.0  # standard tau_eff for YELLOW
    tau_zone_factor_red: float = 0.7     # tighten tau_eff for RED graphs

    # Enox infrastructure patterns (all default OFF for backward compat)
    enox_journal_enabled: bool = False      # Decision journal
    enox_dedup_enabled: bool = False        # Multi-stage dedup
    enox_dedup_epsilon: float = 0.0         # Metric distance threshold (0.0 = exact only)
    enox_sweep_enabled: bool = False        # Post-step sweep
    enox_sweep_threshold: float = 0.05      # Sibling dirty-sig overlap trigger (5%)
    enox_include_uri_map: bool = False      # Include full URI map in result (debug)

    # Diagnostics
    track_seam_score: bool = True
    reject_rate_alert: float = 0.20  # flag if > 20% rejects

    # === Phase 4: Multi-tick ===
    max_ticks: int = 1                       # 1 = backward compat single-pass

    # Issue 1: Convergence detector
    convergence_window: int = 2              # UNVALIDATED, needs sweep (exp19)

    # Issues 2+5: Cold-start + pilot fine-tuning
    pilot_ticks: int = 3                     # UNVALIDATED, needs sweep (exp19)
    pilot_thresh_factor: float = 0.7         # UNVALIDATED, needs sweep (exp19)
    pilot_fsr_floor: float = 0.05            # UNVALIDATED, needs sweep (exp19)

    # Issue 4: EMA rate for rho signal weights
    ema_weight_alpha: float = 0.3            # UNVALIDATED, needs sweep (exp19)

    # Issue 6: ROI threshold
    min_roi_fraction: float = 0.15           # UNVALIDATED, needs sweep (exp19)

    # Governor EMA (restored from exp08)
    governor_ema_enabled: bool = False        # default=False for backward compat
    governor_corridor_hi: float = 1.5         # from exp08 (TARGET * 1.5)
    governor_corridor_lo: float = 0.5         # from exp08 (TARGET * 0.5)
    governor_strictness_clamp: float = 0.05   # from exp08
    governor_warmup_ticks: int = 3            # from exp08 (WARMUP_STEPS=3)
