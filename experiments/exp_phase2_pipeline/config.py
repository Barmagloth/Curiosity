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

    # Diagnostics
    track_seam_score: bool = True
    reject_rate_alert: float = 0.20  # flag if > 20% rejects
