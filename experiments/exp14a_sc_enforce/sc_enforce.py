#!/usr/bin/env python3
"""
Curiosity -- exp14a_sc_enforce: Scale-Consistency Enforcement module.

When the adaptive refinement pipeline computes a delta (refinement) for a unit,
SC-Enforce checks whether the delta leaks low-frequency content into the parent
scale.  If it does (D_parent > tau_parent), the delta is either damped or
rejected outright.

Three-tier response:
  1. D_parent <= tau           -> PASS   (delta applied as-is)
  2. tau < D_parent <= 2*tau   -> DAMP   (project delta onto HF subspace,
                                          repeat up to max_damp_iterations;
                                          if still above tau -> REJECT)
  3. D_parent > 2*tau          -> REJECT (delta discarded, unit skipped)

Strictness-weighted waste budget (Domino Effect protection):
  Each step has a limited waste budget.  Rejected units consume strictness-
  weighted cost.  When the budget is exhausted the step is force-terminated.
  Clean nodes cost ~1 per reject; radioactive hubs with accumulated strictness
  cost 4-5x, burning through the budget instantly.

Local strictness escalation:
  After DAMP/REJECT, the unit's strictness_multiplier grows (effective_tau
  shrinks).  Multiplier decays 0.9x per step without violations.

CPU-only.  Requires numpy.

Roadmap level: SC-enforce (Phase 2)
Depends on: exp12a_tau_parent PASSED (provides per-space per-level thresholds).
"""

import json
import sys
from dataclasses import dataclass, field
from math import floor
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Imports from sc_baseline (add parent to path)
# ---------------------------------------------------------------------------
SC_BASELINE_DIR = Path(__file__).resolve().parent.parent / "sc_baseline"
sys.path.insert(0, str(SC_BASELINE_DIR))


# ═══════════════════════════════════════════════════════════════════════════
# Threshold loader
# ═══════════════════════════════════════════════════════════════════════════

def load_thresholds(json_path: str) -> Dict[Tuple[str, int], float]:
    """Load per-space per-level thresholds from a JSON file.

    Expected JSON format::

        {
          "thresholds": {
            "T1_scalar_L1": 0.4595,
            "T2_vector_L2": 0.2487,
            ...
          }
        }

    Keys are parsed as ``{space_type}_L{level}`` where *space_type* is
    everything before the last ``_L`` token (e.g. ``T1_scalar``) and *level*
    is the integer suffix.

    Args:
        json_path: Path to the threshold JSON file.

    Returns:
        Dictionary mapping ``(space_type, level)`` to the threshold float.

    Raises:
        FileNotFoundError: If *json_path* does not exist.
        KeyError: If the JSON does not contain a ``"thresholds"`` key.
        ValueError: If a key cannot be parsed into (space_type, level).
    """
    path = Path(json_path)
    with open(path, "r") as f:
        data = json.load(f)

    raw = data["thresholds"]
    parsed: Dict[Tuple[str, int], float] = {}
    for key, value in raw.items():
        # Split on the last occurrence of "_L" to get space_type and level
        idx = key.rfind("_L")
        if idx == -1:
            raise ValueError(
                f"Cannot parse threshold key '{key}'; expected format "
                f"'{{space_type}}_L{{level}}'"
            )
        space_type = key[:idx]
        level = int(key[idx + 2:])
        parsed[(space_type, level)] = float(value)

    return parsed


# ═══════════════════════════════════════════════════════════════════════════
# D_parent metric
# ═══════════════════════════════════════════════════════════════════════════

def d_parent_lf_frac(
    delta: np.ndarray,
    R_fn: Callable[[np.ndarray], np.ndarray],
    Up_fn: Callable[[np.ndarray, tuple], np.ndarray],
) -> float:
    """Compute D_parent using the low-frequency fraction variant.

    Measures how much of *delta* lives in the low-frequency (parent-scale)
    subspace by restricting to the coarse grid and prolonging back, then
    taking the ratio of norms.

    .. math::

        D_{\\text{parent}} = \\frac{\\|Up(R(\\delta))\\|}{\\|\\delta\\| + \\epsilon}

    Args:
        delta:  The refinement delta array.
        R_fn:   Restriction operator (fine -> coarse).
        Up_fn:  Prolongation operator (coarse -> fine, needs target shape).

    Returns:
        Scalar D_parent value in [0, 1].
    """
    R_delta = R_fn(delta)
    Up_R_delta = Up_fn(R_delta, delta.shape)
    numer = np.linalg.norm(Up_R_delta.ravel())
    denom = np.linalg.norm(delta.ravel()) + 1e-12
    return float(numer / denom)


# ═══════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EnforceResult:
    """Result of a single SC-enforcement check on a delta.

    Attributes:
        enforced_delta:   The (possibly damped) delta to apply, or None if
                          rejected.
        action:           One of ``'pass'``, ``'damped'``, ``'rejected'``.
        d_parent_value:   Final D_parent after enforcement.
        damp_iterations:  Number of damping iterations applied (0 if pass or
                          immediate reject).
        original_d_parent: D_parent of the original (unmodified) delta.
    """
    enforced_delta: Optional[np.ndarray]
    action: str  # 'pass' | 'damped' | 'rejected'
    d_parent_value: float
    damp_iterations: int
    original_d_parent: float


# ═══════════════════════════════════════════════════════════════════════════
# StrictnessTracker
# ═══════════════════════════════════════════════════════════════════════════

class StrictnessTracker:
    """Per-unit strictness multipliers with escalation and decay.

    After a DAMP or REJECT event the unit's strictness multiplier is
    escalated, making future enforcement tighter (``effective_tau =
    tau / multiplier``).  Multipliers decay toward 1.0 each step that the
    unit does not violate.

    Args:
        escalation_factor: Multiplicative increase on violation (default 1.5).
        decay_factor:      Multiplicative decay per clean step (default 0.9).
                           Applied as ``m = max(1.0, m * decay_factor)``.
    """

    def __init__(
        self,
        escalation_factor: float = 1.5,
        decay_factor: float = 0.9,
    ) -> None:
        self.escalation_factor = escalation_factor
        self.decay_factor = decay_factor
        self._multipliers: Dict[str, float] = {}

    def escalate(self, unit_id: str) -> None:
        """Increase the strictness multiplier for *unit_id*."""
        current = self._multipliers.get(unit_id, 1.0)
        self._multipliers[unit_id] = current * self.escalation_factor

    def get(self, unit_id: str) -> float:
        """Return the current strictness multiplier (>=1.0)."""
        return self._multipliers.get(unit_id, 1.0)

    def decay_all(self) -> None:
        """Decay all multipliers toward 1.0.  Call once per step."""
        to_remove = []
        for uid in self._multipliers:
            new_val = self._multipliers[uid] * self.decay_factor
            if new_val <= 1.0:
                to_remove.append(uid)
            else:
                self._multipliers[uid] = new_val
        for uid in to_remove:
            del self._multipliers[uid]

    def snapshot(self) -> Dict[str, float]:
        """Return a copy of all current multipliers (for diagnostics)."""
        return dict(self._multipliers)


# ═══════════════════════════════════════════════════════════════════════════
# WasteBudget
# ═══════════════════════════════════════════════════════════════════════════

class WasteBudget:
    """Strictness-weighted waste budget for a single refinement step.

    Each rejected unit costs ``strictness_multiplier`` units of budget.
    Clean nodes cost ~1; radioactive hubs with accumulated strictness cost
    4-5x, burning through the budget quickly and forcing early termination.

    The maximum budget per step is ``floor(budget_per_step * omega)``.

    Args:
        budget_per_step: Total unit count (or similar measure) for this step.
        omega:           Fraction of budget allocated to waste tolerance
                         (default 0.2, i.e. 20%).
    """

    def __init__(self, budget_per_step: int, omega: float = 0.2) -> None:
        self.r_max = floor(budget_per_step * omega)
        self.omega = omega
        self._waste: float = 0.0
        self._exhausted: bool = False

    @property
    def waste_current(self) -> float:
        """Current accumulated waste cost."""
        return self._waste

    @property
    def exhausted(self) -> bool:
        """Whether the waste budget has been exceeded."""
        return self._exhausted

    def record_reject(self, unit_id: str, strictness: float) -> bool:
        """Record a rejection and return True if the budget is now exhausted.

        Args:
            unit_id:    Identifier of the rejected unit (for logging).
            strictness: The ``strictness_multiplier`` of the rejected unit.
                        This is the cost charged against the waste budget.

        Returns:
            ``True`` if ``waste_current >= r_max`` after this rejection
            (step should be force-terminated).
        """
        self._waste += strictness
        if self._waste >= self.r_max:
            self._exhausted = True
        return self._exhausted

    def reset_step(self) -> None:
        """Reset the waste counter.  Call at the start of each step."""
        self._waste = 0.0
        self._exhausted = False


# ═══════════════════════════════════════════════════════════════════════════
# SCEnforcer
# ═══════════════════════════════════════════════════════════════════════════

class SCEnforcer:
    """Scale-Consistency enforcer: check and enforce HF-only deltas.

    Given a refinement delta and its coarse context, computes D_parent and
    applies the three-tier response (pass / damp / reject).

    Adaptive threshold for small populations
    -----------------------------------------
    For spaces with few active units (trees, small graphs) the enforcement
    threshold is relaxed by a factor derived from the standard error of
    small samples:

    .. math::

        \\tau_{eff}(N) = \\tau_{base} \\cdot \\left(1 + \\frac{\\beta}{\\sqrt{N}}\\right)

    At small *N* the multiplier is large (2-3x), permitting coarse, high-
    variance splits that are statistically legitimate.  As *N* grows the
    law of large numbers collapses the tolerance to zero and the threshold
    converges asymptotically to ``tau_base``.

    Args:
        tau_parent:          Dict mapping ``(space_type, level) -> float``
                             thresholds, as returned by :func:`load_thresholds`.
        R_fn:                Restriction operator ``(array) -> coarse_array``.
        Up_fn:               Prolongation operator ``(coarse, target_shape) ->
                             fine_array``.
        space_type:          Space type key (e.g. ``"T1_scalar"``).
        damp_factor:         Scaling factor for the LF projection removal
                             during damping (default 0.5).
        max_damp_iterations: Maximum damping iterations before falling back
                             to rejection (default 3).
        n_active:            Current number of active (non-frozen) units.
                             Used for the adaptive threshold relaxation.
                             Default 0 disables the relaxation.
        beta:                Statistical tolerance coefficient for the
                             adaptive threshold.  Default 2.0 gives ~2x
                             relaxation at N=4, ~1.7x at N=8, converging
                             to 1.0x as N -> inf.
    """

    def __init__(
        self,
        tau_parent: Dict[Tuple[str, int], float],
        R_fn: Callable[[np.ndarray], np.ndarray],
        Up_fn: Callable[[np.ndarray, tuple], np.ndarray],
        space_type: str,
        damp_factor: float = 0.5,
        max_damp_iterations: int = 3,
        n_active: int = 0,
        beta: float = 2.0,
    ) -> None:
        self.tau_parent = tau_parent
        self.R_fn = R_fn
        self.Up_fn = Up_fn
        self.space_type = space_type
        self.damp_factor = damp_factor
        self.max_damp_iterations = max_damp_iterations
        self.n_active = n_active
        self.beta = beta

        # Optional strictness tracker — caller may attach one for per-unit
        # threshold tightening.
        self.strictness_tracker: Optional[StrictnessTracker] = None

    # ------------------------------------------------------------------
    # Adaptive threshold multiplier
    # ------------------------------------------------------------------

    def _small_sample_multiplier(self) -> float:
        """Compute tau relaxation factor: 1 + beta / sqrt(N).

        Returns 1.0 (no relaxation) when n_active <= 0 or beta <= 0.
        """
        if self.n_active <= 0 or self.beta <= 0.0:
            return 1.0
        return 1.0 + self.beta / np.sqrt(self.n_active)

    # ------------------------------------------------------------------
    # Threshold lookup
    # ------------------------------------------------------------------

    def get_tau(self, level: int, unit_id: Optional[str] = None) -> float:
        """Look up the enforcement threshold for *level*, adjusted by
        the adaptive small-sample relaxation and per-unit strictness.

        The effective threshold is:

        .. math::

            \\tau_{eff} = \\frac{\\tau_{base} \\cdot (1 + \\beta / \\sqrt{N})}{S_{unit}}

        where *S_unit* is the strictness multiplier (>=1.0, from
        :class:`StrictnessTracker`).  The relaxation inflates the threshold
        for small populations; strictness deflates it for repeat offenders.

        Args:
            level:   Depth level (1, 2, 3, ...).
            unit_id: Optional unit identifier for strictness adjustment.

        Returns:
            Effective tau_parent threshold.

        Raises:
            KeyError: If ``(space_type, level)`` is not in the threshold dict.
        """
        key = (self.space_type, level)
        base_tau = self.tau_parent[key]

        # Adaptive relaxation for small populations
        tau = base_tau * self._small_sample_multiplier()

        # Per-unit strictness tightening
        if unit_id is not None and self.strictness_tracker is not None:
            multiplier = self.strictness_tracker.get(unit_id)
            tau = tau / multiplier

        return tau

    # ------------------------------------------------------------------
    # Core enforcement
    # ------------------------------------------------------------------

    def check_and_enforce(
        self,
        delta: np.ndarray,
        coarse: np.ndarray,
        level: int,
        unit_id: Optional[str] = None,
    ) -> EnforceResult:
        """Check a refinement delta and enforce scale-consistency.

        Args:
            delta:   Refinement delta array (fine-scale).
            coarse:  Coarse representation (currently unused in the lf_frac
                     metric but kept for future metric variants).
            level:   Depth level for threshold lookup.
            unit_id: Optional unit identifier for strictness adjustment.

        Returns:
            :class:`EnforceResult` with the enforcement outcome.
        """
        tau = self.get_tau(level, unit_id=unit_id)
        original_d = d_parent_lf_frac(delta, self.R_fn, self.Up_fn)

        # --- Tier 1: PASS -------------------------------------------------
        if original_d <= tau:
            return EnforceResult(
                enforced_delta=delta,
                action="pass",
                d_parent_value=original_d,
                damp_iterations=0,
                original_d_parent=original_d,
            )

        # --- Tier 3: immediate REJECT --------------------------------------
        if original_d > 2.0 * tau:
            return EnforceResult(
                enforced_delta=None,
                action="rejected",
                d_parent_value=original_d,
                damp_iterations=0,
                original_d_parent=original_d,
            )

        # --- Tier 2: DAMP --------------------------------------------------
        current_delta = delta.copy()
        current_d = original_d
        iters = 0

        for iters in range(1, self.max_damp_iterations + 1):
            # Project out the LF component: delta_new = delta - damp_factor * Up(R(delta))
            R_delta = self.R_fn(current_delta)
            Up_R_delta = self.Up_fn(R_delta, current_delta.shape)
            current_delta = current_delta - self.damp_factor * Up_R_delta

            current_d = d_parent_lf_frac(current_delta, self.R_fn, self.Up_fn)

            if current_d <= tau:
                return EnforceResult(
                    enforced_delta=current_delta,
                    action="damped",
                    d_parent_value=current_d,
                    damp_iterations=iters,
                    original_d_parent=original_d,
                )

        # Damping did not converge — fall back to reject
        return EnforceResult(
            enforced_delta=None,
            action="rejected",
            d_parent_value=current_d,
            damp_iterations=iters,
            original_d_parent=original_d,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: full enforcement step over a batch of units
# ═══════════════════════════════════════════════════════════════════════════

def enforce_step(
    enforcer: SCEnforcer,
    deltas: Dict[str, np.ndarray],
    coarses: Dict[str, np.ndarray],
    levels: Dict[str, int],
    waste_budget: WasteBudget,
    strictness_tracker: StrictnessTracker,
) -> Dict[str, EnforceResult]:
    """Run SC-enforcement over a batch of units for one refinement step.

    This is the main entry-point for the adaptive refinement pipeline.
    It iterates over all units, applies :meth:`SCEnforcer.check_and_enforce`,
    updates the waste budget, escalates strictness on violations, and
    force-terminates early if the waste budget is exhausted.

    Args:
        enforcer:           Configured :class:`SCEnforcer` instance (should
                            have ``strictness_tracker`` set to *strictness_tracker*).
        deltas:             Mapping ``unit_id -> delta array``.
        coarses:            Mapping ``unit_id -> coarse array``.
        levels:             Mapping ``unit_id -> depth level``.
        waste_budget:       :class:`WasteBudget` for this step (call
                            :meth:`WasteBudget.reset_step` before invoking).
        strictness_tracker: :class:`StrictnessTracker` for per-unit
                            multiplier management.

    Returns:
        Dict mapping ``unit_id -> EnforceResult`` for all processed units.
        Units not reached due to early termination are absent from the dict.
    """
    # Wire up strictness tracker so get_tau picks it up
    enforcer.strictness_tracker = strictness_tracker

    results: Dict[str, EnforceResult] = {}

    for unit_id in deltas:
        if waste_budget.exhausted:
            break

        delta = deltas[unit_id]
        coarse = coarses[unit_id]
        level = levels[unit_id]

        result = enforcer.check_and_enforce(delta, coarse, level, unit_id=unit_id)
        results[unit_id] = result

        if result.action in ("damped", "rejected"):
            strictness_tracker.escalate(unit_id)

        if result.action == "rejected":
            s = strictness_tracker.get(unit_id)
            waste_budget.record_reject(unit_id, s)

    return results
