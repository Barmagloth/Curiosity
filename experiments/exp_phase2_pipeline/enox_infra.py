"""
Curiosity -- Enox Infrastructure Patterns.

Four infrastructure patterns adapted from Enox (github.com/Disentinel/enox):

1. RegionURI      -- deterministic content-addressed ID for refinement units
2. DecisionJournal -- append-only structured log of every pipeline decision
3. MultiStageDedup -- 3-level dedup (exact hash / metric distance / policy rule)
4. PostStepSweep   -- bush dedup + alias collapse after refinement loop

All patterns are pure observation/annotation -- they never modify pipeline state
or control flow (except dedup skip, which only fires when state is provably
unchanged). This guarantees PSNR-identical output and DET-1 preservation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════════
# 1. RegionURI -- deterministic content-addressed ID
# ═══════════════════════════════════════════════════════════════════════

def region_uri(parent_id: str, op_type: str, child_idx: int) -> str:
    """SHA256(parent_id|op_type|child_idx), truncated to 16 hex chars.

    Parameters
    ----------
    parent_id : str
        URI of the parent (or "root" for top-level units).
    op_type : str
        Operation type: "init", "refine", "probe", "skip_d2", "reject".
    child_idx : int
        Position index in traversal or refinement order.

    Returns
    -------
    str
        16-char hex string, deterministic for same inputs.
    """
    raw = f"{parent_id}|{op_type}|{child_idx}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# ═══════════════════════════════════════════════════════════════════════
# 2. DecisionJournal -- append-only structured log
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class JournalEntry:
    """One decision record in the journal."""
    region_id: str          # RegionURI
    tick: int               # refinement step counter
    gate_stage: str         # "stage1_healthy" / "stage2_utility"
    decision: str           # "pass"/"damped"/"rejected"/"skip_d2"/"skip_budget"/"skip_dedup"
    metrics_snapshot: dict  # {rho, d_parent, damp_iterations, governor_ema}
    thresholds_used: dict   # {tau_effective, waste_omega, ...}


class DecisionJournal:
    """Append-only log of pipeline decisions.

    When disabled, append() is a no-op (zero overhead path).
    """

    def __init__(self, enabled: bool = True):
        self._entries: List[JournalEntry] = []
        self._enabled = enabled

    def append(self, entry: JournalEntry) -> None:
        """Record a decision. No-op if disabled."""
        if self._enabled:
            self._entries.append(entry)

    def entries(self) -> List[dict]:
        """Return all entries as a list of dicts (serializable)."""
        return [asdict(e) for e in self._entries]

    def summary(self) -> dict:
        """Aggregate summary: total count and breakdown by decision type."""
        by_decision: Dict[str, int] = {}
        for e in self._entries:
            by_decision[e.decision] = by_decision.get(e.decision, 0) + 1
        return {
            "total": len(self._entries),
            "by_decision": by_decision,
        }

    def __len__(self) -> int:
        return len(self._entries)


# ═══════════════════════════════════════════════════════════════════════
# 3. MultiStageDedup -- 3-level dedup tracker
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DedupStats:
    """Statistics for multi-stage dedup."""
    n_checked: int = 0
    n_exact_hash_skips: int = 0
    n_metric_distance_skips: int = 0
    n_policy_skips: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class MultiStageDedup:
    """Three-level deduplication tracker.

    Level (a): Exact hash -- skip if unit's local state hash is unchanged.
    Level (b): Metric distance -- skip if rho change < epsilon.
    Level (c): Policy rule -- caller signals skip via policy_skip flag.

    For a single-pass pipeline with epsilon=0.0, levels (a) and (b)
    never fire. This is intentional scaffolding for future multi-pass
    or iterative refinement.

    Parameters
    ----------
    enabled : bool
        If False, check() always returns 'process'.
    epsilon : float
        Threshold for level (b) metric distance skip.
        Default 0.0 = only skip on exact rho match.
    """

    def __init__(self, enabled: bool = True, epsilon: float = 0.0):
        self._enabled = enabled
        self._epsilon = epsilon
        self._state_hashes: Dict[str, str] = {}  # uri -> sha256 hex
        self._prev_rho: Dict[str, float] = {}     # uri -> last rho
        self.stats = DedupStats()

    def check(self, uri: str, state_bytes: bytes, rho: float,
              policy_skip: bool = False) -> str:
        """Check if this unit should be skipped.

        Parameters
        ----------
        uri : str
            RegionURI of the unit.
        state_bytes : bytes
            Local state slice as bytes (for hash comparison).
        rho : float
            Current rho value for the unit.
        policy_skip : bool
            If True, skip due to external policy rule.

        Returns
        -------
        str
            'process' | 'skip_exact' | 'skip_metric' | 'skip_policy'
        """
        if not self._enabled:
            return "process"

        self.stats.n_checked += 1

        # Level (c): policy rule (checked first -- cheapest)
        if policy_skip:
            self.stats.n_policy_skips += 1
            return "skip_policy"

        # Level (a): exact hash match
        new_hash = hashlib.sha256(state_bytes).hexdigest()[:16]
        if uri in self._state_hashes and self._state_hashes[uri] == new_hash:
            self.stats.n_exact_hash_skips += 1
            return "skip_exact"
        self._state_hashes[uri] = new_hash

        # Level (b): metric distance
        if uri in self._prev_rho:
            if abs(rho - self._prev_rho[uri]) <= self._epsilon:
                self.stats.n_metric_distance_skips += 1
                return "skip_metric"
        self._prev_rho[uri] = rho

        return "process"


# ═══════════════════════════════════════════════════════════════════════
# 4. PostStepSweep -- bush dedup + alias collapse
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SweepStats:
    """Statistics for post-step sweep."""
    triggered: bool = False
    n_siblings_checked: int = 0
    n_dirty_sig_groups: int = 0
    n_merge_candidates: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class PostStepSweep:
    """Post-step sweep: detect sibling groups with identical dirty signatures.

    Condition-triggered: only runs if >threshold fraction of siblings share
    the same dirty signature. Currently implemented for tree_hierarchy only
    (explicit parent-child structure).

    This is a pure observation pass -- it identifies merge candidates
    but does NOT modify state. Downstream consumers can use the results.

    Parameters
    ----------
    enabled : bool
        If False, run() returns empty stats immediately.
    sibling_threshold : float
        Fraction of siblings that must share a signature to trigger.
        Default 0.05 (5%).
    """

    def __init__(self, enabled: bool = True, sibling_threshold: float = 0.05):
        self._enabled = enabled
        self._threshold = sibling_threshold
        self.stats = SweepStats()

    def _unit_sig(self, state, unit, n_total: int) -> str:
        """Compute dirty signature hash for a unit's local state."""
        if hasattr(unit, '__len__'):
            local = state[unit]
        elif isinstance(unit, (int,)) and unit < len(state):
            local = state[unit:unit+1]
        else:
            local = state.ravel()[unit:unit+1]
        return hashlib.sha256(local.tobytes()).hexdigest()[:12]

    def run(self, space_type: str, state, units: list,
            n_total: int) -> SweepStats:
        """Run post-step sweep.

        Parameters
        ----------
        space_type : str
            Space type name.
        state : np.ndarray
            Final state after refinement.
        units : list
            Ordered list of unit identifiers.
        n_total : int
            Total number of units.

        Returns
        -------
        SweepStats
        """
        if not self._enabled:
            return self.stats

        # Currently only tree_hierarchy has explicit sibling structure
        if space_type != "tree_hierarchy":
            return self.stats

        # Build sibling groups: nodes sharing the same parent
        sibling_groups: Dict[int, List[int]] = {}
        for u in range(n_total):
            parent = (u - 1) // 2 if u > 0 else -1
            if parent not in sibling_groups:
                sibling_groups[parent] = []
            sibling_groups[parent].append(u)

        # Compute dirty signatures
        sigs: Dict[int, str] = {}
        for u in range(n_total):
            sigs[u] = self._unit_sig(state, u, n_total)

        # Check sibling groups for shared signatures
        total_siblings = 0
        shared_siblings = 0
        merge_candidates = 0
        dirty_sig_groups = 0

        for parent, children in sibling_groups.items():
            if len(children) < 2:
                continue
            total_siblings += len(children)

            # Group children by signature
            sig_groups: Dict[str, List[int]] = {}
            for c in children:
                s = sigs[c]
                if s not in sig_groups:
                    sig_groups[s] = []
                sig_groups[s].append(c)

            for sig, group in sig_groups.items():
                if len(group) > 1:
                    shared_siblings += len(group)
                    dirty_sig_groups += 1
                    merge_candidates += len(group) - 1  # all but one

        # Check threshold
        fraction = shared_siblings / max(total_siblings, 1)
        triggered = fraction > self._threshold

        self.stats = SweepStats(
            triggered=triggered,
            n_siblings_checked=total_siblings,
            n_dirty_sig_groups=dirty_sig_groups,
            n_merge_candidates=merge_candidates if triggered else 0,
        )
        return self.stats
