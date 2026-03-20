#!/usr/bin/env python3
"""
Curiosity -- exp13 segment_compress: Segment Compression module (P1-B1).

Compresses degree-2 chains in adaptive refinement trees by merging
consecutive nodes whose dirty signatures are stable.  This reduces
per-step bookkeeping from O(N) to O(segments) for pass-through nodes.

Dirty Signature layout (12 bits total, from exp11):
  bits [11:8]  seam_risk  (4 bits, 0-15)
  bits  [7:4]  uncert     (4 bits, 0-15)
  bits  [3:0]  mass       (4 bits, 0-15)

Segment definition:
  A segment is a sequence of nodes [n0, n1, ..., nk] where:
  - Each ni has degree 2 (one parent, one child) in the tree
  - Dirty signatures are stable: Hamming distance between current and
    baseline < 3 bits AND no single component change >= 4 levels,
    for at least `stability_window` consecutive steps
  - Length <= max_segment_length (default 8)

Classes:
  Segment                  -- dataclass for a compressed segment
  RefinementTree           -- tree structure analysis (degree-2 chains)
  SignatureStabilityChecker -- tracks per-node signature stability
  SegmentTree              -- compression engine (merge / split / update)

Helper functions:
  hamming12, component_diff, unpack_signature
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# ═══════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════

def unpack_signature(sig: int) -> Tuple[int, int, int]:
    """Unpack a 12-bit dirty signature into its three 4-bit components.

    Parameters
    ----------
    sig : int
        12-bit packed signature.

    Returns
    -------
    (seam_risk, uncert, mass) : tuple of int
        Each in [0, 15].
    """
    seam = (sig >> 8) & 0xF
    uncert = (sig >> 4) & 0xF
    mass = sig & 0xF
    return seam, uncert, mass


def hamming12(a: int, b: int) -> int:
    """Hamming distance between two 12-bit integers.

    Counts the number of bit positions where a and b differ.

    Parameters
    ----------
    a, b : int
        12-bit packed signatures.

    Returns
    -------
    int
        Number of differing bits (0-12).
    """
    xor = (a ^ b) & 0xFFF
    return bin(xor).count("1")


def component_diff(a: int, b: int) -> int:
    """Maximum component-wise difference between two 12-bit signatures.

    Unpacks both signatures and returns the largest absolute difference
    among the three 4-bit fields (seam_risk, uncert, mass).

    Parameters
    ----------
    a, b : int
        12-bit packed signatures.

    Returns
    -------
    int
        Max absolute difference across components (0-15).
    """
    sa, ua, ma = unpack_signature(a)
    sb, ub, mb = unpack_signature(b)
    return max(abs(sa - sb), abs(ua - ub), abs(ma - mb))


# ═══════════════════════════════════════════════════════════════════════
# Segment dataclass
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Segment:
    """A compressed segment of degree-2 nodes.

    Attributes
    ----------
    node_ids : list of int
        Ordered list of node IDs in this segment.
    representative_signature : int
        12-bit signature representing the segment (from the first node
        at creation time, or majority vote on merge).
    creation_step : int
        Simulation step when this segment was first formed.
    last_update_step : int
        Most recent step when the segment was modified.
    """
    node_ids: List[int]
    representative_signature: int
    creation_step: int
    last_update_step: int

    def __len__(self) -> int:
        return len(self.node_ids)

    def contains(self, node_id: int) -> bool:
        """Check if a node belongs to this segment."""
        return node_id in self.node_ids


# ═══════════════════════════════════════════════════════════════════════
# RefinementTree
# ═══════════════════════════════════════════════════════════════════════

class RefinementTree:
    """Represents a tree structure for compression analysis.

    Analyses the tree topology to find degree-2 nodes (exactly one parent
    and one child) which are candidates for segment compression.

    Parameters
    ----------
    parent : dict
        Mapping node_id -> parent_node_id.  Root has no entry or maps
        to None.
    children : dict
        Mapping node_id -> list of child_node_ids.  Leaves map to [].
    """

    def __init__(self, parent: Dict[int, Optional[int]],
                 children: Dict[int, List[int]]):
        self.parent = dict(parent)
        self.children = {k: list(v) for k, v in children.items()}
        # Collect all node IDs
        self._nodes: Set[int] = set(self.parent.keys()) | set(self.children.keys())
        for clist in self.children.values():
            self._nodes.update(clist)

    @property
    def nodes(self) -> Set[int]:
        return self._nodes

    def degree(self, node: int) -> int:
        """Number of connections for a node (parent edges + child edges).

        The root has no parent edge; leaves have no child edges.

        Parameters
        ----------
        node : int
            Node ID.

        Returns
        -------
        int
            Total degree (parent count + children count).
        """
        has_parent = 1 if (node in self.parent and
                           self.parent[node] is not None) else 0
        n_children = len(self.children.get(node, []))
        return has_parent + n_children

    def find_degree2_chains(self) -> List[List[int]]:
        """Find all maximal chains of consecutive degree-2 nodes.

        A chain is a maximal connected sequence of nodes where every
        node has degree exactly 2.  Chains are ordered parent-to-child.

        Returns
        -------
        list of list of int
            Each inner list is one maximal chain of degree-2 node IDs.
        """
        degree2_set = {n for n in self._nodes if self.degree(n) == 2}
        if not degree2_set:
            return []

        visited: Set[int] = set()
        chains: List[List[int]] = []

        for node in sorted(degree2_set):
            if node in visited:
                continue

            # Walk up to the start of the chain (towards root)
            current = node
            while True:
                p = self.parent.get(current)
                if p is not None and p in degree2_set and p not in visited:
                    current = p
                else:
                    break

            # Walk down the chain collecting nodes
            chain = []
            while current in degree2_set and current not in visited:
                chain.append(current)
                visited.add(current)
                kids = self.children.get(current, [])
                if len(kids) == 1 and kids[0] in degree2_set and kids[0] not in visited:
                    current = kids[0]
                else:
                    break

            if chain:
                chains.append(chain)

        return chains

    def chain_statistics(self) -> dict:
        """Compute statistics about degree-2 chains in the tree.

        Returns
        -------
        dict with keys:
            total_nodes     : int   -- total nodes in tree
            degree2_nodes   : int   -- nodes with degree exactly 2
            num_chains      : int   -- number of maximal chains
            chain_lengths   : list  -- length of each chain
            max_chain       : int   -- longest chain
            mean_chain      : float -- mean chain length
            fraction_degree2: float -- degree2_nodes / total_nodes
        """
        total = len(self._nodes)
        d2_count = sum(1 for n in self._nodes if self.degree(n) == 2)
        chains = self.find_degree2_chains()
        lengths = [len(c) for c in chains]

        return {
            "total_nodes": total,
            "degree2_nodes": d2_count,
            "num_chains": len(chains),
            "chain_lengths": lengths,
            "max_chain": max(lengths) if lengths else 0,
            "mean_chain": float(np.mean(lengths)) if lengths else 0.0,
            "fraction_degree2": d2_count / max(total, 1),
        }


# ═══════════════════════════════════════════════════════════════════════
# SignatureStabilityChecker
# ═══════════════════════════════════════════════════════════════════════

class SignatureStabilityChecker:
    """Tracks per-node dirty signature stability over time.

    A node is considered stable if its signature has remained within
    thresholds for `stability_window` consecutive steps, compared to
    a rolling baseline (the signature at the start of the current
    stability window).

    Parameters
    ----------
    hamming_threshold : int
        Maximum allowed Hamming distance from baseline (default 3).
    component_threshold : int
        Maximum allowed single-component change from baseline (default 4).
    stability_window : int
        Number of consecutive stable steps required (default 3).
    """

    def __init__(self, hamming_threshold: int = 3,
                 component_threshold: int = 4,
                 stability_window: int = 3):
        self.hamming_threshold = hamming_threshold
        self.component_threshold = component_threshold
        self.stability_window = stability_window

        # Per-node tracking
        self._baseline: Dict[int, int] = {}       # node -> baseline signature
        self._stable_count: Dict[int, int] = {}    # node -> consecutive stable steps
        self._last_sig: Dict[int, int] = {}        # node -> most recent signature
        self._last_step: Dict[int, int] = {}       # node -> most recent step

    def _is_within_threshold(self, sig_a: int, sig_b: int) -> bool:
        """Check if two signatures are within stability thresholds."""
        hd = hamming12(sig_a, sig_b)
        cd = component_diff(sig_a, sig_b)
        return hd < self.hamming_threshold and cd < self.component_threshold

    def record(self, node_id: int, signature: int, step: int) -> None:
        """Record a signature observation for a node at a given step.

        Updates the internal stability counter.  If the signature is
        within thresholds of the baseline, the counter increments;
        otherwise it resets and the baseline updates.

        Parameters
        ----------
        node_id : int
            Node identifier.
        signature : int
            12-bit packed dirty signature.
        step : int
            Current simulation step.
        """
        if node_id not in self._baseline:
            # First observation: set baseline
            self._baseline[node_id] = signature
            self._stable_count[node_id] = 1
            self._last_sig[node_id] = signature
            self._last_step[node_id] = step
            return

        if self._is_within_threshold(signature, self._baseline[node_id]):
            self._stable_count[node_id] += 1
        else:
            # Signature drifted -- reset baseline to current
            self._baseline[node_id] = signature
            self._stable_count[node_id] = 1

        self._last_sig[node_id] = signature
        self._last_step[node_id] = step

    def is_stable(self, node_id: int) -> bool:
        """Check if a node has been stable for the required window.

        Parameters
        ----------
        node_id : int
            Node identifier.

        Returns
        -------
        bool
            True if stable for >= stability_window consecutive steps.
        """
        return self._stable_count.get(node_id, 0) >= self.stability_window

    def reset(self, node_id: int) -> None:
        """Reset the stability counter for a node.

        Call this when a segment containing the node is split.

        Parameters
        ----------
        node_id : int
            Node identifier.
        """
        self._stable_count.pop(node_id, None)
        self._baseline.pop(node_id, None)
        self._last_sig.pop(node_id, None)
        self._last_step.pop(node_id, None)

    def get_baseline(self, node_id: int) -> Optional[int]:
        """Return the current baseline signature for a node, or None."""
        return self._baseline.get(node_id)

    def get_stable_count(self, node_id: int) -> int:
        """Return the current consecutive-stable count."""
        return self._stable_count.get(node_id, 0)


# ═══════════════════════════════════════════════════════════════════════
# SegmentTree — compression engine
# ═══════════════════════════════════════════════════════════════════════

# ── Breakeven threshold from bench_breakeven.py profiling ──────────
# N_critical = ceil(C_init / (C_refine * (1 - 1/L_avg) - C_track))
# With measured constants: C_refine=15900ns, C_track=1900ns,
# C_init=100500ns, L_avg=3 → N_critical = 12.
N_CRITICAL_D2 = 12


def should_compress(
    n_active: int,
    budget_step: int,
    n_stable_d2: int,
) -> Tuple[bool, str]:
    """Profiler-calibrated viability check for segment compression.

    Two guards, evaluated in order:

    1. **Structural (bombardment density):** If ``budget_step >= 0.5 *
       n_active``, degree-2 chains cannot survive carpet-bombing
       refinement — skip compression entirely.
    2. **Breakeven (N_critical):** If ``n_stable_d2 < N_CRITICAL_D2``,
       the fixed cost of instantiating and maintaining the compression
       apparatus exceeds the savings from skipping transit nodes.
       ``N_CRITICAL_D2`` is derived from empirical profiling::

           N_critical = ceil(C_init / (C_refine*(1 - 1/L_avg) - C_track))

       With measured C_refine=15.9us, C_track=1.9us, C_init=100.5us,
       L_avg=3 → **N_critical = 12**.

    All inputs come from metrics the pipeline already computes — no new
    configuration parameters.

    Parameters
    ----------
    n_active : int
        Total active (non-frozen) nodes in the tree this step.
    budget_step : int
        Number of units the governor plans to refine this step.
    n_stable_d2 : int
        Number of degree-2 nodes currently passing the stability check.

    Returns
    -------
    (viable, reason) : tuple of (bool, str)
        ``viable`` is True if compression should proceed.
        ``reason`` is a short diagnostic string (empty when viable).
    """
    # Guard 1: structural — high budget saturates the tree
    if n_active > 0 and budget_step >= 0.5 * n_active:
        return False, "bombardment_density"

    # Guard 2: breakeven — not enough material to recoup C_init
    if n_stable_d2 < N_CRITICAL_D2:
        return False, "below_n_critical"

    return True, ""


class SegmentTree:
    """Compression engine for degree-2 chains in a refinement tree.

    Maintains a collection of Segment objects, merging adjacent stable
    degree-2 nodes and splitting segments when nodes become unstable.

    Parameters
    ----------
    tree : RefinementTree
        The underlying tree structure.
    max_length : int
        Maximum number of nodes per segment (default 8).
    stability_window : int
        Consecutive stable steps required before merging (default 3).
    """

    def __init__(self, tree: RefinementTree, max_length: int = 8,
                 stability_window: int = 3):
        self.tree = tree
        self.max_length = max_length
        self.stability_window = stability_window

        self.checker = SignatureStabilityChecker(
            hamming_threshold=4,
            component_threshold=4,
            stability_window=stability_window,
        )

        # Active segments: list of Segment objects
        self.segments: List[Segment] = []

        # Lookup: node_id -> index into self.segments (or None)
        self._node_to_seg: Dict[int, int] = {}

        # Degree-2 chains (cached)
        self._chains: List[List[int]] = tree.find_degree2_chains()
        self._degree2_nodes: Set[int] = set()
        for chain in self._chains:
            self._degree2_nodes.update(chain)

        # Event log for analysis
        self.merge_events: List[dict] = []
        self.split_events: List[dict] = []

    def _rebuild_lookup(self) -> None:
        """Rebuild the node-to-segment index."""
        self._node_to_seg.clear()
        for idx, seg in enumerate(self.segments):
            for nid in seg.node_ids:
                self._node_to_seg[nid] = idx

    def try_merge(self, node_a: int, node_b: int,
                  signatures: Dict[int, int], step: int) -> bool:
        """Check if two adjacent degree-2 nodes can merge into a segment.

        Both nodes must be degree-2, stable, and not already in the same
        segment.  If they are in different segments, the combined length
        must not exceed max_length.

        Parameters
        ----------
        node_a, node_b : int
            Adjacent node IDs (parent-child relationship).
        signatures : dict
            Current signatures: node_id -> 12-bit int.
        step : int
            Current simulation step.

        Returns
        -------
        bool
            True if the merge was performed.
        """
        # Both must be degree-2
        if node_a not in self._degree2_nodes or node_b not in self._degree2_nodes:
            return False

        # Both must be stable
        if not self.checker.is_stable(node_a) or not self.checker.is_stable(node_b):
            return False

        seg_idx_a = self._node_to_seg.get(node_a)
        seg_idx_b = self._node_to_seg.get(node_b)

        # Already in the same segment
        if seg_idx_a is not None and seg_idx_a == seg_idx_b:
            return False

        # Determine the combined node list
        if seg_idx_a is not None and seg_idx_b is not None:
            seg_a = self.segments[seg_idx_a]
            seg_b = self.segments[seg_idx_b]
            combined = seg_a.node_ids + seg_b.node_ids
            if len(combined) > self.max_length:
                return False
            # Merge b into a
            rep_sig = signatures.get(combined[0], seg_a.representative_signature)
            new_seg = Segment(
                node_ids=combined,
                representative_signature=rep_sig,
                creation_step=min(seg_a.creation_step, seg_b.creation_step),
                last_update_step=step,
            )
            # Remove both old segments (remove higher index first)
            idxs = sorted([seg_idx_a, seg_idx_b], reverse=True)
            for i in idxs:
                self.segments.pop(i)
            self.segments.append(new_seg)

        elif seg_idx_a is not None:
            seg_a = self.segments[seg_idx_a]
            if len(seg_a) + 1 > self.max_length:
                return False
            seg_a.node_ids.append(node_b)
            seg_a.last_update_step = step

        elif seg_idx_b is not None:
            seg_b = self.segments[seg_idx_b]
            if len(seg_b) + 1 > self.max_length:
                return False
            seg_b.node_ids.insert(0, node_a)
            seg_b.last_update_step = step

        else:
            # Neither is in a segment -- create new
            rep_sig = signatures.get(node_a, 0)
            new_seg = Segment(
                node_ids=[node_a, node_b],
                representative_signature=rep_sig,
                creation_step=step,
                last_update_step=step,
            )
            self.segments.append(new_seg)

        self._rebuild_lookup()
        self.merge_events.append({
            "step": step,
            "nodes": [node_a, node_b],
            "segment_size": len(self.segments[-1]) if seg_idx_a is None and seg_idx_b is None else None,
        })
        return True

    def split_segment(self, segment: Segment,
                      node_id: int) -> Tuple[Optional[Segment], Optional[Segment]]:
        """Split a segment at a node that has become unstable.

        The unstable node is removed from the segment, producing up to
        two sub-segments (before and after the unstable node).

        Parameters
        ----------
        segment : Segment
            The segment to split.
        node_id : int
            The unstable node ID (must be in the segment).

        Returns
        -------
        (left, right) : tuple of Optional[Segment]
            The two resulting sub-segments.  Either may be None if the
            split leaves fewer than 2 nodes on that side.  A single-node
            "segment" is still returned for bookkeeping (it may grow
            later).
        """
        if node_id not in segment.node_ids:
            raise ValueError(f"Node {node_id} not in segment {segment.node_ids}")

        idx = segment.node_ids.index(node_id)
        left_ids = segment.node_ids[:idx]
        right_ids = segment.node_ids[idx + 1:]

        step = segment.last_update_step

        left_seg = None
        if left_ids:
            left_seg = Segment(
                node_ids=left_ids,
                representative_signature=segment.representative_signature,
                creation_step=segment.creation_step,
                last_update_step=step,
            )

        right_seg = None
        if right_ids:
            right_seg = Segment(
                node_ids=right_ids,
                representative_signature=segment.representative_signature,
                creation_step=segment.creation_step,
                last_update_step=step,
            )

        return left_seg, right_seg

    def build_segments(self, signatures: Dict[int, int],
                       step: int) -> List[Segment]:
        """Scan all degree-2 chains and build initial segments.

        Groups consecutive stable nodes within each chain into segments
        respecting max_length.  Called once at initialization or for a
        full rebuild.

        Parameters
        ----------
        signatures : dict
            Current signatures: node_id -> 12-bit int.
        step : int
            Current simulation step.

        Returns
        -------
        list of Segment
            Newly created segments.
        """
        self.segments.clear()
        self._node_to_seg.clear()

        for chain in self._chains:
            current_run: List[int] = []
            for nid in chain:
                if self.checker.is_stable(nid):
                    current_run.append(nid)
                    if len(current_run) >= self.max_length:
                        # Flush the current run as a segment
                        rep = signatures.get(current_run[0], 0)
                        seg = Segment(
                            node_ids=list(current_run),
                            representative_signature=rep,
                            creation_step=step,
                            last_update_step=step,
                        )
                        self.segments.append(seg)
                        current_run.clear()
                else:
                    # Flush any accumulated stable run
                    if len(current_run) >= 2:
                        rep = signatures.get(current_run[0], 0)
                        seg = Segment(
                            node_ids=list(current_run),
                            representative_signature=rep,
                            creation_step=step,
                            last_update_step=step,
                        )
                        self.segments.append(seg)
                    current_run.clear()

            # Flush remaining
            if len(current_run) >= 2:
                rep = signatures.get(current_run[0], 0)
                seg = Segment(
                    node_ids=list(current_run),
                    representative_signature=rep,
                    creation_step=step,
                    last_update_step=step,
                )
                self.segments.append(seg)

        self._rebuild_lookup()
        return list(self.segments)

    def update_step(self, signatures: Dict[int, int], step: int) -> None:
        """Process one simulation step: record signatures, merge, split.

        1. Record all signatures into the stability checker.
        2. Check existing segments for unstable nodes -> split.
        3. Attempt to extend existing segments or form new ones.

        Runs in O(total_degree2_nodes) per step, with per-segment
        operations O(segment_length).

        Parameters
        ----------
        signatures : dict
            Current signatures: node_id -> 12-bit int.
        step : int
            Current simulation step.
        """
        # Step 1: Record all signatures
        for nid in self._degree2_nodes:
            sig = signatures.get(nid)
            if sig is not None:
                self.checker.record(nid, sig, step)

        # Step 2: Check for instability in existing segments -> split
        new_segments: List[Segment] = []
        to_remove: List[int] = []

        for seg_idx, seg in enumerate(self.segments):
            unstable_nodes = [
                nid for nid in seg.node_ids
                if not self.checker.is_stable(nid)
            ]
            if not unstable_nodes:
                new_segments.append(seg)
                continue

            # Split at each unstable node
            to_remove.append(seg_idx)
            remaining_ids = list(seg.node_ids)
            fragments: List[List[int]] = []
            current_frag: List[int] = []

            for nid in remaining_ids:
                if nid in unstable_nodes:
                    if current_frag:
                        fragments.append(current_frag)
                        current_frag = []
                    self.checker.reset(nid)
                    self.split_events.append({
                        "step": step,
                        "node": nid,
                        "segment_size_before": len(seg),
                    })
                else:
                    current_frag.append(nid)
            if current_frag:
                fragments.append(current_frag)

            # Create sub-segments from fragments (keep even single-node
            # fragments to allow re-growth)
            for frag in fragments:
                if frag:
                    new_segments.append(Segment(
                        node_ids=frag,
                        representative_signature=signatures.get(frag[0], 0),
                        creation_step=seg.creation_step,
                        last_update_step=step,
                    ))

        self.segments = new_segments
        self._rebuild_lookup()

        # Step 3: Try to extend segments or form new ones along chains
        for chain in self._chains:
            for i in range(len(chain) - 1):
                na, nb = chain[i], chain[i + 1]
                seg_a = self._node_to_seg.get(na)
                seg_b = self._node_to_seg.get(nb)

                # Skip if both already in the same segment
                if seg_a is not None and seg_a == seg_b:
                    continue

                self.try_merge(na, nb, signatures, step)

    def compression_ratio(self) -> float:
        """Fraction of degree-2 nodes that are in segments.

        Returns
        -------
        float
            total_segmented_nodes / total_degree2_nodes, or 0.0 if
            there are no degree-2 nodes.
        """
        total_d2 = len(self._degree2_nodes)
        if total_d2 == 0:
            return 0.0
        segmented = sum(len(seg) for seg in self.segments)
        return segmented / total_d2

    def net_overhead(self) -> float:
        """Net cost/benefit of compression as fraction of baseline cost.

        Uses profiler-calibrated constants (from bench_breakeven.py) to
        compute the actual time delta between running with vs without
        compression, normalised by baseline pipeline cost.

        Negative values mean compression saves time (profitable).
        Positive values mean compression costs more than it saves.

        .. math::

            \\text{net} = \\frac{C_{init}/N_{steps} + N_{d2} \\cdot C_{track}
                          - \\text{segmented} \\cdot (1 - 1/L_{avg})
                          \\cdot C_{refine}}{N_{total} \\cdot C_{refine}}

        Returns
        -------
        float
            Net overhead fraction.  < 0 means profitable,
            > 0 means waste.
        """
        # Profiler constants (ns) — from bench_breakeven.py
        C_REFINE = 15_900
        C_TRACK = 1_900
        C_INIT = 100_500
        N_STEPS_AMORT = 40  # amortise init over typical step count

        total_d2 = len(self._degree2_nodes)
        total_nodes = len(self.tree.nodes)
        if total_nodes == 0 or total_d2 == 0:
            return 0.0

        segmented = sum(len(seg) for seg in self.segments)
        n_seg = len(self.segments)
        L_avg = segmented / max(n_seg, 1) if n_seg > 0 else 1.0

        # Per-step costs
        init_amort = C_INIT / N_STEPS_AMORT
        tracking_cost = total_d2 * C_TRACK
        savings = segmented * (1 - 1 / max(L_avg, 1.01)) * C_REFINE

        net_cost = init_amort + tracking_cost - savings
        baseline = total_nodes * C_REFINE
        return net_cost / baseline
