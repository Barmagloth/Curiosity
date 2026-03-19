#!/usr/bin/env python3
"""
Merge chunk results from exp10i_chunk.py into final summary + report.

Reads all chunk_*.json files from results/, combines them, generates
verdicts and writes exp10i_summary.json + exp10i_report.md.
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"

KILL_THRESH = 0.20
CROSS_BLOCK_THRESH = 0.50
PADDING_WASTE_THRESH = 0.50
FEATURE_DIM = 8

GRAPH_TYPES = ["random_geometric", "barabasi_albert", "grid_graph"]
PARTITION_METHODS = ["random_partition", "spatial_partition", "greedy_partition"]
BLOCK_SIZES = [8, 16, 32, 64]
SPARSITIES = [0.05, 0.1, 0.3, 0.5]

CHUNK_FILES = [
    "chunk_rg256.json",
    "chunk_rg1024.json",
    "chunk_ba256.json",
    "chunk_ba1024.json",
    "chunk_gg256.json",
    "chunk_gg1024.json",
]


def load_chunks() -> List[Dict[str, Any]]:
    all_results = []
    for fname in CHUNK_FILES:
        fpath = RESULTS_DIR / fname
        if not fpath.exists():
            print(f"WARNING: Missing chunk file {fpath}")
            continue
        with open(fpath) as f:
            data = json.load(f)
        print(f"  Loaded {fname}: {len(data)} configs")
        all_results.extend(data)
    return all_results


def recompute_verdicts(all_results: List[Dict]) -> List[Dict]:
    """Recompute contour verdicts.

    On CPU runs, memory is 0 for both candidates, making memory-based
    contours meaningless. We focus on:
      - Contour A: time_overhead < 20% AND cross_block_ratio < 0.50
      - Contour B: time_overhead < 20%
    (Memory conditions are relaxed for CPU-only runs.)
    """
    for r in all_results:
        to = r["time_overhead_frac"]
        cbr = r["cross_block_ratio"]
        pw = r["padding_waste"]

        # CPU-adapted contours (no memory comparison possible)
        r["contour_A"] = "PASS" if (to < KILL_THRESH and cbr < CROSS_BLOCK_THRESH) else "FAIL"
        r["contour_B"] = "PASS" if to < KILL_THRESH else "FAIL"
        r["padding_warning"] = pw > PADDING_WASTE_THRESH

    return all_results


def write_summary(all_results: List[Dict]):
    pass_a = sum(1 for r in all_results if r["contour_A"] == "PASS")
    pass_b = sum(1 for r in all_results if r["contour_B"] == "PASS")
    n_warnings = sum(1 for r in all_results if r["padding_warning"])

    N_nodes_seen = sorted(set(r["N_nodes"] for r in all_results))

    summary = {
        "experiment": "exp10i_graph_blocks",
        "device": "cpu",
        "dtype": "torch.float32",
        "n_seeds": 10,
        "feature_dim": FEATURE_DIM,
        "kill_threshold": KILL_THRESH,
        "cross_block_threshold": CROSS_BLOCK_THRESH,
        "padding_waste_threshold": PADDING_WASTE_THRESH,
        "note": "CPU-only run; memory contours adapted (no VRAM comparison)",
        "config": {
            "N_nodes_list": N_nodes_seen,
            "sparsities": SPARSITIES,
            "graph_types": GRAPH_TYPES,
            "block_sizes": BLOCK_SIZES,
            "partition_methods": PARTITION_METHODS,
        },
        "results": all_results,
        "contour_A_pass": pass_a,
        "contour_A_total": len(all_results),
        "contour_B_pass": pass_b,
        "contour_B_total": len(all_results),
        "padding_warnings": n_warnings,
    }

    out_path = RESULTS_DIR / "exp10i_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Wrote {out_path}")
    return summary


def write_report(all_results: List[Dict], summary: Dict):
    pass_a = summary["contour_A_pass"]
    pass_b = summary["contour_B_pass"]
    n_warnings = summary["padding_warnings"]
    total = len(all_results)

    lines = [
        "# exp10i: Block-Based Addressing for Irregular Graphs -- Report",
        "",
        "## Summary",
        "",
        f"- **Total configs**: {total}",
        f"- **Device**: CPU (parallel chunked run)",
        f"- **Contour A** (time overhead < 20% AND cross_block_ratio < 0.50): "
        f"**{pass_a}/{total}** PASS ({pass_a/total*100:.0f}%)",
        f"- **Contour B** (time overhead < 20%): "
        f"**{pass_b}/{total}** PASS ({pass_b/total*100:.0f}%)",
        f"- **Padding warnings** (waste > 50%): {n_warnings}/{total}",
        "",
        "> Note: Run on CPU; memory contours (resident/peak VRAM) are not",
        "> evaluated. Verdicts focus on wall-clock overhead and blocking quality.",
        "",
    ]

    # ---- Per graph type summary ----
    lines += ["## Results by graph type", ""]

    for gtype in GRAPH_TYPES:
        gt_recs = [r for r in all_results if r["graph_type"] == gtype]
        if not gt_recs:
            continue
        gt_a = sum(1 for r in gt_recs if r["contour_A"] == "PASS")
        gt_b = sum(1 for r in gt_recs if r["contour_B"] == "PASS")
        gt_w = sum(1 for r in gt_recs if r["padding_warning"])

        lines += [
            f"### {gtype}",
            "",
            f"Contour A: {gt_a}/{len(gt_recs)} | "
            f"Contour B: {gt_b}/{len(gt_recs)} | "
            f"Padding warnings: {gt_w}",
            "",
            "| N | sp | bs | partition | A | B | time_oh | cbr | pw |",
            "|---|----|----|----------|---|---|---------|-----|-----|",
        ]

        for r in gt_recs:
            pw_flag = " !!" if r["padding_warning"] else ""
            lines.append(
                f"| {r['N_nodes']} | {r['sparsity']} | {r['block_size']} "
                f"| {r['partition_method'].replace('_partition', '')} "
                f"| {r['contour_A']} | {r['contour_B']} "
                f"| {r['time_overhead_frac']:+.1%} "
                f"| {r['cross_block_ratio']:.2f} "
                f"| {r['padding_waste']:.2f}{pw_flag} |"
            )
        lines.append("")

    # ---- Partition method comparison ----
    lines += ["## Partition method comparison", ""]

    for pm in PARTITION_METHODS:
        pm_recs = [r for r in all_results if r["partition_method"] == pm]
        if not pm_recs:
            continue
        mean_cbr = np.mean([r["cross_block_ratio"] for r in pm_recs])
        mean_pw = np.mean([r["padding_waste"] for r in pm_recs])
        mean_to = np.mean([r["time_overhead_frac"] for r in pm_recs])
        pm_a = sum(1 for r in pm_recs if r["contour_A"] == "PASS")
        lines.append(
            f"- **{pm}**: mean cbr={mean_cbr:.3f}, mean pw={mean_pw:.3f}, "
            f"mean time_oh={mean_to:+.1%}, Contour A: {pm_a}/{len(pm_recs)}")

    lines.append("")

    # ---- Block size analysis ----
    lines += ["## Block size analysis", ""]

    for bs in BLOCK_SIZES:
        bs_recs = [r for r in all_results if r["block_size"] == bs]
        if not bs_recs:
            continue
        mean_cbr = np.mean([r["cross_block_ratio"] for r in bs_recs])
        mean_pw = np.mean([r["padding_waste"] for r in bs_recs])
        bs_a = sum(1 for r in bs_recs if r["contour_A"] == "PASS")
        lines.append(
            f"- **block_size={bs}**: mean cbr={mean_cbr:.3f}, "
            f"mean pw={mean_pw:.3f}, Contour A: {bs_a}/{len(bs_recs)}")

    lines.append("")

    # ---- Cross-block ratio by graph_type x partition ----
    lines += [
        "## Cross-block ratio heatmap (mean cbr by graph_type x partition)",
        "",
        "| graph_type | random | spatial | greedy |",
        "|------------|--------|---------|--------|",
    ]

    for gtype in GRAPH_TYPES:
        row = f"| {gtype} "
        for pm in PARTITION_METHODS:
            recs = [r for r in all_results
                    if r["graph_type"] == gtype and r["partition_method"] == pm]
            if recs:
                mean_cbr = np.mean([r["cross_block_ratio"] for r in recs])
                row += f"| {mean_cbr:.3f} "
            else:
                row += "| - "
        row += "|"
        lines.append(row)

    lines.append("")

    # ---- Key findings ----
    lines += ["## Key Findings", ""]

    # 1. Which graph types work?
    for gtype in GRAPH_TYPES:
        gt_recs = [r for r in all_results if r["graph_type"] == gtype]
        gt_a = sum(1 for r in gt_recs if r["contour_A"] == "PASS")
        pct = gt_a / len(gt_recs) * 100 if gt_recs else 0
        status = "VIABLE" if pct > 50 else "MARGINAL" if pct > 20 else "NOT VIABLE"
        lines.append(f"- **{gtype}**: {status} ({gt_a}/{len(gt_recs)} = {pct:.0f}% Contour A)")

    lines.append("")

    # 2. Best partition method for non-spatial graphs
    ba_recs = [r for r in all_results if r["graph_type"] == "barabasi_albert"]
    if ba_recs:
        lines.append("### Does spatial/greedy help for non-spatial graphs (barabasi_albert)?")
        lines.append("")
        for pm in PARTITION_METHODS:
            pm_ba = [r for r in ba_recs if r["partition_method"] == pm]
            if pm_ba:
                mean_cbr = np.mean([r["cross_block_ratio"] for r in pm_ba])
                lines.append(f"  - {pm}: mean cbr = {mean_cbr:.3f}")
        lines.append("")

    # 3. Padding waste by block size
    lines.append("### Padding waste vs block size")
    lines.append("")
    for bs in BLOCK_SIZES:
        bs_recs = [r for r in all_results if r["block_size"] == bs]
        if bs_recs:
            mean_pw = np.mean([r["padding_waste"] for r in bs_recs])
            n_warn = sum(1 for r in bs_recs if r["padding_warning"])
            lines.append(f"  - bs={bs}: mean pw = {mean_pw:.3f}, warnings: {n_warn}/{len(bs_recs)}")
    lines.append("")

    # ---- Overall verdict ----
    lines += ["## Verdict", ""]

    if pass_a > total * 0.5:
        if pass_a == total:
            lines.append(
                "**OVERALL: PASS** -- Block-based addressing works for irregular graphs "
                "across all tested configurations.")
        else:
            lines.append(
                f"**OVERALL: PARTIAL PASS** -- Block-based addressing works for "
                f"{pass_a}/{total} ({pass_a/total*100:.0f}%) configurations. "
                f"Viable with the right partition method and block size.")
    else:
        lines.append(
            f"**OVERALL: FAIL** -- Block-based addressing only works for "
            f"{pass_a}/{total} ({pass_a/total*100:.0f}%) configurations. "
            f"Graph structure resists blocking in most cases.")

    lines.append("")

    report_text = "\n".join(lines)
    out_path = RESULTS_DIR / "exp10i_report.md"
    with open(out_path, "w") as f:
        f.write(report_text)
    print(f"  Wrote {out_path}")
    return report_text


def main():
    print("=" * 60)
    print("exp10i: Merging chunk results")
    print("=" * 60)

    all_results = load_chunks()
    print(f"\nTotal configs loaded: {len(all_results)}")

    if len(all_results) == 0:
        print("ERROR: No results to merge!")
        return

    all_results = recompute_verdicts(all_results)
    summary = write_summary(all_results)
    write_report(all_results, summary)

    print("\n" + "=" * 60)
    print("MERGE COMPLETE")
    print("=" * 60)
    print(f"Contour A: {summary['contour_A_pass']}/{summary['contour_A_total']}")
    print(f"Contour B: {summary['contour_B_pass']}/{summary['contour_B_total']}")
    print(f"Padding warnings: {summary['padding_warnings']}")


if __name__ == "__main__":
    main()
