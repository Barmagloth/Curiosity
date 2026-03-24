#!/usr/bin/env python3
"""
exp19a — Aggregate results and find max_ticks scaling law.
"""

import json
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_all_chunks():
    """Load all exp19a chunk files."""
    results = []
    for f in sorted(RESULTS_DIR.glob("exp19a_chunk_*.json")):
        with open(f) as fh:
            chunk = json.load(fh)
            results.extend(chunk)
    return results


def analyze():
    results = load_all_chunks()
    valid = [r for r in results if "error" not in r]
    print(f"Loaded {len(results)} results ({len(valid)} valid)")

    # Group by (scale, space_type)
    groups = {}
    for r in valid:
        key = (r["scale"], r["space_type"])
        groups.setdefault(key, []).append(r)

    print("\n" + "=" * 90)
    print(f"{'Scale':<8} {'Space':<20} {'n_units':>7} {'mt=1 PSNR':>10} "
          f"{'Best mt':>7} {'Best PSNR':>10} {'Ratio':>7} {'Ticks used':>10}")
    print("=" * 90)

    scaling_data = []

    for (scale, space_type), group in sorted(groups.items()):
        # Single-tick baseline
        single = [r for r in group if r["max_ticks"] == 1]
        single_psnr = np.median([r["psnr_gain"] for r in single]) if single else 0
        n_units = single[0]["n_total"] if single else 0

        # Find best multi-tick
        best_mt = 1
        best_psnr = single_psnr
        best_ticks_used = 1

        for mt in [2, 3, 5, 8, 13, 20]:
            mt_results = [r for r in group if r["max_ticks"] == mt]
            if mt_results:
                mt_psnr = np.median([r["psnr_gain"] for r in mt_results])
                if mt_psnr > best_psnr:
                    best_psnr = mt_psnr
                    best_mt = mt
                    best_ticks_used = np.median([r["n_ticks_executed"] for r in mt_results])

        ratio = best_psnr / max(single_psnr, 0.01)

        print(f"{scale:<8} {space_type:<20} {n_units:>7} {single_psnr:>10.2f} "
              f"{best_mt:>7} {best_psnr:>10.2f} {ratio:>7.0%} {best_ticks_used:>10.0f}")

        scaling_data.append({
            "scale": scale,
            "space_type": space_type,
            "n_units": n_units,
            "single_tick_psnr_gain": float(single_psnr),
            "best_max_ticks": best_mt,
            "best_psnr_gain": float(best_psnr),
            "ratio": float(ratio),
            "ticks_actually_used": float(best_ticks_used),
        })

    # Detailed per-max_ticks table
    print("\n\nDetailed: median PSNR gain by (scale, space, max_ticks)")
    print("-" * 90)
    for (scale, space_type), group in sorted(groups.items()):
        n_units = group[0]["n_total"]
        row = f"{scale:<8} {space_type:<20} n={n_units:<5}"
        for mt in [1, 2, 3, 5, 8, 13, 20]:
            mt_results = [r for r in group if r["max_ticks"] == mt]
            if mt_results:
                psnr = np.median([r["psnr_gain"] for r in mt_results])
                row += f"  mt{mt:>2}={psnr:>5.1f}"
        print(row)

    # Save
    with open(RESULTS_DIR / "exp19a_scaling_summary.json", "w") as f:
        json.dump(scaling_data, f, indent=2)
    print(f"\nSummary saved to {RESULTS_DIR / 'exp19a_scaling_summary.json'}")


if __name__ == "__main__":
    analyze()
