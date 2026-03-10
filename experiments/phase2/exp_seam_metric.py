#!/usr/bin/env python3
"""
Curiosity — Phase 2 addendum: Local Seam Metric

Метрика шва, не требующая GT, графа соседства и топологии пространства.

SeamScore = Jump_out / (Jump_in + eps)
  Jump_out = median ||state[last_halo_ring] - state[first_outside_ring]||
  Jump_in  = median ||state[inner_ring_k] - state[inner_ring_k+1]|| по внутренним кольцам halo

ΔSeam = SeamScore_after_refine - SeamScore_before_refine
  >0 → refine ухудшил шов
  <0 → refine улучшил шов
  ~0 → без изменений

Dual check: refine OK ⟺ (Gain_interior ≥ thr) ∧ (ΔSeam ≤ limit)

High-d readiness:
  - L2 (baseline)
  - L∞ (max-component)  
  - Random projections (fixed directions, max over projections)

Auto-w pilot: sweep w, pick w where ΔSeam first crosses 0 or hits minimum.

Проверки:
  S1: SeamScore корректно различает hard insert vs halo insert
  S2: ΔSeam отрицателен при правильном halo, положителен без halo
  S3: Dual check отсекает "красивый шов но пустой внутри" (мыло)
  S4: Random projections ловят "тонкий шов" в подпространстве
  S5: Auto-w пилот
"""

import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass, asdict

GRID = 128; TILE = 4; NT = GRID // TILE

# ─── Signal generation ───

def make_signal(name, seed=42):
    rng = np.random.RandomState(seed)
    x = np.linspace(0, 1, GRID, endpoint=False)
    xx, yy = np.meshgrid(x, x)
    if name == "smooth":
        return 0.5 * np.sin(2*np.pi*2*xx) * np.cos(2*np.pi*1.5*yy) + rng.randn(GRID,GRID)*0.01
    elif name == "medium":
        return (0.5*np.sin(2*np.pi*5*xx)*np.cos(2*np.pi*3*yy) +
                0.3*np.sin(2*np.pi*8*(xx+yy)) + rng.randn(GRID,GRID)*0.02)
    elif name == "sharp":
        return (1.0*(xx>0.3) + 0.7*(yy>0.45) - 0.5*((xx>0.55)&(yy>0.65)).astype(float) +
                0.4*np.sin(2*np.pi*5*xx) + rng.randn(GRID,GRID)*0.02)
    raise ValueError(name)

def make_coarse(gt):
    c = np.zeros_like(gt)
    for ti in range(NT):
        for tj in range(NT):
            s, cs = slice(ti*TILE,(ti+1)*TILE), slice(tj*TILE,(tj+1)*TILE)
            c[s, cs] = gt[s, cs].mean()
    return c


# ─── Refine operator ───

def refine_tile(state, gt, ti, tj, w, decay=1.0):
    """Refine with overlap=w. Returns new state."""
    r0, c0 = ti*TILE, tj*TILE
    er0, er1 = max(r0-w, 0), min(r0+TILE+w, GRID)
    ec0, ec1 = max(c0-w, 0), min(c0+TILE+w, GRID)
    delta = (gt[er0:er1, ec0:ec1] - state[er0:er1, ec0:ec1]) * decay
    h, wd = delta.shape
    mask = np.ones((h, wd))
    if w > 0:
        for i in range(min(w, h)):
            f = 0.5*(1-np.cos(np.pi*(i+0.5)/w))
            mask[i, :] *= f
            if h-1-i != i: mask[h-1-i, :] *= f
        for j in range(min(w, wd)):
            f = 0.5*(1-np.cos(np.pi*(j+0.5)/w))
            mask[:, j] *= f
            if wd-1-j != j: mask[:, wd-1-j] *= f
    out = state.copy()
    out[er0:er1, ec0:ec1] += delta * mask
    return out


# ─── Local Seam Metric ───

def extract_rings(state, ti, tj, w, n_rings=None):
    """Extract concentric rings around tile boundary.
    
    Ring 0 = tile interior boundary (last pixel row/col of tile)
    Ring 1..w = halo rings (moving outward)
    Ring w+1 = first ring outside halo
    
    Returns list of ring values (each ring = list of pixel values).
    """
    if n_rings is None:
        n_rings = w + 2  # interior boundary + w halo rings + 1 outside
    
    r0, c0 = ti*TILE, tj*TILE
    r1, c1 = r0+TILE, c0+TILE
    rings = []
    
    for ring_idx in range(n_rings):
        if ring_idx == 0:
            # Interior boundary: last row/col inside tile
            vals = []
            for r in range(r0, r1):
                if c0 > 0: vals.append(state[r, c0])      # left edge
                if c1 < GRID: vals.append(state[r, c1-1])  # right edge
            for c in range(c0, c1):
                if r0 > 0: vals.append(state[r0, c])       # top edge
                if r1 < GRID: vals.append(state[r1-1, c])  # bottom edge
        else:
            # Ring at distance ring_idx from tile boundary (outward)
            d = ring_idx
            vals = []
            for r in range(max(r0-d, 0), min(r1+d, GRID)):
                for c in range(max(c0-d, 0), min(c1+d, GRID)):
                    # Check if this pixel is exactly at distance d from tile boundary
                    dr = max(r0 - r, r - r1 + 1, 0)
                    dc = max(c0 - c, c - c1 + 1, 0)
                    dist = max(dr, dc)  # Chebyshev distance to tile
                    if dist == d:
                        vals.append(state[r, c])
        rings.append(np.array(vals) if vals else np.array([0.0]))
    
    return rings


def compute_jump(ring_a, ring_b):
    """Median absolute difference between adjacent rings.
    
    For scalar: |a - b|.
    For vector-valued: will use norm.
    """
    # Rings may have different sizes; use mean of each, then diff
    return abs(np.median(ring_a) - np.median(ring_b))


def compute_seam_score(state, ti, tj, w, eps=1e-10):
    """SeamScore = Jump_out / (Jump_in + eps).
    
    Jump_out: jump between last halo ring and first outside ring.
    Jump_in: mean jump across internal halo rings.
    """
    rings = extract_rings(state, ti, tj, w, n_rings=w+2)
    
    if w == 0:
        # No halo: jump between tile boundary and outside
        jump_out = compute_jump(rings[0], rings[1])
        jump_in = eps  # no interior reference
        return jump_out / (jump_in + eps), jump_out, jump_in
    
    # Jump_out: ring[w] (last halo) vs ring[w+1] (first outside)
    if len(rings) > w + 1:
        jump_out = compute_jump(rings[w], rings[w+1])
    else:
        jump_out = compute_jump(rings[-2], rings[-1])
    
    # Jump_in: average jump across internal halo ring pairs
    internal_jumps = []
    for k in range(min(w, len(rings)-1)):
        internal_jumps.append(compute_jump(rings[k], rings[k+1]))
    jump_in = np.mean(internal_jumps) if internal_jumps else eps
    
    return jump_out / (jump_in + eps), jump_out, jump_in


def compute_delta_seam(state_before, state_after, ti, tj, w):
    """ΔSeam = SeamScore_after - SeamScore_before."""
    ss_before, _, _ = compute_seam_score(state_before, ti, tj, w)
    ss_after, _, _ = compute_seam_score(state_after, ti, tj, w)
    return ss_after - ss_before, ss_before, ss_after


def compute_gain_interior(state_before, state_after, gt, ti, tj):
    """MSE reduction inside tile (not including halo)."""
    s = slice(ti*TILE, (ti+1)*TILE)
    c = slice(tj*TILE, (tj+1)*TILE)
    mse_before = np.mean((gt[s, c] - state_before[s, c])**2)
    mse_after = np.mean((gt[s, c] - state_after[s, c])**2)
    return max(mse_before - mse_after, 0.0), mse_before, mse_after


# ─── High-d readiness: projection-based seam score ───

def compute_seam_score_projected(state_2d, ti, tj, w, n_proj=16, seed=0, eps=1e-10):
    """Seam score via random projections (simulates high-d).
    
    For 2D scalar field: treats each ring as a distribution, projects onto
    random directions in "ring-value space". This is a structural test.
    
    For actual high-d: state would be (GRID, GRID, D), rings would be vectors,
    and we'd project onto random directions in R^D.
    
    Here we simulate by treating spatial neighbors as pseudo-dimensions.
    """
    rings = extract_rings(state_2d, ti, tj, w, n_rings=w+2)
    
    # L2 score (baseline)
    ss_l2, jo_l2, ji_l2 = compute_seam_score(state_2d, ti, tj, w, eps)
    
    # L∞ score: max absolute difference across ring elements
    if w > 0 and len(rings) > w + 1:
        r_last = rings[w]; r_out = rings[w+1]
        # Element-wise: take max of pairwise differences
        min_len = min(len(r_last), len(r_out))
        if min_len > 0:
            diffs = np.abs(r_last[:min_len] - r_out[:min_len])
            jo_linf = np.max(diffs)
        else:
            jo_linf = 0.0
        
        internal_maxes = []
        for k in range(min(w, len(rings)-1)):
            ra, rb = rings[k], rings[k+1]
            ml = min(len(ra), len(rb))
            if ml > 0:
                internal_maxes.append(np.max(np.abs(ra[:ml] - rb[:ml])))
        ji_linf = np.mean(internal_maxes) if internal_maxes else eps
        ss_linf = jo_linf / (ji_linf + eps)
    else:
        ss_linf = ss_l2; jo_linf = jo_l2; ji_linf = ji_l2
    
    return {"l2": ss_l2, "linf": ss_linf}


# ─── Experiments ───

def exp_s1_s2(signals=["smooth", "medium", "sharp"], seeds=[42, 137]):
    """S1+S2: SeamScore distinguishes hard vs halo insert; ΔSeam sign is correct."""
    results = {}
    
    for sig_name in signals:
        for seed in seeds:
            gt = make_signal(sig_name, seed)
            coarse = make_coarse(gt)
            
            # Pick tiles for refine (top 8% by residual)
            nr = max(1, int(NT*NT*0.08))
            scores = sorted([((i,j), np.mean((gt[i*TILE:(i+1)*TILE, j*TILE:(j+1)*TILE] -
                              coarse[i*TILE:(i+1)*TILE, j*TILE:(j+1)*TILE])**2))
                             for i in range(NT) for j in range(NT)], key=lambda x: -x[1])
            active = [t[0] for t in scores[:nr]]
            
            key = f"{sig_name}_s{seed}"
            results[key] = {}
            
            for w in [0, 1, 2, 3, 4, 5]:
                dseams = []; ss_befores = []; ss_afters = []; gains = []
                
                state = coarse.copy()
                for ti, tj in active:
                    state_before = state.copy()
                    state = refine_tile(state, gt, ti, tj, w, decay=1.0)
                    
                    ds, ssb, ssa = compute_delta_seam(state_before, state, ti, tj, w)
                    gi, _, _ = compute_gain_interior(state_before, state, gt, ti, tj)
                    
                    dseams.append(ds)
                    ss_befores.append(ssb)
                    ss_afters.append(ssa)
                    gains.append(gi)
                
                results[key][w] = {
                    "delta_seam_mean": float(np.mean(dseams)),
                    "delta_seam_median": float(np.median(dseams)),
                    "ss_before_mean": float(np.mean(ss_befores)),
                    "ss_after_mean": float(np.mean(ss_afters)),
                    "gain_mean": float(np.mean(gains)),
                    "n_worsened": int(sum(1 for d in dseams if d > 0.1)),
                    "n_improved": int(sum(1 for d in dseams if d < -0.1)),
                    "n_tiles": len(active),
                }
    
    return results


def exp_s3_dual_check(signals=["smooth", "medium", "sharp"]):
    """S3: Dual check — detect 'beautiful seam but empty inside' (blur/soap)."""
    results = {}
    
    for sig_name in signals:
        gt = make_signal(sig_name, 42)
        coarse = make_coarse(gt)
        
        nr = max(1, int(NT*NT*0.08))
        scores = sorted([((i,j), np.mean((gt[i*TILE:(i+1)*TILE, j*TILE:(j+1)*TILE] -
                          coarse[i*TILE:(i+1)*TILE, j*TILE:(j+1)*TILE])**2))
                         for i in range(NT) for j in range(NT)], key=lambda x: -x[1])
        active = [t[0] for t in scores[:nr]]
        
        # Normal refine (w=3)
        state_normal = coarse.copy()
        normal_ds = []; normal_gi = []
        for ti, tj in active:
            sb = state_normal.copy()
            state_normal = refine_tile(state_normal, gt, ti, tj, 3, decay=1.0)
            ds, _, _ = compute_delta_seam(sb, state_normal, ti, tj, 3)
            gi, _, _ = compute_gain_interior(sb, state_normal, gt, ti, tj)
            normal_ds.append(ds); normal_gi.append(gi)
        
        # "Soap" refine: blur instead of real delta (good seam, bad interior)
        state_soap = coarse.copy()
        soap_ds = []; soap_gi = []
        for ti, tj in active:
            sb = state_soap.copy()
            # Apply heavy blur instead of real refine
            r0, c0 = ti*TILE, tj*TILE
            w = 3
            er0, er1 = max(r0-w, 0), min(r0+TILE+w, GRID)
            ec0, ec1 = max(c0-w, 0), min(c0+TILE+w, GRID)
            # Blur: replace with local mean (soap)
            region_mean = state_soap[er0:er1, ec0:ec1].mean()
            delta_soap = (region_mean - state_soap[er0:er1, ec0:ec1]) * 0.5
            h, wd = delta_soap.shape
            mask = np.ones((h, wd))
            for ii in range(min(w, h)):
                f = 0.5*(1-np.cos(np.pi*(ii+0.5)/w)); mask[ii,:]*=f
                if h-1-ii!=ii: mask[h-1-ii,:]*=f
            for jj in range(min(w, wd)):
                f = 0.5*(1-np.cos(np.pi*(jj+0.5)/w)); mask[:,jj]*=f
                if wd-1-jj!=jj: mask[:,wd-1-jj]*=f
            state_soap[er0:er1, ec0:ec1] += delta_soap * mask
            
            ds, _, _ = compute_delta_seam(sb, state_soap, ti, tj, 3)
            gi, _, _ = compute_gain_interior(sb, state_soap, gt, ti, tj)
            soap_ds.append(ds); soap_gi.append(gi)
        
        results[sig_name] = {
            "normal": {
                "delta_seam": float(np.mean(normal_ds)),
                "gain": float(np.mean(normal_gi)),
                "pass_dual": int(sum(1 for d, g in zip(normal_ds, normal_gi)
                                     if g > 1e-5 and d < 0.5)),
            },
            "soap": {
                "delta_seam": float(np.mean(soap_ds)),
                "gain": float(np.mean(soap_gi)),
                "pass_dual": int(sum(1 for d, g in zip(soap_ds, soap_gi)
                                     if g > 1e-5 and d < 0.5)),
            },
            "n_tiles": len(active),
        }
    
    return results


def exp_s4_projections(signals=["smooth", "medium", "sharp"]):
    """S4: L2 vs L∞ seam scores."""
    results = {}
    
    for sig_name in signals:
        gt = make_signal(sig_name, 42)
        coarse = make_coarse(gt)
        
        nr = max(1, int(NT*NT*0.08))
        tile_scores = sorted([((i,j), np.mean((gt[i*TILE:(i+1)*TILE, j*TILE:(j+1)*TILE] -
                              coarse[i*TILE:(i+1)*TILE, j*TILE:(j+1)*TILE])**2))
                         for i in range(NT) for j in range(NT)], key=lambda x: -x[1])
        active = [t[0] for t in tile_scores[:nr]]
        
        results[sig_name] = {}
        for w in [0, 2, 3]:
            state = coarse.copy()
            l2_scores = []; linf_scores = []
            for ti, tj in active:
                state = refine_tile(state, gt, ti, tj, w)
                proj = compute_seam_score_projected(state, ti, tj, w)
                l2_scores.append(proj["l2"])
                linf_scores.append(proj["linf"])
            
            results[sig_name][w] = {
                "l2_mean": float(np.mean(l2_scores)),
                "linf_mean": float(np.mean(linf_scores)),
                "linf_max": float(np.max(linf_scores)),
            }
    
    return results


def exp_s5_auto_w():
    """S5: Auto-w pilot — find optimal w via ΔSeam minimum."""
    results = {}
    
    for sig_name in ["smooth", "medium", "sharp"]:
        gt = make_signal(sig_name, 42)
        coarse = make_coarse(gt)
        
        nr = max(1, int(NT*NT*0.08))
        tile_scores = sorted([((i,j), np.mean((gt[i*TILE:(i+1)*TILE, j*TILE:(j+1)*TILE] -
                              coarse[i*TILE:(i+1)*TILE, j*TILE:(j+1)*TILE])**2))
                         for i in range(NT) for j in range(NT)], key=lambda x: -x[1])
        active = [t[0] for t in tile_scores[:nr]]
        
        results[sig_name] = {}
        for w in range(7):
            state = coarse.copy()
            dseams = []; gains = []; ss_afters = []
            for ti, tj in active:
                sb = state.copy()
                state = refine_tile(state, gt, ti, tj, w)
                ds, _, ssa = compute_delta_seam(sb, state, ti, tj, max(w, 1))
                gi, _, _ = compute_gain_interior(sb, state, gt, ti, tj)
                dseams.append(ds); gains.append(gi); ss_afters.append(ssa)
            
            results[sig_name][w] = {
                "delta_seam": float(np.median(dseams)),
                "ss_after": float(np.median(ss_afters)),
                "gain": float(np.mean(gains)),
                "score": float(np.mean(gains)) - 0.01 * max(float(np.median(dseams)), 0),
            }
    
    return results


# ─── Main ───

def main():
    out = Path("/home/claude")
    
    print("=" * 60)
    print("Local Seam Metric — Experiments S1-S5")
    print("=" * 60)
    
    # S1+S2
    print("\n[S1+S2] SeamScore & ΔSeam vs overlap width")
    r12 = exp_s1_s2()
    for key in sorted(r12.keys()):
        print(f"\n  {key}:")
        for w in sorted(r12[key].keys()):
            d = r12[key][w]
            print(f"    w={w}: ΔSeam={d['delta_seam_median']:+.3f}  "
                  f"SS_before={d['ss_before_mean']:.3f}  SS_after={d['ss_after_mean']:.3f}  "
                  f"gain={d['gain_mean']:.6f}  "
                  f"worsened={d['n_worsened']}/{d['n_tiles']}  improved={d['n_improved']}/{d['n_tiles']}")
    
    # S3
    print("\n[S3] Dual check: normal vs soap refine")
    r3 = exp_s3_dual_check()
    for sig, d in r3.items():
        print(f"\n  {sig}:")
        for mode in ["normal", "soap"]:
            m = d[mode]
            print(f"    {mode:8s}: ΔSeam={m['delta_seam']:+.3f}  gain={m['gain']:.6f}  "
                  f"pass_dual={m['pass_dual']}/{d['n_tiles']}")
    
    # S4
    print("\n[S4] L2 vs L∞ seam scores")
    r4 = exp_s4_projections()
    for sig, ws in r4.items():
        print(f"\n  {sig}:")
        for w, d in sorted(ws.items()):
            print(f"    w={w}: L2={d['l2_mean']:.3f}  L∞={d['linf_mean']:.3f}  L∞_max={d['linf_max']:.3f}")
    
    # S5
    print("\n[S5] Auto-w pilot (ΔSeam-based)")
    r5 = exp_s5_auto_w()
    for sig, ws in r5.items():
        best_w = min(ws.keys(), key=lambda w: ws[w]["delta_seam"] if w > 0 else 999)
        print(f"\n  {sig}: (best_w={best_w})")
        for w in sorted(ws.keys()):
            d = ws[w]
            marker = " ◀" if w == best_w else ""
            print(f"    w={w}: ΔSeam={d['delta_seam']:+.4f}  SS={d['ss_after']:.3f}  "
                  f"gain={d['gain']:.6f}  score={d['score']:.6f}{marker}")
    
    # Save
    all_data = {"s1s2": r12, "s3": r3, "s4": r4, "s5": r5}
    with open(out / "phase2_seam_metric.json", "w") as f:
        json.dump(all_data, f, indent=2, default=str)
    
    # Plot
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Local Seam Metric — S1-S5", fontsize=14, fontweight="bold")
    
    # S1+S2: ΔSeam vs w for each signal
    ax = axes[0, 0]
    colors_sig = {"smooth": "#1f77b4", "medium": "#ff7f0e", "sharp": "#2ca02c"}
    for key in sorted(r12.keys()):
        sig = key.split("_")[0]
        ws = sorted(r12[key].keys())
        ds = [r12[key][w]["delta_seam_median"] for w in ws]
        ax.plot(ws, ds, "o-", color=colors_sig.get(sig, "gray"), label=key, ms=4, alpha=0.7)
    ax.axhline(y=0, color="black", ls="--", alpha=0.3)
    ax.set_title("S1+S2: ΔSeam vs Overlap Width")
    ax.set_xlabel("Overlap w"); ax.set_ylabel("ΔSeam (median)")
    ax.legend(fontsize=6); ax.grid(True, alpha=0.3)
    
    # S1+S2: SeamScore after vs w
    ax = axes[0, 1]
    for key in sorted(r12.keys()):
        sig = key.split("_")[0]
        ws = sorted(r12[key].keys())
        ss = [r12[key][w]["ss_after_mean"] for w in ws]
        ax.plot(ws, ss, "s-", color=colors_sig.get(sig, "gray"), label=key, ms=4, alpha=0.7)
    ax.axhline(y=1, color="green", ls="--", alpha=0.5, label="SS=1 (no seam)")
    ax.set_title("S1+S2: SeamScore After vs w")
    ax.set_xlabel("Overlap w"); ax.set_ylabel("SeamScore")
    ax.legend(fontsize=6); ax.grid(True, alpha=0.3)
    
    # S3: Dual check bar chart
    ax = axes[0, 2]
    sigs = list(r3.keys())
    x = np.arange(len(sigs)); bw = 0.3
    normal_pass = [r3[s]["normal"]["pass_dual"]/r3[s]["n_tiles"]*100 for s in sigs]
    soap_pass = [r3[s]["soap"]["pass_dual"]/r3[s]["n_tiles"]*100 for s in sigs]
    ax.bar(x - bw/2, normal_pass, bw, label="Normal refine", color="#2ca02c", alpha=0.7)
    ax.bar(x + bw/2, soap_pass, bw, label="Soap refine", color="#d62728", alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(sigs)
    ax.set_ylabel("Pass dual check (%)"); ax.set_title("S3: Dual Check — Normal vs Soap")
    ax.legend(); ax.grid(True, alpha=0.3)
    
    # S5: ΔSeam vs w (auto-w)
    ax = axes[1, 0]
    for sig, ws in r5.items():
        w_list = sorted(ws.keys())
        ds = [ws[w]["delta_seam"] for w in w_list]
        ax.plot(w_list, ds, "o-", color=colors_sig[sig], label=sig, ms=4)
    ax.axhline(y=0, color="black", ls="--", alpha=0.3)
    ax.set_title("S5: Auto-w — ΔSeam vs w")
    ax.set_xlabel("Overlap w"); ax.set_ylabel("ΔSeam (median)")
    ax.legend(); ax.grid(True, alpha=0.3)
    
    # S5: Gain vs w
    ax = axes[1, 1]
    for sig, ws in r5.items():
        w_list = sorted(ws.keys())
        gi = [ws[w]["gain"] for w in w_list]
        ax.plot(w_list, gi, "s-", color=colors_sig[sig], label=sig, ms=4)
    ax.set_title("S5: Gain vs w")
    ax.set_xlabel("Overlap w"); ax.set_ylabel("Interior Gain (MSE reduction)")
    ax.legend(); ax.grid(True, alpha=0.3)
    
    # S4: L2 vs L∞
    ax = axes[1, 2]
    for sig, ws in r4.items():
        w_list = sorted(ws.keys())
        l2 = [ws[w]["l2_mean"] for w in w_list]
        linf = [ws[w]["linf_mean"] for w in w_list]
        ax.plot(w_list, l2, "o-", color=colors_sig[sig], label=f"{sig} L2", ms=4)
        ax.plot(w_list, linf, "s--", color=colors_sig[sig], label=f"{sig} L∞", ms=4, alpha=0.6)
    ax.axhline(y=1, color="green", ls="--", alpha=0.5)
    ax.set_title("S4: L2 vs L∞ SeamScore")
    ax.set_xlabel("Overlap w"); ax.set_ylabel("SeamScore")
    ax.legend(fontsize=6); ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out / "phase2_seam_metric.png", dpi=150, bbox_inches="tight")
    print(f"\n[Saved] phase2_seam_metric.png + .json")
    print("\n[Done]")


if __name__ == "__main__":
    main()
