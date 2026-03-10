#!/usr/bin/env python3
"""
A2 extension v2: Better r_min detection

Problem with v1: 5% threshold gives r_min clustered at 0.425/0.475 because
seam_ratio curves have a sharp knee followed by slow asymptotic tail.
The "real" working point is at the knee, not at the tail.

Solution:
1. r_knee: point of maximum slope change (second derivative of seam_ratio vs r)
2. r_80: where 80% of total seam reduction is achieved
3. r_95: where 95% reduction is achieved (conservative)

These capture the actual behavior better than an arbitrary threshold.
"""

import numpy as np
import os
import json
from typing import Dict, List, Tuple


# ─── Config ──────────────────────────────────────────────────────────────────

N_TILES = 32
TILE_SIZE = 16
N = N_TILES * TILE_SIZE
R_VALUES = np.arange(0, 0.51, 0.025).tolist()
SEED = 42


# ─── Signals (reuse from previous) ──────────────────────────────────────────

def make_coords(n):
    ax = np.linspace(0, 1, n, endpoint=False) + 0.5/n
    return np.meshgrid(ax, ax)


def make_signal_freq(X, Y, base_freq, n_harm=3, seed=42):
    rng = np.random.default_rng(seed)
    s = np.zeros_like(X)
    for h in range(n_harm):
        f = base_freq * (h+1)
        s += (1/(h+1)**1.5) * np.sin(2*np.pi*f*X + rng.uniform(0,2*np.pi)) * \
             np.cos(2*np.pi*f*Y + rng.uniform(0,2*np.pi))
    s *= np.exp(-((X-0.5)**2+(Y-0.5)**2)/(2*0.3**2))
    return (s - s.min()) / (s.max() - s.min() + 1e-10)


def make_mixed(X, Y, lf_w, hf_freq=16):
    lf = np.exp(-((X-0.4)**2+(Y-0.5)**2)/(2*0.2**2)) + \
         0.5*np.exp(-((X-0.7)**2+(Y-0.3)**2)/(2*0.15**2))
    hf = 0.3*np.sin(2*np.pi*hf_freq*X)*np.cos(2*np.pi*hf_freq*Y*0.8)
    s = lf_w*lf + (1-lf_w)*hf
    return (s - s.min()) / (s.max() - s.min() + 1e-10)


def sig_smooth(X, Y):
    return (np.exp(-((X-0.3)**2+(Y-0.4)**2)/(2*0.15**2)) +
            0.6*np.exp(-((X-0.7)**2+(Y-0.6)**2)/(2*0.12**2)))

def sig_medium(X, Y):
    return sig_smooth(X, Y) + 0.2*np.sin(14*np.pi*X)*np.cos(10*np.pi*Y)

def sig_sharp(X, Y):
    base = sig_smooth(X, Y) * 0.4
    circle = ((X-0.4)**2+(Y-0.5)**2 < 0.12**2).astype(float) * 0.5
    rect = ((X>0.55)&(X<0.8)&(Y>0.25)&(Y<0.55)).astype(float) * 0.4
    return base + circle + rect


# ─── Core functions ──────────────────────────────────────────────────────────

def compute_coarse(signal, block):
    h, w = signal.shape
    bh, bw = h//block, w//block
    small = signal[:bh*block,:bw*block].reshape(bh, block, bw, block).mean(axis=(1,3))
    return np.repeat(np.repeat(small, block, axis=0), block, axis=1)


def select_tiles(delta, nt, ts, budget):
    energy = np.zeros((nt, nt))
    for i in range(nt):
        for j in range(nt):
            energy[i,j] = np.mean(delta[i*ts:(i+1)*ts, j*ts:(j+1)*ts]**2)
    k = max(1, int(budget * nt**2))
    idx = np.argpartition(energy.ravel(), -k)[-k:]
    mask = np.zeros(nt**2, dtype=bool)
    mask[idx] = True
    return mask.reshape(nt, nt)


def _cos_ramp(n):
    if n <= 0: return np.array([])
    return 0.5*(1-np.cos(np.pi*np.arange(n)/n))


def assemble(coarse, delta, mask, nt, ts, r):
    overlap = max(0, round(r*ts))
    H, W = coarse.shape
    if overlap == 0:
        out = coarse.copy()
        for i in range(nt):
            for j in range(nt):
                if mask[i,j]:
                    out[i*ts:(i+1)*ts, j*ts:(j+1)*ts] += delta[i*ts:(i+1)*ts, j*ts:(j+1)*ts]
        return out
    
    d_acc = np.zeros((H,W), dtype=np.float64)
    w_acc = np.zeros((H,W), dtype=np.float64)
    for i in range(nt):
        for j in range(nt):
            if not mask[i,j]: continue
            er0, er1 = max(0,i*ts-overlap), min(H,(i+1)*ts+overlap)
            ec0, ec1 = max(0,j*ts-overlap), min(W,(j+1)*ts+overlap)
            ph, pw = er1-er0, ec1-ec0
            wy = np.ones(ph); wx = np.ones(pw)
            top = i*ts-er0
            if top > 0: wy[:top] = _cos_ramp(top)
            bot = er1-(i+1)*ts
            if bot > 0: wy[-bot:] = _cos_ramp(bot)[::-1]
            left = j*ts-ec0
            if left > 0: wx[:left] = _cos_ramp(left)
            right = ec1-(j+1)*ts
            if right > 0: wx[-right:] = _cos_ramp(right)[::-1]
            W2d = np.outer(wy, wx)
            d_acc[er0:er1,ec0:ec1] += W2d * delta[er0:er1,ec0:ec1]
            w_acc[er0:er1,ec0:ec1] += W2d
    out = coarse.copy()
    valid = w_acc > 1e-12
    out[valid] = coarse[valid] + d_acc[valid] / w_acc[valid]
    return out


def seam_ratio(output, nt, ts, band=2):
    H, W = output.shape
    gy = np.abs(np.diff(output, axis=0))
    gx = np.abs(np.diff(output, axis=1))
    h, w = min(gy.shape[0],gx.shape[0]), min(gy.shape[1],gx.shape[1])
    grad = np.sqrt(gy[:h,:w]**2 + gx[:h,:w]**2)
    bmask = np.zeros((h,w), dtype=bool)
    for i in range(1, nt):
        row = i*ts-1
        if row < h: bmask[max(0,row-band):min(h,row+band+1),:] = True
    for j in range(1, nt):
        col = j*ts-1
        if col < w: bmask[:,max(0,col-band):min(w,col+band+1)] = True
    imask = ~bmask
    if bmask.sum()==0 or imask.sum()==0: return 1.0
    return float(np.mean(grad[bmask]) / (np.mean(grad[imask]) + 1e-15))


# ─── Spectral characteristics ────────────────────────────────────────────────

def spectral_info(delta):
    F = np.fft.fft2(delta)
    power = np.abs(F)**2
    h, w = delta.shape
    fy, fx = np.fft.fftfreq(h), np.fft.fftfreq(w)
    FY, FX = np.meshgrid(fy, fx, indexing='ij')
    freq_mag = np.sqrt(FX**2 + FY**2)
    total = power.sum()
    if total < 1e-15:
        return {"centroid": 0, "hf_ratio": 0, "lf_ratio": 1}
    centroid = float((freq_mag * power).sum() / total)
    hf = float(power[freq_mag > 0.125].sum() / total)
    lf = float(power[freq_mag < 0.03125].sum() / total)
    return {"centroid": centroid, "hf_ratio": hf, "lf_ratio": lf}


def edge_amplitude(delta, nt, ts):
    vals = []
    for i in range(1, nt):
        row = i*ts
        if row < delta.shape[0]:
            vals.append(np.abs(delta[row,:] - delta[row-1,:]).mean())
    for j in range(1, nt):
        col = j*ts
        if col < delta.shape[1]:
            vals.append(np.abs(delta[:,col] - delta[:,col-1]).mean())
    return float(np.mean(vals)) if vals else 0.0


# ─── Knee detection ──────────────────────────────────────────────────────────

def find_knee_and_thresholds(r_vals, sr_vals):
    """
    Find:
    - r_knee: point of maximum negative slope change (steepest descent start)
    - r_80: 80% of total reduction achieved
    - r_95: 95% of total reduction achieved
    
    Returns dict with all three + the seam reduction curve.
    """
    r_arr = np.array(r_vals)
    sr_arr = np.array(sr_vals)
    
    # Total reduction
    sr_max = sr_arr[0]  # at r=0 (hard insert)
    sr_min = sr_arr[-1]  # at max r
    total_reduction = sr_max - sr_min
    
    if total_reduction < 1e-10:
        # No seam to speak of
        return {"r_knee": 0.0, "r_80": 0.0, "r_95": 0.0,
                "total_reduction": 0.0, "sr_at_0": sr_max, "sr_at_max": sr_min,
                "has_seam": False}
    
    # r_80, r_95: interpolate
    target_80 = sr_max - 0.80 * total_reduction
    target_95 = sr_max - 0.95 * total_reduction
    
    r_80 = r_arr[-1]
    r_95 = r_arr[-1]
    for i in range(len(sr_arr)-1):
        if sr_arr[i] >= target_80 >= sr_arr[i+1]:
            # Linear interpolation
            frac = (sr_arr[i] - target_80) / (sr_arr[i] - sr_arr[i+1] + 1e-15)
            r_80 = r_arr[i] + frac * (r_arr[i+1] - r_arr[i])
            break
    
    for i in range(len(sr_arr)-1):
        if sr_arr[i] >= target_95 >= sr_arr[i+1]:
            frac = (sr_arr[i] - target_95) / (sr_arr[i] - sr_arr[i+1] + 1e-15)
            r_95 = r_arr[i] + frac * (r_arr[i+1] - r_arr[i])
            break
    
    # r_knee: maximum |d(sr)/dr| — steepest descent
    if len(r_arr) >= 3:
        dr = np.diff(r_arr)
        dsr = np.diff(sr_arr)
        slope = dsr / (dr + 1e-15)
        # Knee = where negative slope is steepest
        knee_idx = np.argmin(slope)  # most negative
        r_knee = float(r_arr[knee_idx])
    else:
        r_knee = 0.0
    
    return {
        "r_knee": round(float(r_knee), 4),
        "r_80": round(float(r_80), 4),
        "r_95": round(float(r_95), 4),
        "total_reduction": float(total_reduction),
        "reduction_pct": float(total_reduction / sr_max * 100),
        "sr_at_0": float(sr_max),
        "sr_at_max": float(sr_min),
        "has_seam": total_reduction > 0.05 * sr_max,  # >5% reduction means there was a seam
    }


# ─── Run full analysis ──────────────────────────────────────────────────────

def analyze_signal(name, signal, budget=0.2):
    coarse = compute_coarse(signal, TILE_SIZE)
    delta = signal - coarse
    mask = select_tiles(delta, N_TILES, TILE_SIZE, budget)
    
    r_vals = []
    sr_vals = []
    for r in R_VALUES:
        out = assemble(coarse, delta, mask, N_TILES, TILE_SIZE, r=r)
        sr = seam_ratio(out, N_TILES, TILE_SIZE)
        r_vals.append(r)
        sr_vals.append(sr)
    
    knee = find_knee_and_thresholds(r_vals, sr_vals)
    spec = spectral_info(delta)
    ea = edge_amplitude(delta, N_TILES, TILE_SIZE)
    rms = float(np.sqrt(np.mean(delta**2)))
    
    return {
        "name": name,
        "knee": knee,
        "spectral": spec,
        "edge_amp": ea,
        "delta_rms": rms,
        "norm_edge": ea / (rms + 1e-15),
        "r_vals": r_vals,
        "sr_vals": sr_vals,
    }


def main():
    outdir = "/home/claude/results"
    os.makedirs(outdir, exist_ok=True)
    
    X, Y = make_coords(N)
    
    results = []
    
    print("="*80)
    print("A2 ext v2: r_knee / r_80 / r_95 analysis")
    print("="*80)
    
    # Frequency sweep
    print(f"\n{'Signal':20s} {'r_knee':>8s} {'r_80':>8s} {'r_95':>8s} {'reduc%':>8s} {'centroid':>10s} {'norm_edge':>10s} {'seam?':>6s}")
    print("-"*90)
    
    for bf in [1, 2, 3, 4, 6, 8, 10, 14, 18, 24, 32, 48, 64]:
        sig = make_signal_freq(X, Y, bf)
        r = analyze_signal(f"freq_{bf}", sig)
        results.append(r)
        k = r["knee"]
        print(f"  {r['name']:18s} {k['r_knee']:8.3f} {k['r_80']:8.3f} {k['r_95']:8.3f} "
              f"{k['reduction_pct']:7.1f}% {r['spectral']['centroid']:10.4f} {r['norm_edge']:10.3f} "
              f"{'YES' if k['has_seam'] else 'no':>6s}")
    
    # Mix sweep
    print()
    for lf_w in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
        sig = make_mixed(X, Y, lf_w)
        r = analyze_signal(f"mix_lf{lf_w:.1f}", sig)
        results.append(r)
        k = r["knee"]
        print(f"  {r['name']:18s} {k['r_knee']:8.3f} {k['r_80']:8.3f} {k['r_95']:8.3f} "
              f"{k['reduction_pct']:7.1f}% {r['spectral']['centroid']:10.4f} {r['norm_edge']:10.3f} "
              f"{'YES' if k['has_seam'] else 'no':>6s}")
    
    # Steps
    print()
    for ns in [1, 2, 3, 5]:
        rng_s = np.random.default_rng(SEED + ns)
        sig = np.zeros_like(X)
        for _ in range(ns):
            cx, cy = rng_s.uniform(0.2, 0.8, 2)
            rad = rng_s.uniform(0.05, 0.2)
            sig += rng_s.uniform(0.3, 0.8) * ((X-cx)**2+(Y-cy)**2 < rad**2).astype(float)
        sig = np.clip(sig, 0, 1)
        # Add small smooth component to avoid trivial delta
        sig += 0.1 * sig_smooth(X, Y)
        r = analyze_signal(f"steps_{ns}", sig)
        results.append(r)
        k = r["knee"]
        print(f"  {r['name']:18s} {k['r_knee']:8.3f} {k['r_80']:8.3f} {k['r_95']:8.3f} "
              f"{k['reduction_pct']:7.1f}% {r['spectral']['centroid']:10.4f} {r['norm_edge']:10.3f} "
              f"{'YES' if k['has_seam'] else 'no':>6s}")
    
    # Originals
    print()
    for name, fn in [("smooth", sig_smooth), ("medium", sig_medium), ("sharp", sig_sharp)]:
        sig = fn(X, Y)
        r = analyze_signal(f"orig_{name}", sig)
        results.append(r)
        k = r["knee"]
        print(f"  {r['name']:18s} {k['r_knee']:8.3f} {k['r_80']:8.3f} {k['r_95']:8.3f} "
              f"{k['reduction_pct']:7.1f}% {r['spectral']['centroid']:10.4f} {r['norm_edge']:10.3f} "
              f"{'YES' if k['has_seam'] else 'no':>6s}")
    
    # ── Correlation analysis (only signals with seams) ──
    seam_results = [r for r in results if r["knee"]["has_seam"]]
    
    print(f"\n{'='*80}")
    print(f"CORRELATION ANALYSIS (N={len(seam_results)} signals with seams)")
    print(f"{'='*80}")
    
    if len(seam_results) >= 5:
        r_knees = np.array([r["knee"]["r_knee"] for r in seam_results])
        r_80s = np.array([r["knee"]["r_80"] for r in seam_results])
        r_95s = np.array([r["knee"]["r_95"] for r in seam_results])
        centroids = np.array([r["spectral"]["centroid"] for r in seam_results])
        hf_ratios = np.array([r["spectral"]["hf_ratio"] for r in seam_results])
        norm_edges = np.array([r["norm_edge"] for r in seam_results])
        reductions = np.array([r["knee"]["reduction_pct"] for r in seam_results])
        
        def corr(a, b):
            if np.std(a) < 1e-10 or np.std(b) < 1e-10: return 0
            return float(np.corrcoef(a, b)[0, 1])
        
        print(f"\n  Correlations with r_80 (practical working point):")
        print(f"    centroid:    r = {corr(r_80s, centroids):+.3f}")
        print(f"    hf_ratio:    r = {corr(r_80s, hf_ratios):+.3f}")
        print(f"    norm_edge:   r = {corr(r_80s, norm_edges):+.3f}")
        print(f"    reduction%:  r = {corr(r_80s, reductions):+.3f}")
        
        print(f"\n  Correlations with r_knee (steepest descent):")
        print(f"    centroid:    r = {corr(r_knees, centroids):+.3f}")
        print(f"    hf_ratio:    r = {corr(r_knees, hf_ratios):+.3f}")
        print(f"    norm_edge:   r = {corr(r_knees, norm_edges):+.3f}")
        
        # Stats
        print(f"\n  r_knee: mean={r_knees.mean():.3f} std={r_knees.std():.3f} range=[{r_knees.min():.3f}, {r_knees.max():.3f}]")
        print(f"  r_80:   mean={r_80s.mean():.3f} std={r_80s.std():.3f} range=[{r_80s.min():.3f}, {r_80s.max():.3f}]")
        print(f"  r_95:   mean={r_95s.mean():.3f} std={r_95s.std():.3f} range=[{r_95s.min():.3f}, {r_95s.max():.3f}]")
    
    # ── Plots ──
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("A2 ext v2: r_knee / r_80 / r_95 — parametrizing overlap", fontsize=14, fontweight='bold')
    
    # 1: All seam curves
    ax = axes[0, 0]
    for r in results:
        if r["knee"]["has_seam"]:
            ax.plot(r["r_vals"], r["sr_vals"], '-', alpha=0.5, linewidth=1)
    ax.set_xlabel("Overlap ratio r"); ax.set_ylabel("Seam ratio")
    ax.set_title("All seam ratio curves (seam-producing signals)")
    ax.grid(True, alpha=0.3)
    
    # 2: r_80 vs centroid
    ax = axes[0, 1]
    for r in seam_results:
        c = 'blue' if 'freq' in r['name'] else ('orange' if 'mix' in r['name'] else 'green')
        ax.scatter(r["spectral"]["centroid"], r["knee"]["r_80"], c=c, s=60, alpha=0.7, edgecolors='k', lw=0.5)
    ax.set_xlabel("Spectral centroid"); ax.set_ylabel("r_80")
    ax.set_title("r_80 vs spectral centroid"); ax.grid(True, alpha=0.3)
    
    # 3: r_80 vs norm_edge
    ax = axes[0, 2]
    for r in seam_results:
        c = 'blue' if 'freq' in r['name'] else ('orange' if 'mix' in r['name'] else 'green')
        ax.scatter(r["norm_edge"], r["knee"]["r_80"], c=c, s=60, alpha=0.7, edgecolors='k', lw=0.5)
    ax.set_xlabel("Normalized edge amplitude"); ax.set_ylabel("r_80")
    ax.set_title("r_80 vs norm edge amplitude"); ax.grid(True, alpha=0.3)
    
    # 4: r_knee distribution
    ax = axes[1, 0]
    knees = [r["knee"]["r_knee"] for r in seam_results]
    ax.hist(knees, bins=15, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(knees), color='red', ls='--', label=f'mean={np.mean(knees):.3f}')
    ax.set_xlabel("r_knee"); ax.set_ylabel("Count")
    ax.set_title("Distribution of r_knee"); ax.legend(); ax.grid(True, alpha=0.3)
    
    # 5: r_80 distribution
    ax = axes[1, 1]
    r80s = [r["knee"]["r_80"] for r in seam_results]
    ax.hist(r80s, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(np.mean(r80s), color='red', ls='--', label=f'mean={np.mean(r80s):.3f}')
    ax.set_xlabel("r_80"); ax.set_ylabel("Count")
    ax.set_title("Distribution of r_80"); ax.legend(); ax.grid(True, alpha=0.3)
    
    # 6: Selected curves with knee/r80/r95 marked
    ax = axes[1, 2]
    show = [("orig_smooth", 'green'), ("orig_medium", 'orange'), ("freq_2", 'blue'), ("freq_14", 'steelblue')]
    for name, color in show:
        for r in results:
            if r["name"] == name and r["knee"]["has_seam"]:
                ax.plot(r["r_vals"], r["sr_vals"], 'o-', color=color, label=name, markersize=3)
                k = r["knee"]
                ax.axvline(k["r_knee"], color=color, ls=':', alpha=0.5)
                ax.axvline(k["r_80"], color=color, ls='--', alpha=0.5)
    ax.set_xlabel("Overlap ratio r"); ax.set_ylabel("Seam ratio")
    ax.set_title("Selected curves (dotted=knee, dashed=r_80)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "phase1_a2ext_v2.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save
    def conv(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating, np.float64)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (bool, np.bool_)): return bool(o)
        if isinstance(o, dict): return {k: conv(v) for k, v in o.items()}
        if isinstance(o, list): return [conv(v) for v in o]
        return o
    
    with open(os.path.join(outdir, "phase1_a2ext_v2.json"), 'w') as f:
        json.dump(conv([{k:v for k,v in r.items() if k != 'sr_vals'} for r in results]), f, indent=2)
    
    print(f"\nPlots & data saved to {outdir}/")


if __name__ == "__main__":
    main()
