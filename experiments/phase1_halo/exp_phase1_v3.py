#!/usr/bin/env python3
"""
Curiosity — Phase 1 v3: Boundary-aware refinement

Design decisions for v3:
1. A1/A2 use IDEAL delta (no noise). The seam IS the artifact we want to detect.
   Noise masks the seam. Noisy delta is tested in A3 (drift) and later experiments.

2. elements_per_tile = 16 (not 8). With 8 elements, overlap of r=0.1 is 0-1 elements,
   making feather meaningless. With 16, r=0.1 gives 1-2 elements of feather.

3. Seam metric: max |gradient| at tile edges normalized by mean |gradient| interior.
   This directly captures the discontinuity artifact.

4. For A1, the seam happens because: refined tile has (coarse + delta) = signal,
   but adjacent non-refined tile has just coarse. The step at the boundary is
   (signal - coarse) at the edge = delta at the edge. With halo, this step is
   smoothed by the feather ramp.
"""

import numpy as np
import os
from dataclasses import dataclass, field
from typing import List, Dict
import json


@dataclass
class Config:
    n_tiles: int = 32
    elements_per_tile: int = 16  # more elements → meaningful overlap
    
    r_values: List[float] = field(default_factory=lambda: 
        [0.0, 0.05, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3])
    budget_fractions: List[float] = field(default_factory=lambda: [0.1, 0.15, 0.2, 0.3, 0.5])
    
    # For A3 only
    delta_noise_std: float = 0.03
    delta_scale: float = 0.85
    
    boundary_band: int = 2  # elements from tile edge for boundary zone
    seed: int = 42

CFG = Config()


# ─── Signals ─────────────────────────────────────────────────────────────────

def make_coords(n):
    ax = np.linspace(0, 1, n, endpoint=False) + 0.5/n
    return np.meshgrid(ax, ax)

def sig_constant(X, Y): return np.full_like(X, 0.5)

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

SIGNALS = {"constant": sig_constant, "smooth": sig_smooth,
           "medium": sig_medium, "sharp": sig_sharp}


# ─── Coarse ──────────────────────────────────────────────────────────────────

def compute_coarse(signal, block):
    h, w = signal.shape
    bh, bw = h//block, w//block
    small = signal[:bh*block, :bw*block].reshape(bh, block, bw, block).mean(axis=(1,3))
    return np.repeat(np.repeat(small, block, axis=0), block, axis=1)


# ─── Tile selection ──────────────────────────────────────────────────────────

def select_tiles(delta, n_tiles, ts, budget_frac):
    """Deterministic top-k by energy."""
    energy = np.zeros((n_tiles, n_tiles))
    for i in range(n_tiles):
        for j in range(n_tiles):
            energy[i,j] = np.mean(delta[i*ts:(i+1)*ts, j*ts:(j+1)*ts]**2)
    k = max(1, int(budget_frac * n_tiles**2))
    flat = energy.ravel()
    idx = np.argpartition(flat, -k)[-k:]
    mask = np.zeros(n_tiles**2, dtype=bool)
    mask[idx] = True
    return mask.reshape(n_tiles, n_tiles)


# ─── Assembly ────────────────────────────────────────────────────────────────

def assemble_hard(coarse, delta, mask, nt, ts):
    """HARD INSERT: no overlap, no feather, no smoothing whatsoever."""
    out = coarse.copy()
    for i in range(nt):
        for j in range(nt):
            if mask[i,j]:
                out[i*ts:(i+1)*ts, j*ts:(j+1)*ts] += delta[i*ts:(i+1)*ts, j*ts:(j+1)*ts]
    return out


def _cos_ramp(n):
    if n <= 0: return np.array([])
    return 0.5 * (1 - np.cos(np.pi * np.arange(n) / n))


def assemble_halo(coarse, delta, mask, nt, ts, r, blend_base="coarse"):
    """BOUNDARY-AWARE: overlap + cosine feather."""
    overlap = max(0, round(r * ts))
    if overlap == 0:
        return assemble_hard(coarse, delta, mask, nt, ts)
    
    H, W = coarse.shape
    delta_acc = np.zeros((H, W), dtype=np.float64)
    w_acc = np.zeros((H, W), dtype=np.float64)
    
    for i in range(nt):
        for j in range(nt):
            if not mask[i,j]: continue
            
            # Expanded region
            er0 = max(0, i*ts - overlap)
            er1 = min(H, (i+1)*ts + overlap)
            ec0 = max(0, j*ts - overlap)
            ec1 = min(W, (j+1)*ts + overlap)
            
            ph, pw = er1-er0, ec1-ec0
            
            # Weights: 1 in core, cosine ramp in overlap
            wy = np.ones(ph)
            top = i*ts - er0
            if top > 0: wy[:top] = _cos_ramp(top)
            bot = er1 - (i+1)*ts
            if bot > 0: wy[-bot:] = _cos_ramp(bot)[::-1]
            
            wx = np.ones(pw)
            left = j*ts - ec0
            if left > 0: wx[:left] = _cos_ramp(left)
            right = ec1 - (j+1)*ts
            if right > 0: wx[-right:] = _cos_ramp(right)[::-1]
            
            W2d = np.outer(wy, wx)
            delta_acc[er0:er1, ec0:ec1] += W2d * delta[er0:er1, ec0:ec1]
            w_acc[er0:er1, ec0:ec1] += W2d
    
    out = coarse.copy() if blend_base == "coarse" else coarse.copy()
    valid = w_acc > 1e-12
    out[valid] = coarse[valid] + delta_acc[valid] / w_acc[valid]
    return out


# ─── Metrics ─────────────────────────────────────────────────────────────────

def boundary_gradient_stats(output, nt, ts, band):
    """
    Compute gradient magnitude at tile boundaries vs interior.
    Returns: (mean_boundary_grad, mean_interior_grad, max_boundary_grad, seam_ratio)
    """
    H, W = output.shape
    gy = np.abs(np.diff(output, axis=0))
    gx = np.abs(np.diff(output, axis=1))
    
    # Gradient magnitude (approximate, using available sizes)
    h = min(gy.shape[0], gx.shape[0])
    w = min(gy.shape[1], gx.shape[1])
    grad_mag = np.sqrt(gy[:h,:w]**2 + gx[:h,:w]**2)
    
    # Boundary mask
    bmask = np.zeros((h, w), dtype=bool)
    for i in range(1, nt):
        row = i*ts - 1  # gradient straddles this boundary
        if row < h:
            r0, r1 = max(0, row-band), min(h, row+band+1)
            bmask[r0:r1, :] = True
    for j in range(1, nt):
        col = j*ts - 1
        if col < w:
            c0, c1 = max(0, col-band), min(w, col+band+1)
            bmask[:, c0:c1] = True
    
    imask = ~bmask
    
    if bmask.sum() == 0 or imask.sum() == 0:
        return 0, 0, 0, 1.0
    
    mg_b = float(np.mean(grad_mag[bmask]))
    mg_i = float(np.mean(grad_mag[imask]))
    mx_b = float(np.max(grad_mag[bmask]))
    
    return mg_b, mg_i, mx_b, mg_b / (mg_i + 1e-15)


def laplacian_energy(f):
    ly = np.diff(f, n=2, axis=0)
    lx = np.diff(f, n=2, axis=1)
    h, w = min(ly.shape[0], lx.shape[0]), min(ly.shape[1], lx.shape[1])
    return float(np.mean((ly[:h,:w] + lx[:h,:w])**2))


def mse(a, b): return float(np.mean((a-b)**2))

def psnr_val(a, b):
    m = mse(a, b)
    if m < 1e-15: return 100.0
    dr = max(a.max()-a.min(), b.max()-b.min(), 1e-10)
    return float(10*np.log10(dr**2/m))


# ─── A1: Seam detection ─────────────────────────────────────────────────────

def run_A1(cfg):
    print("\n" + "="*70)
    print("A1: Hard insert vs boundary-aware — seam at tile edges")
    print("  Using IDEAL delta (no noise). Seam = step at refined/non-refined boundary.")
    print("="*70)
    
    N = cfg.n_tiles * cfg.elements_per_tile
    X, Y = make_coords(N)
    ts = cfg.elements_per_tile
    band = cfg.boundary_band
    
    results = {}
    
    for sn, sfn in SIGNALS.items():
        signal = sfn(X, Y)
        coarse = compute_coarse(signal, ts)
        delta = signal - coarse  # IDEAL delta
        
        gt_stats = boundary_gradient_stats(signal, cfg.n_tiles, ts, band)
        co_stats = boundary_gradient_stats(coarse, cfg.n_tiles, ts, band)
        
        results[sn] = {"budgets": {}, "gt_seam_ratio": gt_stats[3], "coarse_seam_ratio": co_stats[3]}
        
        print(f"\n  Signal: {sn}  (GT seam_ratio={gt_stats[3]:.3f}, coarse seam_ratio={co_stats[3]:.3f})")
        
        for budget in cfg.budget_fractions:
            mask = select_tiles(delta, cfg.n_tiles, ts, budget)
            
            hard = assemble_hard(coarse, delta, mask, cfg.n_tiles, ts)
            aware = assemble_halo(coarse, delta, mask, cfg.n_tiles, ts, r=0.2)
            
            h_stats = boundary_gradient_stats(hard, cfg.n_tiles, ts, band)
            a_stats = boundary_gradient_stats(aware, cfg.n_tiles, ts, band)
            
            entry = {
                "n_active": int(mask.sum()),
                "hard": {"mean_bgrad": h_stats[0], "mean_igrad": h_stats[1], 
                         "max_bgrad": h_stats[2], "seam_ratio": h_stats[3]},
                "aware": {"mean_bgrad": a_stats[0], "mean_igrad": a_stats[1],
                          "max_bgrad": a_stats[2], "seam_ratio": a_stats[3]},
                "seam_ratio_reduction": h_stats[3] / (a_stats[3] + 1e-15),
                "psnr_hard": psnr_val(hard, signal),
                "psnr_aware": psnr_val(aware, signal),
            }
            results[sn]["budgets"][str(budget)] = entry
            
            print(f"    budget={budget:.0%}  n_active={mask.sum():4d}  "
                  f"seam_ratio: hard={h_stats[3]:.3f} aware={a_stats[3]:.3f} "
                  f"(hard/aware={entry['seam_ratio_reduction']:.2f}x)  "
                  f"max_bgrad: hard={h_stats[2]:.4f} aware={a_stats[2]:.4f}  "
                  f"PSNR={entry['psnr_hard']:.1f}/{entry['psnr_aware']:.1f}")
    
    return results


# ─── A2: r_min sweep ─────────────────────────────────────────────────────────

def run_A2(cfg):
    print("\n" + "="*70)
    print("A2: Overlap ratio sweep — r_min per signal class")
    print("  Using IDEAL delta, budget=20%")
    print("="*70)
    
    N = cfg.n_tiles * cfg.elements_per_tile
    X, Y = make_coords(N)
    ts = cfg.elements_per_tile
    band = cfg.boundary_band
    budget = 0.2
    
    results = {}
    
    for sn, sfn in SIGNALS.items():
        signal = sfn(X, Y)
        coarse = compute_coarse(signal, ts)
        delta = signal - coarse
        mask = select_tiles(delta, cfg.n_tiles, ts, budget)
        
        gt_sr = boundary_gradient_stats(signal, cfg.n_tiles, ts, band)[3]
        
        # Best reference: r=0.3
        best = assemble_halo(coarse, delta, mask, cfg.n_tiles, ts, r=0.3)
        best_sr = boundary_gradient_stats(best, cfg.n_tiles, ts, band)[3]
        
        sweep = {}
        r_min = None
        
        print(f"\n  Signal: {sn}  (GT seam_ratio={gt_sr:.3f}, best(r=0.3) seam_ratio={best_sr:.3f})")
        
        for r in cfg.r_values:
            if r == 0:
                out = assemble_hard(coarse, delta, mask, cfg.n_tiles, ts)
            else:
                out = assemble_halo(coarse, delta, mask, cfg.n_tiles, ts, r=r)
            
            stats = boundary_gradient_stats(out, cfg.n_tiles, ts, band)
            p = psnr_val(out, signal)
            
            # r_min: seam_ratio within 5% of best
            deg = (stats[3] - best_sr) / (best_sr + 1e-15)
            clean = deg < 0.05
            if clean and r_min is None:
                r_min = r
            
            sweep[str(r)] = {"seam_ratio": stats[3], "mean_bgrad": stats[0],
                             "max_bgrad": stats[2], "psnr": p, "degradation": deg}
            
            marker = " ← r_min" if (r == r_min and clean) else ""
            print(f"    r={r:.3f}  seam_ratio={stats[3]:.3f}  max_bgrad={stats[2]:.4f}  "
                  f"deg={deg:+.4f}  PSNR={p:.1f}{marker}")
        
        results[sn] = {"r_min": r_min, "gt_sr": gt_sr, "best_sr": best_sr, "sweep": sweep}
    
    return results


# ─── A3: Drift test ──────────────────────────────────────────────────────────

def run_A3(cfg):
    print("\n" + "="*70)
    print("A3: Blending base — coarse vs output (drift over iterations)")
    print("  Using NOISY delta to simulate imperfect refinement")
    print("="*70)
    
    rng = np.random.default_rng(cfg.seed)
    N = cfg.n_tiles * cfg.elements_per_tile
    X, Y = make_coords(N)
    ts = cfg.elements_per_tile
    band = cfg.boundary_band
    r = 0.2
    budget = 0.2
    n_iters = 5
    
    results = {}
    
    for sn, sfn in SIGNALS.items():
        if sn == "constant": continue
        
        signal = sfn(X, Y)
        coarse = compute_coarse(signal, ts)
        true_delta = signal - coarse
        
        out_c = coarse.copy()  # coarse-base accumulator
        out_o = coarse.copy()  # output-base accumulator
        
        data_c, data_o = [], []
        
        for it in range(n_iters):
            rng_it = np.random.default_rng(cfg.seed + it + 100)
            noise = rng_it.normal(0, cfg.delta_noise_std * (1 + 0.1*it), true_delta.shape)
            imp_delta = true_delta * cfg.delta_scale * (0.7**it) + noise
            
            mask = select_tiles(true_delta, cfg.n_tiles, ts, budget)
            
            # Coarse base
            out_c = assemble_halo(coarse, imp_delta, mask, cfg.n_tiles, ts, r=r)
            
            # Output base: blend on top of previous
            # Need custom logic: use out_o as base instead of coarse
            overlap = round(r * ts)
            H, W = coarse.shape
            d_acc = np.zeros((H,W), dtype=np.float64)
            w_acc = np.zeros((H,W), dtype=np.float64)
            
            for i in range(cfg.n_tiles):
                for j in range(cfg.n_tiles):
                    if not mask[i,j]: continue
                    er0 = max(0, i*ts-overlap)
                    er1 = min(H, (i+1)*ts+overlap)
                    ec0 = max(0, j*ts-overlap)
                    ec1 = min(W, (j+1)*ts+overlap)
                    ph, pw = er1-er0, ec1-ec0
                    
                    wy = np.ones(ph)
                    top = i*ts - er0
                    if top > 0: wy[:top] = _cos_ramp(top)
                    bot = er1 - (i+1)*ts
                    if bot > 0: wy[-bot:] = _cos_ramp(bot)[::-1]
                    wx = np.ones(pw)
                    left = j*ts - ec0
                    if left > 0: wx[:left] = _cos_ramp(left)
                    right = ec1 - (j+1)*ts
                    if right > 0: wx[-right:] = _cos_ramp(right)[::-1]
                    
                    W2d = np.outer(wy, wx)
                    d_acc[er0:er1, ec0:ec1] += W2d * imp_delta[er0:er1, ec0:ec1]
                    w_acc[er0:er1, ec0:ec1] += W2d
            
            valid = w_acc > 1e-12
            out_o_new = out_o.copy()
            out_o_new[valid] = out_o[valid] + d_acc[valid] / w_acc[valid]
            out_o = out_o_new
            
            sc = boundary_gradient_stats(out_c, cfg.n_tiles, ts, band)
            so = boundary_gradient_stats(out_o, cfg.n_tiles, ts, band)
            
            data_c.append({"iter": it, "seam_ratio": sc[3], "psnr": psnr_val(out_c, signal),
                           "mean_bgrad": sc[0], "lap_E": laplacian_energy(out_c - signal)})
            data_o.append({"iter": it, "seam_ratio": so[3], "psnr": psnr_val(out_o, signal),
                           "mean_bgrad": so[0], "lap_E": laplacian_energy(out_o - signal)})
        
        drift = data_o[-1]["seam_ratio"] / (data_c[-1]["seam_ratio"] + 1e-15)
        results[sn] = {"coarse_base": data_c, "output_base": data_o, "drift": drift}
        
        print(f"  {sn:10s}  iters={n_iters}  seam_ratio: coarse={data_c[-1]['seam_ratio']:.3f} "
              f"output={data_o[-1]['seam_ratio']:.3f}  drift={drift:.2f}x  "
              f"PSNR: {data_c[-1]['psnr']:.1f}/{data_o[-1]['psnr']:.1f}")
    
    return results


# ─── Plots ───────────────────────────────────────────────────────────────────

def plot_all(a1, a2, a3, cfg, outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Phase 1 v3: Boundary-aware refinement — seam detection",
                 fontsize=14, fontweight='bold')
    
    # A1: seam_ratio hard vs aware by budget
    ax = axes[0, 0]
    for sn in ["smooth", "medium", "sharp"]:
        bs = sorted(a1[sn]["budgets"].keys(), key=float)
        bv = [float(b) for b in bs]
        sr_h = [a1[sn]["budgets"][b]["hard"]["seam_ratio"] for b in bs]
        sr_a = [a1[sn]["budgets"][b]["aware"]["seam_ratio"] for b in bs]
        ax.plot(bv, sr_h, 's--', label=f"{sn} hard")
        ax.plot(bv, sr_a, 'o-', label=f"{sn} aware")
    ax.axhline(1.0, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel("Budget fraction"); ax.set_ylabel("Seam ratio (boundary/interior grad)")
    ax.set_title("A1: Seam ratio by budget"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    
    # A1: seam_ratio_reduction (hard/aware ratio)
    ax = axes[0, 1]
    for sn in ["smooth", "medium", "sharp"]:
        bs = sorted(a1[sn]["budgets"].keys(), key=float)
        bv = [float(b) for b in bs]
        red = [a1[sn]["budgets"][b]["seam_ratio_reduction"] for b in bs]
        ax.plot(bv, red, 'o-', label=sn)
    ax.axhline(1.0, color='gray', ls='--', alpha=0.5, label='no effect')
    ax.set_xlabel("Budget fraction"); ax.set_ylabel("Hard/Aware seam ratio")
    ax.set_title("A1: Seam reduction by halo"); ax.legend(); ax.grid(True, alpha=0.3)
    
    # A1: constant control — max boundary gradient
    ax = axes[0, 2]
    bs = sorted(a1["constant"]["budgets"].keys(), key=float)
    bv = [float(b) for b in bs]
    mx_h = [a1["constant"]["budgets"][b]["hard"]["max_bgrad"] for b in bs]
    mx_a = [a1["constant"]["budgets"][b]["aware"]["max_bgrad"] for b in bs]
    ax.plot(bv, mx_h, 's-', label="hard", color='red')
    ax.plot(bv, mx_a, 'o-', label="aware", color='green')
    ax.set_xlabel("Budget fraction"); ax.set_ylabel("Max boundary gradient")
    ax.set_title("A1 control: Constant (should be 0)")
    ax.legend(); ax.grid(True, alpha=0.3)
    
    # A2: seam_ratio vs r
    ax = axes[1, 0]
    for sn in SIGNALS:
        rs = sorted(a2[sn]["sweep"].keys(), key=float)
        rv = [float(r) for r in rs]
        sr = [a2[sn]["sweep"][r]["seam_ratio"] for r in rs]
        ax.plot(rv, sr, 'o-', label=f"{sn} (r_min={a2[sn]['r_min']})")
    ax.set_xlabel("Overlap ratio r"); ax.set_ylabel("Seam ratio")
    ax.set_title("A2: r_min sweep"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    
    # A2: max boundary gradient vs r
    ax = axes[1, 1]
    for sn in ["smooth", "medium", "sharp"]:
        rs = sorted(a2[sn]["sweep"].keys(), key=float)
        rv = [float(r) for r in rs]
        mx = [a2[sn]["sweep"][r]["max_bgrad"] for r in rs]
        ax.plot(rv, mx, 'o-', label=sn)
    ax.set_xlabel("Overlap ratio r"); ax.set_ylabel("Max boundary gradient")
    ax.set_title("A2: Peak seam gradient"); ax.legend(); ax.grid(True, alpha=0.3)
    
    # A3: drift
    ax = axes[1, 2]
    for sn in a3:
        its = [e["iter"] for e in a3[sn]["coarse_base"]]
        sr_c = [e["seam_ratio"] for e in a3[sn]["coarse_base"]]
        sr_o = [e["seam_ratio"] for e in a3[sn]["output_base"]]
        ax.plot(its, sr_c, 'o-', label=f"{sn} coarse-base")
        ax.plot(its, sr_o, 's--', label=f"{sn} output-base")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Seam ratio")
    ax.set_title("A3: Drift — coarse vs output base")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "phase1_v3_summary.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Spatial map for sharp signal
    fig, axes2 = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle("A1 spatial: Sharp signal, budget=20%, ideal delta", fontsize=13)
    
    N = cfg.n_tiles * cfg.elements_per_tile
    X, Y = make_coords(N)
    ts = cfg.elements_per_tile
    signal = sig_sharp(X, Y)
    coarse = compute_coarse(signal, ts)
    delta = signal - coarse
    mask = select_tiles(delta, cfg.n_tiles, ts, 0.2)
    hard = assemble_hard(coarse, delta, mask, cfg.n_tiles, ts)
    aware = assemble_halo(coarse, delta, mask, cfg.n_tiles, ts, r=0.2)
    
    # Gradient magnitude maps
    def grad_mag(f):
        gy = np.diff(f, axis=0)
        gx = np.diff(f, axis=1)
        h,w = min(gy.shape[0],gx.shape[0]), min(gy.shape[1],gx.shape[1])
        return np.sqrt(gy[:h,:w]**2 + gx[:h,:w]**2)
    
    gm_gt = grad_mag(signal)
    gm_hard = grad_mag(hard)
    gm_aware = grad_mag(aware)
    
    plots = [
        ("Ground truth", signal, 'viridis'),
        ("Coarse", coarse, 'viridis'),
        ("Active tiles", np.kron(mask.astype(float), np.ones((ts,ts))), 'gray'),
        ("Delta (true)", delta, 'RdBu_r'),
        ("Hard insert", hard, 'viridis'),
        ("Boundary-aware (r=0.2)", aware, 'viridis'),
        ("|∇| Hard", gm_hard, 'hot'),
        ("|∇| Aware", gm_aware, 'hot'),
    ]
    
    for ax, (t, d, cm) in zip(axes2.ravel(), plots):
        im = ax.imshow(d, cmap=cm, aspect='equal')
        ax.set_title(t, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "phase1_v3_spatial.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Plots saved to {outdir}/")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    outdir = "/home/claude/results"
    os.makedirs(outdir, exist_ok=True)
    
    a1 = run_A1(CFG)
    a2 = run_A2(CFG)
    a3 = run_A3(CFG)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print("\nA1 — Seam detection (ideal delta):")
    for sn in SIGNALS:
        if sn == "constant":
            mx = max(a1[sn]["budgets"][b]["hard"]["max_bgrad"] for b in a1[sn]["budgets"])
            print(f"  {sn:10s} (CONTROL): max boundary grad = {mx:.6f} (should be 0 for const signal)")
        else:
            srs = [a1[sn]["budgets"][b]["seam_ratio_reduction"] for b in a1[sn]["budgets"]]
            print(f"  {sn:10s}: seam_ratio(hard/aware) = {min(srs):.2f}–{max(srs):.2f}x "
                  f"({'SEAM DETECTED' if max(srs) > 1.05 else 'NO CLEAR SEAM'})")
    
    print("\nA2 — r_min:")
    for sn in SIGNALS:
        print(f"  {sn:10s}: r_min = {a2[sn]['r_min']}")
    
    print("\nA3 — Drift:")
    for sn in a3:
        print(f"  {sn:10s}: drift = {a3[sn]['drift']:.2f}x")
    
    # Save JSON
    def conv(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, dict): return {k: conv(v) for k, v in o.items()}
        if isinstance(o, list): return [conv(v) for v in o]
        return o
    
    with open(os.path.join(outdir, "phase1_v3_results.json"), 'w') as f:
        json.dump(conv({"A1": a1, "A2": a2, "A3": a3}), f, indent=2)
    
    plot_all(a1, a2, a3, CFG, outdir)
    print(f"\nDone. All in {outdir}/")


if __name__ == "__main__":
    main()
