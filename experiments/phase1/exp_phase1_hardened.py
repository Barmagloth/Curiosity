#!/usr/bin/env python3
"""
Phase 1 — hardened version

Three fixes:
1. Seam metric only at active/inactive boundary (not all tile edges)
2. Sweep by integer overlap widths (2..8 elements), not fractional r
3. A3 with decaying delta per iteration (multi-scale simulation)
"""

import numpy as np
import os
import json


N_TILES = 32
TILE_SIZE = 16
N = N_TILES * TILE_SIZE
SEED = 42
BUDGETS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
# Integer overlap widths in elements
OVERLAP_WIDTHS = [0, 1, 2, 3, 4, 5, 6, 7, 8]


# ─── Signals ─────────────────────────────────────────────────────────────────

def make_coords(n):
    ax = np.linspace(0, 1, n, endpoint=False) + 0.5/n
    return np.meshgrid(ax, ax)

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

def sig_constant(X, Y):
    return np.full_like(X, 0.5)

SIGNALS = {"constant": sig_constant, "smooth": sig_smooth,
           "medium": sig_medium, "sharp": sig_sharp}


# ─── Core ────────────────────────────────────────────────────────────────────

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

def assemble_hard(coarse, delta, mask, nt, ts):
    out = coarse.copy()
    for i in range(nt):
        for j in range(nt):
            if mask[i,j]:
                out[i*ts:(i+1)*ts, j*ts:(j+1)*ts] += delta[i*ts:(i+1)*ts, j*ts:(j+1)*ts]
    return out

def assemble_halo(coarse, delta, mask, nt, ts, overlap_w):
    """Assemble with integer overlap width (in elements)."""
    if overlap_w <= 0:
        return assemble_hard(coarse, delta, mask, nt, ts)
    H, W = coarse.shape
    d_acc = np.zeros((H,W), dtype=np.float64)
    w_acc = np.zeros((H,W), dtype=np.float64)
    for i in range(nt):
        for j in range(nt):
            if not mask[i,j]: continue
            er0, er1 = max(0, i*ts-overlap_w), min(H, (i+1)*ts+overlap_w)
            ec0, ec1 = max(0, j*ts-overlap_w), min(W, (j+1)*ts+overlap_w)
            ph, pw = er1-er0, ec1-ec0
            wy = np.ones(ph); wx = np.ones(pw)
            top = i*ts - er0
            if top > 0: wy[:top] = _cos_ramp(top)
            bot = er1 - (i+1)*ts
            if bot > 0: wy[-bot:] = _cos_ramp(bot)[::-1]
            left = j*ts - ec0
            if left > 0: wx[:left] = _cos_ramp(left)
            right = ec1 - (j+1)*ts
            if right > 0: wx[-right:] = _cos_ramp(right)[::-1]
            W2d = np.outer(wy, wx)
            d_acc[er0:er1,ec0:ec1] += W2d * delta[er0:er1,ec0:ec1]
            w_acc[er0:er1,ec0:ec1] += W2d
    out = coarse.copy()
    valid = w_acc > 1e-12
    out[valid] = coarse[valid] + d_acc[valid] / w_acc[valid]
    return out

def assemble_halo_on_base(base, delta, mask, nt, ts, overlap_w):
    """Assemble with blending relative to arbitrary base (for A3 output-base test)."""
    if overlap_w <= 0:
        out = base.copy()
        for i in range(nt):
            for j in range(nt):
                if mask[i,j]:
                    out[i*ts:(i+1)*ts, j*ts:(j+1)*ts] += delta[i*ts:(i+1)*ts, j*ts:(j+1)*ts]
        return out
    H, W = base.shape
    d_acc = np.zeros((H,W), dtype=np.float64)
    w_acc = np.zeros((H,W), dtype=np.float64)
    for i in range(nt):
        for j in range(nt):
            if not mask[i,j]: continue
            er0, er1 = max(0, i*ts-overlap_w), min(H, (i+1)*ts+overlap_w)
            ec0, ec1 = max(0, j*ts-overlap_w), min(W, (j+1)*ts+overlap_w)
            ph, pw = er1-er0, ec1-ec0
            wy = np.ones(ph); wx = np.ones(pw)
            top = i*ts - er0
            if top > 0: wy[:top] = _cos_ramp(top)
            bot = er1 - (i+1)*ts
            if bot > 0: wy[-bot:] = _cos_ramp(bot)[::-1]
            left = j*ts - ec0
            if left > 0: wx[:left] = _cos_ramp(left)
            right = ec1 - (j+1)*ts
            if right > 0: wx[-right:] = _cos_ramp(right)[::-1]
            W2d = np.outer(wy, wx)
            d_acc[er0:er1,ec0:ec1] += W2d * delta[er0:er1,ec0:ec1]
            w_acc[er0:er1,ec0:ec1] += W2d
    out = base.copy()
    valid = w_acc > 1e-12
    out[valid] = base[valid] + d_acc[valid] / w_acc[valid]
    return out


# ─── FIX 1: Seam metric only at active/inactive boundary ────────────────────

def active_boundary_seam_metric(output, mask, nt, ts, band=1):
    """
    Gradient energy measured ONLY at edges where active tile meets inactive tile.
    This is the actual seam location. Everything else is natural signal gradient.
    
    Returns: (mean_seam_grad, mean_elsewhere_grad, seam_ratio, n_seam_edges)
    """
    H, W = output.shape
    gy = np.abs(np.diff(output, axis=0))
    gx = np.abs(np.diff(output, axis=1))
    h = min(gy.shape[0], gx.shape[0])
    w = min(gy.shape[1], gx.shape[1])
    grad = np.sqrt(gy[:h,:w]**2 + gx[:h,:w]**2)
    
    # Build seam mask: only at active/inactive tile boundaries
    seam_mask = np.zeros((h, w), dtype=bool)
    n_seam_edges = 0
    
    for i in range(nt):
        for j in range(nt):
            if not mask[i, j]:
                continue
            # Check each neighbor
            for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                ni, nj = i+di, j+dj
                if 0 <= ni < nt and 0 <= nj < nt and not mask[ni, nj]:
                    n_seam_edges += 1
                    # Mark the boundary zone
                    if di == -1:  # seam at top of tile i
                        row = i * ts - 1
                        if 0 <= row < h:
                            seam_mask[max(0,row-band):min(h,row+band+1), j*ts:min(w,(j+1)*ts)] = True
                    elif di == 1:  # seam at bottom of tile i
                        row = (i+1)*ts - 1
                        if 0 <= row < h:
                            seam_mask[max(0,row-band):min(h,row+band+1), j*ts:min(w,(j+1)*ts)] = True
                    elif dj == -1:  # seam at left of tile j
                        col = j * ts - 1
                        if 0 <= col < w:
                            seam_mask[i*ts:min(h,(i+1)*ts), max(0,col-band):min(w,col+band+1)] = True
                    elif dj == 1:  # seam at right of tile j
                        col = (j+1)*ts - 1
                        if 0 <= col < w:
                            seam_mask[i*ts:min(h,(i+1)*ts), max(0,col-band):min(w,col+band+1)] = True
    
    elsewhere_mask = ~seam_mask
    
    if seam_mask.sum() == 0 or elsewhere_mask.sum() == 0:
        return 0, 0, 1.0, n_seam_edges
    
    mg_seam = float(np.mean(grad[seam_mask]))
    mg_else = float(np.mean(grad[elsewhere_mask]))
    
    return mg_seam, mg_else, mg_seam / (mg_else + 1e-15), n_seam_edges


def mse(a, b): return float(np.mean((a-b)**2))
def psnr(a, b):
    m = mse(a, b)
    if m < 1e-15: return 100.0
    dr = max(a.max()-a.min(), b.max()-b.min(), 1e-10)
    return float(10*np.log10(dr**2/m))


# ─── A1: Hard vs halo at active/inactive boundary ───────────────────────────

def run_A1(signals, nt, ts):
    print("\n" + "="*100)
    print("A1: Hard insert vs boundary-aware (seam only at active/inactive edges)")
    print("  overlap = 5 elements (working point)")
    print("="*100)
    
    X, Y = make_coords(N)
    overlap_w = 5  # working point
    
    results = {}
    
    print(f"\n{'Signal':10s} {'Budget':>7s} | {'SR hard':>8s} {'SR halo':>8s} {'ratio':>7s} | "
          f"{'seamG_h':>8s} {'seamG_a':>8s} {'elseG':>8s} | {'PSNR_h':>7s} {'PSNR_a':>7s} {'edges':>6s}")
    print("-"*105)
    
    for sn, sfn in signals.items():
        signal = sfn(X, Y)
        coarse = compute_coarse(signal, ts)
        delta = signal - coarse
        
        results[sn] = {}
        
        for budget in BUDGETS:
            mask = select_tiles(delta, nt, ts, budget)
            
            hard = assemble_hard(coarse, delta, mask, nt, ts)
            halo = assemble_halo(coarse, delta, mask, nt, ts, overlap_w)
            
            sg_h, eg_h, sr_h, ne_h = active_boundary_seam_metric(hard, mask, nt, ts)
            sg_a, eg_a, sr_a, ne_a = active_boundary_seam_metric(halo, mask, nt, ts)
            
            ratio = sr_h / (sr_a + 1e-15)
            p_h = psnr(hard, signal)
            p_a = psnr(halo, signal)
            
            results[sn][str(budget)] = {
                "sr_hard": sr_h, "sr_halo": sr_a, "ratio": ratio,
                "seam_grad_hard": sg_h, "seam_grad_halo": sg_a,
                "elsewhere_grad": eg_h,
                "psnr_hard": p_h, "psnr_halo": p_a,
                "n_edges": ne_h,
            }
            
            print(f"  {sn:8s} {budget:6.0%}  | {sr_h:8.3f} {sr_a:8.3f} {ratio:6.2f}x | "
                  f"{sg_h:8.5f} {sg_a:8.5f} {eg_h:8.5f} | {p_h:7.1f} {p_a:7.1f} {ne_h:>6d}")
        print()
    
    return results


# ─── A2: Integer overlap sweep ───────────────────────────────────────────────

def run_A2(signals, nt, ts):
    print("\n" + "="*100)
    print("A2: Integer overlap sweep (elements: 0..8)")
    print("  budget = 20%, seam metric at active/inactive boundary only")
    print("="*100)
    
    X, Y = make_coords(N)
    budget = 0.2
    
    results = {}
    
    for sn, sfn in signals.items():
        signal = sfn(X, Y)
        coarse = compute_coarse(signal, ts)
        delta = signal - coarse
        mask = select_tiles(delta, nt, ts, budget)
        
        sweep = {}
        
        # GT seam ratio (natural)
        _, _, gt_sr, _ = active_boundary_seam_metric(signal, mask, nt, ts)
        
        print(f"\n  {sn} (GT active-boundary seam_ratio = {gt_sr:.3f}):")
        print(f"    {'w':>3s} {'r':>6s} | {'SR':>8s} {'seamG':>8s} {'elseG':>8s} | {'PSNR':>7s} {'reduction':>10s}")
        
        sr_at_0 = None
        sr_at_max = None
        
        for w in OVERLAP_WIDTHS:
            out = assemble_halo(coarse, delta, mask, nt, ts, w)
            sg, eg, sr, _ = active_boundary_seam_metric(out, mask, nt, ts)
            p = psnr(out, signal)
            
            if w == 0: sr_at_0 = sr
            sr_at_max = sr
            
            reduction = ""
            if sr_at_0 is not None and sr_at_0 > sr:
                reduction = f"-{(1 - sr/sr_at_0)*100:.0f}%"
            
            sweep[w] = {"sr": sr, "seam_grad": sg, "else_grad": eg, "psnr": p}
            
            print(f"    {w:3d} {w/ts:6.3f} | {sr:8.3f} {sg:8.5f} {eg:8.5f} | {p:7.1f} {reduction:>10s}")
        
        results[sn] = {"gt_sr": gt_sr, "sweep": sweep}
    
    return results


# ─── A3: Decaying delta per iteration ────────────────────────────────────────

def run_A3(signals, nt, ts):
    print("\n" + "="*100)
    print("A3: Blending base — coarse vs output, with DECAYING delta per iteration")
    print("  Simulates multi-level refinement where deeper levels have weaker deltas")
    print("  overlap = 5 elements, budget = 20%")
    print("="*100)
    
    X, Y = make_coords(N)
    overlap_w = 5
    budget = 0.2
    n_iters = 7
    # Delta decay: each iteration delta is scaled by this factor
    decay_factors = [1.0, 0.5, 0.25, 0.12, 0.06, 0.03, 0.015]
    
    results = {}
    
    for sn, sfn in signals.items():
        if sn == "constant":
            continue
        
        signal = sfn(X, Y)
        coarse = compute_coarse(signal, ts)
        true_delta = signal - coarse
        
        out_c = coarse.copy()  # coarse-base: reset each iter
        out_o = coarse.copy()  # output-base: accumulates
        
        data_c, data_o = [], []
        
        print(f"\n  {sn}:")
        print(f"    {'iter':>4s} {'decay':>6s} | {'SR_c':>7s} {'SR_o':>7s} {'drift':>7s} | "
              f"{'PSNR_c':>7s} {'PSNR_o':>7s} | {'seamG_c':>8s} {'seamG_o':>8s}")
        
        for it in range(n_iters):
            rng_it = np.random.default_rng(SEED + it + 100)
            
            # Delta for this "level": decaying + small noise
            scale = decay_factors[it]
            noise = rng_it.normal(0, 0.01 * scale, true_delta.shape)
            iter_delta = true_delta * scale + noise
            
            # Different tiles each iteration
            mask = select_tiles(true_delta * scale, nt, ts, budget)
            
            # Coarse-base: blend relative to original coarse each time
            out_c = assemble_halo(coarse, iter_delta, mask, nt, ts, overlap_w)
            
            # Output-base: blend relative to previous output
            out_o = assemble_halo_on_base(out_o, iter_delta, mask, nt, ts, overlap_w)
            
            sg_c, eg_c, sr_c, _ = active_boundary_seam_metric(out_c, mask, nt, ts)
            sg_o, eg_o, sr_o, _ = active_boundary_seam_metric(out_o, mask, nt, ts)
            p_c = psnr(out_c, signal)
            p_o = psnr(out_o, signal)
            drift = sr_o / (sr_c + 1e-15)
            
            data_c.append({"iter": it, "decay": scale, "sr": sr_c, "psnr": p_c, "seam_grad": sg_c})
            data_o.append({"iter": it, "decay": scale, "sr": sr_o, "psnr": p_o, "seam_grad": sg_o})
            
            print(f"    {it:4d} {scale:6.3f} | {sr_c:7.3f} {sr_o:7.3f} {drift:6.2f}x | "
                  f"{p_c:7.1f} {p_o:7.1f} | {sg_c:8.5f} {sg_o:8.5f}")
        
        results[sn] = {"coarse_base": data_c, "output_base": data_o}
    
    return results


# ─── Plots ───────────────────────────────────────────────────────────────────

def plot_all(a1, a2, a3, outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Phase 1 hardened: active-boundary seam metric, integer overlap, decaying delta",
                 fontsize=13, fontweight='bold')
    
    # A1: seam ratio hard vs halo
    ax = axes[0, 0]
    for sn in ["smooth", "medium", "sharp"]:
        bs = sorted(a1[sn].keys(), key=float)
        bv = [float(b) for b in bs]
        sr_h = [a1[sn][b]["sr_hard"] for b in bs]
        sr_a = [a1[sn][b]["sr_halo"] for b in bs]
        ax.plot(bv, sr_h, 's--', alpha=0.7, label=f"{sn} hard")
        ax.plot(bv, sr_a, 'o-', alpha=0.7, label=f"{sn} halo")
    ax.axhline(1.0, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel("Budget"); ax.set_ylabel("Active-boundary seam ratio")
    ax.set_title("A1: Seam at active/inactive edges"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    
    # A1: reduction factor
    ax = axes[0, 1]
    for sn in ["smooth", "medium", "sharp"]:
        bs = sorted(a1[sn].keys(), key=float)
        bv = [float(b) for b in bs]
        rat = [a1[sn][b]["ratio"] for b in bs]
        ax.plot(bv, rat, 'o-', label=sn)
    ax.axhline(1.0, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel("Budget"); ax.set_ylabel("Hard / Halo seam ratio")
    ax.set_title("A1: Seam reduction factor"); ax.legend(); ax.grid(True, alpha=0.3)
    
    # A2: integer overlap sweep
    ax = axes[0, 2]
    for sn in ["smooth", "medium", "sharp"]:
        ws = sorted(a2[sn]["sweep"].keys())
        wv = [int(w) for w in ws]
        sr = [a2[sn]["sweep"][w]["sr"] for w in ws]
        ax.plot(wv, sr, 'o-', label=f"{sn} (GT={a2[sn]['gt_sr']:.2f})")
    ax.axhline(1.0, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel("Overlap width (elements)"); ax.set_ylabel("Active-boundary seam ratio")
    ax.set_title("A2: Integer overlap sweep"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    
    # A2: PSNR vs overlap
    ax = axes[1, 0]
    for sn in ["smooth", "medium", "sharp"]:
        ws = sorted(a2[sn]["sweep"].keys())
        wv = [int(w) for w in ws]
        p = [a2[sn]["sweep"][w]["psnr"] for w in ws]
        ax.plot(wv, p, 'o-', label=sn)
    ax.set_xlabel("Overlap width (elements)"); ax.set_ylabel("PSNR")
    ax.set_title("A2: PSNR vs overlap"); ax.legend(); ax.grid(True, alpha=0.3)
    
    # A3: drift with decaying delta
    ax = axes[1, 1]
    for sn in a3:
        iters = [d["iter"] for d in a3[sn]["coarse_base"]]
        sr_c = [d["sr"] for d in a3[sn]["coarse_base"]]
        sr_o = [d["sr"] for d in a3[sn]["output_base"]]
        ax.plot(iters, sr_c, 'o-', label=f"{sn} coarse-base")
        ax.plot(iters, sr_o, 's--', label=f"{sn} output-base")
    ax.set_xlabel("Iteration (delta decays)"); ax.set_ylabel("Active-boundary seam ratio")
    ax.set_title("A3: Drift with decaying delta"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    
    # A3: PSNR drift
    ax = axes[1, 2]
    for sn in a3:
        iters = [d["iter"] for d in a3[sn]["coarse_base"]]
        p_c = [d["psnr"] for d in a3[sn]["coarse_base"]]
        p_o = [d["psnr"] for d in a3[sn]["output_base"]]
        ax.plot(iters, p_c, 'o-', label=f"{sn} coarse-base")
        ax.plot(iters, p_o, 's--', label=f"{sn} output-base")
    ax.set_xlabel("Iteration (delta decays)"); ax.set_ylabel("PSNR")
    ax.set_title("A3: PSNR with decaying delta"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "phase1_hardened.png"), dpi=150, bbox_inches='tight')
    plt.close()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    outdir = "/home/claude/results"
    os.makedirs(outdir, exist_ok=True)
    
    a1 = run_A1(SIGNALS, N_TILES, TILE_SIZE)
    a2 = run_A2(SIGNALS, N_TILES, TILE_SIZE)
    a3 = run_A3(SIGNALS, N_TILES, TILE_SIZE)
    
    # ── Summary ──
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    
    print("\nA1 — Seam at active/inactive boundary (overlap=5 elements):")
    for sn in ["smooth", "medium", "sharp"]:
        ratios = [a1[sn][b]["ratio"] for b in a1[sn]]
        psnr_gains = [a1[sn][b]["psnr_halo"] - a1[sn][b]["psnr_hard"] for b in a1[sn]]
        print(f"  {sn:10s}: reduction = {min(ratios):.2f}–{max(ratios):.2f}x, "
              f"PSNR gain = {min(psnr_gains):+.1f} to {max(psnr_gains):+.1f} dB")
    
    print("\nA2 — Integer overlap sweep (budget=20%):")
    for sn in ["smooth", "medium", "sharp"]:
        ws = sorted(a2[sn]["sweep"].keys())
        sr_0 = a2[sn]["sweep"][0]["sr"]
        gt = a2[sn]["gt_sr"]
        print(f"  {sn:10s}: GT_sr={gt:.3f}, hard(w=0)={sr_0:.3f}")
        for w in ws:
            sr = a2[sn]["sweep"][w]["sr"]
            red = (1 - sr/sr_0) * 100 if sr_0 > 0 else 0
            print(f"    w={w:2d}  sr={sr:.3f}  reduction={red:+.0f}%  PSNR={a2[sn]['sweep'][w]['psnr']:.1f}")
    
    print("\nA3 — Drift with decaying delta:")
    for sn in a3:
        final_c = a3[sn]["coarse_base"][-1]
        final_o = a3[sn]["output_base"][-1]
        drift_sr = final_o["sr"] / (final_c["sr"] + 1e-15)
        drift_psnr = final_o["psnr"] - final_c["psnr"]
        print(f"  {sn:10s}: SR drift = {drift_sr:.2f}x, PSNR diff = {drift_psnr:+.1f} dB "
              f"(coarse={final_c['psnr']:.1f}, output={final_o['psnr']:.1f})")
    
    plot_all(a1, a2, a3, outdir)
    
    def conv(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating, np.float64)): return float(o)
        if isinstance(o, (bool, np.bool_)): return bool(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, dict): return {k: conv(v) for k, v in o.items()}
        if isinstance(o, list): return [conv(v) for v in o]
        return o
    
    with open(os.path.join(outdir, "phase1_hardened.json"), 'w') as f:
        json.dump(conv({"A1": a1, "A2": a2, "A3": a3}), f, indent=2)
    
    print(f"\nAll saved to {outdir}/")


if __name__ == "__main__":
    main()
