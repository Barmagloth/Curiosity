#!/usr/bin/env python3
"""
Curiosity — Local Seam Metric v2

v1 → v2 fixes:
  - Кольца убиты. Теперь edge strips: пары пикселей (B_in, B_out) через halo boundary.
  - Guard для w=0: SeamScore = Jump_out_after / (Jump_out_before + eps), ΔSeam = SS - 1.
  - Для w>0: SeamScore = Jump_out / (Jump_in + eps) как раньше, но Jump_in = 
    разности по парам на 1 шаг глубже внутрь halo (inner strip), не кольца.
  - Robust stat: median для Jump_out/Jump_in.
  - L∞: p95 вместо max.
  - Клиппинг: B_in/B_out строго по halo-маске, не выходят за патч.

Эксперименты:
  S1: SeamScore разделяет hard insert vs halo insert (sweep w=0..6)
  S2: ΔSeam монотонен по w (регрессия v1: немонотонность из-за колец)
  S3: Dual check ловит soap (gain + seam)
  S4: L2 vs L∞(p95)
  S5: Auto-w через ΔSeam minimum
"""

import numpy as np
import json
from pathlib import Path

GRID = 128; TILE = 4; NT = GRID // TILE


# ─── Signal ───

def make_signal(name, seed=42):
    rng = np.random.RandomState(seed)
    x = np.linspace(0, 1, GRID, endpoint=False)
    xx, yy = np.meshgrid(x, x)
    if name == "smooth":
        return 0.5*np.sin(2*np.pi*2*xx)*np.cos(2*np.pi*1.5*yy) + rng.randn(GRID,GRID)*0.01
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
            c[ti*TILE:(ti+1)*TILE, tj*TILE:(tj+1)*TILE] = gt[ti*TILE:(ti+1)*TILE, tj*TILE:(tj+1)*TILE].mean()
    return c


# ─── Refine ───

def refine_tile(state, gt, ti, tj, w, decay=1.0):
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

def soap_refine(state, ti, tj, w):
    """Soap: blur toward local mean (good seam, bad interior)."""
    r0, c0 = ti*TILE, tj*TILE
    er0, er1 = max(r0-w, 0), min(r0+TILE+w, GRID)
    ec0, ec1 = max(c0-w, 0), min(c0+TILE+w, GRID)
    region_mean = state[er0:er1, ec0:ec1].mean()
    delta = (region_mean - state[er0:er1, ec0:ec1]) * 0.5
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


# ─── Edge strips seam metric ───

def get_boundary_pairs(ti, tj, w):
    """Get (B_in, B_out) pixel pairs at halo boundary.
    
    B_in: pixels on the outer ring of halo (last layer before outside).
    B_out: pixels just outside halo (first layer of outside).
    
    Returns list of ((r_in, c_in), (r_out, c_out)) pairs.
    Also returns inner pairs: ((r_deeper, c_deeper), (r_in, c_in)) 
    for Jump_in normalization.
    """
    r0, c0 = ti * TILE, tj * TILE
    r1, c1 = r0 + TILE, c0 + TILE
    
    # Halo region: [r0-w, r1+w) x [c0-w, c1+w), clipped to grid
    hr0 = max(r0 - w, 0)
    hr1 = min(r1 + w, GRID)
    hc0 = max(c0 - w, 0)
    hc1 = min(c1 + w, GRID)
    
    boundary_pairs = []  # (inside_halo, outside_halo)
    inner_pairs = []     # (deeper_inside, boundary_inside)
    
    # Four edges of the halo rectangle
    # Top edge: row = hr0
    if hr0 > 0:  # there IS an outside above
        for c in range(hc0, hc1):
            boundary_pairs.append(((hr0, c), (hr0 - 1, c)))
            if hr0 + 1 < hr1:
                inner_pairs.append(((hr0 + 1, c), (hr0, c)))
    
    # Bottom edge: row = hr1 - 1
    if hr1 < GRID:
        for c in range(hc0, hc1):
            boundary_pairs.append(((hr1 - 1, c), (hr1, c)))
            if hr1 - 2 >= hr0:
                inner_pairs.append(((hr1 - 2, c), (hr1 - 1, c)))
    
    # Left edge: col = hc0
    if hc0 > 0:
        for r in range(hr0, hr1):
            boundary_pairs.append(((r, hc0), (r, hc0 - 1)))
            if hc0 + 1 < hc1:
                inner_pairs.append(((r, hc0 + 1), (r, hc0)))
    
    # Right edge: col = hc1 - 1
    if hc1 < GRID:
        for r in range(hr0, hr1):
            boundary_pairs.append(((r, hc1 - 1), (r, hc1)))
            if hc1 - 2 >= hc0:
                inner_pairs.append(((r, hc1 - 2), (r, hc1 - 1)))
    
    return boundary_pairs, inner_pairs


def compute_jump(state, pairs):
    """Median absolute difference across pairs."""
    if not pairs:
        return 0.0
    diffs = [abs(state[p1] - state[p2]) for p1, p2 in pairs]
    return float(np.median(diffs))


def compute_jump_p95(state, pairs):
    """P95 absolute difference (L∞-like, robust)."""
    if not pairs:
        return 0.0
    diffs = [abs(state[p1] - state[p2]) for p1, p2 in pairs]
    return float(np.percentile(diffs, 95))


def seam_score(state, ti, tj, w, eps=1e-10):
    """SeamScore = Jump_out / (Jump_in + eps).
    
    For w=0: returns Jump_out only (Jump_in undefined).
    """
    if w == 0:
        # w=0: no halo, just measure the boundary jump
        # Use tile boundary vs immediate outside
        bp, _ = get_boundary_pairs(ti, tj, 0)
        # For w=0, "halo boundary" = tile boundary
        # Recompute: tile edges vs neighbors
        r0, c0 = ti*TILE, tj*TILE
        r1, c1 = r0+TILE, c0+TILE
        pairs = []
        if r0 > 0:
            for c in range(c0, c1): pairs.append(((r0, c), (r0-1, c)))
        if r1 < GRID:
            for c in range(c0, c1): pairs.append(((r1-1, c), (r1, c)))
        if c0 > 0:
            for r in range(r0, r1): pairs.append(((r, c0), (r, c0-1)))
        if c1 < GRID:
            for r in range(r0, r1): pairs.append(((r, c1-1), (r, c1)))
        j_out = compute_jump(state, pairs)
        j_out_p95 = compute_jump_p95(state, pairs)
        return j_out, j_out_p95, 0.0, len(pairs)
    
    bp, ip = get_boundary_pairs(ti, tj, w)
    j_out = compute_jump(state, bp)
    j_in = compute_jump(state, ip)
    j_out_p95 = compute_jump_p95(state, bp)
    
    ss = j_out / (j_in + eps)
    return ss, j_out_p95, j_in, len(bp)


def delta_seam(state_before, state_after, ti, tj, w, eps=1e-10):
    """ΔSeam: how much did refine change the seam quality.
    
    For w=0: ΔSeam = Jump_out_after / (Jump_out_before + eps) - 1
      >0 = worsened, <0 = improved, 0 = unchanged.
    
    For w>0: ΔSeam = SS_after - SS_before
      where SS = Jump_out / (Jump_in + eps).
    """
    if w == 0:
        j_out_before, _, _, _ = seam_score(state_before, ti, tj, 0)
        j_out_after, _, _, _ = seam_score(state_after, ti, tj, 0)
        ds = j_out_after / (j_out_before + eps) - 1.0
        return ds, j_out_before, j_out_after
    
    ss_before, _, _, _ = seam_score(state_before, ti, tj, w)
    ss_after, _, _, _ = seam_score(state_after, ti, tj, w)
    return ss_after - ss_before, ss_before, ss_after


def gain_interior(state_before, state_after, gt, ti, tj):
    s = slice(ti*TILE, (ti+1)*TILE)
    c = slice(tj*TILE, (tj+1)*TILE)
    mse_b = np.mean((gt[s,c] - state_before[s,c])**2)
    mse_a = np.mean((gt[s,c] - state_after[s,c])**2)
    return max(mse_b - mse_a, 0.0)


# ─── Experiments ───

def get_active_tiles(gt, coarse, frac=0.08):
    nr = max(1, int(NT*NT*frac))
    scores = sorted([((i,j), np.mean((gt[i*TILE:(i+1)*TILE, j*TILE:(j+1)*TILE] -
                      coarse[i*TILE:(i+1)*TILE, j*TILE:(j+1)*TILE])**2))
                     for i in range(NT) for j in range(NT)], key=lambda x: -x[1])
    return [t[0] for t in scores[:nr]]


def exp_s1s2():
    """S1+S2: SeamScore & ΔSeam vs w, now with edge strips."""
    results = {}
    for sig in ["smooth", "medium", "sharp"]:
        for seed in [42, 137]:
            gt = make_signal(sig, seed)
            coarse = make_coarse(gt)
            active = get_active_tiles(gt, coarse)
            key = f"{sig}_s{seed}"
            results[key] = {}
            
            for w in range(7):
                state = coarse.copy()
                dseams = []; ss_bef = []; ss_aft = []; gains = []
                ss_vals = []; p95_vals = []
                
                for ti, tj in active:
                    sb = state.copy()
                    state = refine_tile(state, gt, ti, tj, w)
                    
                    ds, ssb, ssa = delta_seam(sb, state, ti, tj, w)
                    gi = gain_interior(sb, state, gt, ti, tj)
                    ss, p95, ji, n = seam_score(state, ti, tj, max(w, 1))
                    
                    dseams.append(ds)
                    ss_bef.append(ssb)
                    ss_aft.append(ssa)
                    gains.append(gi)
                    ss_vals.append(ss)
                    p95_vals.append(p95)
                
                results[key][w] = {
                    "dseam_med": float(np.median(dseams)),
                    "dseam_mean": float(np.mean(dseams)),
                    "ss_bef": float(np.median(ss_bef)),
                    "ss_aft": float(np.median(ss_aft)),
                    "ss_p95_med": float(np.median(p95_vals)),
                    "gain": float(np.mean(gains)),
                    "n_worse": int(sum(1 for d in dseams if d > 0.05)),
                    "n_better": int(sum(1 for d in dseams if d < -0.05)),
                    "n": len(active),
                }
    return results


def exp_s3():
    """S3: Dual check — normal vs soap."""
    results = {}
    for sig in ["smooth", "medium", "sharp"]:
        gt = make_signal(sig, 42)
        coarse = make_coarse(gt)
        active = get_active_tiles(gt, coarse)
        w = 3
        
        # Normal
        state_n = coarse.copy()
        n_ds = []; n_gi = []
        for ti, tj in active:
            sb = state_n.copy()
            state_n = refine_tile(state_n, gt, ti, tj, w)
            ds, _, _ = delta_seam(sb, state_n, ti, tj, w)
            gi = gain_interior(sb, state_n, gt, ti, tj)
            n_ds.append(ds); n_gi.append(gi)
        
        # Soap
        state_s = coarse.copy()
        s_ds = []; s_gi = []
        for ti, tj in active:
            sb = state_s.copy()
            state_s = soap_refine(state_s, ti, tj, w)
            ds, _, _ = delta_seam(sb, state_s, ti, tj, w)
            gi = gain_interior(sb, state_s, gt, ti, tj)
            s_ds.append(ds); s_gi.append(gi)
        
        # Dual check: gain > 1e-5 AND ΔSeam < 0.5
        n_pass_n = sum(1 for d, g in zip(n_ds, n_gi) if g > 1e-5 and d < 0.5)
        n_pass_s = sum(1 for d, g in zip(s_ds, s_gi) if g > 1e-5 and d < 0.5)
        
        results[sig] = {
            "normal": {"dseam": float(np.median(n_ds)), "gain": float(np.mean(n_gi)),
                       "pass": n_pass_n, "n": len(active)},
            "soap":   {"dseam": float(np.median(s_ds)), "gain": float(np.mean(s_gi)),
                       "pass": n_pass_s, "n": len(active)},
        }
    return results


def exp_s4():
    """S4: L2 (median) vs L∞ (p95) at seam."""
    results = {}
    for sig in ["smooth", "medium", "sharp"]:
        gt = make_signal(sig, 42)
        coarse = make_coarse(gt)
        active = get_active_tiles(gt, coarse)
        results[sig] = {}
        
        for w in [0, 1, 2, 3, 4, 5]:
            state = coarse.copy()
            l2s = []; p95s = []
            for ti, tj in active:
                state = refine_tile(state, gt, ti, tj, w)
                ss_l2, ss_p95, ji, n = seam_score(state, ti, tj, max(w, 1))
                l2s.append(ss_l2)
                p95s.append(ss_p95)
            
            results[sig][w] = {
                "l2_med": float(np.median(l2s)),
                "p95_med": float(np.median(p95s)),
                "p95_max": float(np.max(p95s)),
            }
    return results


def exp_s5():
    """S5: Auto-w pilot via ΔSeam."""
    results = {}
    for sig in ["smooth", "medium", "sharp"]:
        gt = make_signal(sig, 42)
        coarse = make_coarse(gt)
        active = get_active_tiles(gt, coarse)
        results[sig] = {}
        
        for w in range(7):
            state = coarse.copy()
            dseams = []; gains = []; ss_afts = []
            for ti, tj in active:
                sb = state.copy()
                state = refine_tile(state, gt, ti, tj, w)
                ds, _, ssa = delta_seam(sb, state, ti, tj, max(w, 1))
                gi = gain_interior(sb, state, gt, ti, tj)
                dseams.append(ds); gains.append(gi); ss_afts.append(ssa)
            
            ds_med = float(np.median(dseams))
            results[sig][w] = {
                "dseam": ds_med,
                "ss_aft": float(np.median(ss_afts)),
                "gain": float(np.mean(gains)),
            }
    return results


# ─── Main ───

def main():
    out = Path("/home/claude")
    
    print("=" * 60)
    print("Local Seam Metric v2 — Edge Strips")
    print("=" * 60)
    
    # S1+S2
    print("\n[S1+S2] ΔSeam vs w (edge strips, median)")
    r12 = exp_s1s2()
    for key in sorted(r12.keys()):
        print(f"\n  {key}:")
        for w in sorted(r12[key].keys()):
            d = r12[key][w]
            mono = "✓" if w == 0 or d["dseam_med"] <= r12[key].get(w-1, d)["dseam_med"] + 0.01 else "✗"
            print(f"    w={w}: ΔSeam={d['dseam_med']:+.4f}  SS_aft={d['ss_aft']:.3f}  "
                  f"gain={d['gain']:.6f}  worse={d['n_worse']}/{d['n']}  "
                  f"better={d['n_better']}/{d['n']}  {mono}")
    
    # Monotonicity check
    print("\n  Monotonicity summary (ΔSeam should decrease with w):")
    for key in sorted(r12.keys()):
        ws = sorted(r12[key].keys())
        ds = [r12[key][w]["dseam_med"] for w in ws[1:]]  # skip w=0
        is_mono = all(ds[i] >= ds[i+1] - 0.02 for i in range(len(ds)-1))
        trend = "monotone ✓" if is_mono else "NON-MONOTONE ✗"
        print(f"    {key}: {trend}  [{', '.join(f'{d:+.3f}' for d in ds)}]")
    
    # S3
    print("\n[S3] Dual check")
    r3 = exp_s3()
    for sig, d in r3.items():
        print(f"  {sig}:")
        for mode in ["normal", "soap"]:
            m = d[mode]
            print(f"    {mode:8s}: ΔSeam={m['dseam']:+.4f}  gain={m['gain']:.6f}  "
                  f"pass={m['pass']}/{m['n']}")
    
    # S4
    print("\n[S4] L2(median) vs L∞(p95)")
    r4 = exp_s4()
    for sig, ws in r4.items():
        print(f"  {sig}:")
        for w in sorted(ws.keys()):
            d = ws[w]
            ratio = d["p95_med"] / max(d["l2_med"], 1e-10)
            print(f"    w={w}: L2={d['l2_med']:.4f}  p95={d['p95_med']:.4f}  ratio={ratio:.2f}")
    
    # S5
    print("\n[S5] Auto-w")
    r5 = exp_s5()
    for sig, ws in r5.items():
        ws_list = sorted(ws.keys())
        ds_list = [(w, ws[w]["dseam"]) for w in ws_list if w > 0]
        best_w = min(ds_list, key=lambda x: x[1])[0] if ds_list else 0
        print(f"  {sig}: best_w={best_w}")
        for w in ws_list:
            d = ws[w]
            mark = " ◀" if w == best_w else ""
            print(f"    w={w}: ΔSeam={d['dseam']:+.5f}  SS={d['ss_aft']:.4f}  "
                  f"gain={d['gain']:.6f}{mark}")
    
    # Save
    all_data = {"s1s2": r12, "s3": r3, "s4": r4, "s5": r5}
    with open(out / "seam_metric_v2.json", "w") as f:
        json.dump(all_data, f, indent=2, default=str)
    
    # Plot
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    
    colors = {"smooth": "#1f77b4", "medium": "#ff7f0e", "sharp": "#2ca02c"}
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Local Seam Metric v2 — Edge Strips", fontsize=14, fontweight="bold")
    
    # S1+S2: ΔSeam vs w
    ax = axes[0, 0]
    for key in sorted(r12.keys()):
        sig = key.split("_")[0]
        ws = sorted(r12[key].keys())
        ds = [r12[key][w]["dseam_med"] for w in ws]
        ax.plot(ws, ds, "o-", color=colors.get(sig, "gray"), label=key, ms=4, alpha=0.7)
    ax.axhline(y=0, color="black", ls="--", alpha=0.3, label="ΔSeam=0")
    ax.set_title("S1+S2: ΔSeam vs w"); ax.set_xlabel("w"); ax.set_ylabel("ΔSeam (median)")
    ax.legend(fontsize=6); ax.grid(True, alpha=0.3)
    
    # S1+S2: SS_after vs w
    ax = axes[0, 1]
    for key in sorted(r12.keys()):
        sig = key.split("_")[0]
        ws = sorted(r12[key].keys())
        ss = [r12[key][w]["ss_aft"] for w in ws]
        ax.plot(ws, ss, "s-", color=colors.get(sig, "gray"), label=key, ms=4, alpha=0.7)
    ax.axhline(y=1, color="green", ls="--", alpha=0.5, label="SS=1 (no seam)")
    ax.set_title("S1+S2: SeamScore After vs w"); ax.set_xlabel("w"); ax.set_ylabel("SS")
    ax.legend(fontsize=6); ax.grid(True, alpha=0.3)
    
    # S3: Dual check
    ax = axes[0, 2]
    sigs = list(r3.keys()); x = np.arange(len(sigs)); bw = 0.3
    norm_p = [r3[s]["normal"]["pass"]/r3[s]["normal"]["n"]*100 for s in sigs]
    soap_p = [r3[s]["soap"]["pass"]/r3[s]["soap"]["n"]*100 for s in sigs]
    ax.bar(x-bw/2, norm_p, bw, label="Normal", color="#2ca02c", alpha=0.7)
    ax.bar(x+bw/2, soap_p, bw, label="Soap", color="#d62728", alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(sigs)
    ax.set_ylabel("Pass dual check (%)"); ax.set_title("S3: Dual Check")
    ax.legend(); ax.grid(True, alpha=0.3)
    
    # S5: ΔSeam vs w
    ax = axes[1, 0]
    for sig, ws in r5.items():
        w_list = sorted(ws.keys())
        ax.plot(w_list, [ws[w]["dseam"] for w in w_list], "o-", color=colors[sig], label=sig, ms=4)
    ax.axhline(y=0, color="black", ls="--", alpha=0.3)
    ax.set_title("S5: ΔSeam vs w"); ax.set_xlabel("w"); ax.set_ylabel("ΔSeam")
    ax.legend(); ax.grid(True, alpha=0.3)
    
    # S5: Gain vs w
    ax = axes[1, 1]
    for sig, ws in r5.items():
        w_list = sorted(ws.keys())
        ax.plot(w_list, [ws[w]["gain"] for w in w_list], "s-", color=colors[sig], label=sig, ms=4)
    ax.set_title("S5: Gain vs w"); ax.set_xlabel("w"); ax.set_ylabel("Gain")
    ax.legend(); ax.grid(True, alpha=0.3)
    
    # S4: L2 vs p95
    ax = axes[1, 2]
    for sig, ws in r4.items():
        w_list = sorted(ws.keys())
        ax.plot(w_list, [ws[w]["l2_med"] for w in w_list], "o-", color=colors[sig],
                label=f"{sig} L2", ms=4)
        ax.plot(w_list, [ws[w]["p95_med"] for w in w_list], "s--", color=colors[sig],
                label=f"{sig} p95", ms=4, alpha=0.6)
    ax.axhline(y=1, color="green", ls="--", alpha=0.5)
    ax.set_title("S4: L2 vs p95 SeamScore"); ax.set_xlabel("w"); ax.set_ylabel("Score")
    ax.legend(fontsize=6); ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out / "seam_metric_v2.png", dpi=150, bbox_inches="tight")
    print(f"\n[Saved] seam_metric_v2.png + .json")
    print("\n[Done]")


if __name__ == "__main__":
    main()
