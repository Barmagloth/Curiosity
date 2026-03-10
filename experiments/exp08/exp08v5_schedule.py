"""
Exp0.8v5 — Нужен ли schedule? (FIX: larger field so strictness matters)

Проблема v4: 256 тайлов, даже при strictness=0.95 → 13 кандидатов > cap.
Контроллеру нечего регулировать.

Фикс: поле 256×256 → 1024 тайла (32×32).
Target = 6/step, budget = 100.
Теперь strictness 0.90 → ~100 кандидатов, 0.99 → ~10.
Контроллер может работать в диапазоне [0.90, 0.99].
"""

import numpy as np
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, zoom, laplace, uniform_filter
from scipy.stats import rankdata, wilcoxon

TILE_SIZE = 8
FIELD_SIZE = 256
NTS = FIELD_SIZE // TILE_SIZE        # 32
N_TILES = NTS ** 2                    # 1024

N_STEPS = 25
TOTAL_BUDGET = 120
TARGET_PER_STEP = TOTAL_BUDGET / N_STEPS  # 4.8
HARD_CAP = int(3 * TARGET_PER_STEP)       # 14
CORRIDOR_TOL = 0.3
CORRIDOR_LO = TARGET_PER_STEP * (1 - CORRIDOR_TOL)  # 3.36
CORRIDOR_HI = TARGET_PER_STEP * (1 + CORRIDOR_TOL)  # 6.24
OVER_PENALTY_W = 2.0
UNDER_PENALTY_W = 1.0

PROBE_FRAC = 0.1
PROBE_MIN = 1
EMA_ALPHA = 0.3
STRICTNESS_CLAMP = 0.02   # меньше шаг → меньше осцилляций
WARMUP_STEPS = 3
N_SEEDS = 20
SCENES = ["clean", "noise", "spatvar", "shift"]


def phase_weights(step):
    t = step / max(N_STEPS - 1, 1)
    return {"resid": 0.7 - 0.4*t, "var": 0.15 + 0.2*t, "hf": 0.15 + 0.2*t}

FIXED_W = {"resid": 0.5, "var": 0.25, "hf": 0.25}


# ═══════════════════════════════════════════════════════════════
# Scene generation
# ═══════════════════════════════════════════════════════════════

def make_gt(size=FIELD_SIZE, seed=42):
    rng = np.random.RandomState(seed)
    field = np.zeros((size, size), dtype=np.float64)
    Y, X = np.mgrid[:size, :size].astype(np.float64)
    # large gaussians
    for _ in range(5):
        cx, cy = rng.randint(30, size-30, 2)
        sigma = rng.uniform(15, 50)
        field += rng.uniform(0.4, 1.0)*np.exp(-((X-cx)**2+(Y-cy)**2)/(2*sigma**2))
    # small gaussians
    for _ in range(10):
        cx, cy = rng.randint(10, size-10, 2)
        sigma = rng.uniform(3, 10)
        field += rng.uniform(0.2, 0.8)*np.exp(-((X-cx)**2+(Y-cy)**2)/(2*sigma**2))
    # shapes
    rc, rr = rng.randint(40, size-40, 2)
    mask_c = ((X-rc)**2+(Y-rr)**2) < rng.uniform(15, 40)**2
    field[mask_c] += 0.7
    x0, y0 = rng.randint(10, size//2, 2)
    w, h = rng.randint(20, 60, 2)
    field[y0:y0+h, x0:x0+w] += 0.5
    return field


def make_shifting_gt(size, seed, step, total):
    rng = np.random.RandomState(seed)
    field = np.zeros((size, size), dtype=np.float64)
    Y, X = np.mgrid[:size, :size].astype(np.float64)
    for _ in range(5):
        cx, cy = rng.randint(30, size-30, 2)
        sigma = rng.uniform(15, 40)
        field += rng.uniform(0.5, 1.0)*np.exp(-((X-cx)**2+(Y-cy)**2)/(2*sigma**2))
    if step >= total//2:
        rng2 = np.random.RandomState(seed+5000+step)
        for _ in range(15):
            cx, cy = rng2.randint(5, size-5, 2)
            sigma = rng2.uniform(2, 8)
            field += rng2.uniform(0.3, 0.9)*np.exp(-((X-cx)**2+(Y-cy)**2)/(2*sigma**2))
        field += rng2.normal(0, 0.05, field.shape)
    return field


def apply_deg(gt, deg, rng):
    if deg == "clean": return gt.copy()
    elif deg == "noise": return gt + rng.normal(0, 0.15, gt.shape)
    elif deg == "spatvar":
        out = gt.copy()
        mid = gt.shape[1]//2
        out[:,:mid] += rng.normal(0, 0.15, (gt.shape[0], mid))
        out[:,mid:] = gaussian_filter(out[:,mid:], sigma=2.5)
        return out
    return gt.copy()


def make_coarse(field, factor=4):
    small = field[::factor, ::factor]
    return zoom(small, factor, order=1)[:field.shape[0], :field.shape[1]]


def compute_psnr(gt, ap):
    mse = np.mean((gt-ap)**2)
    if mse < 1e-15: return 100.0
    return 10*np.log10(np.max(gt)**2/mse)


def refine_tile(gt, coarse, output, i, j):
    sl = (slice(i*TILE_SIZE,(i+1)*TILE_SIZE), slice(j*TILE_SIZE,(j+1)*TILE_SIZE))
    output[sl] = gt[sl]


# ═══════════════════════════════════════════════════════════════
# Rho
# ═══════════════════════════════════════════════════════════════

def tile_mse(gt, ap):
    errs = np.zeros((NTS, NTS))
    for i in range(NTS):
        for j in range(NTS):
            sl = (slice(i*TILE_SIZE,(i+1)*TILE_SIZE),
                  slice(j*TILE_SIZE,(j+1)*TILE_SIZE))
            errs[i,j] = np.mean((gt[sl]-ap[sl])**2)
    return errs

def tile_means(f):
    c = np.zeros((NTS, NTS))
    for i in range(NTS):
        for j in range(NTS):
            sl = (slice(i*TILE_SIZE,(i+1)*TILE_SIZE),
                  slice(j*TILE_SIZE,(j+1)*TILE_SIZE))
            c[i,j] = f[sl].mean()
    return c

def qnorm(x):
    f = x.ravel()
    if f.max()-f.min() < 1e-12: return np.zeros_like(x)
    return (rankdata(f)/len(f)).reshape(x.shape)

def compute_rho(gt, out, w):
    res = tile_mse(gt, out)
    ct = tile_means(out)
    mf = uniform_filter(ct, 3, mode='reflect')
    mf2 = uniform_filter(ct**2, 3, mode='reflect')
    var = np.maximum(mf2-mf**2, 0)
    hf = tile_means(np.abs(laplace(out)))
    return w["resid"]*qnorm(res) + w["var"]*qnorm(var) + w["hf"]*qnorm(hf)


# ═══════════════════════════════════════════════════════════════
# Cold start calibration
# ═══════════════════════════════════════════════════════════════

def calibrate(rho, already, target):
    vals = rho[~already]
    if len(vals) == 0: return 0.5
    best_q, best_d = 0.5, 1e9
    for q in np.arange(0.50, 0.995, 0.005):
        thr = np.quantile(vals, q)
        cnt = np.sum(vals >= thr)
        d = abs(cnt - target)
        if d < best_d:
            best_d = d
            best_q = q
    return best_q


# ═══════════════════════════════════════════════════════════════
# Tile selection
# ═══════════════════════════════════════════════════════════════

def select_tiles(rho, already, rng, strictness, budget_rem):
    avail = ~already
    vals = rho[avail]
    if len(vals) == 0:
        return [], set(), 0

    thr = np.quantile(vals, strictness)

    cands, below = [], []
    for i in range(NTS):
        for j in range(NTS):
            if already[i,j]: continue
            if rho[i,j] >= thr:
                cands.append((i, j, rho[i,j]))
            else:
                below.append((i, j))

    cands.sort(key=lambda x: -x[2])
    n_above = len(cands)

    n_probe = max(PROBE_MIN, int(TARGET_PER_STEP * PROBE_FRAC))
    max_ex = min(HARD_CAP - n_probe, int(budget_rem) - n_probe)
    max_ex = max(0, max_ex)
    exploit = [(c[0],c[1]) for c in cands[:max_ex]]

    probe_pool = below + [(c[0],c[1]) for c in cands[max_ex:]]
    n_pr = min(n_probe, len(probe_pool), max(0, int(budget_rem)-len(exploit)))
    if n_pr > 0 and probe_pool:
        idx = rng.choice(len(probe_pool), size=n_pr, replace=False)
        probe_tiles = [probe_pool[k] for k in idx]
    else:
        probe_tiles = []

    return exploit + probe_tiles, set(probe_tiles), n_above


# ═══════════════════════════════════════════════════════════════
# Compliance
# ═══════════════════════════════════════════════════════════════

def step_penalty(cost):
    return (max(0, cost - CORRIDOR_HI) * OVER_PENALTY_W
            + max(0, CORRIDOR_LO - cost) * UNDER_PENALTY_W)


# ═══════════════════════════════════════════════════════════════
# Episode runner
# ═══════════════════════════════════════════════════════════════

def run_episode(gt_func, variant, seed, apply_probe=True):
    rng = np.random.RandomState(seed)
    gt0 = gt_func(0)
    coarse = make_coarse(gt0)
    output = coarse.copy()
    already = np.zeros((NTS, NTS), dtype=bool)
    budget_rem = float(TOTAL_BUDGET)

    rho0 = compute_rho(gt0, output, FIXED_W)
    strictness = calibrate(rho0, already, TARGET_PER_STEP)
    ema_cost = TARGET_PER_STEP

    log = {"psnr":[], "cost":[], "strictness":[], "n_above":[],
           "penalty":[], "cumcost":[], "disc_count":[], "cap_trig":[]}

    for step in range(N_STEPS):
        if budget_rem <= 0:
            gt_c = gt_func(step)
            log["psnr"].append(compute_psnr(gt_c, output))
            for k in ["cost","n_above","penalty","disc_count"]:
                log[k].append(0)
            log["strictness"].append(strictness)
            prev = log["cumcost"][-1] if log["cumcost"] else 0
            log["cumcost"].append(prev)
            log["cap_trig"].append(False)
            continue

        gt_c = gt_func(step)
        w = phase_weights(step) if variant == "C" else FIXED_W
        rho = compute_rho(gt_c, output, w)
        selected, probe_set, n_above = select_tiles(
            rho, already, rng, strictness, budget_rem)
        cap_trig = n_above > HARD_CAP

        # discovery
        err_true = tile_mse(gt_c, output)
        err_avail = err_true[~already]
        q75 = np.quantile(err_avail, 0.75) if len(err_avail) > 0 else 0
        disc = sum(1 for ti,tj in probe_set if err_true[ti,tj] >= q75)

        # apply
        applied = 0
        for ti,tj in selected:
            if (ti,tj) in probe_set and not apply_probe:
                continue
            refine_tile(gt_c, coarse, output, ti, tj)
            already[ti,tj] = True
            applied += 1
        cost = applied
        budget_rem -= cost

        # EMA controller (B, C) — after warmup
        if variant in ("B","C") and step >= WARMUP_STEPS:
            ema_cost = EMA_ALPHA * cost + (1-EMA_ALPHA) * ema_cost
            if ema_cost > CORRIDOR_HI:
                strictness = min(0.995, strictness + STRICTNESS_CLAMP)
            elif ema_cost < CORRIDOR_LO:
                strictness = max(0.5, strictness - STRICTNESS_CLAMP)

        log["psnr"].append(compute_psnr(gt_c, output))
        log["cost"].append(cost)
        log["strictness"].append(strictness)
        log["n_above"].append(n_above)
        log["penalty"].append(step_penalty(cost))
        prev = log["cumcost"][-1] if log["cumcost"] else 0
        log["cumcost"].append(prev + cost)
        log["disc_count"].append(disc)
        log["cap_trig"].append(cap_trig)

    # summary
    pw = slice(WARMUP_STEPS, None)
    cpw = log["cost"][pw]
    log["final_psnr"] = log["psnr"][-1]
    log["total_cost"] = sum(log["cost"])
    log["mean_cost_pw"] = np.mean(cpw)
    log["std_cost_pw"] = np.std(cpw)
    log["p95_cost_pw"] = np.percentile(cpw, 95) if len(cpw) else 0
    log["mean_penalty_pw"] = np.mean(log["penalty"][pw])
    log["total_disc"] = sum(log["disc_count"])
    log["cap_count"] = sum(log["cap_trig"])
    log["efficiency"] = log["final_psnr"] / max(log["total_cost"], 1)
    return log


def make_gt_func(scene, seed):
    if scene == "shift":
        return lambda step: make_shifting_gt(FIELD_SIZE, seed, step, N_STEPS)
    gt = make_gt(FIELD_SIZE, seed)
    rng = np.random.RandomState(seed+1000)
    deg = apply_deg(gt, scene, rng)
    return lambda step: deg


# ═══════════════════════════════════════════════════════════════
# Full experiment
# ═══════════════════════════════════════════════════════════════

def run_full():
    out = Path("/home/claude/exp08v5_results")
    out.mkdir(exist_ok=True)
    R = {}
    for sc in SCENES:
        R[sc] = {}
        for v in ["A","B","C"]:
            runs, runs_np = [], []
            for s in range(N_SEEDS):
                gf = make_gt_func(sc, s)
                runs.append(run_episode(gf, v, s+2000, True))
                runs_np.append(run_episode(gf, v, s+2000, False))
            R[sc][v] = {"runs": runs, "runs_np": runs_np}
    return R, out


def ex(R, sc, v, k):
    return [r[k] for r in R[sc][v]["runs"]]


def print_summary(R):
    print(f"{'Sc':<8} {'V':>1} {'PSNR':>7} {'IQR':>5} "
          f"{'Tot':>4} {'MnC':>5} {'Std':>5} {'P95':>5} "
          f"{'Pen':>6} {'Dsc':>3} {'Cap':>3}")
    print("="*62)
    rows = []
    for sc in SCENES:
        for v in ["A","B","C"]:
            psnrs = ex(R,sc,v,"final_psnr")
            r = {
                "sc":sc, "v":v,
                "psnr_med": np.median(psnrs),
                "psnr_iqr": np.subtract(*np.percentile(psnrs,[75,25])),
                "tot": np.median(ex(R,sc,v,"total_cost")),
                "mn": np.median(ex(R,sc,v,"mean_cost_pw")),
                "std": np.median(ex(R,sc,v,"std_cost_pw")),
                "p95": np.median(ex(R,sc,v,"p95_cost_pw")),
                "pen": np.median(ex(R,sc,v,"mean_penalty_pw")),
                "dsc": np.median(ex(R,sc,v,"total_disc")),
                "cap": np.median(ex(R,sc,v,"cap_count")),
                "_psnr": psnrs,
                "_pen": ex(R,sc,v,"mean_penalty_pw"),
                "_std": ex(R,sc,v,"std_cost_pw"),
                "_tot": ex(R,sc,v,"total_cost"),
            }
            rows.append(r)
            print(f"{sc:<8} {v:>1} {r['psnr_med']:>7.2f} {r['psnr_iqr']:>5.2f} "
                  f"{r['tot']:>4.0f} {r['mn']:>5.1f} {r['std']:>5.2f} "
                  f"{r['p95']:>5.1f} {r['pen']:>6.2f} {r['dsc']:>3.0f} "
                  f"{r['cap']:>3.0f}")
        print("-"*62)
    return rows


def stat_verdict(rows):
    print("\n" + "="*65)
    print("ВЕРДИКТ (Wilcoxon, p<0.05)")
    print("="*65)
    vv = []
    for sc in SCENES:
        a = [r for r in rows if r["sc"]==sc and r["v"]=="A"][0]
        b = [r for r in rows if r["sc"]==sc and r["v"]=="B"][0]
        c = [r for r in rows if r["sc"]==sc and r["v"]=="C"][0]

        print(f"\n  [{sc}] Governor B vs A:")
        wb = 0
        for name, ka, kb, better in [
            ("PSNR", "_psnr", "_psnr", "higher"),
            ("Penalty", "_pen", "_pen", "lower"),
            ("StdCost", "_std", "_std", "lower"),
        ]:
            da, db = np.array(a[ka]), np.array(b[kb])
            diff = db - da
            if np.all(diff == 0):
                print(f"    {name}: identical")
                continue
            try:
                _, p = wilcoxon(diff)
                med_d = np.median(diff)
                win = (p < 0.05 and
                       ((better=="higher" and med_d > 0) or
                        (better=="lower" and med_d < 0)))
                if win: wb += 1
                print(f"    {name}: Δmed={med_d:+.3f} p={p:.4f} "
                      f"{'✓' if win else '✗'}")
            except ValueError:
                print(f"    {name}: cannot test")

        gv = "B" if wb >= 2 else "A"
        print(f"    → {gv} (wins={wb}/3)")

        print(f"  [{sc}] Policy C vs B:")
        wc = 0
        dp = np.array(c["_psnr"]) - np.array(b["_psnr"])
        if np.all(dp == 0):
            print(f"    PSNR: identical")
        else:
            try:
                _, p = wilcoxon(dp)
                md = np.median(dp)
                win = p < 0.05 and md > 0
                if win: wc += 1
                print(f"    PSNR: Δmed={md:+.3f} p={p:.4f} "
                      f"{'✓' if win else '✗'}")
            except ValueError:
                print(f"    PSNR: cannot test")

        dt = np.median(c["_tot"]) - np.median(b["_tot"])
        overspend = dt > np.median(b["_tot"]) * 0.05
        if overspend: wc -= 1
        print(f"    TotCost Δmed={dt:+.1f} "
              f"{'OVERSPEND' if overspend else 'ok'}")

        pv = "C" if wc >= 1 else "B"
        print(f"    → {pv} (wins={wc})")

        vv.append({"sc":sc, "gov":gv, "pol":pv, "wb":wb, "wc":wc})

    print("\n" + "="*65)
    gn = sum(1 for v in vv if v["gov"]=="B")
    pn = sum(1 for v in vv if v["pol"]=="C")
    print(f"Governor EMA: {gn}/{len(SCENES)}")
    if gn >= len(SCENES)*0.5:
        print("→ EMA-контроллер НУЖЕН.")
    elif gn >= 1:
        print("→ EMA-контроллер ПОЛЕЗЕН для некоторых сцен.")
    else:
        print("→ EMA-контроллер НЕ НУЖЕН.")
    print(f"Phase schedule: {pn}/{len(SCENES)}")
    if pn >= len(SCENES)*0.5:
        print("→ Phase schedule НУЖЕН.")
    elif pn >= 1:
        print("→ Phase schedule ПОЛЕЗЕН.")
    else:
        print("→ Phase schedule НЕ НУЖЕН.")
    return vv


def plot_results(R, out_dir):
    colors = {"A":"#e74c3c","B":"#2ecc71","C":"#3498db"}
    labels = {"A":"Fixed","B":"EMA","C":"EMA+Phase"}
    for sc in SCENES:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"Exp0.8v5 — {sc}", fontsize=14)
        metrics = [
            ("psnr","PSNR","dB"), ("cost","Cost/step","tiles"),
            ("strictness","Strictness","quantile"),
            ("n_above","Candidates","count"),
            ("penalty","Penalty",""), ("cumcost","Cumulative cost","tiles"),
        ]
        for idx, (m, title, yl) in enumerate(metrics):
            ax = axes[idx//3, idx%3]
            for v in ["A","B","C"]:
                mat = np.array([r[m] for r in R[sc][v]["runs"]])
                med = np.median(mat, 0)
                q25 = np.percentile(mat, 25, 0)
                q75 = np.percentile(mat, 75, 0)
                ax.plot(med, color=colors[v], label=labels[v], lw=2)
                ax.fill_between(range(N_STEPS), q25, q75,
                                color=colors[v], alpha=0.15)
            if m == "cost":
                ax.axhline(TARGET_PER_STEP, color='gray', ls='--', alpha=0.5)
                ax.axhspan(CORRIDOR_LO, CORRIDOR_HI, color='gray', alpha=0.08)
            if m == "cumcost":
                ax.axhline(TOTAL_BUDGET, color='gray', ls='--', alpha=0.5)
            ax.axvline(WARMUP_STEPS-0.5, color='orange', ls=':', alpha=0.3)
            ax.set_title(title); ax.set_ylabel(yl)
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(out_dir/f"exp08_{sc}.png", dpi=150, bbox_inches='tight')
        plt.close()

    # probe ablation
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(SCENES)); w = 0.25
    for i, v in enumerate(["A","B","C"]):
        vals = []
        for sc in SCENES:
            pw = [a["final_psnr"]-b["final_psnr"]
                  for a, b in zip(R[sc][v]["runs"], R[sc][v]["runs_np"])]
            vals.append(np.median(pw))
        ax.bar(x+i*w, vals, w, label=labels[v], color=colors[v], alpha=0.8)
    ax.set_xticks(x+w); ax.set_xticklabels(SCENES)
    ax.set_title("Probe contribution (ablation)")
    ax.set_ylabel("ΔPSNR (dB)"); ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(out_dir/"probe_ablation.png", dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    print("="*65)
    print("Exp0.8v5 — Schedule test (256×256, 1024 tiles)")
    print(f"  Budget={TOTAL_BUDGET} Target/step={TARGET_PER_STEP:.1f} "
          f"Cap={HARD_CAP} Corridor=[{CORRIDOR_LO:.1f},{CORRIDOR_HI:.1f}]")
    print("="*65 + "\n")

    R, out = run_full()
    rows = print_summary(R)
    vv = stat_verdict(rows)
    plot_results(R, out)
    with open(out/"summary.json","w") as f:
        json.dump({
            "rows": [{k:v for k,v in r.items() if not k.startswith("_")}
                     for r in rows],
            "verdicts": vv,
        }, f, indent=2, default=str)
    print(f"\nРезультаты: {out}/")
