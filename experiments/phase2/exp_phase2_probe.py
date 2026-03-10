#!/usr/bin/env python3
"""
Curiosity — Phase 2 v3: Probe-budget exploration (B1 + B2)

TP: loud regions dominate exploitation budget; quiet regions have real structure
    but exploitation never reaches them. Probe is needed.
TN: quiet regions are truly empty. Probe wastes budget.
"""
import numpy as np
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Set
from pathlib import Path

GRID = 128; TILE = 4; NT = GRID // TILE; OV = 3
N_STEPS = 25; BUDGET_FRAC = 0.06; PROBE_FRAC = 0.10; N_SEEDS = 7
GAIN_THR = 1e-4; TILE_COST = float((TILE + 2*OV)**2)  # raised from 1e-6 to avoid TN false discoveries

def _quiet_mask():
    x = np.linspace(0,1,GRID,endpoint=False); xx,yy = np.meshgrid(x,x)
    m = np.zeros((GRID,GRID),dtype=bool)
    for cx,cy,r in [(0.2,0.8,0.07),(0.75,0.25,0.06),(0.5,0.6,0.05),(0.85,0.85,0.05)]:
        m |= ((xx-cx)**2+(yy-cy)**2 < r**2)
    return m

def _quiet_tiles():
    m = _quiet_mask(); s = set()
    for ti in range(NT):
        for tj in range(NT):
            if m[ti*TILE:(ti+1)*TILE, tj*TILE:(tj+1)*TILE].any(): s.add((ti,tj))
    return s

QUIET_TILES = _quiet_tiles()

def make_tp(seed):
    rng = np.random.RandomState(seed)
    x = np.linspace(0,1,GRID,endpoint=False); xx,yy = np.meshgrid(x,x)
    # Loud background: sharp structures with high residual
    gt = 1.0*(xx>0.3) + 0.7*(yy>0.45) - 0.5*((xx>0.55)&(yy>0.65)).astype(float)
    gt += 0.4*np.sin(2*np.pi*5*xx)*np.cos(2*np.pi*3*yy)
    # Quiet texture: smooth sub-tile gradient (NOT zero-mean per tile!)
    # Period chosen so variation within tile is real but low amplitude
    quiet_tex = 0.15 * np.sin(2*np.pi*8*xx + rng.uniform(0,2*np.pi))
    quiet_tex *= np.cos(2*np.pi*6*yy + rng.uniform(0,2*np.pi))
    quiet_tex += 0.10 * np.sin(2*np.pi*11*xx) * np.cos(2*np.pi*9*yy)
    quiet_tex += rng.randn(GRID,GRID)*0.02
    gt += _quiet_mask().astype(float) * quiet_tex
    return gt

def make_tn(seed):
    rng = np.random.RandomState(seed)
    x = np.linspace(0,1,GRID,endpoint=False); xx,yy = np.meshgrid(x,x)
    gt = 1.0*(xx>0.3) + 0.7*(yy>0.45) - 0.5*((xx>0.55)&(yy>0.65)).astype(float)
    gt += 0.4*np.sin(2*np.pi*5*xx)*np.cos(2*np.pi*3*yy)
    gt += _quiet_mask().astype(float) * rng.randn(GRID,GRID)*0.0005
    return gt

def make_coarse(gt):
    c = np.zeros_like(gt)
    for ti in range(NT):
        for tj in range(NT):
            c[ti*TILE:(ti+1)*TILE, tj*TILE:(tj+1)*TILE] = gt[ti*TILE:(ti+1)*TILE, tj*TILE:(tj+1)*TILE].mean()
    return c

def refine_tile(state,gt,ti,tj,decay=1.0):
    r0,c0 = ti*TILE, tj*TILE
    er0,er1 = max(r0-OV,0), min(r0+TILE+OV,GRID)
    ec0,ec1 = max(c0-OV,0), min(c0+TILE+OV,GRID)
    delta = (gt[er0:er1,ec0:ec1]-state[er0:er1,ec0:ec1])*decay
    h,w = delta.shape
    mask = np.ones((h,w))
    for i in range(min(OV,h)):
        f=0.5*(1-np.cos(np.pi*(i+0.5)/OV)); mask[i,:]*=f
        if h-1-i!=i: mask[h-1-i,:]*=f
    for j in range(min(OV,w)):
        f=0.5*(1-np.cos(np.pi*(j+0.5)/OV)); mask[:,j]*=f
        if w-1-j!=j: mask[:,w-1-j]*=f
    out = state.copy(); out[er0:er1,ec0:ec1] += delta*mask
    return out

def tile_res(s,g,ti,tj):
    return float(np.mean((g[ti*TILE:(ti+1)*TILE,tj*TILE:(tj+1)*TILE]-s[ti*TILE:(ti+1)*TILE,tj*TILE:(tj+1)*TILE])**2))

def tile_var(s,g,ti,tj):
    return float(np.var(g[ti*TILE:(ti+1)*TILE,tj*TILE:(tj+1)*TILE]-s[ti*TILE:(ti+1)*TILE,tj*TILE:(tj+1)*TILE]))

def tile_gain(sb,sa,g,ti,tj):
    r = slice(ti*TILE,(ti+1)*TILE); c = slice(tj*TILE,(tj+1)*TILE)
    return max(float(np.mean((g[r,c]-sb[r,c])**2)-np.mean((g[r,c]-sa[r,c])**2)),0.0)

def calc_psnr(g,s):
    mse=np.mean((g-s)**2)
    if mse<1e-15: return 80.0
    return float(10*np.log10((g.max()-g.min())**2/mse))

def calc_qpsnr(g,s):
    m=_quiet_mask()
    if not m.any(): return 80.0
    mse=np.mean((g[m]-s[m])**2)
    if mse<1e-15: return 80.0
    return float(10*np.log10((g.max()-g.min())**2/mse))

# Strategies
class Strat:
    def __init__(self,name): self.name=name; self.lc=np.zeros((NT,NT),dtype=int); self.br=np.zeros((NT,NT)); self.bc=np.ones((NT,NT))
    def pick(self,step,state,gt,expl,n): return []
    def upd(self,step,t,g): self.lc[t]=step

class NoProbe(Strat):
    def __init__(self): super().__init__("no_probe")

class Uniform(Strat):
    def __init__(self,rng): super().__init__("uniform"); self.rng=rng
    def pick(self,step,state,gt,expl,n):
        p=[(i,j) for i in range(NT) for j in range(NT) if (i,j) not in expl]
        if not p or n==0: return []
        return [p[i] for i in self.rng.choice(len(p),min(n,len(p)),replace=False)]

class Age(Strat):
    def __init__(self,rng,ttl=5): super().__init__("age"); self.rng=rng; self.ttl=ttl
    def pick(self,step,state,gt,expl,n):
        p=sorted([(i,j,step-self.lc[i,j]) for i in range(NT) for j in range(NT) if (i,j) not in expl],key=lambda x:-x[2])
        return [(x[0],x[1]) for x in p[:n]]

class Uncert(Strat):
    def __init__(self,rng): super().__init__("uncert"); self.rng=rng
    def pick(self,step,state,gt,expl,n):
        p=sorted([((i,j),tile_var(state,gt,i,j)) for i in range(NT) for j in range(NT) if (i,j) not in expl],key=lambda x:-x[1])
        return [x[0] for x in p[:n]]

class Bandit(Strat):
    def __init__(self,rng,c=1.0): super().__init__("bandit"); self.rng=rng; self.c=c
    def pick(self,step,state,gt,expl,n):
        tot=max(self.bc.sum(),1)
        p=sorted([((i,j),self.br[i,j]/self.bc[i,j]+self.c*np.sqrt(np.log(tot)/self.bc[i,j])) for i in range(NT) for j in range(NT) if (i,j) not in expl],key=lambda x:-x[1])
        return [x[0] for x in p[:n]]
    def upd(self,step,t,g): super().upd(step,t,g); self.br[t]+=g/TILE_COST; self.bc[t]+=1

@dataclass
class StepRec:
    step:int; psnr_g:float; psnr_q:float; cost:float
    n_exploit:int; n_probe:int; n_false:int
    probe_gain:float; probe_cost:float; hidden_refined:int; discovered:bool

@dataclass
class Run:
    strat:str; scene:str; seed:int
    steps:List[StepRec]=field(default_factory=list)
    t_discover:int=-1; final_psnr:float=0; final_qpsnr:float=0
    total_cost:float=0; total_false:int=0; probe_roi:float=0

def run_one(scene,strat,seed):
    gt = make_tp(seed) if scene=="TP" else make_tn(seed)
    coarse = make_coarse(gt); state = coarse.copy()
    budget = max(1,int(NT*NT*BUDGET_FRAC))
    strat.lc[:]=0; strat.br[:]=0; strat.bc[:]=1
    res = Run(strat=strat.name,scene=scene,seed=seed)
    cum=0.0; found=False

    for step in range(N_STEPS):
        decay = 1.0/(1.0+step*0.4)
        if strat.name=="no_probe": ne,np_=budget,0
        else: np_=max(1,int(budget*PROBE_FRAC)); ne=budget-np_

        scores=sorted([((i,j),tile_res(state,gt,i,j)) for i in range(NT) for j in range(NT)],key=lambda x:-x[1])
        expl=set(t[0] for t in scores[:ne])
        probes=set(strat.pick(step,state,gt,expl,np_))
        active=expl|probes

        sc=sp_g=sp_c=0.0; nf=nh=0; disc=False
        exploit_gains=[]; probe_gains=[]
        for t in active:
            sb=state.copy()
            state=refine_tile(state,gt,t[0],t[1],decay)
            g=tile_gain(sb,state,gt,t[0],t[1])
            sc+=TILE_COST
            if t in expl: exploit_gains.append(g)
            if t in probes:
                sp_g+=g; sp_c+=TILE_COST; strat.upd(step,t,g)
                probe_gains.append((t,g))
            if t in QUIET_TILES: nh+=1
        # False activation: probe tile gain < 10th percentile of exploitation gains
        # Discovery: probe found quiet tile with gain >= median exploitation gain
        e_p10=np.percentile(exploit_gains,10) if exploit_gains else 0
        e_med=np.median(exploit_gains) if exploit_gains else 0
        for t,g in probe_gains:
            if g < e_p10: nf+=1
            if t in QUIET_TILES and g >= e_med: disc=True
        cum+=sc
        if disc and not found: found=True; res.t_discover=step
        res.steps.append(StepRec(step=step,psnr_g=calc_psnr(gt,state),psnr_q=calc_qpsnr(gt,state),
            cost=cum,n_exploit=len(expl),n_probe=len(probes),n_false=nf,
            probe_gain=sp_g,probe_cost=sp_c,hidden_refined=nh,discovered=found))

    res.final_psnr=res.steps[-1].psnr_g; res.final_qpsnr=res.steps[-1].psnr_q
    res.total_cost=cum; res.total_false=sum(s.n_false for s in res.steps)
    tpg=sum(s.probe_gain for s in res.steps); tpc=sum(s.probe_cost for s in res.steps)
    res.probe_roi=tpg/max(tpc,1e-10)
    return res

def smoke_overlap(seeds=[42,137,2025]):
    """Smoke test: seam ratio = E[grad^2 at seam] / E[grad^2 at GT seam].
    SR=1 → no artificial seam. SR>>1 → hard insert artifact."""
    out={}
    sigs={"smooth":None,"medium":None,"sharp":None}
    x=np.linspace(0,1,GRID,endpoint=False); xx,yy=np.meshgrid(x,x)
    sigs["smooth"]=0.5*np.sin(2*np.pi*2*xx)*np.cos(2*np.pi*1.5*yy)
    sigs["medium"]=0.5*np.sin(2*np.pi*5*xx)*np.cos(2*np.pi*3*yy)+0.3*np.sin(2*np.pi*8*(xx+yy))
    sigs["sharp"]=1.0*(xx>0.3)+0.7*(yy>0.45)-0.5*((xx>0.55)&(yy>0.65)).astype(float)+0.4*np.sin(2*np.pi*5*xx)

    for seed in seeds:
        rng=np.random.RandomState(seed)
        out[seed]={}
        for sig_name,gt_base in sigs.items():
            gt=gt_base+rng.randn(GRID,GRID)*0.02
            coarse=np.zeros_like(gt)
            for ti in range(NT):
                for tj in range(NT):
                    s=slice(ti*TILE,(ti+1)*TILE);c=slice(tj*TILE,(tj+1)*TILE)
                    coarse[s,c]=gt[s,c].mean()
            nr=max(1,int(NT*NT*0.08))
            out[seed][sig_name]={}
            for w in range(6):
                state=coarse.copy()
                scores=sorted([((i,j),tile_res(state,gt,i,j)) for i in range(NT) for j in range(NT)],key=lambda x:-x[1])
                active=set(t[0] for t in scores[:nr])
                for ti,tj in active:
                    r0,c0=ti*TILE,tj*TILE
                    er0,er1=max(r0-w,0),min(r0+TILE+w,GRID)
                    ec0,ec1=max(c0-w,0),min(c0+TILE+w,GRID)
                    d=gt[er0:er1,ec0:ec1]-state[er0:er1,ec0:ec1]
                    h,wd=d.shape;mk=np.ones((h,wd))
                    for ii in range(min(w,h)):
                        f=0.5*(1-np.cos(np.pi*(ii+0.5)/max(w,1)));mk[ii,:]*=f
                        if h-1-ii!=ii:mk[h-1-ii,:]*=f
                    for jj in range(min(w,wd)):
                        f=0.5*(1-np.cos(np.pi*(jj+0.5)/max(w,1)));mk[:,jj]*=f
                        if wd-1-jj!=jj:mk[:,wd-1-jj]*=f
                    state[er0:er1,ec0:ec1]+=d*mk
                # SR = seam_grad / gt_grad at seam pixels
                ss=[];sg=[]
                for i in range(GRID-1):
                    for j in range(GRID):
                        ta,tb=(i//TILE,j//TILE),((i+1)//TILE,j//TILE)
                        if (ta in active)!=(tb in active):
                            ss.append((state[i,j]-state[i+1,j])**2)
                            sg.append((gt[i,j]-gt[i+1,j])**2)
                for i in range(GRID):
                    for j in range(GRID-1):
                        ta,tb=(i//TILE,j//TILE),(i//TILE,(j+1)//TILE)
                        if (ta in active)!=(tb in active):
                            ss.append((state[i,j]-state[i,j+1])**2)
                            sg.append((gt[i,j]-gt[i,j+1])**2)
                sr=np.mean(ss)/max(np.mean(sg),1e-10) if ss else 0
                out[seed][sig_name][w]={"sr":float(sr),"psnr":float(calc_psnr(gt,state))}
    return out

def main():
    odir=Path("/home/claude"); all_runs=[]
    print("="*60); print("Phase 2 v3: Probe Exploration"); print("="*60)

    # Scene check
    gt=make_tp(0); coarse=make_coarse(gt)
    budget=max(1,int(NT*NT*BUDGET_FRAC))
    scores=sorted([((i,j),tile_res(coarse,gt,i,j)) for i in range(NT) for j in range(NT)],key=lambda x:-x[1])
    top=set(t[0] for t in scores[:budget])
    qr=[tile_res(coarse,gt,i,j) for i,j in QUIET_TILES]
    lr=[tile_res(coarse,gt,i,j) for i in range(NT) for j in range(NT) if (i,j) not in QUIET_TILES]
    print(f"\n[Scene] quiet={len(QUIET_TILES)} budget={budget} quiet_in_top={len(QUIET_TILES&top)}")
    print(f"  quiet_res={np.mean(qr):.6f} loud_res={np.mean(lr):.6f} cutoff={scores[budget-1][1]:.6f}")

    # Smoke
    print("\n[Smoke Test] SR = E[grad²_seam] / E[grad²_GT_seam]  (SR=1 ideal)")
    smoke=smoke_overlap()
    for seed,sigs in smoke.items():
        print(f"  seed={seed}:")
        for sig,ws in sigs.items():
            vals=[ws[w]["sr"] for w in range(6)]
            w_opt=int(np.argmin(vals))
            print(f"    {sig:8s}: "+" ".join(f"w{w}={v:.2f}" for w,v in zip(range(6),vals))
                  +f"  min@w={w_opt}({vals[w_opt]:.2f})")

    # Main
    facs=[("no_probe",lambda r:NoProbe()),("uniform",lambda r:Uniform(r)),
          ("age",lambda r:Age(r)),("uncert",lambda r:Uncert(r)),("bandit",lambda r:Bandit(r))]
    for scene in ["TP","TN"]:
        print(f"\n{'─'*50}\n  {scene}\n{'─'*50}")
        for sn,fac in facs:
            ms={"p":[],"q":[],"d":[],"f":[],"r":[]}
            for seed in range(N_SEEDS):
                rng=np.random.RandomState(seed*37+hash(sn)%97)
                r=run_one(scene,fac(rng),seed); all_runs.append(r)
                ms["p"].append(r.final_psnr); ms["q"].append(r.final_qpsnr)
                ms["d"].append(r.t_discover); ms["f"].append(r.total_false); ms["r"].append(r.probe_roi)
            dr=sum(1 for d in ms["d"] if d>=0)/N_SEEDS
            md=np.mean([d for d in ms["d"] if d>=0]) if dr>0 else -1
            print(f"  {sn:12s} PSNR={np.mean(ms['p']):.2f}±{np.std(ms['p']):.2f} qPSNR={np.mean(ms['q']):.2f}"
                  f" disc={dr*100:.0f}%(s{md:.1f}) false={np.mean(ms['f']):.1f} ROI={np.mean(ms['r']):.2e}")

    # Save JSON
    jdata={"config":{"grid":GRID,"tile":TILE,"nt":NT,"ov":OV,"steps":N_STEPS,
        "budget":BUDGET_FRAC,"probe":PROBE_FRAC,"seeds":N_SEEDS,"quiet":len(QUIET_TILES)},
        "smoke":{str(k):v for k,v in smoke.items()},
        "runs":[{"strat":r.strat,"scene":r.scene,"seed":r.seed,"t_discover":r.t_discover,
            "final_psnr":r.final_psnr,"final_qpsnr":r.final_qpsnr,"total_cost":r.total_cost,
            "total_false":r.total_false,"probe_roi":r.probe_roi,
            "steps":[asdict(s) for s in r.steps]} for r in all_runs]}
    with open(odir/"phase2_probe.json","w") as f: json.dump(jdata,f,indent=2,default=str)

    # Plots
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    colors={"no_probe":"#333","uniform":"#1f77b4","age":"#ff7f0e","uncert":"#2ca02c","bandit":"#d62728"}

    fig,axes=plt.subplots(2,3,figsize=(18,11))
    fig.suptitle("Phase 2 v3: Probe Exploration",fontsize=14,fontweight="bold")
    for ci,scene in enumerate(["TP","TN"]):
        sr=[r for r in all_runs if r.scene==scene]
        for row,metric,ylabel in [(0,"psnr_g","Global PSNR"),(1,"psnr_q","Quiet PSNR")]:
            ax=axes[row,ci]
            for sn,_ in facs:
                by_s={}
                for r in [x for x in sr if x.strat==sn]:
                    for s in r.steps: by_s.setdefault(s.step,[]).append(getattr(s,metric))
                steps=sorted(by_s)
                ax.plot(steps,[np.mean(by_s[s]) for s in steps],label=sn,color=colors[sn],lw=1.5)
            ax.set_title(f"{scene}: {ylabel} vs Step"); ax.set_xlabel("Step"); ax.set_ylabel("dB")
            ax.legend(fontsize=7); ax.grid(True,alpha=0.3)

    ax=axes[0,2]
    styles={"smooth":"o-","medium":"s-","sharp":"^-"}
    for seed,sigs in smoke.items():
        for sig,ws in sigs.items():
            w_=sorted(ws.keys())
            ax.plot(w_,[ws[w]["sr"] for w in w_],styles.get(sig,"o-"),
                    label=f"{sig}/s{seed}",ms=3,alpha=0.7)
    ax.axhline(y=1.0,color="green",ls="--",alpha=0.5,label="SR=1 (ideal)")
    ax.set_title("Smoke: Seam Ratio vs Overlap")
    ax.set_xlabel("Overlap"); ax.set_ylabel("SR (state/GT)"); ax.legend(fontsize=5,ncol=2); ax.grid(True,alpha=0.3)

    ax=axes[1,2]; snames=[s[0] for s in facs]; xp=np.arange(len(snames)); bw=0.25
    tp_d=[sum(1 for r in all_runs if r.strat==sn and r.scene=="TP" and r.t_discover>=0)/max(sum(1 for r in all_runs if r.strat==sn and r.scene=="TP"),1)*100 for sn in snames]
    tp_q=[np.mean([r.final_qpsnr for r in all_runs if r.strat==sn and r.scene=="TP"]) for sn in snames]
    tn_f=[np.mean([r.total_false for r in all_runs if r.strat==sn and r.scene=="TN"]) for sn in snames]
    ax.bar(xp-bw,tp_d,bw,label="TP Disc%",color="#2ca02c",alpha=0.7)
    ax.bar(xp,tp_q,bw,label="TP qPSNR",color="#1f77b4",alpha=0.7)
    ax3=ax.twinx(); ax3.bar(xp+bw,tn_f,bw,label="TN False",color="#d62728",alpha=0.7)
    ax.set_xticks(xp); ax.set_xticklabels(snames,fontsize=7,rotation=15)
    ax.set_ylabel("Disc%/qPSNR"); ax3.set_ylabel("False(TN)")
    ax.set_title("Summary"); ax.legend(loc="upper left",fontsize=6); ax3.legend(loc="upper right",fontsize=6)
    ax.grid(True,alpha=0.3)
    plt.tight_layout(); plt.savefig(odir/"phase2_probe.png",dpi=150,bbox_inches="tight"); plt.close()

    fig2,(ax1,ax2)=plt.subplots(1,2,figsize=(14,5))
    for sn_,ax_ in [("TP",ax1),("TN",ax2)]:
        for sn,_ in facs:
            by_s={}
            for r in [x for x in all_runs if x.strat==sn and x.scene==sn_]:
                for s in r.steps: by_s.setdefault(s.step,[]).append(s.hidden_refined)
            steps=sorted(by_s)
            ax_.plot(steps,[np.mean(by_s[s]) for s in steps],label=sn,color=colors[sn],lw=1.5)
        ax_.set_title(f"{sn_}: Quiet Tiles/Step"); ax_.set_xlabel("Step"); ax_.set_ylabel("Count")
        ax_.legend(fontsize=7); ax_.grid(True,alpha=0.3)
    plt.tight_layout(); plt.savefig(odir/"phase2_quiet_timeline.png",dpi=150,bbox_inches="tight"); plt.close()

    print("\n[Done]")

if __name__=="__main__": main()
