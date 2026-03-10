#!/usr/bin/env python3
"""
Curiosity — Seam Metric: Cross-space validation

Проверяем что SeamScore = Jump_out / (Jump_in + eps) работает не только на пикселях.

Пространства:
  T1: 2D scalar grid (baseline, контроль)
  T2: 2D vector-valued grid (dim=32, имитация feature map)
  T3: Irregular graph (k-NN точечное облако)
  T4: Tree hierarchy (узлы дерева с parent/children)

Для каждого:
  - Строим GT, coarse, refine с halo
  - Считаем SeamScore, ΔSeam
  - Hard insert (w=0) vs halo insert (w>0)
  - Dual check (gain + ΔSeam)
  - Soap test

Метрика: SeamScore = Jump_out / (Jump_in + eps)
  Jump = robust_stat(||x[p] - x[q]|| по парам через boundary)
  Норма: L2 для scalar, L2 + L∞(p95) для vector, L2 для graph/tree.
"""

import numpy as np
import json
from pathlib import Path
from collections import defaultdict

EPS = 1e-10


# ═══════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════

def robust_jump(diffs):
    """Median of absolute differences."""
    if len(diffs) == 0: return 0.0
    return float(np.median(diffs))

def robust_jump_p95(diffs):
    if len(diffs) == 0: return 0.0
    return float(np.percentile(diffs, 95))


# ═══════════════════════════════════════════════
# T1: 2D Scalar Grid (baseline)
# ═══════════════════════════════════════════════

class ScalarGrid:
    """2D scalar grid, tile-based refinement."""
    
    def __init__(self, N=64, tile=8, seed=42):
        self.N = N; self.T = tile; self.NT = N // tile
        rng = np.random.RandomState(seed)
        x = np.linspace(0, 1, N, endpoint=False)
        xx, yy = np.meshgrid(x, x)
        self.gt = (0.8*(xx > 0.35) + 0.5*np.sin(2*np.pi*4*xx)*np.cos(2*np.pi*3*yy)
                   + rng.randn(N, N)*0.02)
        self.coarse = self._make_coarse(self.gt)
    
    def _make_coarse(self, gt):
        c = np.zeros_like(gt)
        for ti in range(self.NT):
            for tj in range(self.NT):
                s = slice(ti*self.T, (ti+1)*self.T)
                cs = slice(tj*self.T, (tj+1)*self.T)
                c[s, cs] = gt[s, cs].mean()
        return c
    
    def refine(self, state, ti, tj, w, decay=1.0):
        r0, c0 = ti*self.T, tj*self.T
        er0 = max(r0-w, 0); er1 = min(r0+self.T+w, self.N)
        ec0 = max(c0-w, 0); ec1 = min(c0+self.T+w, self.N)
        delta = (self.gt[er0:er1, ec0:ec1] - state[er0:er1, ec0:ec1]) * decay
        h, wd = delta.shape
        mask = np.ones((h, wd))
        if w > 0:
            for i in range(min(w, h)):
                f = 0.5*(1-np.cos(np.pi*(i+0.5)/w))
                mask[i,:]*=f
                if h-1-i!=i: mask[h-1-i,:]*=f
            for j in range(min(w, wd)):
                f = 0.5*(1-np.cos(np.pi*(j+0.5)/w))
                mask[:,j]*=f
                if wd-1-j!=j: mask[:,wd-1-j]*=f
        out = state.copy()
        out[er0:er1, ec0:ec1] += delta * mask
        return out
    
    def soap(self, state, ti, tj, w):
        r0, c0 = ti*self.T, tj*self.T
        er0 = max(r0-w, 0); er1 = min(r0+self.T+w, self.N)
        ec0 = max(c0-w, 0); ec1 = min(c0+self.T+w, self.N)
        m = state[er0:er1, ec0:ec1].mean()
        delta = (m - state[er0:er1, ec0:ec1]) * 0.5
        h, wd = delta.shape
        mask = np.ones((h, wd))
        if w > 0:
            for i in range(min(w, h)):
                f = 0.5*(1-np.cos(np.pi*(i+0.5)/w))
                mask[i,:]*=f
                if h-1-i!=i: mask[h-1-i,:]*=f
            for j in range(min(w, wd)):
                f = 0.5*(1-np.cos(np.pi*(j+0.5)/w))
                mask[:,j]*=f
                if wd-1-j!=j: mask[:,wd-1-j]*=f
        out = state.copy()
        out[er0:er1, ec0:ec1] += delta * mask
        return out
    
    def boundary_pairs(self, ti, tj, w):
        """Returns (boundary_pairs, inner_pairs) as index tuples."""
        r0, c0 = ti*self.T, tj*self.T
        hr0 = max(r0-w, 0); hr1 = min(r0+self.T+w, self.N)
        hc0 = max(c0-w, 0); hc1 = min(c0+self.T+w, self.N)
        bp = []; ip = []
        if w == 0:
            r1, c1 = r0+self.T, c0+self.T
            if r0 > 0:
                for c in range(c0,c1): bp.append(((r0,c),(r0-1,c)))
            if r1 < self.N:
                for c in range(c0,c1): bp.append(((r1-1,c),(r1,c)))
            if c0 > 0:
                for r in range(r0,r1): bp.append(((r,c0),(r,c0-1)))
            if c1 < self.N:
                for r in range(r0,r1): bp.append(((r,c1-1),(r,c1)))
            return bp, ip
        
        # w > 0: boundary at halo edge, inner = one step deeper
        if hr0 > 0:
            for c in range(hc0,hc1):
                bp.append(((hr0,c),(hr0-1,c)))
                if hr0+1<hr1: ip.append(((hr0+1,c),(hr0,c)))
        if hr1 < self.N:
            for c in range(hc0,hc1):
                bp.append(((hr1-1,c),(hr1,c)))
                if hr1-2>=hr0: ip.append(((hr1-2,c),(hr1-1,c)))
        if hc0 > 0:
            for r in range(hr0,hr1):
                bp.append(((r,hc0),(r,hc0-1)))
                if hc0+1<hc1: ip.append(((r,hc0+1),(r,hc0)))
        if hc1 < self.N:
            for r in range(hr0,hr1):
                bp.append(((r,hc1-1),(r,hc1)))
                if hc1-2>=hc0: ip.append(((r,hc1-2),(r,hc1-1)))
        return bp, ip
    
    def seam_score(self, state, ti, tj, w):
        bp, ip = self.boundary_pairs(ti, tj, w)
        d_out = [abs(state[a]-state[b]) for a,b in bp]
        d_in = [abs(state[a]-state[b]) for a,b in ip] if ip else []
        jo = robust_jump(d_out)
        ji = robust_jump(d_in) if d_in else 0.0
        return jo / (ji + EPS), jo, ji
    
    def gain(self, sb, sa, ti, tj):
        s=slice(ti*self.T,(ti+1)*self.T); c=slice(tj*self.T,(tj+1)*self.T)
        return max(float(np.mean((self.gt[s,c]-sb[s,c])**2)-np.mean((self.gt[s,c]-sa[s,c])**2)),0)
    
    def active_tiles(self, state, frac=0.10):
        nr = max(1, int(self.NT**2 * frac))
        scores = sorted([((i,j), np.mean((self.gt[i*self.T:(i+1)*self.T,j*self.T:(j+1)*self.T]-
                          state[i*self.T:(i+1)*self.T,j*self.T:(j+1)*self.T])**2))
                         for i in range(self.NT) for j in range(self.NT)],key=lambda x:-x[1])
        return [t[0] for t in scores[:nr]]


# ═══════════════════════════════════════════════
# T2: Vector-valued Grid (feature map)
# ═══════════════════════════════════════════════

class VectorGrid:
    """2D grid where each cell holds a vector of dim D."""
    
    def __init__(self, N=32, tile=8, D=32, seed=42):
        self.N = N; self.T = tile; self.NT = N//tile; self.D = D
        rng = np.random.RandomState(seed)
        x = np.linspace(0, 1, N, endpoint=False)
        xx, yy = np.meshgrid(x, x)
        
        # GT: each channel has different spatial structure
        self.gt = np.zeros((N, N, D))
        for d in range(D):
            freq = 2 + d * 0.5
            phase = rng.uniform(0, 2*np.pi)
            amp = 0.5 / (1 + d * 0.1)
            self.gt[:,:,d] = amp * np.sin(2*np.pi*freq*xx + phase) * np.cos(2*np.pi*(freq*0.7)*yy)
            if d % 3 == 0:
                self.gt[:,:,d] += 0.3 * (xx > rng.uniform(0.2, 0.8)).astype(float)
        self.gt += rng.randn(N, N, D) * 0.02
        
        self.coarse = self._make_coarse(self.gt)
    
    def _make_coarse(self, gt):
        c = np.zeros_like(gt)
        for ti in range(self.NT):
            for tj in range(self.NT):
                s=slice(ti*self.T,(ti+1)*self.T); cs=slice(tj*self.T,(tj+1)*self.T)
                c[s,cs,:] = gt[s,cs,:].mean(axis=(0,1), keepdims=True)
        return c
    
    def refine(self, state, ti, tj, w, decay=1.0):
        r0, c0 = ti*self.T, tj*self.T
        er0=max(r0-w,0); er1=min(r0+self.T+w,self.N)
        ec0=max(c0-w,0); ec1=min(c0+self.T+w,self.N)
        delta = (self.gt[er0:er1,ec0:ec1,:] - state[er0:er1,ec0:ec1,:]) * decay
        h,wd = delta.shape[:2]
        mask = np.ones((h,wd,1))
        if w > 0:
            for i in range(min(w,h)):
                f=0.5*(1-np.cos(np.pi*(i+0.5)/w))
                mask[i,:,:]*=f
                if h-1-i!=i: mask[h-1-i,:,:]*=f
            for j in range(min(w,wd)):
                f=0.5*(1-np.cos(np.pi*(j+0.5)/w))
                mask[:,j,:]*=f
                if wd-1-j!=j: mask[:,wd-1-j,:]*=f
        out = state.copy()
        out[er0:er1,ec0:ec1,:] += delta * mask
        return out
    
    def soap(self, state, ti, tj, w):
        r0, c0 = ti*self.T, tj*self.T
        er0=max(r0-w,0); er1=min(r0+self.T+w,self.N)
        ec0=max(c0-w,0); ec1=min(c0+self.T+w,self.N)
        m = state[er0:er1,ec0:ec1,:].mean(axis=(0,1), keepdims=True)
        delta = (m - state[er0:er1,ec0:ec1,:]) * 0.5
        h,wd = delta.shape[:2]
        mask = np.ones((h,wd,1))
        if w > 0:
            for i in range(min(w,h)):
                f=0.5*(1-np.cos(np.pi*(i+0.5)/w))
                mask[i,:,:]*=f
                if h-1-i!=i: mask[h-1-i,:,:]*=f
            for j in range(min(w,wd)):
                f=0.5*(1-np.cos(np.pi*(j+0.5)/w))
                mask[:,j,:]*=f
                if wd-1-j!=j: mask[:,wd-1-j,:]*=f
        out = state.copy()
        out[er0:er1,ec0:ec1,:] += delta * mask
        return out
    
    def boundary_pairs(self, ti, tj, w):
        # Same grid topology as ScalarGrid
        r0, c0 = ti*self.T, tj*self.T
        hr0=max(r0-w,0); hr1=min(r0+self.T+w,self.N)
        hc0=max(c0-w,0); hc1=min(c0+self.T+w,self.N)
        bp=[]; ip=[]
        if w == 0:
            r1,c1 = r0+self.T, c0+self.T
            if r0>0:
                for c in range(c0,c1): bp.append(((r0,c),(r0-1,c)))
            if r1<self.N:
                for c in range(c0,c1): bp.append(((r1-1,c),(r1,c)))
            if c0>0:
                for r in range(r0,r1): bp.append(((r,c0),(r,c0-1)))
            if c1<self.N:
                for r in range(r0,r1): bp.append(((r,c1-1),(r,c1)))
            return bp, ip
        if hr0>0:
            for c in range(hc0,hc1):
                bp.append(((hr0,c),(hr0-1,c)))
                if hr0+1<hr1: ip.append(((hr0+1,c),(hr0,c)))
        if hr1<self.N:
            for c in range(hc0,hc1):
                bp.append(((hr1-1,c),(hr1,c)))
                if hr1-2>=hr0: ip.append(((hr1-2,c),(hr1-1,c)))
        if hc0>0:
            for r in range(hr0,hr1):
                bp.append(((r,hc0),(r,hc0-1)))
                if hc0+1<hc1: ip.append(((r,hc0+1),(r,hc0)))
        if hc1<self.N:
            for r in range(hr0,hr1):
                bp.append(((r,hc1-1),(r,hc1)))
                if hc1-2>=hc0: ip.append(((r,hc1-2),(r,hc1-1)))
        return bp, ip
    
    def _norm(self, state, a, b):
        return np.linalg.norm(state[a[0],a[1],:] - state[b[0],b[1],:])
    
    def _linf(self, state, a, b):
        return np.max(np.abs(state[a[0],a[1],:] - state[b[0],b[1],:]))
    
    def seam_score(self, state, ti, tj, w):
        bp, ip = self.boundary_pairs(ti, tj, w)
        d_out_l2 = [self._norm(state,a,b) for a,b in bp]
        d_in_l2 = [self._norm(state,a,b) for a,b in ip] if ip else []
        d_out_linf = [self._linf(state,a,b) for a,b in bp]
        jo_l2 = robust_jump(d_out_l2); ji_l2 = robust_jump(d_in_l2)
        jo_linf = robust_jump_p95(d_out_linf)
        return jo_l2/(ji_l2+EPS), jo_linf, ji_l2
    
    def gain(self, sb, sa, ti, tj):
        s=slice(ti*self.T,(ti+1)*self.T); c=slice(tj*self.T,(tj+1)*self.T)
        return max(float(np.mean((self.gt[s,c,:]-sb[s,c,:])**2) -
                         np.mean((self.gt[s,c,:]-sa[s,c,:])**2)), 0)
    
    def active_tiles(self, state, frac=0.10):
        nr = max(1, int(self.NT**2 * frac))
        scores = sorted([((i,j), np.mean((self.gt[i*self.T:(i+1)*self.T,j*self.T:(j+1)*self.T,:]-
                          state[i*self.T:(i+1)*self.T,j*self.T:(j+1)*self.T,:])**2))
                         for i in range(self.NT) for j in range(self.NT)],key=lambda x:-x[1])
        return [t[0] for t in scores[:nr]]


# ═══════════════════════════════════════════════
# T3: Irregular Graph (k-NN point cloud)
# ═══════════════════════════════════════════════

class IrregularGraph:
    """Point cloud in R^2 with k-NN edges. Values are scalars.
    
    "Tile" = cluster of nearby points. 
    "Halo" = k-hop neighbors of cluster.
    Boundary = edges between halo and outside.
    """
    
    def __init__(self, n_points=200, k=6, n_clusters=10, seed=42):
        rng = np.random.RandomState(seed)
        self.n = n_points; self.k = k
        
        # Random points in [0,1]^2
        self.pos = rng.rand(n_points, 2)
        
        # k-NN graph
        from scipy.spatial import cKDTree
        tree = cKDTree(self.pos)
        _, idx = tree.query(self.pos, k=k+1)  # +1 for self
        self.neighbors = {i: set(idx[i, 1:]) for i in range(n_points)}
        
        # GT: smooth function + discontinuity
        self.gt = (0.5 * np.sin(4*np.pi*self.pos[:,0]) * np.cos(3*np.pi*self.pos[:,1])
                   + 0.7 * (self.pos[:,0] > 0.4).astype(float)
                   + rng.randn(n_points) * 0.03)
        
        # Coarse: cluster means
        from scipy.cluster.vq import kmeans2
        _, self.labels = kmeans2(self.pos, n_clusters, minit='points', seed=seed)
        self.n_clusters = n_clusters
        
        self.coarse = np.zeros(n_points)
        for c in range(n_clusters):
            mask = self.labels == c
            self.coarse[mask] = self.gt[mask].mean()
    
    def get_cluster_points(self, cluster_id):
        return set(np.where(self.labels == cluster_id)[0])
    
    def get_halo(self, cluster_points, hops=1):
        """h-hop neighbors of cluster that are NOT in cluster."""
        halo = set()
        frontier = set(cluster_points)
        for _ in range(hops):
            new_frontier = set()
            for p in frontier:
                for n in self.neighbors[p]:
                    if n not in cluster_points and n not in halo:
                        new_frontier.add(n)
                        halo.add(n)
            frontier = new_frontier
        return halo
    
    def refine(self, state, cluster_id, hops=1, decay=1.0):
        """Refine cluster + halo with feather."""
        cluster = self.get_cluster_points(cluster_id)
        halo = self.get_halo(cluster, hops)
        
        out = state.copy()
        # Full correction inside cluster
        for p in cluster:
            out[p] = state[p] + (self.gt[p] - state[p]) * decay
        # Faded correction in halo (linear fade by hop distance)
        for p in halo:
            # Approximate fade: distance to cluster center
            fade = 0.3  # simple constant fade for halo
            out[p] = state[p] + (self.gt[p] - state[p]) * decay * fade
        return out
    
    def soap(self, state, cluster_id, hops=1):
        cluster = self.get_cluster_points(cluster_id)
        halo = self.get_halo(cluster, hops)
        all_pts = cluster | halo
        m = np.mean([state[p] for p in all_pts])
        out = state.copy()
        for p in cluster:
            out[p] = state[p] + (m - state[p]) * 0.5
        for p in halo:
            out[p] = state[p] + (m - state[p]) * 0.15
        return out
    
    def boundary_pairs(self, cluster_id, hops):
        """Boundary = edges between halo and outside.
        Inner = edges between cluster and halo."""
        cluster = self.get_cluster_points(cluster_id)
        halo = self.get_halo(cluster, hops)
        outside = set(range(self.n)) - cluster - halo
        
        bp = []  # halo ↔ outside
        ip = []  # cluster ↔ halo
        
        for p in halo:
            for n in self.neighbors[p]:
                if n in outside:
                    bp.append((p, n))
                if n in cluster:
                    ip.append((n, p))  # cluster→halo
        
        return bp, ip
    
    def seam_score(self, state, cluster_id, hops):
        bp, ip = self.boundary_pairs(cluster_id, hops)
        d_out = [abs(state[a] - state[b]) for a, b in bp]
        d_in = [abs(state[a] - state[b]) for a, b in ip] if ip else []
        jo = robust_jump(d_out)
        ji = robust_jump(d_in) if d_in else 0.0
        return jo / (ji + EPS), jo, ji
    
    def gain(self, sb, sa, cluster_id):
        pts = self.get_cluster_points(cluster_id)
        mse_b = np.mean([(self.gt[p]-sb[p])**2 for p in pts])
        mse_a = np.mean([(self.gt[p]-sa[p])**2 for p in pts])
        return max(mse_b - mse_a, 0.0)
    
    def active_clusters(self, state, frac=0.3):
        nr = max(1, int(self.n_clusters * frac))
        scores = []
        for c in range(self.n_clusters):
            pts = self.get_cluster_points(c)
            mse = np.mean([(self.gt[p]-state[p])**2 for p in pts])
            scores.append((c, mse))
        scores.sort(key=lambda x: -x[1])
        return [s[0] for s in scores[:nr]]


# ═══════════════════════════════════════════════
# T4: Tree hierarchy
# ═══════════════════════════════════════════════

class TreeSpace:
    """Binary tree where each node holds a scalar value.
    
    "Tile" = subtree. "Halo" = parent + siblings.
    Boundary = edge between halo and rest of tree.
    """
    
    def __init__(self, depth=6, seed=42):
        rng = np.random.RandomState(seed)
        self.depth = depth
        self.n = 2**depth - 1  # complete binary tree
        
        # GT: depth-dependent structure
        self.gt = np.zeros(self.n)
        for i in range(self.n):
            d = int(np.log2(i + 1))  # depth of node
            self.gt[i] = 0.5 * np.sin(i * 0.3) / (1 + d * 0.2) + rng.randn() * 0.05
            if d >= 3:
                self.gt[i] += 0.3 * ((i % 7) > 3)  # discontinuity
        
        # Coarse: depth-truncated (only first 3 levels resolved)
        self.coarse_depth = 3
        self.coarse = np.zeros(self.n)
        for i in range(self.n):
            d = int(np.log2(i + 1))
            if d < self.coarse_depth:
                self.coarse[i] = self.gt[i]
            else:
                # Average of subtree under ancestor at coarse_depth
                ancestor = i
                while int(np.log2(ancestor + 1)) >= self.coarse_depth:
                    ancestor = (ancestor - 1) // 2
                subtree = self._subtree(ancestor)
                self.coarse[i] = np.mean([self.gt[j] for j in subtree])
    
    def _children(self, i):
        left = 2*i + 1; right = 2*i + 2
        c = []
        if left < self.n: c.append(left)
        if right < self.n: c.append(right)
        return c
    
    def _parent(self, i):
        return (i-1)//2 if i > 0 else None
    
    def _subtree(self, i):
        """All nodes in subtree rooted at i."""
        nodes = [i]; q = [i]
        while q:
            curr = q.pop()
            for c in self._children(curr):
                nodes.append(c); q.append(c)
        return nodes
    
    def _neighbors(self, i):
        """Parent + children + siblings."""
        n = set()
        p = self._parent(i)
        if p is not None:
            n.add(p)
            for c in self._children(p):
                if c != i: n.add(c)
        for c in self._children(i):
            n.add(c)
        return n
    
    def refine(self, state, subtree_root, halo_hops=1, decay=1.0):
        """Refine subtree with halo = h-hop tree neighbors."""
        core = set(self._subtree(subtree_root))
        halo = set()
        frontier = set(core)
        for _ in range(halo_hops):
            nf = set()
            for p in frontier:
                for n in self._neighbors(p):
                    if n not in core and n not in halo:
                        halo.add(n); nf.add(n)
            frontier = nf
        
        out = state.copy()
        for p in core:
            out[p] = state[p] + (self.gt[p] - state[p]) * decay
        for p in halo:
            out[p] = state[p] + (self.gt[p] - state[p]) * decay * 0.3
        return out
    
    def soap(self, state, subtree_root, halo_hops=1):
        core = set(self._subtree(subtree_root))
        halo = set()
        frontier = set(core)
        for _ in range(halo_hops):
            nf = set()
            for p in frontier:
                for n in self._neighbors(p):
                    if n not in core and n not in halo:
                        halo.add(n); nf.add(n)
            frontier = nf
        
        m = np.mean([state[p] for p in core | halo])
        out = state.copy()
        for p in core:
            out[p] = state[p] + (m - state[p]) * 0.5
        for p in halo:
            out[p] = state[p] + (m - state[p]) * 0.15
        return out
    
    def boundary_pairs(self, subtree_root, halo_hops):
        core = set(self._subtree(subtree_root))
        halo = set()
        frontier = set(core)
        for _ in range(halo_hops):
            nf = set()
            for p in frontier:
                for n in self._neighbors(p):
                    if n not in core and n not in halo:
                        halo.add(n); nf.add(n)
            frontier = nf
        outside = set(range(self.n)) - core - halo
        
        bp = []  # halo ↔ outside
        ip = []  # core ↔ halo
        for p in halo:
            for n in self._neighbors(p):
                if n in outside: bp.append((p, n))
                if n in core: ip.append((n, p))
        return bp, ip
    
    def seam_score(self, state, subtree_root, halo_hops):
        bp, ip = self.boundary_pairs(subtree_root, halo_hops)
        d_out = [abs(state[a]-state[b]) for a,b in bp]
        d_in = [abs(state[a]-state[b]) for a,b in ip] if ip else []
        jo = robust_jump(d_out); ji = robust_jump(d_in)
        return jo/(ji+EPS), jo, ji
    
    def gain(self, sb, sa, subtree_root):
        pts = self._subtree(subtree_root)
        mse_b = np.mean([(self.gt[p]-sb[p])**2 for p in pts])
        mse_a = np.mean([(self.gt[p]-sa[p])**2 for p in pts])
        return max(mse_b - mse_a, 0.0)
    
    def active_subtrees(self, state, depth_level=3, frac=0.3):
        """Subtrees rooted at nodes of given depth."""
        roots = [i for i in range(self.n) if int(np.log2(i+1)) == depth_level]
        scores = []
        for r in roots:
            pts = self._subtree(r)
            mse = np.mean([(self.gt[p]-state[p])**2 for p in pts])
            scores.append((r, mse))
        scores.sort(key=lambda x: -x[1])
        nr = max(1, int(len(roots) * frac))
        return [s[0] for s in scores[:nr]]


# ═══════════════════════════════════════════════
# Universal test harness
# ═══════════════════════════════════════════════

def run_test(space_name, space, state_init, active_units, 
             refine_fn, soap_fn, seam_fn, gain_fn, halo_params):
    """Test normal vs soap refine across halo sizes.
    
    Returns dict with results per halo param.
    """
    results = {"name": space_name, "params": {}}
    
    for hp in halo_params:
        state = state_init.copy() if isinstance(state_init, np.ndarray) else state_init.copy()
        
        normal_ds = []; normal_gi = []
        soap_ds = []; soap_gi = []
        ss_normal = []; ss_soap = []
        
        # Normal refine
        state_n = state.copy() if isinstance(state, np.ndarray) else state.copy()
        for unit in active_units:
            sb = state_n.copy()
            state_n = refine_fn(state_n, unit, hp)
            ss_b, _, _ = seam_fn(sb, unit, hp)
            ss_a, _, _ = seam_fn(state_n, unit, hp)
            ds = ss_a - ss_b
            gi = gain_fn(sb, state_n, unit)
            normal_ds.append(ds); normal_gi.append(gi); ss_normal.append(ss_a)
        
        # Soap refine
        state_s = state.copy() if isinstance(state, np.ndarray) else state.copy()
        for unit in active_units:
            sb = state_s.copy()
            state_s = soap_fn(state_s, unit, hp)
            ss_b, _, _ = seam_fn(sb, unit, hp)
            ss_a, _, _ = seam_fn(state_s, unit, hp)
            ds = ss_a - ss_b
            gi = gain_fn(sb, state_s, unit)
            soap_ds.append(ds); soap_gi.append(gi); ss_soap.append(ss_a)
        
        # Dual check
        n_pass_n = sum(1 for d,g in zip(normal_ds, normal_gi) if g > 1e-5 and d < 0.5)
        n_pass_s = sum(1 for d,g in zip(soap_ds, soap_gi) if g > 1e-5 and d < 0.5)
        
        results["params"][hp] = {
            "normal": {
                "dseam": float(np.median(normal_ds)),
                "gain": float(np.mean(normal_gi)),
                "ss": float(np.median(ss_normal)),
                "pass": n_pass_n,
            },
            "soap": {
                "dseam": float(np.median(soap_ds)),
                "gain": float(np.mean(soap_gi)),
                "ss": float(np.median(ss_soap)),
                "pass": n_pass_s,
            },
            "n": len(active_units),
        }
    
    return results


# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════

def main():
    out = Path("/home/claude")
    
    print("=" * 60)
    print("Seam Metric — Cross-Space Validation")
    print("=" * 60)
    
    all_results = {}
    
    # T1: Scalar Grid
    print("\n[T1] 2D Scalar Grid (64x64, tile=8)")
    sg = ScalarGrid(64, 8)
    active = sg.active_tiles(sg.coarse)
    r1 = run_test("scalar_grid", sg, sg.coarse, active,
                  lambda s,u,w: sg.refine(s, u[0], u[1], w),
                  lambda s,u,w: sg.soap(s, u[0], u[1], w),
                  lambda s,u,w: sg.seam_score(s, u[0], u[1], max(w,1)),
                  lambda sb,sa,u: sg.gain(sb, sa, u[0], u[1]),
                  [0, 1, 2, 3])
    all_results["T1"] = r1
    
    for hp in [0,1,2,3]:
        d = r1["params"][hp]
        print(f"  w={hp}: normal(ΔS={d['normal']['dseam']:+.3f} gain={d['normal']['gain']:.5f} pass={d['normal']['pass']}/{d['n']})"
              f"  soap(ΔS={d['soap']['dseam']:+.3f} gain={d['soap']['gain']:.5f} pass={d['soap']['pass']}/{d['n']})")
    
    # T2: Vector Grid
    print("\n[T2] Vector Grid (32x32, tile=8, dim=32)")
    vg = VectorGrid(32, 8, 32)
    active_v = vg.active_tiles(vg.coarse)
    r2 = run_test("vector_grid", vg, vg.coarse, active_v,
                  lambda s,u,w: vg.refine(s, u[0], u[1], w),
                  lambda s,u,w: vg.soap(s, u[0], u[1], w),
                  lambda s,u,w: vg.seam_score(s, u[0], u[1], max(w,1)),
                  lambda sb,sa,u: vg.gain(sb, sa, u[0], u[1]),
                  [0, 1, 2, 3])
    all_results["T2"] = r2
    
    for hp in [0,1,2,3]:
        d = r2["params"][hp]
        print(f"  w={hp}: normal(ΔS={d['normal']['dseam']:+.3f} gain={d['normal']['gain']:.5f} pass={d['normal']['pass']}/{d['n']})"
              f"  soap(ΔS={d['soap']['dseam']:+.3f} gain={d['soap']['gain']:.5f} pass={d['soap']['pass']}/{d['n']})")
    
    # T3: Irregular Graph
    print("\n[T3] Irregular Graph (200 points, k=6, 10 clusters)")
    ig = IrregularGraph(200, 6, 10)
    active_g = ig.active_clusters(ig.coarse)
    r3 = run_test("irregular_graph", ig, ig.coarse, active_g,
                  lambda s,u,h: ig.refine(s, u, h),
                  lambda s,u,h: ig.soap(s, u, h),
                  lambda s,u,h: ig.seam_score(s, u, max(h,1)),
                  lambda sb,sa,u: ig.gain(sb, sa, u),
                  [0, 1, 2])
    all_results["T3"] = r3
    
    for hp in [0,1,2]:
        d = r3["params"][hp]
        print(f"  hops={hp}: normal(ΔS={d['normal']['dseam']:+.3f} gain={d['normal']['gain']:.5f} pass={d['normal']['pass']}/{d['n']})"
              f"  soap(ΔS={d['soap']['dseam']:+.3f} gain={d['soap']['gain']:.5f} pass={d['soap']['pass']}/{d['n']})")
    
    # T4: Tree
    print("\n[T4] Tree hierarchy (depth=6, 63 nodes)")
    ts = TreeSpace(6)
    active_t = ts.active_subtrees(ts.coarse, depth_level=3)
    r4 = run_test("tree", ts, ts.coarse, active_t,
                  lambda s,u,h: ts.refine(s, u, h),
                  lambda s,u,h: ts.soap(s, u, h),
                  lambda s,u,h: ts.seam_score(s, u, max(h,1)),
                  lambda sb,sa,u: ts.gain(sb, sa, u),
                  [0, 1, 2])
    all_results["T4"] = r4
    
    for hp in [0,1,2]:
        d = r4["params"][hp]
        print(f"  hops={hp}: normal(ΔS={d['normal']['dseam']:+.3f} gain={d['normal']['gain']:.5f} pass={d['normal']['pass']}/{d['n']})"
              f"  soap(ΔS={d['soap']['dseam']:+.3f} gain={d['soap']['gain']:.5f} pass={d['soap']['pass']}/{d['n']})")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Dual Check (pass rates)")
    print("=" * 60)
    print(f"{'Space':20s} {'Halo':6s} {'Normal':8s} {'Soap':8s} {'Separates?':10s}")
    print("-" * 55)
    for tkey in ["T1", "T2", "T3", "T4"]:
        r = all_results[tkey]
        for hp in sorted(r["params"].keys()):
            d = r["params"][hp]
            n = d["n"]
            np_ = d["normal"]["pass"]; sp = d["soap"]["pass"]
            sep = "YES" if np_ > sp + 1 else ("WEAK" if np_ > sp else "NO")
            print(f"  {r['name']:18s} {hp:6d} {np_:4d}/{n:<3d} {sp:4d}/{n:<3d} {sep}")
    
    # Save
    with open(out / "seam_crossspace.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Plot
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Seam Metric — Cross-Space Validation", fontsize=14, fontweight="bold")
    
    titles = {"T1": "Scalar Grid", "T2": "Vector Grid (dim=32)",
              "T3": "Irregular Graph", "T4": "Tree Hierarchy"}
    
    for idx, tkey in enumerate(["T1", "T2", "T3", "T4"]):
        ax = axes[idx//2, idx%2]
        r = all_results[tkey]
        hps = sorted(r["params"].keys())
        n = r["params"][hps[0]]["n"]
        
        norm_pass = [r["params"][h]["normal"]["pass"]/n*100 for h in hps]
        soap_pass = [r["params"][h]["soap"]["pass"]/n*100 for h in hps]
        
        x = np.arange(len(hps)); bw = 0.3
        ax.bar(x-bw/2, norm_pass, bw, label="Normal", color="#2ca02c", alpha=0.7)
        ax.bar(x+bw/2, soap_pass, bw, label="Soap", color="#d62728", alpha=0.7)
        ax.set_xticks(x); ax.set_xticklabels([str(h) for h in hps])
        ax.set_xlabel("Halo size"); ax.set_ylabel("Pass dual check (%)")
        ax.set_title(f"{tkey}: {titles[tkey]}")
        ax.legend(); ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 110)
    
    plt.tight_layout()
    plt.savefig(out / "seam_crossspace.png", dpi=150, bbox_inches="tight")
    print(f"\n[Saved] seam_crossspace.png + .json")
    print("\n[Done]")


if __name__ == "__main__":
    main()
