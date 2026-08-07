"""Replay the notebook's exact logic across many seeds."""
import numpy as np, json
from dataclasses import dataclass
from typing import Tuple

@dataclass
class Cfg:
    initial_price=100.0; drift=0.0001; volatility=0.02; dt=1.0
    poisson_rate=0.1; order_size=10; n_ticks=10000
    naive_spread=0.10; risk_aversion=0.1; max_inventory=100
cfg=Cfg()

def gbm(c,rng):
    eps=rng.standard_normal(c.n_ticks-1)
    lr=(c.drift-0.5*c.volatility**2)*c.dt + c.volatility*np.sqrt(c.dt)*eps
    p=np.zeros(c.n_ticks); p[0]=c.initial_price
    p[1:]=c.initial_price*np.exp(np.cumsum(lr)); return p

def flow(c,rng):
    ap=1-np.exp(-c.poisson_rate*c.dt)
    return rng.random(c.n_ticks)<ap, rng.choice([1,-1],size=c.n_ticks)

class MM:
    def __init__(s,c): s.c=c; s.inv=0; s.cash=0.0; s.pnl=[]; s.invh=[]
    def quote(s,m): raise NotImplementedError
    def tick(s,m,arr,d,rng):
        bid,ask=s.quote(m); kappa=10.0
        pb=np.exp(-kappa*(m-bid)); pa=np.exp(-kappa*(ask-m))
        if arr:
            if d==1:
                if s.inv>-s.c.max_inventory and rng.random()<pa:
                    s.inv-=s.c.order_size; s.cash+=ask*s.c.order_size
            else:
                if s.inv<s.c.max_inventory and rng.random()<pb:
                    s.inv+=s.c.order_size; s.cash-=bid*s.c.order_size
        s.pnl.append(s.cash+s.inv*m); s.invh.append(s.inv)

class Naive(MM):
    def quote(s,m): h=s.c.naive_spread/2; return m-h,m+h
class InvAware(MM):
    def quote(s,m):
        g=0.5; skew=-s.inv*g*(s.c.volatility**2); r=m+skew
        h=s.c.naive_spread/2; return r-h,r+h

def metrics(s):
    p=np.array(s.pnl); i=np.array(s.invh); ret=np.diff(p)
    return dict(pnl=p[-1], sharpe=np.mean(ret)/(np.std(ret)+1e-6)*np.sqrt(252),
                mdd=np.min(p-np.maximum.accumulate(p)),
                avg_inv=np.mean(np.abs(i)), inv_std=np.std(i))

rows=[]
for s in range(200):
    rng=np.random.default_rng(s)
    pr=gbm(cfg,rng); arr,dirs=flow(cfg,rng)
    a,b=Naive(cfg),InvAware(cfg)
    for t in range(cfg.n_ticks):
        a.tick(pr[t],arr[t],dirs[t],rng)
    rng2=np.random.default_rng(s+10**6)
    for t in range(cfg.n_ticks):
        b.tick(pr[t],arr[t],dirs[t],rng2)
    ma,mb=metrics(a),metrics(b)
    rows.append(dict(pnl_a=ma['pnl'],pnl_b=mb['pnl'],
                     rel=(mb['pnl']-ma['pnl'])/abs(ma['pnl'])*100 if ma['pnl'] else np.nan,
                     mdd_a=ma['mdd'],mdd_b=mb['mdd'],
                     istd_a=ma['inv_std'],istd_b=mb['inv_std'],
                     sh_a=ma['sharpe'],sh_b=mb['sharpe']))
import statistics as st
def q(k): 
    v=np.array([r[k] for r in rows]); return v
print("200 independent seeds, notebook's exact logic\n")
print(f"{'metric':<26}{'p5':>12}{'median':>12}{'p95':>12}")
for k,lab in (('pnl_a','Strategy A final PnL'),('pnl_b','Strategy B final PnL'),('rel','rel. improvement %')):
    v=q(k); print(f"{lab:<26}{np.percentile(v,5):>12,.0f}{np.median(v):>12,.0f}{np.percentile(v,95):>12,.0f}")
print()
print(f"Strategy A profitable in {(q('pnl_a')>0).mean()*100:.0f}% of seeds; B in {(q('pnl_b')>0).mean()*100:.0f}%")
print(f"B beats A in {(q('pnl_b')>q('pnl_a')).mean()*100:.0f}% of seeds")
va,vb=q('istd_a'),q('istd_b')
print(f"inventory-std reduction: median {np.median((1-vb/va)*100):+.0f}%  (p5 {np.percentile((1-vb/va)*100,5):+.0f}%, p95 {np.percentile((1-vb/va)*100,95):+.0f}%)")
da,db=q('mdd_a'),q('mdd_b')
print(f"max-drawdown reduction : median {np.median((1-db/da)*100):+.0f}%  (p5 {np.percentile((1-db/da)*100,5):+.0f}%, p95 {np.percentile((1-db/da)*100,95):+.0f}%)")
