import numpy as np
exec(open('_audit.py').read().split('rows=[]')[0])
np.random.seed(42)                      # exactly as the notebook does
class R:                                # legacy global RNG shim
    def standard_normal(s,n): return np.random.randn(n)
    def random(s,n=None): return np.random.rand(n) if n else np.random.random()
    def choice(s,a,size=None): return np.random.choice(a,size=size)
rng=R()
pr=gbm(cfg,rng); arr,dirs=flow(cfg,rng)
a,b=Naive(cfg),InvAware(cfg)
for t in range(cfg.n_ticks): a.tick(pr[t],arr[t],dirs[t],rng)
for t in range(cfg.n_ticks): b.tick(pr[t],arr[t],dirs[t],rng)
ma,mb=metrics(a),metrics(b)
rel=(mb['pnl']-ma['pnl'])/abs(ma['pnl'])*100
print(f"seed 42 (the notebook's own draw):")
print(f"  Strategy A PnL {ma['pnl']:>12,.0f}   Strategy B PnL {mb['pnl']:>12,.0f}")
print(f"  relative improvement {rel:+.1f}%")
print(f"  inventory std  A {ma['inv_std']:.1f} -> B {mb['inv_std']:.1f}  ({(1-mb['inv_std']/ma['inv_std'])*100:+.0f}%)")
print(f"  max drawdown   A {ma['mdd']:,.0f} -> B {mb['mdd']:,.0f}  ({(1-mb['mdd']/ma['mdd'])*100:+.0f}%)")
