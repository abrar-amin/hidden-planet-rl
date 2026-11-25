"""Audit: power of the FULL v3 success criterion (3 ES CIs x 2 seeds + realised D2,D3 point >= 0 + futility continue), from dev-reference noise levels."""
import numpy as np
from math import erf, sqrt
rng=np.random.default_rng(1); N=200000; n=192
def run(delta, sd_es=0.20, sd_re=0.47, infl=1.0, fut_se=0.039, d_real=None):
    d_real = delta if d_real is None else d_real
    out={}
    def arm_noise(k, sd): return rng.normal(0, infl*sd/np.sqrt(k*n), N)
    ok=np.ones(N,bool)
    res={}
    for key,sd,d in (("es",sd_es,delta),("re",sd_re,d_real)):
        init=arm_noise(2,sd); ctrl=arm_noise(2,sd); demo=arm_noise(3,sd)
        rl=[arm_noise(2,sd) for _ in range(2)]
        se1=infl*sd*np.sqrt(1/(2*n)+1/(2*n)); se2=infl*sd*np.sqrt(1/(2*n)+1/(3*n))
        res[key]=[(d+rl[s]-init, d+rl[s]-demo, d+rl[s]-ctrl, se1, se2) for s in range(2)]
    single=[]
    for s in range(2):
        D1,D2,D3,se1,se2=res["es"][s]
        es_ok=(D1-1.96*se1>0)&(D2-1.96*se2>0)&(D3-1.96*se1>0)
        single.append((D1-1.96*se1>0).mean())
        R1,R2,R3,_,_=res["re"][s]
        ok&=es_ok&(R2>=0)&(R3>=0)
        if s==0: one=es_ok&(R2>=0)&(R3>=0)
    Phi=lambda z:0.5*(1+erf(z/sqrt(2)))
    pf=Phi(delta/fut_se)
    return single[0], one.mean(), ok.mean(), ok.mean()*pf*pf
print("true effect (ES = realised) | P(single CI>0) | P(one seed meets ES+sanity) | P(both seeds) | x P(both pass futility)")
for d in (0.0,0.03,0.04,0.05,0.06,0.08,0.10):
    print(f"  {d:+.2f}: "+"  ".join(f"{x:.3f}" for x in run(d)))
print("same, LLM sampling noise 1.5x the scripts':")
for d in (0.05,0.08,0.10):
    print(f"  {d:+.2f}: "+"  ".join(f"{x:.3f}" for x in run(d,infl=1.5,fut_se=0.058)))
print("FALSE-POSITIVE scenario: ES effect is metric bias only (true realised effect 0):")
for d in (0.03,0.05,0.06):
    print(f"  ES bias {d:+.2f}, realised 0: "+"  ".join(f"{x:.3f}" for x in run(d,d_real=0.0)))
