import json, sys, numpy as np
sys.path.insert(0,'.')
from hidden_planet.expected_score import es_from_metrics
D='results/dev192_avg3/'
def load(a):
    by={}
    for l in open(D+f'refs_dev_{a}_episodes.jsonl'):
        r=json.loads(l); r['es']=es_from_metrics(r); by.setdefault(r['seed'],{})[r['sample']]=r
    return by
P={a:load(a) for a in ['fixed','teacher','eig','compact','compact_smart','random']}
seeds=sorted(P['fixed'])
def arr(a,k,key): return np.array([P[a][s][k][key] for s in seeds])
def se(d): return d.std(ddof=1)/np.sqrt(len(d))
print("n systems",len(seeds))
for key in ['es','reward']:
    print("==",key)
    for a in ['fixed','teacher','eig']:
        x=np.stack([arr(a,k,key) for k in range(3)])
        print(f" {a}: mean {x.mean():+.3f} per-episode SD {x.std():.3f}; within-system SD across plays {np.sqrt(x.var(0,ddof=1).mean()):.3f}; between-system SD of means {x.mean(0).std(ddof=1):.3f}")
    # same-RNG pairing vs different-RNG pairing
    for a,b in [('teacher','fixed'),('eig','fixed')]:
        same=[se(arr(a,k,key)-arr(b,k,key)) for k in range(3)]
        diff=[se(arr(a,k,key)-arr(b,(k+1)%3,key)) for k in range(3)]
        print(f" {a}-{b}: paired SE same policy-seed {np.round(same,4)}, different policy-seed {np.round(diff,4)}")
    # null noise floor: same policy, different plays
    for a in ['fixed','teacher']:
        d=arr(a,0,key)-arr(a,1,key)
        print(f" {a} play0-play1 (true diff 0): mean {d.mean():+.3f} SE {se(d):.4f}  SD {d.std(ddof=1):.3f}")
    # 2 samples vs 2 samples (independent policies proxied by teacher k0,k1 vs fixed k2 & ... ) 
    d=(arr('teacher',0,key)+arr('teacher',1,key))/2-(arr('fixed',2,key)+arr('fixed',1,key))/2
    print(f" teacher(2 plays) - fixed(2 plays, partly different rng): SE {se(d):.4f}")
    d=(arr('fixed',0,key)+arr('fixed',1,key))/2-np.stack([arr('fixed',k,key) for k in range(3)]).mean(0)
    print(f" [D2-like, null, CORRELATED plays] fixed(2 plays) - fixed(avg3): SE {se(d):.4f}")
# futility: 48-system, 1 sample each, independent plays
print("== futility proxy (48 systems, single sample at step 0 and step 15)")
for a in ['fixed','teacher']:
    d=arr(a,0,'es')-arr(a,1,'es'); sd=d.std(ddof=1); se48=sd/np.sqrt(48)
    from math import erf,sqrt
    Phi=lambda z:0.5*(1+erf(z/sqrt(2)))
    print(f" {a}: SD of per-system diff {sd:.3f} -> SE(48) {se48:.3f}; P(continue | true delta): " + ", ".join(f"{dl:+.2f}: {Phi(dl/se48):.2f}" for dl in (0,0.02,0.05,0.10)))
# ES - realised by policy, paired differences between policies
print("== ES minus realised (bias) by policy and differential vs fixed, dev192 avg3")
rng=np.random.default_rng(0)
def ci(d):
    b=d[rng.integers(0,len(d),(20000,len(d)))].mean(1); return d.mean(),*np.percentile(b,[2.5,97.5])
g={a:np.stack([arr(a,k,'es')-arr(a,k,'reward') for k in range(3)]).mean(0) for a in P}
for a in P:
    print(f" {a:14s} ES-realised %+.3f [%+.3f,%+.3f]"%ci(g[a]), "| differential vs fixed %+.3f [%+.3f,%+.3f]"%ci(g[a]-g['fixed']))
# where do fixed commits sit
for a in ['fixed','teacher','eig']:
    rows=[P[a][s][k] for s in seeds for k in range(3)]
    pl=[r for r in rows if r.get('verdict_planet',0)>.5]; no=[r for r in rows if r.get('verdict_none',0)>.5]
    f=lambda rs,lo,hi: sum(lo<=r['post_conf']<hi for r in rs)/len(rows)
    print(f" {a}: planet commits {len(pl)/len(rows):.2f} (p<0.5 {f(pl,0,.5):.3f}, 0.5-0.7 {f(pl,.5,.7):.3f}, 0.7-0.9 {f(pl,.7,.9):.3f}, >=0.9 {f(pl,.9,2):.3f}); none commits {len(no)/len(rows):.2f} (0.5-0.7 {f(no,.5,.7):.3f}); cannot {sum(r.get('verdict_cannot',0)>.5 for r in rows)/len(rows):.2f}")
    for lo,hi in [(.5,.7),(.7,.9),(.9,2)]:
        sel=[r for r in pl if lo<=r['post_conf']<hi]
        if sel: print(f"     planet p[{lo},{hi}): n={len(sel)} mean 2p-1 {np.mean([2*r['post_conf']-1 for r in sel]):+.3f} realised score {np.mean([r['score'] for r in sel]):+.3f}")
