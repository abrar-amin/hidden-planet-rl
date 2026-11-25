"""Audit: ES vs realised differences between every pair of scripted policies on dev (references averaged over 3 plays)."""
import json, sys, itertools, numpy as np
sys.path.insert(0,'.')
from hidden_planet.expected_score import es_from_metrics
D='results/dev192_avg3/'
def load(a):
    by={}
    for l in open(D+f'refs_dev_{a}_episodes.jsonl'):
        r=json.loads(l); by.setdefault(r['seed'],[]).append((es_from_metrics(r),r['reward']))
    return {s:np.mean(v,0) for s,v in by.items()}
P={a:load(a) for a in ['fixed','teacher','eig','compact','compact_smart']}
S=sorted(P['fixed']); rng=np.random.default_rng(0)
def ci(d):
    b=d[rng.integers(0,len(d),(20000,len(d)))].mean(1); return "%+.3f [%+.3f,%+.3f]"%(d.mean(),*np.percentile(b,[2.5,97.5]))
for a,b in itertools.combinations(P,2):
    e=np.array([P[a][s][0]-P[b][s][0] for s in S]); r=np.array([P[a][s][1]-P[b][s][1] for s in S])
    print(f"{a:14s}-{b:14s} ES {ci(e)} | realised {ci(r)} | ES-minus-realised differential {ci(e-r)}")
