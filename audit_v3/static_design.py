"""Audit: is there a better STATIC design inside the SFT data's support (14-18 transits, 7-9 RVs) than the D2 reference (16, 8)? Train seeds 940000+ only."""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS","1")
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np; np.seterr(all="ignore")
from multiprocessing import Pool
from hidden_planet.tasks import sample_system
from hidden_planet.text_env import HiddenPlanetSession
from hidden_planet.text_policies import fixed_fit_agent
from es_loophole_test import run_gen
KW=dict(reward_mode="strict0", wrong_penalty=1.0, cost_per_unit=0.008)
DES=[(16,8),(14,7),(14,9),(18,7),(18,9),(12,6),(10,5),(16,6),(16,4),"diverse"]
def play(arg):
    d,seed=arg
    s=HiddenPlanetSession(sample_system(seed,5,"train"),**KW); rng=np.random.default_rng(seed+17)
    gen=fixed_fit_agent(s.user_prompt(),rng,max_turns=s.max_turns,diverse=True) if d=="diverse" else fixed_fit_agent(s.user_prompt(),rng,max_turns=s.max_turns,n_transits=d[0],n_rv=d[1])
    run_gen(s,gen); r,m=s.grade()
    return str(d),seed,float(m["reward_realised"]),float(m["es"]),float(m["spent"])
if __name__=="__main__":
    n=int(sys.argv[1]); jobs=[(d,940000+i) for d in DES for i in range(n)]
    with Pool(9) as p: res=p.map(play,jobs,chunksize=4)
    json.dump(res,open("audit_v3/static_design.json","w"))
    def ci(d):
        b=d[np.random.default_rng(0).integers(0,len(d),(10000,len(d)))].mean(1); lo,hi=np.percentile(b,[2.5,97.5]); return f"{d.mean():+.3f} [{lo:+.3f},{hi:+.3f}]"
    A=lambda d,i: np.array([x[i] for x in res if x[0]==str(d)])
    print(f"n={n} train seeds 940000+\ndesign      realised   ES   spent | ES - ES(16,8)           | realised - realised(16,8)")
    for d in DES: print(f"{str(d):10s} {A(d,2).mean():+.3f} {A(d,3).mean():+.3f} {A(d,4).mean():5.1f} | {ci(A(d,3)-A((16,8),3))} | {ci(A(d,2)-A((16,8),2))}")
