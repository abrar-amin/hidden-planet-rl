import sys, json, pickle, os, glob
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SP=os.environ.get("ARENA_DIR", "arena_workspace")  # directory holding the arena transcripts (not published)
REF=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dev")
refs={a:{json.loads(l)["seed"]:json.loads(l) for l in open(f"{REF}/refs_dev_{a}_episodes.jsonl")} for a in ["teacher","eig","compact","compact_smart","lite_teacher","always_cannot","random"]}
def grade(arena):
    d=f"{SP}/{arena}/keeper"; M=json.load(open(d+"/map.json")); out={}
    for eid,seed in M.items():
        p=f"{d}/state_{eid}.pkl"
        if not os.path.exists(p): continue
        s=pickle.load(open(p,"rb"))
        if not s.done: continue
        r,g=s.grade(); v="planet" if g["verdict_planet"] else ("none" if g["verdict_none"] else ("cannot" if g["verdict_cannot"] else "?"))
        out[seed]=dict(reward=r,correct=g["correct"],spent=g["spent"],v=v)
    return out
groups={"Fable (oracle)":["arena","arena_fableO2","arena_fableO3"],"Fable (lite)":["arena_fableL"],"Opus (oracle)":["arena_opusO"],"Sonnet (oracle)":["arena_sonnetO"],"Haiku (oracle)":["arena_haikuO"]}
res={}
for name,ars in groups.items():
    o={}
    for a in ars:
        if os.path.exists(f"{SP}/{a}/keeper/map.json"): o.update(grade(a))
    res[name]=o
json.dump({k:{str(s):v for s,v in o.items()} for k,o in res.items()},open(f"{SP}/arena_results.json","w"),indent=1)
def summ(rows):
    n=len(rows); import math
    if n==0: return "n=0"
    m=sum(r["reward"] for r in rows)/n; sd=(sum((r["reward"]-m)**2 for r in rows)/max(n-1,1))**.5
    return f"n={n:3d} reward {m:+.3f} ±{1.96*sd/math.sqrt(n):.2f}  correct {sum(r['correct'] for r in rows)/n:.2f}  cannot {sum(r.get('v')=='cannot' or r.get('verdict_cannot',0)==1 for r in rows)/n:.2f}  spent {sum(r['spent'] for r in rows)/n:.1f}"
for name,o in res.items():
    seeds=sorted(o)
    print(f"{name:18s} {summ([o[s] for s in seeds])}   (seeds {seeds[0] if seeds else '-'}-{seeds[-1] if seeds else '-'})")
    for a in ["teacher","eig","compact","compact_smart","lite_teacher"]:
        rr=[refs[a][s] for s in seeds if s in refs[a]]
        if rr: print(f"   same seeds: {a:14s} {summ([dict(reward=x['reward'],correct=x.get('correct',0),spent=x.get('spent',0),verdict_cannot=x.get('verdict_cannot',0)) for x in rr])}")
