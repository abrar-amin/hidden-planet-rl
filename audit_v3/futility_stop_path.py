"""Audit: exercise the real watchdog STOP path (never reached in dry_event.sh because the mock policy always improves)."""
import sys, os, json, asyncio, tempfile, types
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import train_rl
async def main():
    d=tempfile.mkdtemp(dir='audit_v3')
    async def fake_train():
        for step,es in [(0,0.30),(5,0.35),(10,0.36),(15,0.30),(20,0.5)]:
            with open(os.path.join(d,'metrics.jsonl'),'a') as f:
                f.write(json.dumps({"step":step,"test/env/all/es":es,"test/env/all/reward/total":es,"env/all/reward/total":0.1})+"\n")
            await asyncio.sleep(1.2)
        return "finished"
    a=types.SimpleNamespace(log=d,dry_run=True,max_dollars=24.0,max_minutes=40.0,collapse_guard=10,futility_step=15,pilot_stop_step=0)
    t=asyncio.create_task(fake_train()); wd=asyncio.create_task(train_rl.watchdog(a,"Qwen/Qwen3.5-9B",t))
    try: r=await t
    except asyncio.CancelledError: r="CANCELLED"
    await wd
    print("train task:",r); 
    for fn in ("futility_check.txt","futility_stop.txt"):
        p=os.path.join(d,fn); print(fn,":",open(p).read().strip() if os.path.exists(p) else "MISSING")
    print("rows logged:",sum(1 for _ in open(os.path.join(d,'metrics.jsonl'))))
asyncio.run(main())
