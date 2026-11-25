"""The sealed confirmation set and its one-use lock.

Sealed set: test-split seeds 20000-20095. Seeds 10000-10095 must not be used as a sealed set: scripted references were run
on them during a dry run and their rewards were seen.

Lock: results/SEALED_LOCK.json, outside any run directory. Every entry point that can touch the sealed seeds
(eval_zero_shot.py --eval-set sealed, eval_refs.py --eval-set sealed) calls claim(arm) first:
  - no lock            -> the lock is created 'in progress' for that arm
  - same arm           -> allowed (a failed run can be resumed; finished outputs are skipped by the callers)
  - a different arm    -> refused
finish(arm) marks it complete; it is called only after every sealed evaluation of that arm has succeeded.
"""
from __future__ import annotations

import json
import os
import sys
import time

SEALED_START, SEALED_N = 20000, 96
LOCK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "SEALED_LOCK.json")


def overlaps(start: int, n: int) -> bool:
    return start < SEALED_START + SEALED_N and start + n > SEALED_START


def claim(arm: str | None):
    if not arm:
        raise SystemExit("the sealed set needs --sealed-arm <arm>; use scripts/event.sh sealed, which applies the pre-registered rule")
    if os.path.exists(LOCK):
        cur = json.load(open(LOCK))
        if cur["arm"] != arm:
            raise SystemExit(f"SEALED SET LOCKED: it was claimed for arm '{cur['arm']}' ({cur['status']}, {cur['time']}); refusing arm '{arm}'")
        return cur
    cur = dict(arm=arm, status="in progress", time=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(cur, open(LOCK, "w"))
    return cur


def finish(arm: str):
    cur = claim(arm)
    cur.update(status="complete", finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(cur, open(LOCK, "w"))


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "finish":
        finish(sys.argv[2]); print("sealed set marked complete for", sys.argv[2])
    elif len(sys.argv) >= 3 and sys.argv[1] == "claim":
        print(claim(sys.argv[2]))
    else:
        print(json.load(open(LOCK)) if os.path.exists(LOCK) else "sealed set unused")
