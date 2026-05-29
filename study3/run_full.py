#!/usr/bin/env python3
"""Study 3 full run orchestrator. Iterates the FULL_PLAN datasets, judging each with the
PANEL in parallel (per-model concurrency in harness). Resumable (append-only JSONL).
Run (background):  source secrets.sh && python3 study3/run_full.py"""
import asyncio, sys, time, json
from pathlib import Path
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
import harness
from plan import FULL_PLAN, PANEL, RUN_ID
OUT=HERE/"outputs"; OUT.mkdir(exist_ok=True)
def _count_run(ds):
    f=OUT/f"{ds}.judgments.jsonl"; c=0
    if f.exists():
        for l in open(f):
            try:
                if json.loads(l).get("run_id")==RUN_ID: c+=1
            except: pass
    return c
_baseline=sum(_count_run(ds) for ds,_,_,_ in FULL_PLAN)   # already-done judgments (resume)
(OUT/"full_run.start").write_text(json.dumps({"start":time.time(),"baseline":_baseline}))
status={"started":time.time(),"datasets":{}}
for ds,n,tier,sensitive in FULL_PLAN:
    print(f"\n=== {ds} (n={n}, contamination={tier}, sensitive={sensitive}) x {len(PANEL)} models ===",flush=True)
    t0=time.time(); status["datasets"][ds]={"phase":"running","start":t0}
    (OUT/"full_run.status").write_text(json.dumps(status))
    try:
        asyncio.run(harness.run(ds,n,PANEL,RUN_ID,0.0))
        status["datasets"][ds]={"phase":"done","start":t0,"end":time.time()}
    except Exception as e:
        status["datasets"][ds]={"phase":"error","start":t0,"err":repr(e)[:200]}
        print("  ! dataset error:",repr(e)[:200],flush=True)
    (OUT/"full_run.status").write_text(json.dumps(status))
status["finished"]=time.time(); (OUT/"full_run.status").write_text(json.dumps(status))
print("\nFULL RUN COMPLETE",flush=True)
