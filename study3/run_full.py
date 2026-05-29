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
(OUT/"full_run.start").write_text(str(time.time()))   # start marker for ETA
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
