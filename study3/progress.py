#!/usr/bin/env python3
"""Print a Study-3 full-run progress table (read-only; safe to call anytime)."""
import json, sys, time
from pathlib import Path
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE)); OUT=HERE/"outputs"
from plan import FULL_PLAN, PANEL, RUN_ID

def counts(ds):
    f=OUT/f"{ds}.judgments.jsonl"
    if not f.exists(): return 0,0,0
    done=err=unp=0
    for l in open(f):
        try:
            d=json.loads(l)
            if d.get("run_id")!=RUN_ID: continue
            done+=1
            if d.get("error"): err+=1
            if d.get("pred") is None: unp+=1
        except: pass
    return done,err,unp

start=None
if (OUT/"full_run.start").exists():
    try: start=float((OUT/"full_run.start").read_text().strip())
    except: pass
elapsed=(time.time()-start) if start else 0
tot_done=tot_exp=0; rows=[]
for ds,n,tier,sens in FULL_PLAN:
    exp=n*len(PANEL); done,err,unp=counts(ds); tot_done+=done; tot_exp+=exp
    pct=100*done/exp if exp else 0
    st="✔ done" if done>=exp else ("▶ running" if done>0 else "… queued")
    rows.append((ds,tier,("Y" if sens else "-"),done,exp,pct,unp,err,st))
rate=tot_done/elapsed if elapsed>0 else 0
eta=(tot_exp-tot_done)/rate if rate>0 else float("inf")
hh=lambda s: ("%dm%02ds"%(s//60,s%60)) if s!=float("inf") and s<1e7 else "—"

print(f"┌─ STUDY 3 FULL RUN — progress @ {time.strftime('%H:%M:%S')} "
      f"({tot_done}/{tot_exp} judgments = {100*tot_done/max(1,tot_exp):4.1f}%) "
      f"elapsed {hh(elapsed)}  rate {rate:4.1f}/s  ETA {hh(eta)}")
print(f"│ {'dataset':15s} {'tier':5s} {'sens':4s} {'done':>6} {'exp':>6} {'%':>6} {'unparsed':>8} {'err':>4}  status")
for ds,tier,sens,done,exp,pct,unp,err,st in rows:
    print(f"│ {ds:15s} {tier:5s} {sens:^4s} {done:>6} {exp:>6} {pct:>5.1f}% {unp:>8} {err:>4}  {st}")
print(f"└─ panel: {', '.join(PANEL)}")
done_all = tot_exp>0 and tot_done>=tot_exp
fin = (OUT/"full_run.status").exists() and json.loads((OUT/"full_run.status").read_text()).get("finished")
print("STATUS:", "COMPLETE" if (done_all or fin) else "IN PROGRESS")
