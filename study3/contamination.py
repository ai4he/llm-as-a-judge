#!/usr/bin/env python3
"""Study 3 contamination probe (verbatim quote-completion), generic over datasets.
Gives the model the dataset name + a prefix of an item's text and asks it to reproduce the
rest verbatim; high normalized overlap => likely seen in pretraining (would inflate agreement).
Aggregate-only outputs. Usage:
  source secrets.sh; python3 study3/contamination.py --all --n 25 --models gemma-4-31b qwen3.5-9b
  python3 study3/contamination.py --dataset hatexplain --n 25"""
import os, json, asyncio, argparse, re, difflib, sys
from pathlib import Path
import aiohttp
ROOT=Path(__file__).resolve().parent.parent; OUT=ROOT/"study3"/"outputs"
sys.path.insert(0,str(ROOT/"study3")); from harness import BASE,KEY,call_chat,LOADERS,MODEL_CONCURRENCY
from plan import FULL_PLAN
DSNAME={"chaosnli_snli":"SNLI natural-language-inference","chaosnli_mnli":"MultiNLI natural-language-inference",
        "hatexplain":"HateXplain hate-speech detection","go_emotions":"GoEmotions emotion",
        "social_bias_frames":"Social Bias Frames (SBIC)"}
def norm(s): return re.sub(r"\s+"," ",re.sub(r"[^a-z0-9 ]","",str(s).lower())).strip()
def overlap(pred,truth):
    a,b=norm(pred),norm(truth)
    return 0.0 if not b else difflib.SequenceMatcher(None,a,b).ratio()
def get_text(it): return str(it.get("text") or it.get("premise") or "")

async def probe_item(session,sem,model,ds,item):
    txt=get_text(item); words=txt.split()
    if len(words)<4: return None
    prefix=" ".join(words[:max(3,len(words)//3)])
    sysp=("You are being tested on verbatim recall of a known public dataset. "
          "If you recognize the text, reproduce it exactly.")
    usr=(f"The following is the beginning of an item from the well-known '{DSNAME.get(ds,ds)}' dataset. "
         f"Reproduce the COMPLETE original text verbatim (only the text, nothing else):\n\n\"{prefix} ...\"")
    async with sem:
        r=await call_chat(session,model,[{"role":"system","content":sysp},{"role":"user","content":usr}],400,0.0)
    out=r.get("content") or r.get("reasoning") or ""
    return {"model":model,"overlap":overlap(out,txt)}

async def run_ds(ds,n,models):
    items=LOADERS[ds](n)
    sems={m:asyncio.Semaphore(min(16,MODEL_CONCURRENCY.get(m,16))) for m in models}
    rows=[]
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=64)) as s:
        tasks=[probe_item(s,sems[m],m,ds,it) for it in items for m in models]
        for f in asyncio.as_completed(tasks):
            r=await f
            if r: rows.append(r)
    rep={}
    for m in models:
        ov=[r["overlap"] for r in rows if r["model"]==m]
        if not ov: continue
        mean=sum(ov)/len(ov); hi=sum(1 for x in ov if x>=0.9)/len(ov)
        rep[m]={"n":len(ov),"mean_overlap":round(mean,3),"frac_verbatim_ge0.9":round(hi,3),
                "max_overlap":round(max(ov),3),
                "flag":"HIGH" if mean>0.6 or hi>0.2 else ("MEDIUM" if mean>0.35 else "LOW")}
    json.dump(rep,open(OUT/f"{ds}.contamination.json","w"),indent=2)
    print(f"  {ds:20s} "+" | ".join(f"{m}:{rep[m]['mean_overlap']}({rep[m]['flag']})" for m in rep))
    return rep

async def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset"); ap.add_argument("--all",action="store_true")
    ap.add_argument("--n",type=int,default=25); ap.add_argument("--models",nargs="*",default=["gemma-4-31b","qwen3.5-9b"])
    a=ap.parse_args(); assert KEY,"source secrets.sh"
    dsl=[d for d,_,_,_ in FULL_PLAN] if a.all else [a.dataset]
    print("CONTAMINATION PROBE  models=%s  n=%d"%(a.models,a.n))
    for ds in dsl: await run_ds(ds,a.n,a.models)

if __name__=="__main__": asyncio.run(main())
