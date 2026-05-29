#!/usr/bin/env python3
"""Study 3 contamination probe (quote-completion / verbatim-recall).
Gives the model the dataset name + a short prefix of an item and asks it to reproduce the
rest verbatim. High normalized overlap => the item was likely seen in pretraining (contamination),
which would inflate LLM-as-judge agreement. Aggregate-only outputs (no raw text committed).
Usage: source secrets.sh; python3 study3/contamination.py --dataset chaosnli_snli --n 25 --models gemma-4-31b qwen3.5-9b
"""
import os, json, asyncio, argparse, re, difflib
from pathlib import Path
import aiohttp, sys
ROOT=Path(__file__).resolve().parent.parent; OUT=ROOT/"study3"/"outputs"
sys.path.insert(0,str(ROOT/"study3")); from harness import BASE,KEY,call_chat,LOADERS,MODEL_CONCURRENCY

def norm(s): return re.sub(r"\s+"," ",re.sub(r"[^a-z0-9 ]","",str(s).lower())).strip()
def overlap(pred, truth):
    a,b=norm(pred),norm(truth)
    if not b: return 0.0
    return difflib.SequenceMatcher(None,a,b).ratio()              # 0..1 verbatim similarity

async def probe_item(session, sem, model, ds, item):
    prem=item["premise"]; words=prem.split()
    prefix=" ".join(words[:max(3,len(words)//3)])
    sys_p=("You are being tested on verbatim recall of a known public dataset. "
           "If you recognize the text, reproduce it exactly.")
    usr=(f"The following is the beginning of a PREMISE sentence from the well-known "
         f"'{ds}' / SNLI natural-language-inference dataset. Reproduce the COMPLETE original premise "
         f"verbatim (only the sentence, nothing else):\n\n\"{prefix} ...\"")
    async with sem:
        r=await call_chat(session,model,[{"role":"system","content":sys_p},{"role":"user","content":usr}],400,0.0)
    out=r.get("content") or r.get("reasoning") or ""
    return {"model":model,"item_id":item["id"],"overlap":overlap(out,prem),
            "len_truth":len(words)}

async def main_async(ds,n,models):
    items=LOADERS[ds](n)
    sems={m:asyncio.Semaphore(min(16,MODEL_CONCURRENCY.get(m,16))) for m in models}
    conn=aiohttp.TCPConnector(limit=64); rows=[]
    async with aiohttp.ClientSession(connector=conn) as s:
        tasks=[probe_item(s,sems[m],m,ds,it) for it in items for m in models]
        for fut in asyncio.as_completed(tasks): rows.append(await fut)
    rep={}
    for m in models:
        ov=[r["overlap"] for r in rows if r["model"]==m]
        hi=sum(1 for x in ov if x>=0.9)
        rep[m]={"n":len(ov),"mean_overlap":round(sum(ov)/len(ov),3),
                "frac_verbatim>=0.9":round(hi/len(ov),3),
                "max_overlap":round(max(ov),3),
                "contamination_flag":"HIGH" if sum(ov)/len(ov)>0.6 or hi/len(ov)>0.2 else
                                     ("MEDIUM" if sum(ov)/len(ov)>0.35 else "LOW")}
    (OUT/f"{ds}.contamination.json").write_text(json.dumps(rep,indent=2))
    print(f"CONTAMINATION PROBE  dataset={ds}  n={n}")
    print(f"{'model':20s} {'mean_ov':>8} {'verbatim>=.9':>13} {'max':>6}  flag")
    for m,s_ in rep.items():
        print(f"{m:20s} {s_['mean_overlap']:>8} {s_['frac_verbatim>=0.9']:>13} {s_['max_overlap']:>6}  {s_['contamination_flag']}")
    print(f"\nwrote {OUT}/{ds}.contamination.json")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset",default="chaosnli_snli")
    ap.add_argument("--n",type=int,default=25); ap.add_argument("--models",nargs="*",default=["gemma-4-31b","qwen3.5-9b"])
    a=ap.parse_args(); assert KEY,"source secrets.sh"; asyncio.run(main_async(a.dataset,a.n,a.models))
