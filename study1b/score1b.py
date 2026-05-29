#!/usr/bin/env python3
"""Score Study 1b: summary-level Spearman per (model,aspect) on SummEval, compared to the
REPORTED G-Eval/GPT-4 correlations.  Summary-level = per-document Spearman over its 16
summaries, averaged across documents (the canonical SummEval metric, Fabbri 2021 / Liu 2023).
Also reports pooled Spearman, system-level Spearman, coverage, and a 'surpasses reported?' verdict.
Writes study1b/outputs/<dataset>.<run>.scores.json and prints a comparison table.
"""
import json, sys, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr, bootstrap
ROOT=Path(__file__).resolve().parent.parent; OUT=ROOT/"study1b"/"outputs"
sys.path.insert(0,str(ROOT/"study1b")); from harness1b import REPORTED, ASPECTS

def per_doc_spearman(pairs_by_doc):
    """pairs_by_doc: {docid:[(model,human),...]} -> mean per-doc spearman (skip nan/const docs)."""
    rs=[]
    for d,ps in pairs_by_doc.items():
        if len(ps)<3: continue
        m=[p[0] for p in ps]; h=[p[1] for p in ps]
        if len(set(m))<2 or len(set(h))<2: continue   # need variance both sides
        r=spearmanr(m,h).correlation
        if r==r: rs.append(r)
    return (float(np.mean(rs)), len(rs), rs) if rs else (float("nan"),0,[])

def main(dataset,run):
    path=OUT/f"{dataset}.{run}.judgments.jsonl"
    rows=[json.loads(l) for l in open(path)]
    rep=REPORTED.get(dataset,{})
    aspects=list(ASPECTS)
    # only score models with near-complete coverage (exclude a still-running slow model)
    import collections
    cnt=collections.Counter(r["model"] for r in rows); expected=1600*len(aspects)
    models=sorted(m for m,c in cnt.items() if c>=0.95*expected)
    skipped={m:cnt[m] for m in cnt if m not in models}
    if skipped: print("  (skipped incomplete models:",{m:f"{c}/{expected}" for m,c in skipped.items()},")")
    res={"dataset":dataset,"run":run,"reported":rep,"models":{}}
    print(f"\nSTUDY 1b  {dataset}  (run={run})   reported = {rep.get('source','?')}")
    hdr=f"{'model':22s} "+" ".join(f"{a[:5]:>14s}" for a in aspects)+f"  {'avg':>6s}  cov"
    print(hdr); print("-"*len(hdr))
    print(f"{'REPORTED (G-Eval-4)':22s} "+" ".join(f"{rep.get(a,float('nan')):>14.3f}" for a in aspects)
          +f"  {np.mean([rep[a] for a in aspects if a in rep]):>6.3f}")
    for m in models:
        mr={}; line=f"{m:22s} "; covs=[]
        for a in aspects:
            sub=[r for r in rows if r["model"]==m and r["aspect"]==a]
            parsed=[r for r in sub if r.get("score") is not None]
            cov=len(parsed)/len(sub) if sub else 0; covs.append(cov)
            by_doc=defaultdict(list)
            for r in parsed: by_doc[r["docid"]].append((r["score"],r["human"]))
            mean_r,ndoc,rs=per_doc_spearman(by_doc)
            # pooled spearman over all parsed pairs
            mm=[r["score"] for r in parsed]; hh=[r["human"] for r in parsed]
            pooled=spearmanr(mm,hh).correlation if len(set(mm))>1 else float("nan")
            beat = (mean_r>rep[a]) if a in rep else None
            mr[a]={"summary_spearman":round(mean_r,4),"n_docs":ndoc,
                   "pooled_spearman":round(float(pooled),4),"coverage":round(cov,4),
                   "reported":rep.get(a),"surpasses_reported":beat}
            mark="*" if beat else (" " if beat is None else " ")
            line+=f"{mean_r:>13.3f}{mark} "
        avg=np.mean([mr[a]["summary_spearman"] for a in aspects])
        ravg=np.mean([rep[a] for a in aspects if a in rep]) if rep else float("nan")
        mr["_avg_summary_spearman"]=round(float(avg),4)
        mr["_avg_reported"]=round(float(ravg),4)
        mr["_surpasses_avg"]=bool(avg>ravg)
        mr["_n_aspects_surpassed"]=sum(1 for a in aspects if mr[a]["surpasses_reported"])
        res["models"][m]=mr
        print(line+f" {avg:>6.3f}  {np.mean(covs):.2f}"+("  <= BEATS avg" if avg>ravg else ""))
    # verdict
    best=max(res["models"], key=lambda m:res["models"][m]["_avg_summary_spearman"])
    bestavg=res["models"][best]["_avg_summary_spearman"]; ravg=res["models"][best]["_avg_reported"]
    any_beat=any(res["models"][m]["_surpasses_avg"] for m in models)
    res["verdict"]={"best_model":best,"best_avg":bestavg,"reported_avg":ravg,
        "any_model_surpasses_avg":any_beat,
        "interpretation":("MATTER OF TIME: open models already match/exceed the reported proprietary judge"
                          if any_beat else
                          "STRUCTURAL: even the most powerful open models fall short of the reported judge")}
    json.dump(res,open(OUT/f"{dataset}.{run}.scores.json","w"),indent=2)
    print(f"\n  * = surpasses the reported per-aspect number")
    print(f"  best open model: {best}  avg ρ={bestavg:.3f}  vs reported {ravg:.3f}  -> {res['verdict']['interpretation']}")
    print(f"  wrote {dataset}.{run}.scores.json")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset",default="summeval"); ap.add_argument("--run",default="powerful")
    a=ap.parse_args(); main(a.dataset,a.run)
