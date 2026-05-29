#!/usr/bin/env python3
"""Cycle-1 revision (R4 annotator-bias): on MultiPICo (which ships annotator demographics), recompute
the human ceiling and the LLM panel's agreement separately for annotator subgroups (Gender, Generation),
testing whether judges agree more with majority-group annotators than with minority-group annotators.
Reuses the exact load_multipico grouping so item ids match the existing judgments. No new API calls."""
import json, sys
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
ROOT=Path(__file__).resolve().parent.parent; OUT=ROOT/"study3"/"outputs"; DATA=ROOT/"data"
RUN="full"
def load_multipico_demo(lang="en"):
    from datasets import load_dataset
    ds=load_dataset("Multilingual-Perspectivist-NLU/MultiPICo","default",split="train",streaming=True)
    byk=defaultdict(lambda:{"labs":[],"gen":[],"age":[]})
    for ex in ds:
        if ex.get("language")!=lang: continue
        try: lab="iro" if int(ex["label"])==1 else "noiro"
        except: continue
        d=byk[(ex.get("post_id"),ex.get("reply_id"))]
        d["labs"].append(lab); d["gen"].append(str(ex.get("Gender"))); d["age"].append(str(ex.get("GenerationAggregated")))
    items={}; i=0
    for k,d in byk.items():
        if len(d["labs"])<3: continue
        cnt=Counter(d["labs"]); top,tc=cnt.most_common(1)[0]
        if list(cnt.values()).count(tc)>1: continue
        items[f"mpc_{i}"]={"labs":d["labs"],"gen":d["gen"],"age":d["age"]}; i+=1
    # replicate the shuffle+truncation to n=500 used by the harness loader (seed 20260529)
    import random
    ids=list(items); random.Random(20260529).shuffle(ids); ids=ids[:500]
    return {idx:items[idx] for idx in ids}
def preds():
    d=defaultdict(dict)
    for l in open(OUT/"multipico.judgments.jsonl"):
        try: r=json.loads(l)
        except: continue
        if r.get("run_id")==RUN and r.get("pred") is not None: d[r["item_id"]][r["model"]]=r["pred"]
    return d
def maj(labs):
    c=Counter(labs); top,tc=c.most_common(1)[0]
    return top if list(c.values()).count(tc)==1 else None
def main():
    items=load_multipico_demo(); P=preds()
    mods=sorted({m for v in P.values() for m in v})
    def subgroup_stats(field):
        out={}
        # collect subgroup values with enough support
        vals=Counter(v for it in items.values() for v in it[field])
        groups=[g for g,c in vals.items() if g not in ("None","nan","DATA_EXPIRED") and c>=200]
        for g in groups:
            ceil=[]; agree=[]
            for idx,it in items.items():
                sub=[l for l,gg in zip(it["labs"],it[field]) if gg==g]
                if len(sub)<2: continue
                m=maj(sub)
                if m is None: continue
                c=Counter(sub); ceil.append(c[m]/len(sub))            # subgroup self-ceiling
                if idx in P:
                    mv=[P[idx][mm] for mm in mods if mm in P[idx]]
                    agree.append(np.mean([1.0 if p==m else 0.0 for p in mv]))   # panel-vs-subgroup-majority
            out[g]={"n_items":len(agree),"subgroup_ceiling":round(float(np.mean(ceil)),3),
                    "panel_agreement":round(float(np.mean(agree)),3)}
        return out
    res={"by_gender":subgroup_stats("gen"),"by_generation":subgroup_stats("age")}
    print("MultiPICo annotator-subgroup analysis (panel agreement with each subgroup's majority):")
    for field,st in res.items():
        print(f"  {field}:")
        for g,v in st.items(): print(f"    {g:14s} n={v['n_items']:3d} subgroup_ceiling={v['subgroup_ceiling']} panel_agreement={v['panel_agreement']}")
    # gap between best- and worst-served subgroup
    for field,st in list(res.items()):
        if isinstance(st,dict) and len(st)>=2:
            pa=[v["panel_agreement"] for v in st.values()]
            res[field+"_gap"]=round(max(pa)-min(pa),3)
            print(f"  {field} max-min panel-agreement gap = {res[field+'_gap']}")
    json.dump(res,open(DATA/"rev_multipico_subgroup.json","w"),indent=2)
    print("wrote data/rev_multipico_subgroup.json")
if __name__=="__main__": main()
