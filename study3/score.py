#!/usr/bin/env python3
"""Study 3 scoring: agreement vs human ceiling, count-based alt-test approximation,
bootstrap CIs, jury, and 5-level verdict mapping. Reads study3/outputs/<dataset>.judgments.jsonl.
Usage: python3 study3/score.py --dataset chaosnli_snli --run pilot"""
import os, json, argparse, math, random
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, precision_recall_fscore_support
ROOT=Path(__file__).resolve().parent.parent; OUT=ROOT/"study3"/"outputs"
import sys; sys.path.insert(0,str(ROOT/"study3")); from harness import LOADERS

VLAB={1:"Validated",2:"Promising",3:"Mixed",4:"Caution",5:"Unreliable"}
def verdict(acc, ceiling, kappa):
    # relative to human ceiling + absolute kappa
    if acc>=ceiling-0.02 and kappa>=0.6: return 1
    if acc>=ceiling-0.05 or kappa>=0.6: return 2
    if kappa>=0.4: return 3
    if kappa>=0.2: return 4
    return 5

def human_ceiling(items):
    accs, pair = [], []
    for it in items:
        c=it["label_count"]; N=sum(c)
        if N<2: continue
        accs.append(max(c)/N)                                   # random annotator vs item majority
        pair.append(sum(x*(x-1) for x in c)/(N*(N-1)))          # P(two random annotators agree)
    return float(np.mean(accs)), float(np.mean(pair))

def alt_test_winrate(items_by_id, preds, classes):
    "Count-based approx of the alt-test: P(LLM >= a random excluded human at predicting the rest)."
    wins=[]; idx={c:i for i,c in enumerate(classes)}
    for iid,pred in preds.items():
        if pred is None or iid not in items_by_id: continue
        c=list(items_by_id[iid]["label_count"]); N=sum(c)
        if N<2 or pred not in idx: continue
        wr=0.0
        for j,cj in enumerate(c):
            if cj==0: continue
            rest=c.copy(); rest[j]-=1; rm=int(np.argmax(rest))
            llm=1 if idx[pred]==rm else 0; hum=1 if j==rm else 0
            w=1.0 if llm>hum else (0.5 if llm==hum else 0.0)
            wr += (cj/N)*w
        wins.append(wr)
    return float(np.mean(wins)) if wins else float("nan"), len(wins)

def boot_ci(vals, fn, B=2000, seed=7):
    rng=random.Random(seed); n=len(vals); out=[]
    for _ in range(B):
        s=[vals[rng.randrange(n)] for _ in range(n)]; out.append(fn(s))
    lo,hi=np.percentile(out,[2.5,97.5]); return float(lo),float(hi)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset",default="chaosnli_snli")
    ap.add_argument("--run",default="pilot"); ap.add_argument("--n",type=int,default=250); a=ap.parse_args()
    items=LOADERS[a.dataset](a.n); by_id={it["id"]:it for it in items}
    H_acc,H_pair=human_ceiling(items)
    recs=[json.loads(l) for l in open(OUT/f"{a.dataset}.judgments.jsonl")]
    recs=[r for r in recs if r.get("run_id")==a.run]
    models=sorted({r["model"] for r in recs})
    report={"dataset":a.dataset,"run":a.run,"n_items":len(items),
            "human_ceiling_acc":round(H_acc,3),"human_pairwise_agree":round(H_pair,3),"models":{}}
    classes=items[0]["classes"]; per_item_pred=defaultdict(dict)
    for m in models:
        rm=[r for r in recs if r["model"]==m]
        pairs=[(r["gold"],r["pred"]) for r in rm if r["pred"] in classes and r["gold"] in classes]
        preds_by_id={r["item_id"]:r["pred"] for r in rm}
        for r in rm:
            if r["pred"] in classes: per_item_pred[r["item_id"]][m]=r["pred"]
        parse_rate=round(sum(1 for r in rm if r["pred"] in classes)/max(1,len(rm)),3)
        if len(pairs)<5:
            report["models"][m]={"parse_rate":parse_rate,"n":len(pairs),"note":"insufficient"}; continue
        g=[p[0] for p in pairs]; p=[p[1] for p in pairs]
        acc=float(np.mean([a==b for a,b in pairs])); kap=float(cohen_kappa_score(g,p,labels=classes))
        f1=float(f1_score(g,p,labels=classes,average="macro",zero_division=0))
        pr,rc,_,_=precision_recall_fscore_support(g,p,labels=classes,average="macro",zero_division=0)
        acc_lo,acc_hi=boot_ci(pairs, lambda s: np.mean([x==y for x,y in s]))
        wr,wn=alt_test_winrate(by_id,preds_by_id,classes)
        v=verdict(acc,H_acc,kap)
        report["models"][m]={"parse_rate":parse_rate,"n":len(pairs),"accuracy":round(acc,3),
            "acc_CI":[round(acc_lo,3),round(acc_hi,3)],"cohen_kappa_vs_gold":round(kap,3),
            "macro_f1":round(f1,3),"macro_precision":round(float(pr),3),"macro_recall":round(float(rc),3),
            "alt_test_winrate":round(wr,3),"alt_test_pass":bool(wr>=0.5),"verdict":v,"verdict_label":VLAB[v],
            "vs_ceiling_acc":round(acc-H_acc,3)}
    # jury (majority vote across models present per item)
    jg,jp=[],[]
    for iid,mp in per_item_pred.items():
        if len(mp)<2 or iid not in by_id: continue
        vote=Counter(mp.values()).most_common(1)[0][0]; jg.append(by_id[iid]["gold"]); jp.append(vote)
    if jg:
        jacc=float(np.mean([a==b for a,b in zip(jg,jp)])); jkap=float(cohen_kappa_score(jg,jp,labels=classes))
        report["jury"]={"n":len(jg),"accuracy":round(jacc,3),"cohen_kappa":round(jkap,3),
                        "verdict_label":VLAB[verdict(jacc,H_acc,jkap)],"vs_ceiling_acc":round(jacc-H_acc,3)}
    (OUT/f"{a.dataset}.{a.run}.scores.json").write_text(json.dumps(report,indent=2))
    # print
    print(f"\nDATASET {a.dataset} run={a.run}  items={len(items)}")
    print(f"HUMAN CEILING: random-annotator-vs-majority acc={H_acc:.3f} | pairwise agree={H_pair:.3f}\n")
    print(f"{'model':20s} {'parse':>5} {'acc':>6} {'95% CI':>13} {'kappa':>6} {'mF1':>5} {'altWR':>6} {'verdict'}")
    for m,s in report["models"].items():
        if "accuracy" not in s: print(f"{m:20s} {s['parse_rate']:>5} (insufficient)"); continue
        print(f"{m:20s} {s['parse_rate']:>5} {s['accuracy']:>6} [{s['acc_CI'][0]:.2f},{s['acc_CI'][1]:.2f}] "
              f"{s['cohen_kappa_vs_gold']:>6} {s['macro_f1']:>5} {s['alt_test_winrate']:>6} {s['verdict_label']}")
    if "jury" in report: print(f"{'JURY(majority)':20s} {'':>5} {report['jury']['accuracy']:>6} {'':>13} "
                               f"{report['jury']['cohen_kappa']:>6}  ->  {report['jury']['verdict_label']}")
    print(f"\nwrote {OUT}/{a.dataset}.{a.run}.scores.json")

if __name__=="__main__": main()
