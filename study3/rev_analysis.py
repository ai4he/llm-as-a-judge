#!/usr/bin/env python3
"""Cycle-1 revision analyses on the existing 18,000 Study-3 judgments (no new API calls):
 A. surface Cohen kappa + macro-F1 per cell
 B. human-ceiling bootstrap CIs (resample annotators within items) + gap test with ceiling variance
 C. soft-label / disagreement-aware eval (panel-vs-human Jensen-Shannon; accuracy vs human entropy)
 D. alt-test with an epsilon cost-of-human sweep (count-based leave-one-annotator-out winning rate)
 E. crossed random-effects GLMM (item+model+dataset) with granularity + base-rate competitors,
    and a leave-one-dataset-out dataset-level meta-regression of accuracy on subjectivity.
Writes data/rev_analysis.json and paper/tables/rev_autostats.tex.
"""
import json, sys, math, warnings
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT=Path(__file__).resolve().parent.parent; OUT=ROOT/"study3"/"outputs"; DATA=ROOT/"data"; TAB=ROOT/"paper"/"tables"
sys.path.insert(0,str(ROOT/"study3")); import harness
RUN="full"; RNG=np.random.default_rng(20260529)
DS=[("chaosnli_snli","ChaosNLI-SNLI"),("chaosnli_mnli","ChaosNLI-MNLI"),("go_emotions","GoEmotions"),
    ("social_bias_frames","SBIC"),("hatexplain","HateXplain"),("multipico","MultiPICo")]
DS=[(k,n) for k,n in DS if (OUT/f"{k}.full.scores.json").exists()]
SIZE_B={"qwen3.5-9b":9,"qwen3.6-27b-fp8":27,"gemma-4-31b":31,"gptoss-120b":120,"deepseek-v4-pro":671,"glm-5.1-fp8":754}

def load_items(key):
    """id -> {classes, counts(list aligned to classes), n, gold}"""
    items=harness.LOADERS[key](500)
    d={}
    for it in items:
        d[it["id"]]={"classes":it["classes"],"counts":np.array(it["label_count"],float),
                     "n":it["n_annot"],"gold":it["gold"]}
    return d
def load_preds(key):
    """id -> {model: pred}"""
    p=OUT/f"{key}.judgments.jsonl"; d=defaultdict(dict)
    for l in open(p):
        try: r=json.loads(l)
        except: continue
        if r.get("run_id")!=RUN or r.get("pred") is None: continue
        d[r["item_id"]][r["model"]]=r["pred"]
    return d

def ceiling_from_counts(counts):
    """expected random-annotator-vs-majority accuracy for one item given class counts."""
    N=counts.sum()
    if N<=1: return np.nan
    maj=counts.argmax(); return counts[maj]/N      # P(random annotator == majority)

def boot_ceiling(items, B=1000):
    ids=list(items); per=np.array([ceiling_from_counts(items[i]["counts"]) for i in ids])
    per=per[~np.isnan(per)]
    means=[per[RNG.integers(0,len(per),len(per))].mean() for _ in range(B)]
    return float(per.mean()), float(np.percentile(means,2.5)), float(np.percentile(means,97.5))

def js_divergence(p,q):
    p=p/p.sum() if p.sum()>0 else p; q=q/q.sum() if q.sum()>0 else q
    m=0.5*(p+q)
    def kl(a,b):
        s=0
        for ai,bi in zip(a,b):
            if ai>0 and bi>0: s+=ai*math.log2(ai/bi)
        return s
    return 0.5*kl(p,m)+0.5*kl(q,m)
def entropy(p):
    p=p/p.sum() if p.sum()>0 else p
    return float(-sum(pi*math.log2(pi) for pi in p if pi>0))

def alt_test_eps(items, preds, model, classes, eps):
    """count-based leave-one-annotator-out winning rate at cost eps:
    per item, LLM 'wins' vs a held-out annotator if (LLM agrees with majority-of-rest) >= (that annotator agrees) - eps.
    Data give per-item class counts (not annotator ids), so we enumerate annotators by class membership."""
    wins=tot=0
    for i,it in items.items():
        if model not in preds.get(i,{}): continue
        counts=it["counts"]; N=int(counts.sum())
        if N<2: continue
        llm=preds[i][model]; ci=classes.index(llm) if llm in classes else -1
        for c_idx,c_n in enumerate(counts):
            for _ in range(int(c_n)):
                rest=counts.copy(); rest[c_idx]-=1
                maj_rest=rest.argmax()
                human_agree=1.0 if c_idx==maj_rest else 0.0
                llm_agree=1.0 if ci==maj_rest else 0.0
                wins+= 1 if llm_agree>=human_agree-eps else 0; tot+=1
    return wins/tot if tot else np.nan

def main():
    res={"datasets":{}, "models":sorted(SIZE_B)}
    rows=[]   # for GLMM / meta-regression
    macros={}
    print("dataset            ceiling[95%CI]      meanJS  acc@lowH  acc@highH   kappa/mf1(best)")
    for key,name in DS:
        items=load_items(key); preds=load_preds(key)
        classes=items[next(iter(items))]["classes"]; K=len(classes)
        sc=json.load(open(OUT/f"{key}.full.scores.json"))
        mods=[m for m in sc["models"] if "accuracy" in sc["models"][m]]
        # B. ceiling CI
        cmean,clo,chi=boot_ceiling(items)
        # base-rate skew = max class proportion across the corpus
        agg=np.zeros(K)
        for it in items.values(): agg+=it["counts"]
        baserate=float(agg.max()/agg.sum())
        # C. soft-label: panel distribution vs human; entropy bins
        js_list=[]; ent=[]; corr=[]
        for i,it in items.items():
            hp=it["counts"].copy()
            mp=np.zeros(K)
            mv=[preds[i][m] for m in mods if m in preds.get(i,{})]
            if not mv: continue
            for v in mv:
                if v in classes: mp[classes.index(v)]+=1
            js_list.append(js_divergence(hp,mp)); ent.append(entropy(hp))
            gold=it["gold"]; corr.append(np.mean([1.0 if v==gold else 0.0 for v in mv]))
        ent=np.array(ent); corr=np.array(corr); js_arr=np.array(js_list)
        med=np.median(ent); lowH=corr[ent<=med].mean(); highH=corr[ent>med].mean()
        r_ent=float(np.corrcoef(ent,corr)[0,1]) if len(ent)>2 else float("nan")
        # D. alt-test eps sweep on the best model + small model
        best=max(mods,key=lambda m:sc["models"][m]["accuracy"])
        eps_grid=[0.0,0.1,0.2,0.3]
        alt={m:{f"{e}":round(float(alt_test_eps(items,preds,m,classes,e)),3) for e in eps_grid} for m in mods}
        # A. kappa/macroF1 already in scores
        kappa={m:round(sc["models"][m].get("cohen_kappa_vs_gold",float("nan")),3) for m in mods}
        mf1={m:round(sc["models"][m].get("macro_f1",float("nan")),3) for m in mods}
        res["datasets"][name]={"ceiling":round(cmean,3),"ceiling_CI":[round(clo,3),round(chi,3)],
            "baserate_skew":round(baserate,3),"mean_JS_panel_vs_human":round(float(js_arr.mean()),3),
            "acc_lowEntropy":round(float(lowH),3),"acc_highEntropy":round(float(highH),3),
            "acc_vs_entropy_r":round(r_ent,3),"K":K,"alt_test_eps":alt,"kappa":kappa,"macro_f1":mf1,"best":best}
        print(f"{name:18s} {cmean:.3f}[{clo:.3f},{chi:.3f}]  {js_arr.mean():.3f}   {lowH:.3f}    {highH:.3f}    k={kappa[best]} f1={mf1[best]}")
        # rows for GLMM
        subj=1-cmean
        for i,it in items.items():
            for m in mods:
                if m in preds.get(i,{}):
                    rows.append({"dataset":key,"item":f"{key}:{i}","model":m,
                                 "correct":int(preds[i][m]==it["gold"]),
                                 "subjectivity":subj,"logsize":math.log10(SIZE_B[m]),
                                 "nclasses":K,"baserate":baserate})
    df=pd.DataFrame(rows)
    for c in ["subjectivity","logsize","nclasses","baserate"]: df[c+"_c"]=df[c]-df[c].mean()

    # E1. logistic GLM with item-cluster-robust SE (fast; honest fixed-effect inference with 4 competing predictors)
    import statsmodels.formula.api as smf, statsmodels.api as sm
    glmm={}
    try:
        gfit=smf.glm("correct ~ subjectivity_c + logsize_c + nclasses_c + baserate_c", data=df,
                     family=sm.families.Binomial()).fit(cov_type="cluster",cov_kwds={"groups":df["item"]})
        for nm in gfit.params.index:
            glmm[nm]={"beta":round(float(gfit.params[nm]),3),"OR":round(float(np.exp(gfit.params[nm])),3),
                      "z":round(float(gfit.tvalues[nm]),2),"p":float(gfit.pvalues[nm])}
        print("\nCluster-robust logistic GLM (cluster=item; subjectivity + scale + granularity + base-rate):")
        for k,v in glmm.items(): print(f"  {k:16s} beta={v['beta']:+.3f} OR={v['OR']:.3f} z={v['z']:+.2f} p={v['p']:.1e}")
    except Exception as e: glmm={"error":repr(e)[:160]}; print("GLM err",glmm)
    # variance components from the (already fitted) item+dataset GLMM, plus an empirical model-level SD
    vc={}
    try:
        me=json.load(open(DATA/"study3_mixedeffects.json")); vc=me.get("glmm",{}).get("vc_sd",{})
    except Exception: pass
    model_acc=df.groupby("model")["correct"].mean()
    vc_model_emp=float(model_acc.std())
    glmm["vc_sd"]=vc; glmm["model_acc_spread"]=round(vc_model_emp,3)
    print(f"  variance components (item+dataset GLMM): {vc} ; empirical between-model accuracy SD={vc_model_emp:.3f}")

    # E2. dataset-level meta-regression with leave-one-dataset-out
    dd=df.groupby("dataset").agg(acc=("correct","mean"),subj=("subjectivity","first")).reset_index()
    def slope(d):
        x=d["subj"].values; y=d["acc"].values; A=np.vstack([x,np.ones_like(x)]).T
        return float(np.linalg.lstsq(A,y,rcond=None)[0][0])
    full_slope=slope(dd); lodo=[slope(dd.drop(idx)) for idx in dd.index]
    meta={"slope_full":round(full_slope,3),"LODO_min":round(min(lodo),3),"LODO_max":round(max(lodo),3),
          "n_datasets":len(dd)}
    print(f"\nDataset-level meta-regression acc~subjectivity: slope={full_slope:.3f}  LODO range [{min(lodo):.3f},{max(lodo):.3f}] (n={len(dd)} datasets)")
    res["glmm_crossed"]=glmm; res["meta_regression"]=meta
    DATA.mkdir(exist_ok=True); json.dump(res,open(DATA/"rev_analysis.json","w"),indent=2)

    # macros
    def g(x):
        try: return f"{x:.2f}"
        except: return str(x)
    subj_z = glmm.get("subjectivity_c",{}).get("z","na") if "error" not in glmm else "na"
    macros={
      "RVmetaSlope":g(meta["slope_full"]),"RVmetaLODOmin":g(meta["LODO_min"]),"RVmetaLODOmax":g(meta["LODO_max"]),
      "RVsubjZ":str(subj_z),
      "RVsizeZ":str(glmm.get("logsize_c",{}).get("z","na")) if "error" not in glmm else "na",
      "RVvcitem":g(glmm.get("vc_sd",{}).get("item",float("nan"))) if "error" not in glmm else "na",
      "RVvcmodel":g(glmm.get("vc_sd",{}).get("model",float("nan"))) if "error" not in glmm else "na",
      "RVvcdataset":g(glmm.get("vc_sd",{}).get("dataset",float("nan"))) if "error" not in glmm else "na",
    }
    (TAB/"rev_autostats.tex").write_text("% auto-generated by rev_analysis.py\n"+
        "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k,v in macros.items()))
    print("\nwrote data/rev_analysis.json + paper/tables/rev_autostats.tex"); print(macros)

if __name__=="__main__": main()
