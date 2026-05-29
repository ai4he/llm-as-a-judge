#!/usr/bin/env python3
"""Cycle-2 revision analyses (no new API calls):
 A. Study-1b summary-level Spearman with bootstrap-over-documents 95% CIs + comparison to the
    reported G-Eval numbers (does the open model's CI actually exceed the reference?).
 B. Study-3 alt-test winning-rate bootstrap 95% CIs per model x dataset (flag whether CI excludes 0.5).
 C. Annotator-count ablation: subsample ChaosNLI items to A=3,5,10 annotators; recompute the human
    ceiling and the alt-test, to show the NLI result is not an artifact of ~100 annotators.
 D. Rescaled/standardized fixed-effects logit (per-SD odds ratios) + dataset-cluster bootstrap CI on
    the meta-regression slope (honest inference for few clusters).
Writes data/rev2_analysis.json + paper/tables/rev2_autostats.tex.
"""
import json, sys, math
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import warnings; warnings.filterwarnings("ignore")
ROOT=Path(__file__).resolve().parent.parent; OUT=ROOT/"study3"/"outputs"; S1B=ROOT/"study1b"/"outputs"; DATA=ROOT/"data"; TAB=ROOT/"paper"/"tables"
sys.path.insert(0,str(ROOT/"study3")); import harness
RNG=np.random.default_rng(7)
ASP=["coherence","consistency","fluency","relevance"]
REPORTED={"coherence":0.582,"consistency":0.507,"fluency":0.455,"relevance":0.547}
SIZE_B={"qwen3.5-9b":9,"qwen3.6-27b-fp8":27,"gemma-4-31b":31,"gptoss-120b":120,"deepseek-v4-pro":671,"glm-5.1-fp8":754}
out={}

# ---------- A. Study 1b bootstrap CIs ----------
def study1b_ci():
    rows=[json.loads(l) for l in open(S1B/"summeval.powerful.judgments.jsonl")]
    models=sorted({r["model"] for r in rows})
    # per (model,aspect): docid -> [(score,human)]
    res={}
    for m in models:
        res[m]={}
        for a in ASP:
            bydoc=defaultdict(list)
            for r in rows:
                if r["model"]==m and r["aspect"]==a and r.get("score") is not None:
                    bydoc[r["docid"]].append((r["score"],r["human"]))
            docs=list(bydoc)
            def doc_rho(d):
                ms=[x[0] for x in bydoc[d]]; hs=[x[1] for x in bydoc[d]]
                if len(set(ms))<2 or len(set(hs))<2: return None
                v=spearmanr(ms,hs).correlation; return v if v==v else None
            base=[doc_rho(d) for d in docs]; base=[x for x in base if x is not None]
            pt=float(np.mean(base))
            boots=[]
            for _ in range(1000):
                samp=RNG.choice(base,len(base),replace=True); boots.append(np.mean(samp))
            lo,hi=float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5))
            ref=REPORTED[a]
            res[m][a]={"rho":round(pt,3),"CI":[round(lo,3),round(hi,3)],"reported":ref,
                       "exceeds_reported":bool(lo>ref),"ci_contains_ref":bool(lo<=ref<=hi)}
    return res

# ---------- shared loaders for Study 3 ----------
def load_items(key,n=500):
    d={}
    for it in harness.LOADERS[key](n):
        d[it["id"]]={"classes":it["classes"],"counts":np.array(it["label_count"],float),"n":it["n_annot"],"gold":it["gold"]}
    return d
def load_preds(key):
    d=defaultdict(dict)
    for l in open(OUT/f"{key}.judgments.jsonl"):
        try: r=json.loads(l)
        except: continue
        if r.get("run_id")=="full" and r.get("pred") is not None: d[r["item_id"]][r["model"]]=r["pred"]
    return d
def alt_wr(items,preds,model,classes,counts_key="counts",eps=0.0):
    wins=tot=0
    for i,it in items.items():
        if model not in preds.get(i,{}): continue
        counts=it[counts_key]; N=int(counts.sum())
        if N<2: continue
        ci=classes.index(preds[i][model]) if preds[i][model] in classes else -1
        for c_idx,c_n in enumerate(counts):
            for _ in range(int(c_n)):
                rest=counts.copy(); rest[c_idx]-=1; mr=rest.argmax()
                wins+= 1 if (1.0 if ci==mr else 0.0)>=(1.0 if c_idx==mr else 0.0)-eps else 0; tot+=1
    return wins/tot if tot else float("nan")

# ---------- B. alt-test bootstrap CIs ----------
def alttest_ci(keys):
    res={}
    for key in keys:
        items=load_items(key); preds=load_preds(key)
        classes=items[next(iter(items))]["classes"]
        ids=list(items); res[key]={}
        sc=json.load(open(OUT/f"{key}.full.scores.json")); mods=[m for m in sc["models"] if "accuracy" in sc["models"][m]]
        for m in mods:
            pt=alt_wr(items,preds,m,classes)
            boots=[]
            for _ in range(300):
                samp=RNG.choice(ids,len(ids),replace=True)
                sub={f"{j}":items[i] for j,i in enumerate(samp)}; subp={f"{j}":preds.get(i,{}) for j,i in enumerate(samp)}
                boots.append(alt_wr(sub,subp,m,classes))
            boots=[b for b in boots if b==b]
            lo,hi=float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5))
            res[key][m]={"wr":round(float(pt),3),"CI":[round(lo,3),round(hi,3)],"passes_above_half":bool(lo>0.5)}
    return res

# ---------- C. annotator-count ablation on ChaosNLI ----------
def annotator_ablation(key="chaosnli_snli",levels=(3,5,10)):
    items=load_items(key); preds=load_preds(key); classes=items[next(iter(items))]["classes"]
    full_ceiling=float(np.mean([it["counts"].max()/it["counts"].sum() for it in items.values()]))
    res={"full_ceiling":round(full_ceiling,3),"levels":{}}
    best="gemma-4-31b"
    for A in levels:
        ceils=[]; sub_items={}
        for i,it in items.items():
            p=it["counts"]/it["counts"].sum()
            draw=RNG.multinomial(A,p)   # subsample A annotators from empirical dist
            if draw.sum()<2: continue
            ceils.append(draw.max()/draw.sum())
            ni=dict(it); ni["counts_sub"]=draw.astype(float); sub_items[i]=ni
        # alt-test on subsampled counts for the best model
        wr=alt_wr(sub_items,preds,best,classes,counts_key="counts_sub")
        res["levels"][A]={"ceiling":round(float(np.mean(ceils)),3),"alt_wr_best":round(float(wr),3)}
    res["best_model"]=best
    return res

# ---------- D. standardized GLM + cluster bootstrap ----------
def glm_standardized():
    import statsmodels.formula.api as smf, statsmodels.api as sm
    rows=[]
    keys=["chaosnli_snli","chaosnli_mnli","go_emotions","social_bias_frames","hatexplain","multipico"]
    ceil={}
    for key in keys:
        items=load_items(key); preds=load_preds(key); classes=items[next(iter(items))]["classes"]; K=len(classes)
        c=float(np.mean([it["counts"].max()/it["counts"].sum() for it in items.values()])); ceil[key]=c
        agg=np.zeros(K)
        for it in items.values(): agg+=it["counts"]
        base=float(agg.max()/agg.sum())
        for i,it in items.items():
            for m in preds.get(i,{}):
                rows.append({"dataset":key,"item":f"{key}:{i}","correct":int(preds[i][m]==it["gold"]),
                             "subjectivity":1-c,"logsize":math.log10(SIZE_B[m]),"nclasses":K,"baserate":base})
    df=pd.DataFrame(rows)
    for col in ["subjectivity","logsize","nclasses","baserate"]:
        df[col+"_z"]=(df[col]-df[col].mean())/df[col].std()
    g=smf.glm("correct ~ subjectivity_z + logsize_z + nclasses_z + baserate_z",data=df,family=sm.families.Binomial()).fit(
        cov_type="cluster",cov_kwds={"groups":df["item"]})
    fe={n:{"OR_perSD":round(float(np.exp(g.params[n])),3),"z":round(float(g.tvalues[n]),2),"p":float(g.pvalues[n])} for n in g.params.index}
    # dataset-cluster bootstrap of the meta-regression slope (acc ~ subjectivity over 6 datasets)
    dd=df.groupby("dataset").agg(acc=("correct","mean"),subj=("subjectivity","first")).reset_index()
    def slope(d):
        x=d["subj"].values;y=d["acc"].values;A=np.vstack([x,np.ones_like(x)]).T
        return float(np.linalg.lstsq(A,y,rcond=None)[0][0])
    dsl=list(dd["dataset"]); boots=[]
    for _ in range(2000):
        samp=RNG.choice(dsl,len(dsl),replace=True)
        boots.append(slope(dd.set_index("dataset").loc[samp].reset_index()))
    return {"per_SD_odds_ratios":fe,"meta_slope":round(slope(dd),3),
            "meta_slope_clusterboot_CI":[round(float(np.percentile(boots,2.5)),3),round(float(np.percentile(boots,97.5)),3)]}

print("A. Study-1b bootstrap CIs..."); out["study1b_ci"]=study1b_ci()
beats=[(m,a) for m in out["study1b_ci"] for a in ASP if out["study1b_ci"][m][a]["exceeds_reported"]]
print(f"   per-aspect cells whose 95% CI strictly exceeds reported G-Eval: {len(beats)} of {len(out['study1b_ci'])*4}")
print("B. alt-test bootstrap CIs..."); out["alttest_ci"]=alttest_ci(["chaosnli_snli","chaosnli_mnli","social_bias_frames","multipico","go_emotions","hatexplain"])
print("C. annotator-count ablation (ChaosNLI-SNLI)..."); out["annotator_ablation"]=annotator_ablation()
print("   ",out["annotator_ablation"])
print("D. standardized GLM + cluster bootstrap..."); out["glm_std"]=glm_standardized()
for n,v in out["glm_std"]["per_SD_odds_ratios"].items(): print(f"   {n:16s} OR/SD={v['OR_perSD']:.2f} z={v['z']:+.2f} p={v['p']:.1e}")
print("   meta slope",out["glm_std"]["meta_slope"],"cluster-boot CI",out["glm_std"]["meta_slope_clusterboot_CI"])
DATA.mkdir(exist_ok=True); json.dump(out,open(DATA/"rev2_analysis.json","w"),indent=2)

# macros
def f(x):
    try: return f"{float(x):.2f}"
    except: return str(x)
gz=out["glm_std"]["per_SD_odds_ratios"]
macros={"RTwobeats":str(len(beats)),
  "RTwosubjOR":f(gz["subjectivity_z"]["OR_perSD"]),"RTwosubjP":("%.2f"%gz["subjectivity_z"]["p"]),
  "RTwobaseOR":f(gz["baserate_z"]["OR_perSD"]),"RTwosizeOR":f(gz["logsize_z"]["OR_perSD"]),
  "RTwometaCIlo":f(out["glm_std"]["meta_slope_clusterboot_CI"][0]),"RTwometaCIhi":f(out["glm_std"]["meta_slope_clusterboot_CI"][1]),
  "RTwoablCeilThree":f(out["annotator_ablation"]["levels"][3]["ceiling"]),
  "RTwoablWRThree":f(out["annotator_ablation"]["levels"][3]["alt_wr_best"]),
  "RTwoablCeilFull":f(out["annotator_ablation"]["full_ceiling"])}
(TAB/"rev2_autostats.tex").write_text("% auto-generated by rev2_analysis.py\n"+"".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k,v in macros.items()))
print("\nwrote data/rev2_analysis.json + rev2_autostats.tex"); print(macros)
