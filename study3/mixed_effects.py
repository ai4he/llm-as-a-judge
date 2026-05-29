#!/usr/bin/env python3
"""Study 3 inferential layer: a mixed-effects logistic model of judge correctness +
Benjamini-Hochberg FDR across the per-(dataset x model) ceiling-gap tests.

Design: each observation is one (item i, model m) judgment, correct = 1[pred==gold].
Predictors:
  - subjectivity_c  = (1 - human_ceiling_acc) for the item's dataset, centered  [dataset-level]
  - logsize_c       = log10(model params in B), centered                        [model-level]
Random intercepts: (1|dataset) and (1|item) -- absorb the dataset/item structure so the
fixed effects are estimated within that structure.  We report:
  (a) a Bayesian mixed-effects logistic GLMM (variance components + fixed posteriors), and
  (b) a logistic GLM with item-cluster-robust SEs as an inferential cross-check (clean p-values).
FDR: 30 binomial tests of 'model accuracy == human-ceiling accuracy' (one per dataset x model),
Benjamini-Hochberg q-values at alpha=.05.
Writes data/study3_mixedeffects.json + paper/tables/study3_stats_autostats.tex.
"""
import json, sys, math
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
import statsmodels.api as sm
ROOT=Path(__file__).resolve().parent.parent; OUT=ROOT/"study3"/"outputs"; DATA=ROOT/"data"; TAB=ROOT/"paper"/"tables"
RUN="full"
DS=[("chaosnli_snli","ChaosNLI-SNLI"),("chaosnli_mnli","ChaosNLI-MNLI"),
    ("go_emotions","GoEmotions"),("social_bias_frames","SBIC"),("hatexplain","HateXplain"),
    ("multipico","MultiPICo")]
DS=[d for d in DS if (OUT/f"{d[0]}.full.scores.json").exists()]
SIZE_B={"qwen3.5-9b":9,"qwen3.6-27b-fp8":27,"gemma-4-31b":31,"gptoss-120b":120,
        "deepseek-v4-pro":671,"glm-5.1-fp8":754}  # total params (B); log10 scale

def load():
    rows=[]; ceil={}
    for key,_ in DS:
        sc=json.load(open(OUT/f"{key}.full.scores.json")); ceil[key]=sc["human_ceiling_acc"]
        for l in open(OUT/f"{key}.judgments.jsonl"):
            try: d=json.loads(l)
            except: continue
            if d.get("run_id")!=RUN or d.get("pred") is None or d.get("gold") is None: continue
            rows.append({"dataset":key,"item":f"{key}:{d['item_id']}","model":d["model"],
                         "correct":int(d["pred"]==d["gold"])})
    df=pd.DataFrame(rows).drop_duplicates(["dataset","item","model"])
    df["subjectivity"]=df["dataset"].map(lambda k:1-ceil[k])
    df["logsize"]=df["model"].map(lambda m:math.log10(SIZE_B[m]))
    df["subjectivity_c"]=df["subjectivity"]-df["subjectivity"].mean()
    df["logsize_c"]=df["logsize"]-df["logsize"].mean()
    return df,ceil

def main():
    df,ceil=load()
    print(f"observations: {len(df)}  items: {df['item'].nunique()}  models: {df['model'].nunique()}  datasets: {df['dataset'].nunique()}")
    out={"n_obs":int(len(df)),"n_items":int(df["item"].nunique()),"sizes_B":SIZE_B,"ceiling":ceil}

    # ---- (b) cluster-robust logistic GLM (primary inferential p-values) ----
    glm=smf.glm("correct ~ subjectivity_c + logsize_c",data=df,family=sm.families.Binomial()).fit(
        cov_type="cluster",cov_kwds={"groups":df["item"]})
    cr={}
    for name in glm.params.index:
        cr[name]={"beta":float(glm.params[name]),"se":float(glm.bse[name]),
                  "z":float(glm.tvalues[name]),"p":float(glm.pvalues[name]),
                  "OR":float(np.exp(glm.params[name])),
                  "ci":[float(np.exp(glm.conf_int().loc[name,0])),float(np.exp(glm.conf_int().loc[name,1]))]}
    out["logit_cluster_robust"]=cr
    print("\n-- logistic GLM (item-cluster-robust SE) --")
    for k,v in cr.items(): print(f"   {k:16s} beta={v['beta']:+.3f}  OR={v['OR']:.3f}  z={v['z']:+.2f}  p={v['p']:.1e}")

    # ---- (a) Bayesian mixed-effects logistic GLMM with (1|dataset)+(1|item) ----
    glmm=None
    try:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
        m=BinomialBayesMixedGLM.from_formula(
            "correct ~ subjectivity_c + logsize_c",
            {"dataset":"0+C(dataset)","item":"0+C(item)"}, df)
        r=m.fit_vb(verbose=False)
        fe=list(r.model.exog_names)
        gd={}
        for i,nm in enumerate(fe):
            gd[nm]={"post_mean":float(r.fe_mean[i]),"post_sd":float(r.fe_sd[i]),
                    "z":float(r.fe_mean[i]/r.fe_sd[i])}
        # variance-component posterior means (log-sd parameters -> sd)
        vc={nm:float(np.exp(r.vcp_mean[i])) for i,nm in enumerate(r.model.vcp_names)}
        glmm={"fixed":gd,"vc_sd":vc}
        print("\n-- Bayesian mixed-effects logistic GLMM (1|dataset)+(1|item) --")
        for k,v in gd.items(): print(f"   {k:16s} post_mean={v['post_mean']:+.3f}  sd={v['post_sd']:.3f}  z={v['z']:+.2f}")
        print(f"   random-intercept SDs: "+", ".join(f"{k}={v:.2f}" for k,v in vc.items()))
    except Exception as e:
        print("GLMM failed:",repr(e)[:120]); glmm={"error":repr(e)[:200]}
    out["glmm"]=glmm

    # ---- FDR across per-(dataset x model) ceiling-gap tests ----
    tests=[]
    for key,name in DS:
        sc=json.load(open(OUT/f"{key}.full.scores.json")); c=ceil[key]
        for mod,md in sc["models"].items():
            if "accuracy" not in md: continue
            n=md["n"]; k=int(round(md["accuracy"]*n))
            p=binomtest(k,n,c).pvalue            # H0: model accuracy == ceiling
            tests.append({"dataset":key,"model":mod,"acc":md["accuracy"],"ceiling":c,
                          "delta":md["accuracy"]-c,"n":n,"p":p})
    pvals=[t["p"] for t in tests]
    rej,q,_,_=multipletests(pvals,alpha=0.05,method="fdr_bh")
    for t,r,qq in zip(tests,rej,q): t["q"]=float(qq); t["sig_fdr"]=bool(r)
    n_sig=int(sum(rej)); n_below=int(sum(1 for t in tests if t["sig_fdr"] and t["delta"]<0))
    n_above=int(sum(1 for t in tests if t["sig_fdr"] and t["delta"]>0))
    out["fdr"]={"n_tests":len(tests),"alpha":0.05,"method":"fdr_bh",
                "n_sig":n_sig,"n_sig_below_ceiling":n_below,"n_sig_above_ceiling":n_above,"tests":tests}
    print(f"\n-- FDR (BH, alpha=.05) over {len(tests)} ceiling-gap tests: "
          f"{n_sig} significant ({n_below} below ceiling, {n_above} above) --")

    DATA.mkdir(exist_ok=True); json.dump(out,open(DATA/"study3_mixedeffects.json","w"),indent=2)
    # macros
    subj=cr.get("subjectivity_c",{}); size=cr.get("logsize_c",{})
    def f(x,d=2): return f"{x:.{d}f}"
    macros={
     "SMEsubjOR":f(subj.get("OR",float('nan'))), "SMEsubjP":("%.0e"%subj.get("p",1)).replace("e-0","e-").replace("e+0","e"),
     "SMEsizeOR":f(size.get("OR",float('nan'))), "SMEsizeP":f(size.get("p",float('nan')),3),
     "SMEsizeZ":f(size.get("z",float('nan'))),
     "SMEfdrtests":str(len(tests)),"SMEfdrsig":str(n_sig),"SMEfdrbelow":str(n_below),"SMEfdrabove":str(n_above),
    }
    if glmm and "vc_sd" in glmm:
        macros["SMEvcdataset"]=f(glmm["vc_sd"].get("dataset",float('nan')))
        macros["SMEvcitem"]=f(glmm["vc_sd"].get("item",float('nan')))
        macros["SMEglmmsubj"]=f(glmm["fixed"]["subjectivity_c"]["post_mean"],2)
        macros["SMEglmmsize"]=f(glmm["fixed"]["logsize_c"]["post_mean"],2)
    (TAB/"study3_stats_autostats.tex").write_text("% auto-generated by mixed_effects.py\n"+
        "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k,v in macros.items()))
    print("\nwrote data/study3_mixedeffects.json + paper/tables/study3_stats_autostats.tex"); print(macros)

if __name__=="__main__": main()
