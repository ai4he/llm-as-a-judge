#!/usr/bin/env python3
"""Study 3 inferential figure: (a) fixed-effect odds ratios with 95% CI (cluster-robust logit),
(b) GLMM variance-component SDs, (c) FDR ceiling-gap dot plot (acc - ceiling per dataset x model,
filled = significant at BH q<.05). Reads data/study3_mixedeffects.json."""
import json
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent.parent; FIG=ROOT/"figures"
d=json.load(open(ROOT/"data"/"study3_mixedeffects.json"))
fig=plt.figure(figsize=(15,4.6)); gs=fig.add_gridspec(1,3,width_ratios=[1.05,0.7,1.5])

# (a) odds ratios
ax=fig.add_subplot(gs[0,0]); cr=d["logit_cluster_robust"]
terms=[("subjectivity_c","subjectivity\n(1 - human ceiling)"),("logsize_c","model size\n(log10 params)")]
y=np.arange(len(terms))[::-1]
for i,(k,lab) in enumerate(terms):
    v=cr[k]; ax.plot([v["ci"][0],v["ci"][1]],[y[i],y[i]],color="#333",lw=2,zorder=2)
    ax.scatter([v["OR"]],[y[i]],s=90,color="#1f78b4" if v["OR"]>1 else "#d73027",zorder=3,edgecolor="k")
    ax.text(v["OR"],y[i]+0.14,f"OR={v['OR']:.2f}, p={v['p']:.0e}",ha="center",fontsize=8)
ax.axvline(1,color="#888",ls="--"); ax.set_yticks(y); ax.set_yticklabels([l for _,l in terms],fontsize=9)
ax.set_xscale("log"); ax.set_xlabel("odds ratio for P(judge correct)"); ax.set_title("(a) Fixed effects (cluster-robust logit)",fontsize=10)
ax.set_ylim(-0.6,len(terms)-0.2); ax.grid(alpha=.25,axis="x")

# (b) variance components
ax=fig.add_subplot(gs[0,1]); vc=d.get("glmm",{}).get("vc_sd",{})
if vc:
    ks=["item","dataset"]; vals=[vc.get(k,0) for k in ks]
    ax.bar(ks,vals,color=["#6a3d9a","#33a02c"],edgecolor="k")
    for i,v in enumerate(vals): ax.text(i,v+0.03,f"{v:.2f}",ha="center",fontsize=9)
ax.set_ylabel("random-intercept SD (logit)"); ax.set_title("(b) GLMM variance\ncomponents",fontsize=10); ax.grid(alpha=.25,axis="y")

# (c) ceiling-gap dot plot with FDR
ax=fig.add_subplot(gs[0,2]); tests=d["fdr"]["tests"]
dss=sorted({t["dataset"] for t in tests}); mods=sorted({t["model"] for t in tests})
cmap={m:c for m,c in zip(mods,plt.cm.tab10.colors)}
for j,ds in enumerate(dss):
    for t in [x for x in tests if x["dataset"]==ds]:
        x=t["delta"]; col=cmap[t["model"]]
        ax.scatter([x],[j+ (mods.index(t["model"])-2.5)*0.11],s=55,
                   color=col if t["sig_fdr"] else "white",edgecolor=col,lw=1.6,zorder=3)
ax.axvline(0,color="#1f78b4",ls="--",lw=1.6,label="human ceiling")
ax.set_yticks(range(len(dss))); ax.set_yticklabels([s.replace("_","\n") for s in dss],fontsize=8)
ax.set_xlabel("accuracy - human ceiling  (left = below ceiling)"); ax.grid(alpha=.25,axis="x")
ax.set_title(f"(c) Ceiling gap per dataset x model  (filled = BH q<.05; "
             f"{d['fdr']['n_sig_below_ceiling']} below, {d['fdr']['n_sig_above_ceiling']} above)",fontsize=9.5)
handles=[plt.Line2D([],[],marker='o',ls='',color=cmap[m],label=m.replace('-fp8','')) for m in mods]
ax.legend(handles=handles,fontsize=6.5,loc="lower left",ncol=2)
fig.suptitle("Figure S3-4.  Study 3 inferential layer: what governs LLM-judge agreement",fontsize=12,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig(FIG/"s3_fig04_stats.png",bbox_inches="tight",dpi=300); fig.savefig(FIG/"s3_fig04_stats.pdf",bbox_inches="tight")
print("wrote s3_fig04_stats")
