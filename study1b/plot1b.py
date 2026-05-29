#!/usr/bin/env python3
"""Study-1b figure: per-aspect summary-level Spearman for each powerful open judge on SummEval,
with the reported G-Eval/GPT-4 bar as a dashed line per aspect."""
import json, sys
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent.parent; OUT=ROOT/"study1b"/"outputs"; FIG=ROOT/"figures"
DATASET=sys.argv[1] if len(sys.argv)>1 else "summeval"; RUN=sys.argv[2] if len(sys.argv)>2 else "powerful"
s=json.load(open(OUT/f"{DATASET}.{RUN}.scores.json")); rep=s["reported"]
ASP=["coherence","consistency","fluency","relevance"]
models=sorted(s["models"],key=lambda m:-s["models"][m]["_avg_summary_spearman"])
fig,ax=plt.subplots(figsize=(11,4.6))
x=np.arange(len(ASP)); w=0.8/len(models)
colors=plt.cm.viridis(np.linspace(0.15,0.85,len(models)))
for i,m in enumerate(models):
    vals=[s["models"][m][a]["summary_spearman"] for a in ASP]
    ax.bar(x+i*w-0.4+w/2,vals,w,label=m.replace("-fp8",""),color=colors[i],edgecolor="#222",lw=.5,zorder=3)
for j,a in enumerate(ASP):
    ax.plot([j-0.42,j+0.42],[rep[a],rep[a]],color="#d73027",ls="--",lw=2,zorder=4,
            label="reported G-Eval-4" if j==0 else None)
    ax.text(j,rep[a]+0.01,f"{rep[a]:.2f}",ha="center",color="#d73027",fontsize=8,fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([a.capitalize() for a in ASP],fontsize=10)
ax.set_ylabel("summary-level Spearman $\\rho$ (vs human means)"); ax.set_ylim(0,max(0.7,
    max(s["models"][m][a]["summary_spearman"] for m in models for a in ASP)+0.08))
ax.set_title("Study 1b: most powerful open judges vs reported G-Eval/GPT-4 on SummEval\n"
             "(bars above the red dashed line surpass the reported proprietary judge)",fontsize=12,fontweight="bold")
ax.legend(fontsize=8,ncol=3,loc="upper center"); ax.grid(alpha=.25,axis="y")
fig.savefig(FIG/"s1b_fig01_spearman.png",bbox_inches="tight",dpi=300); fig.savefig(FIG/"s1b_fig01_spearman.pdf",bbox_inches="tight")
print("wrote s1b_fig01_spearman")
