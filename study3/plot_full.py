#!/usr/bin/env python3
"""Study 3 full-run figures: per-dataset accuracy vs human ceiling, and a dataset x model
verdict heatmap. Reads study3/outputs/<dataset>.full.scores.json."""
import json
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
ROOT=Path(__file__).resolve().parent.parent; OUT=ROOT/"study3"/"outputs"; FIG=ROOT/"figures"
VC={"Validated":"#1a9850","Promising":"#91cf60","Mixed":"#fee08b","Caution":"#fc8d59","Unreliable":"#d73027"}
VNUM={"Validated":5,"Promising":4,"Mixed":3,"Caution":2,"Unreliable":1}
RDYLGN=mcolors.LinearSegmentedColormap.from_list("r",["#d73027","#fc8d59","#fee08b","#91cf60","#1a9850"])
DS=[("chaosnli_snli","ChaosNLI-SNLI\n(NLI)"),
    ("chaosnli_mnli","ChaosNLI-MNLI\n(NLI, harder)"),
    ("go_emotions","GoEmotions\n(4-way sentiment)"),
    ("social_bias_frames","SBIC\n(offensive, sensitive)"),
    ("hatexplain","HateXplain\n(hate speech, sensitive)"),
    ("multipico","MultiPICo\n(irony, clean)")]
DS=[d for d in DS if (OUT/f"{d[0]}.full.scores.json").exists()]   # only datasets that have been scored
S={k:json.load(open(OUT/f"{k}.full.scores.json")) for k,_ in DS}
models=sorted({m for s in S.values() for m in s["models"] if "accuracy" in s["models"][m]})

# Fig A: per-dataset accuracy vs ceiling
fig,axes=plt.subplots(1,len(DS),figsize=(4.3*len(DS),5),sharey=True)
for ax,(k,title) in zip(axes,DS):
    s=S[k]; ms=[m for m in models if m in s["models"] and "accuracy" in s["models"][m]]
    acc=[s["models"][m]["accuracy"] for m in ms]
    lo=[s["models"][m]["acc_CI"][0] for m in ms]; hi=[s["models"][m]["acc_CI"][1] for m in ms]
    vl=[s["models"][m]["verdict_label"] for m in ms]
    err=[np.array(acc)-np.array(lo),np.array(hi)-np.array(acc)]
    x=np.arange(len(ms))
    ax.bar(x,acc,color=[VC[v] for v in vl],edgecolor="#333",yerr=err,capsize=3,zorder=3)
    ax.axhline(s["human_ceiling_acc"],color="#1f78b4",ls="--",lw=1.8,label=f"human ceiling {s['human_ceiling_acc']}")
    if "jury" in s: ax.axhline(s["jury"]["accuracy"],color="#6a3d9a",ls="-.",lw=1.4,label=f"jury {s['jury']['accuracy']}")
    for i,m in enumerate(ms):
        ax.text(i,0.02,f"WR{s['models'][m]['alt_test_winrate']:.2f}",ha="center",fontsize=7,rotation=90,color="#333")
    ax.set_xticks(x); ax.set_xticklabels([m.replace("-fp8","").replace("glm-5.1","glm5.1") for m in ms],rotation=40,ha="right",fontsize=8)
    ax.set_title(title,fontsize=10); ax.set_ylim(0,1.0); ax.legend(fontsize=7,loc="upper right"); ax.grid(alpha=.25,axis="y")
axes[0].set_ylabel("agreement with human majority (accuracy)")
fig.suptitle("Figure S3-2.  Study 3: LLM-as-a-judge vs the human ceiling on 3 datasets (n=500 each, 6-model panel)\n"
             "bar colour = reliability verdict; whiskers = 95%% bootstrap CI; WR = alt-test winrate",fontsize=12,fontweight="bold")
fig.savefig(FIG/"s3_fig02_full_accuracy.png",bbox_inches="tight",dpi=300); fig.savefig(FIG/"s3_fig02_full_accuracy.pdf",bbox_inches="tight")
print("wrote s3_fig02_full_accuracy")

# Fig B: dataset x model verdict heatmap
fig,ax=plt.subplots(figsize=(9.5,3.8))
M=np.full((len(DS),len(models)),np.nan)
for i,(k,_) in enumerate(DS):
    for j,m in enumerate(models):
        mm=S[k]["models"].get(m,{})
        if "verdict_label" in mm: M[i,j]=VNUM[mm["verdict_label"]]
im=ax.imshow(M,cmap=RDYLGN,vmin=1,vmax=5,aspect="auto")
ax.set_xticks(range(len(models))); ax.set_xticklabels([m.replace("-fp8","") for m in models],rotation=40,ha="right",fontsize=8)
ax.set_yticks(range(len(DS))); ax.set_yticklabels([k for k,_ in DS],fontsize=9)
for i,(k,_) in enumerate(DS):
    for j,m in enumerate(models):
        mm=S[k]["models"].get(m,{})
        if "accuracy" in mm: ax.text(j,i,f"{mm['accuracy']:.2f}\n{mm['verdict_label'][:4]}",ha="center",va="center",fontsize=7,
                                     color="#222" if 2.2<M[i,j]<4.2 else "w")
cb=fig.colorbar(im,ax=ax,fraction=0.04,pad=0.02,ticks=[1,2,3,4,5]); cb.ax.set_yticklabels(["Unrel","Caut","Mixed","Prom","Valid"])
ax.set_title("Figure S3-3.  Study 3 reliability verdict: dataset x judge model (cell = accuracy + verdict)",fontsize=11)
fig.savefig(FIG/"s3_fig03_full_verdict.png",bbox_inches="tight",dpi=300); fig.savefig(FIG/"s3_fig03_full_verdict.pdf",bbox_inches="tight")
print("wrote s3_fig03_full_verdict")
