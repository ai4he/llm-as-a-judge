#!/usr/bin/env python3
"""Study 3 pilot figure: per-model accuracy (95% CI) vs the human ceiling, alt-test winrate,
and contamination flag. Reads the pilot scores + contamination JSON."""
import json
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent.parent; OUT=ROOT/"study3"/"outputs"; FIG=ROOT/"figures"
VC={"Validated":"#1a9850","Promising":"#91cf60","Mixed":"#fee08b","Caution":"#fc8d59","Unreliable":"#d73027"}
S=json.load(open(OUT/"chaosnli_snli.pilot.scores.json"))
cont=json.load(open(OUT/"chaosnli_snli.contamination.json")) if (OUT/"chaosnli_snli.contamination.json").exists() else {}
ms=[m for m in S["models"] if "accuracy" in S["models"][m]]
ms=sorted(ms,key=lambda m:S["models"][m]["accuracy"])
acc=[S["models"][m]["accuracy"] for m in ms]
lo=[S["models"][m]["acc_CI"][0] for m in ms]; hi=[S["models"][m]["acc_CI"][1] for m in ms]
wr=[S["models"][m]["alt_test_winrate"] for m in ms]; vl=[S["models"][m]["verdict_label"] for m in ms]
fig,ax=plt.subplots(figsize=(9,5.2)); x=np.arange(len(ms))
err=[np.array(acc)-np.array(lo),np.array(hi)-np.array(acc)]
ax.bar(x,acc,color=[VC[v] for v in vl],edgecolor="#333",yerr=err,capsize=4,width=0.62,zorder=3)
ax.axhline(S["human_ceiling_acc"],color="#1f78b4",ls="--",lw=1.8,zorder=2,
           label=f"human ceiling (random annotator vs majority) = {S['human_ceiling_acc']}")
ax.axhline(S["human_pairwise_agree"],color="#888",ls=":",lw=1.5,zorder=2,
           label=f"human pairwise agreement = {S['human_pairwise_agree']}")
for i,m in enumerate(ms):
    tag=f"alt-WR {wr[i]:.2f}"+("  PASS" if S['models'][m]['alt_test_pass'] else "  fail")
    ax.text(i,acc[i]+0.03,tag,ha="center",fontsize=8)
    if m in cont: ax.text(i,0.03,f"contam:{cont[m]['contamination_flag']}",ha="center",fontsize=7,color="#600")
if "jury" in S: ax.axhline(S["jury"]["accuracy"],color="#6a3d9a",ls="-.",lw=1.5,label=f"jury (4-vote) acc = {S['jury']['accuracy']}")
ax.set_xticks(x); ax.set_xticklabels(ms,rotation=15,ha="right"); ax.set_ylim(0,1.0)
ax.set_ylabel("agreement with human majority label (accuracy)")
ax.set_title("Figure S3-1 (pilot).  LLM-as-a-judge vs the human ceiling on ChaosNLI (n=%d)\nbar colour = reliability verdict; whiskers = 95%% bootstrap CI"%S["n_items"],fontsize=11)
ax.legend(fontsize=8,loc="lower right",frameon=True)
fig.savefig(FIG/"s3_fig01_pilot.png",bbox_inches="tight",dpi=300); fig.savefig(FIG/"s3_fig01_pilot.pdf",bbox_inches="tight")
print("wrote figures/s3_fig01_pilot.png/.pdf")
