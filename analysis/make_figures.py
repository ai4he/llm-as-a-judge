#!/usr/bin/env python3
"""
make_figures.py  --  all analyses, figures and data-driven LaTeX tables for
the PRISMA review "LLM-as-a-Judge vs Human Annotation".

Reads : data/included_studies.csv , data/prisma_counts.json
Writes: figures/fig*.png + fig*.pdf
        paper/tables/*.tex   (reproducible, data-driven)
        data/summary_*.csv   (aggregates)
Run   : python3 analysis/make_figures.py
"""
import json, os, textwrap
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
try:
    from adjustText import adjust_text
    HAVE_ADJUST = True
except Exception:
    HAVE_ADJUST = False

ROOT = Path(__file__).resolve().parent.parent
DATA, FIG, TAB = ROOT/"data", ROOT/"figures", ROOT/"paper"/"tables"
for d in (FIG, TAB): d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 11,
    "axes.grid": True, "grid.alpha": .25, "axes.axisbelow": True,
    "font.family": "DejaVu Sans", "figure.autolayout": False,
})

VERDICT_COLORS = {1:"#1a9850", 2:"#91cf60", 3:"#fee08b", 4:"#fc8d59", 5:"#d73027"}
VERDICT_ORDER  = [1,2,3,4,5]
VERDICT_NAME   = {1:"Validated",2:"Promising\n(conditional)",3:"Mixed /\ntask-dependent",
                  4:"Caution /\nlimited",5:"Unreliable"}
RDYLGN = mcolors.LinearSegmentedColormap.from_list("relg",
            ["#d73027","#fc8d59","#fee08b","#91cf60","#1a9850"])

DOMAIN_FAMILY = {
 "General NLG":"Core NLG / Alignment","Summarization":"Core NLG / Alignment","Dialogue":"Core NLG / Alignment",
 "Machine Translation":"Knowledge / Objective","QA & Factuality":"Knowledge / Objective",
 "Fact-Checking":"Knowledge / Objective","Instruction Following":"Knowledge / Objective",
 "Reasoning & Math":"Knowledge / Objective","Code & SW Eng":"Knowledge / Objective",
 "IR & Relevance":"Annotation / Classification","Social-Science Annotation":"Annotation / Classification",
 "Sentiment & Emotion":"Annotation / Classification","Information Extraction (NER/NLI)":"Annotation / Classification",
 "Content Moderation & Toxicity":"Subjective / Social","Qualitative Coding":"Subjective / Social",
 "Argument Quality":"Subjective / Social","Creative Writing":"Subjective / Social",
 "Medicine & Clinical":"High-stakes / Expert","Mental Health":"High-stakes / Expert",
 "Legal":"High-stakes / Expert","Finance":"High-stakes / Expert","Peer Review & Science":"High-stakes / Expert",
 "Education & Grading":"Education","RAG":"Systems / Applied","Reward Modeling":"Systems / Applied",
 "Multimodal (Vision-Language)":"Multimodal","Text-to-Image":"Multimodal",
 "Multilingual":"Cross-lingual","Methodology & Bias":"Methodology",
}
REAL_TASKS = ["classification-annotation","fact-verification","pairwise-comparison",
              "pointwise-scoring","ranking"]
TASK_SHORT = {"classification-annotation":"Classify /\nannotate","fact-verification":"Fact /\nerror verify",
              "pairwise-comparison":"Pairwise\ncompare","pointwise-scoring":"Pointwise\nscore","ranking":"Ranking"}

df = pd.read_csv(DATA/"included_studies.csv")
PR = json.load(open(DATA/"prisma_counts.json"))
df["agreement"] = pd.to_numeric(df["agreement"], errors="coerce")
df["human_baseline"] = pd.to_numeric(df["human_baseline"], errors="coerce")
N = len(df)

def save(fig, name):
    fig.savefig(FIG/f"{name}.png", bbox_inches="tight")
    fig.savefig(FIG/f"{name}.pdf", bbox_inches="tight")
    plt.close(fig); print("wrote", name)

# ===========================================================================
# Fig 01 - PRISMA 2020 flow diagram
# ===========================================================================
def fig_prisma():
    fig, ax = plt.subplots(figsize=(13,9.2)); ax.axis("off"); ax.set_xlim(0,14); ax.set_ylim(0,12)
    def box(x,y,w,h,txt,fc="#eef3fb",ec="#33558b",fs=8.8):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.07",
                     fc=fc,ec=ec,lw=1.5))
        ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=fs)
    def arr(x1,y1,x2,y2):
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=15,lw=1.5,color="#33558b"))
    for y,lab in [(10.7,"Identification"),(7.4,"Screening"),(2.0,"Included")]:
        ax.text(0.12,y,lab,rotation=90,va="center",ha="center",fontsize=11,fontweight="bold",color="#33558b")
    ax.text(3.4,11.7,"Arm A — bibliographic database (OpenAlex API)",ha="center",fontsize=10,fontweight="bold")
    ax.text(10.3,11.7,"Arm B — web deep-search + citation chasing",ha="center",fontsize=10,fontweight="bold")
    # Arm A (left)
    box(1.4,10.0,4.0,1.2,f"Records retrieved from OpenAlex\n(30 topic queries × 50)\nn = {PR['identified_openalex']}")
    box(1.4,8.45,4.0,1.0,f"Unique records after de-duplication\nn = {PR['openalex_unique']}\n(duplicates removed n = {PR['openalex_dup_removed']})",fc="#f4f4f4")
    box(1.4,6.7,4.0,1.1,f"Records topicality-screened\non-topic n = {PR['openalex_topical']}")
    box(1.4,4.9,4.0,1.1,f"Off-topic / generic LLM works\nexcluded at screening\nn = {PR['openalex_excluded_screening']}",fc="#fbecec",ec="#b03030")
    # Arm B (right)
    box(8.3,10.0,4.0,1.2,f"Records identified via web search\n(59 queries) n = {PR['identified_web']}\n+ citation chasing n = {PR['identified_citation']}")
    box(8.3,8.45,4.0,1.0,f"Unique records after de-duplication\nn = {PR['web_unique']}\n(duplicates removed n = {PR['web_dup_removed']})",fc="#f4f4f4")
    box(8.3,6.7,4.0,1.1,f"Records screened\nscreened-in n = {PR['web_screened_in']}")
    box(8.3,4.9,4.0,1.1,f"Excluded at screening\n(off-topic / no comparison / type)\nn = {PR['web_excluded_screening']}",fc="#fbecec",ec="#b03030")
    # merge
    box(4.6,3.05,4.8,1.1,f"Reports assessed for eligibility (merged, cross-source duplicates removed n = {PR['cross_dup_removed']})\nn = {PR['assessed']}",fs=8.6)
    box(10.0,2.7,3.6,1.9,
        "Excluded, with reasons\n"
        f"  no direct human comparison  n = {PR['excl_no_human_compare']}\n"
        f"  LLM judge used as tool only  n = {PR['excl_tool_only']}\n"
        f"  method only, no human eval  n = {PR['excl_method_only']}\n"
        f"  grey / non-peer literature  n = {PR['excl_grey']}",fc="#fbecec",ec="#b03030",fs=8.2)
    box(4.6,0.9,4.8,1.2,f"Studies included in the review\n(quantitative + qualitative synthesis)\nn = {PR['included']}",
        fc="#e7f6ec",ec="#1a9850",fs=9.6)
    # arrows A
    arr(3.4,10.0,3.4,9.45); arr(3.4,8.45,3.4,7.8); arr(3.4,6.7,3.4,6.0); arr(5.4,7.25,5.4,6.0)
    # arrows B
    arr(10.3,10.0,10.3,9.45); arr(10.3,8.45,10.3,7.8); arr(10.3,6.7,10.3,6.0); arr(8.3,7.25,8.3,6.0)
    # merge arrows
    arr(3.4,6.7,5.0,4.15); arr(10.3,6.7,9.0,4.15)
    arr(7.0,3.05,7.0,2.1); arr(9.4,3.6,10.0,3.6)
    ax.set_title("Figure 1.  PRISMA 2020 flow — two-arm identification (OpenAlex database + web/citation)",
                 fontsize=12.5, loc="left")
    save(fig,"fig01_prisma_flow")

# ===========================================================================
# Fig 02 - overall verdict distribution (donut)
# ===========================================================================
def fig_verdict_overall():
    vc = df["verdict"].value_counts().reindex(VERDICT_ORDER).fillna(0).astype(int)
    fig,ax = plt.subplots(figsize=(7.6,6))
    wedges,_ = ax.pie(vc.values, colors=[VERDICT_COLORS[v] for v in VERDICT_ORDER],
                      startangle=90, counterclock=False, wedgeprops=dict(width=0.42,edgecolor="w"))
    ax.text(0,0,f"{N}\nstudies",ha="center",va="center",fontsize=15,fontweight="bold")
    leg = [f"{VERDICT_NAME[v].replace(chr(10),' ')}  —  {vc[v]}  ({vc[v]/N*100:.0f}%)" for v in VERDICT_ORDER]
    ax.legend(wedges,leg,loc="center left",bbox_to_anchor=(1.0,0.5),frameon=False,fontsize=10,
              title="Reliability verdict")
    ax.set_title("Figure 2.  Reliability verdict across all included studies")
    save(fig,"fig02_verdict_overall")

# ===========================================================================
# Fig 03 - studies per year stacked by verdict
# ===========================================================================
def fig_year():
    piv = df.pivot_table(index="year",columns="verdict",values="id",aggfunc="count").fillna(0)
    piv = piv.reindex(columns=VERDICT_ORDER).fillna(0)
    fig,ax=plt.subplots(figsize=(8,5)); bottom=np.zeros(len(piv))
    for v in VERDICT_ORDER:
        ax.bar(piv.index.astype(str),piv[v],bottom=bottom,color=VERDICT_COLORS[v],
               label=VERDICT_NAME[v].replace("\n"," "),edgecolor="w")
        bottom+=piv[v].values
    for i,t in enumerate(bottom): ax.text(i,t+0.4,int(t),ha="center",fontsize=10,fontweight="bold")
    ax.set_ylabel("number of studies"); ax.set_xlabel("publication year")
    ax.legend(fontsize=8.5,ncol=2,frameon=False)
    ax.set_title("Figure 3.  Included studies by year and reliability verdict")
    save(fig,"fig03_studies_per_year")

# ===========================================================================
# Fig 04 - domain coverage counts
# ===========================================================================
def fig_domaincounts():
    dc = df["domain"].value_counts().sort_values()
    fig,ax=plt.subplots(figsize=(8,9))
    ax.barh(dc.index,dc.values,color="#4575b4",edgecolor="w")
    for i,v in enumerate(dc.values): ax.text(v+0.1,i,str(v),va="center",fontsize=9)
    ax.set_xlabel("number of included studies")
    ax.set_title(f"Figure 4.  Domain / task coverage of the corpus ({N} studies, {df['domain'].nunique()} domains)")
    save(fig,"fig04_domain_counts")

# ===========================================================================
# Fig 05 - mean reliability score by domain (KEY, RQ1)
# ===========================================================================
def fig_reliability_by_domain():
    g = df.groupby("domain")["reliability_score"].agg(["mean","count"]).sort_values("mean")
    fig,ax=plt.subplots(figsize=(8.6,9.5))
    colors=[RDYLGN((m-1)/4) for m in g["mean"]]
    ax.barh(g.index,g["mean"],color=colors,edgecolor="#444",lw=.5)
    for i,(m,c) in enumerate(zip(g["mean"],g["count"])):
        ax.text(m+0.05,i,f"{m:.2f}  (n={c})",va="center",fontsize=8.5)
    ax.axvline(3,color="#888",ls="--",lw=1); ax.text(3.02,0.2,"mixed (3)",fontsize=8,color="#555")
    ax.set_xlim(1,5.6); ax.set_xlabel("mean reliability score  (1 = unreliable … 5 = validated)")
    ax.set_title("Figure 5.  Mean LLM-judge reliability by domain")
    save(fig,"fig05_reliability_by_domain")

# ===========================================================================
# Fig 06 - domain x task reliability landscape heatmap (KEY, RQ2)
# ===========================================================================
def fig_landscape():
    d = df[df["task"].isin(REAL_TASKS)]
    piv = d.pivot_table(index="domain",columns="task",values="reliability_score",aggfunc="mean")
    piv = piv.reindex(columns=REAL_TASKS)
    piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]
    cnt = d.pivot_table(index="domain",columns="task",values="id",aggfunc="count").reindex(
            index=piv.index,columns=REAL_TASKS)
    fig,ax=plt.subplots(figsize=(9.5,11))
    data=np.ma.masked_invalid(piv.values)
    im=ax.imshow(data,cmap=RDYLGN,vmin=1,vmax=5,aspect="auto")
    ax.set_xticks(range(len(REAL_TASKS))); ax.set_xticklabels([TASK_SHORT[t] for t in REAL_TASKS],fontsize=9)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index,fontsize=9)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v=piv.values[i,j]
            if not np.isnan(v):
                c=cnt.values[i,j]; c=0 if np.isnan(c) else int(c)
                ax.text(j,i,f"{v:.1f}\n(n={c})",ha="center",va="center",fontsize=7.5,
                        color="#222" if 2.2<v<4.2 else "w")
    cb=fig.colorbar(im,ax=ax,fraction=0.035,pad=0.02); cb.set_label("mean reliability (1–5)")
    ax.set_title("Figure 6.  Reliability landscape: domain × judging task")
    ax.set_xlabel("judging task formulation")
    save(fig,"fig06_reliability_landscape")

# ===========================================================================
# Fig 07 - subjectivity vs reliability (KEY, RQ2)
# ===========================================================================
def _stacked_by(col, order, title, fname, xlabel):
    sub=df[df[col].isin(order)]
    piv=sub.pivot_table(index=col,columns="verdict",values="id",aggfunc="count").reindex(
          index=order,columns=VERDICT_ORDER).fillna(0)
    frac=piv.div(piv.sum(axis=1),axis=0)*100
    fig,ax=plt.subplots(figsize=(8,5)); left=np.zeros(len(order))
    for v in VERDICT_ORDER:
        ax.barh(range(len(order)),frac[v],left=left,color=VERDICT_COLORS[v],
                label=VERDICT_NAME[v].replace("\n"," "),edgecolor="w")
        for i,val in enumerate(frac[v]):
            if val>6: ax.text(left[i]+val/2,i,f"{val:.0f}%",ha="center",va="center",fontsize=8.5)
        left+=frac[v].values
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{o}\n(n={int(piv.loc[o].sum())})" for o in order])
    ax.set_xlabel("share of studies (%)"); ax.set_xlim(0,100)
    ax.legend(fontsize=8,ncol=3,frameon=False,bbox_to_anchor=(0.5,1.12),loc="center")
    ax.set_title(title,loc="left")
    save(fig,fname)

# ===========================================================================
# Fig 09 - agreement values forest (RQ3)
# ===========================================================================
def fig_forest():
    d=df.dropna(subset=["agreement"]).copy()
    fam_order=["raw-agreement","chance-corrected","correlation"]
    d["fam"]=pd.Categorical(d["metric_family"],fam_order,ordered=True)
    d=d.sort_values(["fam","agreement"])
    doms=d["domain"].unique()
    cmap=plt.get_cmap("tab20"); dcol={dm:cmap(i%20) for i,dm in enumerate(sorted(doms))}
    fig,ax=plt.subplots(figsize=(9,11)); y=np.arange(len(d))
    ax.barh(y,d["agreement"],color=[dcol[x] for x in d["domain"]],edgecolor="#333",lw=.4,height=.62)
    for yi,(_,r) in zip(y,d.iterrows()):
        ax.text(min(r["agreement"]+0.01,0.99),yi,f"{r['agreement']:.2f} ({r['metric']})",
                va="center",fontsize=7.3)
        if not np.isnan(r["human_baseline"]):
            ax.plot(r["human_baseline"],yi,marker="D",color="k",ms=6,zorder=5)
    ax.set_yticks(y); ax.set_yticklabels([f"{r.authors} {r.year} — {r.domain}" for _,r in d.iterrows()],fontsize=7)
    ax.axvspan(0.6,0.8,color="#999",alpha=.12); ax.axvline(0.8,color="#1a9850",ls=":",lw=1)
    ax.axvline(0.6,color="#fc8d59",ls=":",lw=1)
    ax.set_xlim(0,1.04); ax.set_xlabel("reported LLM–human agreement (coefficient or proportion)")
    leg=[Line2D([0],[0],marker="s",color="w",markerfacecolor=dcol[dm],markersize=9,label=dm) for dm in sorted(doms)]
    leg.append(Line2D([0],[0],marker="D",color="w",markerfacecolor="k",markersize=8,label="human–human baseline"))
    ax.legend(handles=leg,fontsize=7,loc="lower right",frameon=True,ncol=1)
    ax.set_title("Figure 9.  Reported LLM–human agreement, by study (grouped by metric family)")
    save(fig,"fig09_agreement_forest")

# ===========================================================================
# Fig 10 - parity scatter LLM-human vs human-human (RQ3)
# ===========================================================================
def fig_parity():
    d=df.dropna(subset=["agreement","human_baseline"]).copy()
    fig,ax=plt.subplots(figsize=(7,6.6))
    ax.plot([0,1],[0,1],color="#444",ls="--",label="parity (LLM = human ceiling)")
    ptexts=[]
    for _,r in d.iterrows():
        col=VERDICT_COLORS[int(r["verdict"])]
        ax.scatter(r["human_baseline"],r["agreement"],s=120,color=col,edgecolor="k",zorder=5)
        ptexts.append(ax.text(r["human_baseline"],r["agreement"],
                      f"{r['authors'].split()[0]} {r['year']}\n({r['domain']})",fontsize=7.5))
    if HAVE_ADJUST:
        adjust_text(ptexts,ax=ax,expand=(1.3,1.6),
                    arrowprops=dict(arrowstyle="-",color="#888",lw=.6))
    ax.set_xlabel("human–human agreement (ceiling)"); ax.set_ylabel("LLM–human agreement")
    ax.set_xlim(0.4,0.95); ax.set_ylim(0.4,0.95)
    ax.fill_between([0,1],[0,1],1,color="#1a9850",alpha=.06)
    ax.text(0.46,0.9,"LLM ≥ human ceiling",fontsize=9,color="#1a9850")
    ax.text(0.7,0.46,"LLM < human ceiling",fontsize=9,color="#d73027")
    leg=[Line2D([0],[0],marker="o",color="w",markerfacecolor=VERDICT_COLORS[v],markeredgecolor="k",
                markersize=10,label=VERDICT_NAME[v].replace("\n"," ")) for v in VERDICT_ORDER if v in d["verdict"].values]
    leg.insert(0,Line2D([0],[0],color="#444",ls="--",label="parity line"))
    ax.legend(handles=leg,fontsize=8,loc="lower right",frameon=True)
    ax.set_title("Figure 10.  LLM–human vs human–human agreement (studies reporting both)")
    save(fig,"fig10_parity_scatter")

# ===========================================================================
# Fig 11 - failure-mode / bias frequency (RQ4)
# ===========================================================================
def fig_biases():
    from collections import Counter
    c=Counter()
    for b in df["biases"].dropna():
        for tok in str(b).split(";"):
            t=tok.strip()
            if t and t!="—": c[t]+=1
    # tidy a few labels
    top=dict(sorted(c.items(),key=lambda x:x[1])[-20:])
    fig,ax=plt.subplots(figsize=(8.6,8.2))
    ax.barh(list(top.keys()),list(top.values()),color="#c0504d",edgecolor="w")
    for i,v in enumerate(top.values()): ax.text(v+0.05,i,str(v),va="center",fontsize=9)
    ax.set_xlabel("number of studies reporting the failure mode / bias")
    ax.set_title("Figure 11.  Catalogue of reported LLM-judge failure modes and biases")
    save(fig,"fig11_bias_frequency")

# ===========================================================================
# Fig 12 - verdict by task formulation
# ===========================================================================
def fig_verdict_by_task():
    sub=df[df["task"].isin(REAL_TASKS)]
    piv=sub.pivot_table(index="task",columns="verdict",values="id",aggfunc="count").reindex(
         index=REAL_TASKS,columns=VERDICT_ORDER).fillna(0)
    frac=piv.div(piv.sum(axis=1),axis=0)*100
    fig,ax=plt.subplots(figsize=(8.4,5)); left=np.zeros(len(REAL_TASKS))
    for v in VERDICT_ORDER:
        ax.barh(range(len(REAL_TASKS)),frac[v],left=left,color=VERDICT_COLORS[v],
                label=VERDICT_NAME[v].replace("\n"," "),edgecolor="w")
        for i,val in enumerate(frac[v]):
            if val>6: ax.text(left[i]+val/2,i,f"{val:.0f}%",ha="center",va="center",fontsize=8.5)
        left+=frac[v].values
    ax.set_yticks(range(len(REAL_TASKS)))
    ax.set_yticklabels([f"{TASK_SHORT[t].replace(chr(10),' ')}\n(n={int(piv.loc[t].sum())})" for t in REAL_TASKS])
    ax.set_xlabel("share of studies (%)"); ax.set_xlim(0,100)
    ax.legend(fontsize=8,ncol=3,frameon=False,bbox_to_anchor=(0.5,1.12),loc="center")
    ax.set_title("Figure 12.  Reliability verdict by judging-task formulation",loc="left")
    save(fig,"fig12_verdict_by_task")

# ===========================================================================
# Fig 13 - domain-family reliability (higher-level synthesis)
# ===========================================================================
def fig_family():
    df["family"]=df["domain"].map(DOMAIN_FAMILY)
    g=df.groupby("family")["reliability_score"].agg(["mean","count","std"]).sort_values("mean")
    fig,ax=plt.subplots(figsize=(8.6,5.6))
    colors=[RDYLGN((m-1)/4) for m in g["mean"]]
    ax.barh(g.index,g["mean"],xerr=g["std"].fillna(0),color=colors,edgecolor="#333",
            error_kw=dict(ecolor="#555",lw=1,capsize=3))
    for i,(m,c) in enumerate(zip(g["mean"],g["count"])): ax.text(m+0.06,i,f"{m:.2f} (n={c})",va="center",fontsize=9)
    ax.axvline(3,color="#888",ls="--",lw=1)
    ax.set_xlim(1,5.4); ax.set_xlabel("mean reliability score (1–5)")
    ax.set_title("Figure 13.  Reliability by domain family (mean ± SD)")
    save(fig,"fig13_family_reliability")

# ===========================================================================
# Fig 14 - temporal verdict-share trend
# ===========================================================================
def fig_trend():
    piv=df.pivot_table(index="year",columns="verdict",values="id",aggfunc="count").reindex(
        columns=VERDICT_ORDER).fillna(0)
    frac=piv.div(piv.sum(axis=1),axis=0)
    # reliable share = verdict 1+2 ; problematic = 4+5
    rel=(frac.get(1,0)+frac.get(2,0))*100; prob=(frac.get(4,0)+frac.get(5,0))*100
    fig,ax=plt.subplots(figsize=(8,5))
    ax.plot(frac.index,rel,"-o",color="#1a9850",lw=2,label="reliable share (verdict 1–2)")
    ax.plot(frac.index,prob,"-o",color="#d73027",lw=2,label="problematic share (verdict 4–5)")
    for x,y in zip(frac.index,rel): ax.text(x,y+1.5,f"{y:.0f}%",ha="center",fontsize=8,color="#1a9850")
    for x,y in zip(frac.index,prob): ax.text(x,y+1.5,f"{y:.0f}%",ha="center",fontsize=8,color="#d73027")
    ax.set_ylabel("share of that year's studies (%)"); ax.set_xlabel("publication year")
    ax.set_xticks(frac.index); ax.legend(frameon=False)
    ax.set_title("Figure 14.  Are LLM judges getting more reliable? Verdict shares over time")
    save(fig,"fig14_temporal_trend")

# ===========================================================================
# Fig 15 - decision map: reliability vs stakes/expertise (KEY, RQ6)
# ===========================================================================
def fig_decision_map():
    em={"low":1,"medium":2,"high":3}
    df["exp_num"]=df["expertise"].map(em)
    g=df.groupby("domain").agg(rel=("reliability_score","mean"),
                               stakes=("exp_num","mean"),n=("id","count")).dropna()
    fig,ax=plt.subplots(figsize=(11,8))
    sc=ax.scatter(g["rel"],g["stakes"],s=g["n"]*70,c=[RDYLGN((m-1)/4) for m in g["rel"]],
                  edgecolor="#222",lw=1,alpha=.92,zorder=4)
    texts=[ax.text(r["rel"],r["stakes"],dm,fontsize=8) for dm,r in g.iterrows()]
    if HAVE_ADJUST:
        adjust_text(texts,ax=ax,expand=(1.25,1.7),
                    arrowprops=dict(arrowstyle="-",color="#888",lw=.6))
    ax.axvline(3,color="#888",ls="--",lw=1); ax.axhline(2,color="#888",ls="--",lw=1)
    ax.set_xlim(1.2,5.3); ax.set_ylim(0.7,3.3)
    ax.set_yticks([1,2,3]); ax.set_yticklabels(["low","medium","high"])
    ax.set_xlabel("mean reliability score  (1 = unreliable … 5 = validated)")
    ax.set_ylabel("domain-expertise / stakes required")
    ax.text(4.6,1.05,"AUTOMATE\n(safe to substitute)",color="#1a9850",fontsize=11,fontweight="bold",ha="center")
    ax.text(1.9,1.05,"VALIDATE FIRST\n(easy but unreliable)",color="#b8860b",fontsize=11,fontweight="bold",ha="center")
    ax.text(4.4,3.12,"HUMAN-IN-THE-LOOP\n(reliable but high-stakes)",color="#1f78b4",fontsize=10.5,fontweight="bold",ha="center")
    ax.text(1.95,3.12,"KEEP HUMANS\n(unreliable + high-stakes)",color="#d73027",fontsize=11,fontweight="bold",ha="center")
    ax.set_title("Figure 15.  Decision map — where to deploy LLM-as-a-judge (bubble area ∝ #studies)")
    save(fig,"fig15_decision_map")

# ===========================================================================
# Fig 16 - metric usage + comparator type (methodology insight)
# ===========================================================================
def fig_methodology():
    fig,axes=plt.subplots(1,2,figsize=(12.5,5.2))
    mc=df[df["metric"].astype(str).str.len()>0]["metric"].replace("",np.nan).dropna().value_counts()
    axes[0].bar(mc.index,mc.values,color="#4575b4",edgecolor="w")
    axes[0].set_xticklabels(mc.index,rotation=45,ha="right",fontsize=8)
    axes[0].set_ylabel("number of studies"); axes[0].set_title("(a) Agreement metric reported")
    sub=df["subjectivity"].value_counts().reindex(["objective","mixed","subjective"]).fillna(0)
    axes[1].bar(sub.index,sub.values,color=["#1a9850","#fee08b","#d73027"],edgecolor="w")
    for i,v in enumerate(sub.values): axes[1].text(i,v+0.3,int(v),ha="center",fontsize=10,fontweight="bold")
    axes[1].set_ylabel("number of studies"); axes[1].set_title("(b) Construct subjectivity of judged task")
    fig.suptitle("Figure 16.  Methodological profile of the corpus",fontsize=13,fontweight="bold")
    save(fig,"fig16_methodology")

# ===========================================================================
# Data-driven aggregate tables + LaTeX
# ===========================================================================
def tables():
    esc=lambda x: str(x).replace("&",r"\&").replace("%",r"\%").replace("_",r"\_")
    df["family"]=df["domain"].map(DOMAIN_FAMILY)
    # per-domain summary with traffic-light
    g=df.groupby("domain").agg(n=("id","count"),rel=("reliability_score","mean"),
        verdict_mode=("verdict",lambda s:int(s.mode().iloc[0]))).reset_index().sort_values("rel",ascending=False)
    def light(m):
        return "Green" if m>=3.5 else ("Amber" if m>=2.5 else "Red")
    g["signal"]=g["rel"].map(light)
    g.to_csv(DATA/"summary_by_domain.csv",index=False)
    df.groupby("family")["reliability_score"].agg(["count","mean","std"]).round(2).to_csv(DATA/"summary_by_family.csv")

    # LaTeX: domain decision table
    sig_tex={"Green":r"\cellcolor{g!55}\textsc{green}","Amber":r"\cellcolor{a!60}\textsc{amber}",
             "Red":r"\cellcolor{r!55}\textsc{red}"}
    rows=[]
    for _,r in g.iterrows():
        rows.append(f"{esc(r['domain'])} & {r['n']} & {r['rel']:.2f} & {VERDICT_LABEL_SHORT[r['verdict_mode']]} & {sig_tex[r['signal']]} \\\\")
    tex=(r"\begin{tabular}{@{}l r c l c@{}}"+"\n\\toprule\n"
         r"Domain / task & $n$ & Mean rel. & Modal verdict & Signal \\"+"\n\\midrule\n"
         +"\n".join(rows)+"\n\\bottomrule\n\\end{tabular}")
    (TAB/"tab_domain_decision.tex").write_text(tex)

    # LaTeX: quantitative anchors
    d=df.dropna(subset=["agreement"]).sort_values(["domain","agreement"])
    arows=[]
    for _,r in d.iterrows():
        hb = "" if np.isnan(r["human_baseline"]) else f"{r['human_baseline']:.2f}"
        arows.append(f"{esc(r['authors'])} ({r['year']}) & {esc(r['domain'])} & {esc(r['metric'])} & "
                     f"{r['agreement']:.2f} & {hb} & {VERDICT_LABEL_SHORT[int(r['verdict'])]} \\\\")
    atex=(r"\begin{tabular}{@{}l l l c c l@{}}"+"\n\\toprule\n"
          r"Study & Domain & Metric & LLM--human & Human--human & Verdict \\"+"\n\\midrule\n"
          +"\n".join(arows)+"\n\\bottomrule\n\\end{tabular}")
    (TAB/"tab_anchor_numbers.tex").write_text(atex)

    # ---- auto-stats macros so the manuscript numbers always match the data ----
    vc = df["verdict"].value_counts()
    def vcn(v): return int(vc.get(v,0))
    sig = g["signal"].value_counts()
    n=len(df); nnum=int(df["agreement"].notna().sum())
    pct_rel=round((vcn(1)+vcn(2))/n*100); pct_prob=round((vcn(4)+vcn(5))/n*100)
    def esc_tex(x): return str(x).replace("&","\\&").replace("%","\\%").replace("_","\\_")
    M={"Nstudies":n,"Ndomains":df["domain"].nunique(),"Nvalidated":vcn(1),"Npromising":vcn(2),
       "Nmixed":vcn(3),"Ncaution":vcn(4),"Nunreliable":vcn(5),"Pctreliable":pct_rel,
       "Pctproblematic":pct_prob,"Nnumeric":nnum,"Nprimary":int((df["synthesis_role"]=="primary").sum()),
       "Nsecondary":int((df["synthesis_role"]=="secondary").sum()),
       "Ngreen":int(sig.get("Green",0)),"Namber":int(sig.get("Amber",0)),"Nred":int(sig.get("Red",0)),
       "OAretrieved":PR["identified_openalex"],"OAunique":PR["openalex_unique"],
       "OAtopical":PR["openalex_topical"],"Webhits":PR["identified_web"],"Webunique":PR["web_unique"],
       "Nassessed":PR["assessed"],"Nexclscreen":PR["openalex_excluded_screening"]+PR["web_excluded_screening"],
       "Crossdup":PR["cross_dup_removed"],"Meanrel":round(float(df["reliability_score"].mean()),2),
       "Nparity":int((df["parity"]=="yes").sum())}
    macros="% auto-generated -- do not edit by hand (see analysis/make_figures.py)\n"
    for k,v in M.items(): macros+=f"\\newcommand{{\\{k}}}{{{v}}}\n"
    (TAB/"autostats.tex").write_text(macros)

    # ---- full study appendix longtable ----
    TS={"classification-annotation":"classify","fact-verification":"verify","pairwise-comparison":"pairwise",
        "pointwise-scoring":"pointwise","ranking":"ranking","survey":"survey","review":"review"}
    rows=[]
    for _,r in df.sort_values(["domain","year","id"]).iterrows():
        ag = "--" if pd.isna(r["agreement"]) else f"{r['agreement']:.2f}"
        rows.append(f"\\texttt{{{esc_tex(r['id'])}}} & {esc_tex(r['authors'])} ({r['year']}) & "
                    f"{esc_tex(r['domain'])} & {TS.get(r['task'],r['task'])} & {esc_tex(r['metric']) or '--'} & "
                    f"{ag} & {VERDICT_LABEL_SHORT[int(r['verdict'])]} \\\\")
    lt=(r"\footnotesize"+"\n"+r"\begin{longtable}{@{}l l l l l c l@{}}"+"\n"
        r"\caption{Full extracted-studies corpus ("+str(n)+r" included studies). "
        r"Agreement is the headline coefficient/proportion where reported.}\label{tab:full}\\"+"\n\\toprule\n"
        r"Key & Study & Domain & Task & Metric & Agree & Verdict \\"+"\n\\midrule\n\\endfirsthead\n"
        r"\toprule Key & Study & Domain & Task & Metric & Agree & Verdict \\ \midrule \endhead"+"\n"
        +"\n".join(rows)+"\n\\bottomrule\n\\end{longtable}")
    (TAB/"tab_full_studies.tex").write_text(lt)
    print("wrote tables: tab_domain_decision, tab_anchor_numbers, autostats, tab_full_studies")
    print(g.to_string(index=False))

VERDICT_LABEL_SHORT={1:"Validated",2:"Promising",3:"Mixed",4:"Caution",5:"Unreliable"}

if __name__=="__main__":
    fig_prisma(); fig_verdict_overall(); fig_year(); fig_domaincounts()
    fig_reliability_by_domain(); fig_landscape()
    _stacked_by("subjectivity",["objective","mixed","subjective"],
        "Figure 7.  Reliability verdict by construct subjectivity","fig07_subjectivity","")
    _stacked_by("expertise",["low","medium","high"],
        "Figure 8.  Reliability verdict by required domain expertise","fig08_expertise","")
    fig_forest(); fig_parity(); fig_biases(); fig_verdict_by_task()
    fig_family(); fig_trend(); fig_decision_map(); fig_methodology()
    tables()
    print("\nALL FIGURES + TABLES DONE ->", FIG)
