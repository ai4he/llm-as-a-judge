#!/usr/bin/env python3
"""Study 2 figures + tables + autostats (HCC/HCI/qualitative). Reads data/study2_corpus.csv."""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT=Path(__file__).resolve().parent.parent; DATA,FIG,TAB=ROOT/"data",ROOT/"figures",ROOT/"paper"/"tables"
plt.rcParams.update({"figure.dpi":120,"savefig.dpi":300,"font.size":11,"axes.titlesize":13,
    "axes.titleweight":"bold","axes.grid":True,"grid.alpha":.25,"axes.axisbelow":True,"font.family":"DejaVu Sans"})
VC={1:"#1a9850",2:"#91cf60",3:"#fee08b",4:"#fc8d59",5:"#d73027"}
VORD=[1,2,3,4,5]; VNAME={1:"Validated",2:"Promising",3:"Mixed",4:"Caution",5:"Unreliable"}
RDYLGN=mcolors.LinearSegmentedColormap.from_list("relg",["#d73027","#fc8d59","#fee08b","#91cf60","#1a9850"])
df=pd.read_csv(DATA/"study2_corpus.csv"); PR=json.load(open(DATA/"study2_prisma_counts.json"))
df["agreement"]=pd.to_numeric(df["agreement"],errors="coerce"); N=len(df)
def save(fig,n): fig.savefig(FIG/f"{n}.png",bbox_inches="tight"); fig.savefig(FIG/f"{n}.pdf",bbox_inches="tight"); plt.close(fig); print("wrote",n)

# S2-Fig1: PRISMA-lite flow for Study 2
def fig_prisma():
    fig,ax=plt.subplots(figsize=(10,4.6)); ax.axis("off"); ax.set_xlim(0,12); ax.set_ylim(0,5)
    def box(x,y,w,h,t,fc="#eef3fb",ec="#33558b"):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.06",fc=fc,ec=ec,lw=1.5))
        ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=9)
    def arr(x1,y1,x2,y2): ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=14,lw=1.4,color="#33558b"))
    box(0.3,3.4,3.0,1.3,f"OpenAlex (Study-2 queries)\nunique works\nn = {PR['identified_openalex']}")
    box(0.3,1.0,3.0,1.2,f"Web deep-search\n+ citation chasing\nn = {PR['identified_web']}")
    box(4.3,2.3,3.0,1.4,f"Topicality-screened\n(HCC/HCI/qual + human\ncomparison)\non-topic n = {PR['openalex_topical']}")
    box(8.4,2.5,3.2,1.7,"Excluded at eligibility\n"
        f"  no human comparison  {PR['excluded_eligibility']['no_human_compare']}\n"
        f"  tool only            {PR['excluded_eligibility']['tool_only']}\n"
        f"  off-topic simulation {PR['excluded_eligibility']['sim_only_offtopic']}\n"
        f"  grey literature      {PR['excluded_eligibility']['grey']}",fc="#fbecec",ec="#b03030")
    box(4.3,0.2,3.0,1.2,f"Included in Study 2\nn = {PR['included']}",fc="#e7f6ec",ec="#1a9850")
    arr(3.3,4.0,4.3,3.2); arr(3.3,1.6,4.3,2.6); arr(5.8,2.3,5.8,1.4); arr(7.3,3.0,8.4,3.2)
    ax.set_title("Figure S2-1.  Study 2 identification (HCC/HCI/qualitative methods)",loc="left",fontsize=12)
    save(fig,"s2_fig01_prisma")

# S2-Fig2: studies per method, colored by mean reliability
def fig_methods():
    g=df.groupby("method").agg(n=("id","count"),rel=("reliability_score","mean")).sort_values("n")
    fig,ax=plt.subplots(figsize=(8.4,7))
    ax.barh(g.index,g["n"],color=[RDYLGN((m-1)/4) for m in g["rel"]],edgecolor="#444",lw=.5)
    for i,(nn,m) in enumerate(zip(g["n"],g["rel"])): ax.text(nn+0.1,i,f"{nn} (rel {m:.1f})",va="center",fontsize=8.5)
    ax.set_xlabel("number of studies (colour = mean reliability 1–5)")
    ax.set_title(f"Figure S2-2.  HCC/HCI/qualitative methods studied with LLMs (n={N})")
    save(fig,"s2_fig02_methods")

# S2-Fig3: role + stance distributions
def fig_role_stance():
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    role=df["role"].value_counts().reindex(["augment","simulate","replace"]).fillna(0)
    axes[0].bar(role.index,role.values,color=["#1a9850","#fee08b","#d73027"],edgecolor="w")
    for i,v in enumerate(role.values): axes[0].text(i,v+0.3,int(v),ha="center",fontsize=11,fontweight="bold")
    axes[0].set_ylabel("studies"); axes[0].set_title("(a) Intended role of the LLM")
    st=df["stance"].value_counts().reindex(["enabling","tension","critical"]).fillna(0)
    axes[1].bar(st.index,st.values,color=["#1a9850","#fee08b","#d73027"],edgecolor="w")
    for i,v in enumerate(st.values): axes[1].text(i,v+0.3,int(v),ha="center",fontsize=11,fontweight="bold")
    axes[1].set_ylabel("studies"); axes[1].set_title("(b) Epistemic stance toward substitution")
    fig.suptitle("Figure S2-3.  How HCC research positions LLMs: augment vs replace, and epistemic stance",
                 fontsize=13,fontweight="bold")
    save(fig,"s2_fig03_role_stance")

# S2-Fig4: verdict by role (stacked %)  -- KEY
def fig_verdict_by_role():
    order=["augment","simulate","replace"]
    piv=df.pivot_table(index="role",columns="verdict",values="id",aggfunc="count").reindex(index=order,columns=VORD).fillna(0)
    frac=piv.div(piv.sum(axis=1),axis=0)*100
    fig,ax=plt.subplots(figsize=(8.6,4.6)); left=np.zeros(len(order))
    for v in VORD:
        ax.barh(range(len(order)),frac[v],left=left,color=VC[v],label=VNAME[v],edgecolor="w")
        for i,val in enumerate(frac[v]):
            if val>6: ax.text(left[i]+val/2,i,f"{val:.0f}%",ha="center",va="center",fontsize=9)
        left+=frac[v].values
    ax.set_yticks(range(len(order))); ax.set_yticklabels([f"{o}\n(n={int(piv.loc[o].sum())})" for o in order])
    ax.set_xlabel("share of studies (%)"); ax.set_xlim(0,100)
    ax.legend(ncol=5,fontsize=8.5,frameon=False,bbox_to_anchor=(0.5,1.13),loc="center")
    ax.set_title("Figure S2-4.  Reliability verdict by LLM role: augmentation is reliable, replacement/simulation are not",loc="left",fontsize=11.5)
    save(fig,"s2_fig04_verdict_by_role")

# S2-Fig5: mean reliability by method
def fig_reliability_method():
    g=df.groupby("method")["reliability_score"].agg(["mean","count"]).sort_values("mean")
    fig,ax=plt.subplots(figsize=(8.4,7))
    ax.barh(g.index,g["mean"],color=[RDYLGN((m-1)/4) for m in g["mean"]],edgecolor="#333")
    for i,(m,c) in enumerate(zip(g["mean"],g["count"])): ax.text(m+0.05,i,f"{m:.2f} (n={c})",va="center",fontsize=8.5)
    ax.axvline(3,color="#888",ls="--",lw=1); ax.set_xlim(1,5.3)
    ax.set_xlabel("mean reliability score (1 = unreliable … 5 = validated)")
    ax.set_title("Figure S2-5.  Reliability of LLM substitution by HCC/HCI method")
    save(fig,"s2_fig05_reliability_by_method")

# S2-Fig6: community coverage + role-reliability summary
def fig_community():
    fig,axes=plt.subplots(1,2,figsize=(12.5,4.8))
    cc=df["community"].value_counts()
    axes[0].barh(cc.index[::-1],cc.values[::-1],color="#4575b4",edgecolor="w")
    for i,v in enumerate(cc.values[::-1]): axes[0].text(v+0.1,i,str(v),va="center",fontsize=9)
    axes[0].set_xlabel("studies"); axes[0].set_title("(a) Research community")
    rr=df.groupby("role")["reliability_score"].mean().reindex(["augment","simulate","replace"])
    axes[1].bar(rr.index,rr.values,color=[RDYLGN((m-1)/4) for m in rr.values],edgecolor="#333")
    for i,v in enumerate(rr.values): axes[1].text(i,v+0.05,f"{v:.2f}",ha="center",fontsize=10,fontweight="bold")
    axes[1].axhline(3,color="#888",ls="--",lw=1); axes[1].set_ylim(0,5); axes[1].set_ylabel("mean reliability")
    axes[1].set_title("(b) Mean reliability by LLM role")
    fig.suptitle("Figure S2-6.  Study-2 community coverage and the augment–replace reliability gap",fontsize=13,fontweight="bold")
    save(fig,"s2_fig06_community_role")

def tables_and_macros():
    esc=lambda x:str(x).replace("&",r"\&").replace("%",r"\%").replace("_",r"\_")
    # method decision table
    g=df.groupby("method").agg(n=("id","count"),rel=("reliability_score","mean"),
        role=("role",lambda s:s.mode().iloc[0])).reset_index().sort_values("rel",ascending=False)
    def light(m): return r"\cellcolor{g!55}\textsc{green}" if m>=3.5 else (r"\cellcolor{a!60}\textsc{amber}" if m>=2.5 else r"\cellcolor{r!55}\textsc{red}")
    rows=[f"{esc(r['method'])} & {r['n']} & {esc(r['role'])} & {r['rel']:.2f} & {light(r['rel'])} \\\\" for _,r in g.iterrows()]
    tex=(r"\begin{tabular}{@{}l r l c c@{}}"+"\n\\toprule\nMethod & $n$ & Modal role & Mean rel. & Signal \\\\"+"\n\\midrule\n"
         +"\n".join(rows)+"\n\\bottomrule\n\\end{tabular}")
    (TAB/"tab_study2_methods.tex").write_text(tex)
    # autostats
    rc=df["role"].value_counts(); sc=df["stance"].value_counts(); cc=df["community"].value_counts()
    relrole=df.groupby("role")["reliability_score"].mean()
    M={"STN":N,"STaugment":int(rc.get("augment",0)),"STreplace":int(rc.get("replace",0)),
       "STsimulate":int(rc.get("simulate",0)),"STtension":int(sc.get("tension",0)),
       "STcritical":int(sc.get("critical",0)),"STenabling":int(sc.get("enabling",0)),
       "SThci":int(cc.get("HCI",0)),"STmethods":df["method"].nunique(),
       "STrelaugment":round(float(relrole.get("augment",0)),2),"STrelreplace":round(float(relrole.get("replace",0)),2),
       "STrelsim":round(float(relrole.get("simulate",0)),2),"STincluded":PR["included"],"SToa":PR["identified_openalex"]}
    (TAB/"study2_autostats.tex").write_text("% auto-generated\n"+"".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k,v in M.items()))
    # full study-2 table (appendix)
    RS={"augment":"augment","replace":"replace","simulate":"simulate"}
    lt=[r"\footnotesize"+"\n"+r"\begin{longtable}{@{}l l l l c l@{}}"+"\n"
        r"\caption{Study 2 corpus ("+str(N)+r" studies): HCC/HCI/qualitative methods using LLMs vs human input.}\label{tab:study2full}\\"+"\n\\toprule"+"\n"
        r"Study & Method & Role & Community & Agree & Verdict \\"+"\n\\midrule\n\\endfirsthead"+"\n"
        r"\toprule Study & Method & Role & Community & Agree & Verdict \\ \midrule \endhead"]
    for _,r in df.sort_values(["method","year"]).iterrows():
        ag="-" if pd.isna(r["agreement"]) else f"{r['agreement']:.2f}"
        lt.append(f"{esc(r['authors'])} ({r['year']}) & {esc(r['method'])} & {RS.get(r['role'],r['role'])} & {esc(r['community'])} & {ag} & {VNAME[int(r['verdict'])]} \\\\")
    lt.append("\\bottomrule\n\\end{longtable}")
    (TAB/"tab_study2_full.tex").write_text("\n".join(lt))
    print("wrote study2 tables + autostats"); print(g.to_string(index=False))

if __name__=="__main__":
    fig_prisma(); fig_methods(); fig_role_stance(); fig_verdict_by_role()
    fig_reliability_method(); fig_community(); tables_and_macros()
    print("STUDY 2 FIGURES DONE")
