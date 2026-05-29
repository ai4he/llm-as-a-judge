#!/usr/bin/env python3
"""Build Study-3 LaTeX results table + autostats macros from the 5 scored datasets
(and contamination JSONs if present). Writes paper/tables/tab_study3_results.tex and
paper/tables/study3_autostats.tex."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent; OUT=ROOT/"study3"/"outputs"; TAB=ROOT/"paper"/"tables"
DS=[("chaosnli_snli","ChaosNLI-SNLI","NLI"),("chaosnli_mnli","ChaosNLI-MNLI","NLI"),
    ("go_emotions","GoEmotions","emotion (4-way)"),("social_bias_frames","SBIC","offensive (binary)"),
    ("hatexplain","HateXplain","hate (3-way)"),("multipico","MultiPICo (clean)","irony (binary)")]
esc=lambda s:str(s).replace("&",r"\&").replace("%",r"\%").replace("_",r"\_")
sig={"Validated":r"\cellcolor{g!55}","Promising":r"\cellcolor{g!25}","Mixed":r"\cellcolor{a!60}",
     "Caution":r"\cellcolor{r!30}","Unreliable":r"\cellcolor{r!55}"}
rows=[]; macros={}; n_judg=0; n_pass=0; n_models=0
for key,name,cons in DS:
    p=OUT/f"{key}.full.scores.json"
    if not p.exists(): continue
    s=json.load(open(p)); mods=[m for m in s["models"] if "accuracy" in s["models"][m]]
    n_models=len(mods)
    accs={m:s["models"][m]["accuracy"] for m in mods}
    best=max(accs,key=accs.get); bestacc=accs[best]
    jury=s.get("jury",{}).get("accuracy",float("nan"))
    ceil=s["human_ceiling_acc"]
    # modal verdict across models
    from collections import Counter
    vmode=Counter(s["models"][m]["verdict_label"] for m in mods).most_common(1)[0][0]
    npass=sum(1 for m in mods if s["models"][m].get("alt_test_pass"))
    n_pass+=npass; n_judg+=sum(s["models"][m]["n"] for m in mods)
    cont=OUT/f"{key}.contamination.json"
    cflag="-"
    if cont.exists():
        cj=json.load(open(cont));
        if cj: cflag=max((v["flag"] for v in cj.values()),key=lambda f:{"LOW":0,"MEDIUM":1,"HIGH":2}[f])
    rows.append(f"{esc(name)} & {esc(cons)} & {ceil:.2f} & {bestacc:.2f} ({esc(best.replace('-fp8',''))}) & "
                f"{jury:.2f} & {sig.get(vmode,'')}{vmode} & {npass}/{len(mods)} & {cflag} \\\\")
tex=(r"\begin{tabular}{@{}l l c c c l c c@{}}"+"\n\\toprule"+"\n"
     r"Dataset & Construct & Human & Best model & Jury & Modal verdict & alt-test & Contam. \\"+"\n"
     r" & & ceiling & (acc) & (acc) & & pass & tier \\"+"\n\\midrule"+"\n"+"\n".join(rows)+
     "\n\\bottomrule\n\\end{tabular}")
(TAB/"tab_study3_results.tex").write_text(tex)
macros={"STHRn": n_judg, "STHRdatasets": len([1 for k,_,_ in DS if (OUT/f'{k}.full.scores.json').exists()]),
        "STHRmodels": n_models, "STHRpass": n_pass}
(TAB/"study3_autostats.tex").write_text("% auto-generated\n"+"".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k,v in macros.items()))
print("wrote tab_study3_results.tex + study3_autostats.tex"); print(macros); print("\n".join(rows))
