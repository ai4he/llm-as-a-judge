#!/usr/bin/env python3
"""Generates notebooks/llm_judge_review.ipynb (valid nbformat-4 JSON)."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent

def md(*src):  return {"cell_type":"markdown","metadata":{},"source":_lines(src)}
def code(*src):return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":_lines(src)}
def _lines(src):
    txt="\n".join(src); L=txt.split("\n")
    return [l+"\n" for l in L[:-1]]+[L[-1]]

cells=[
md("# LLM-as-a-Judge vs Human Annotation — Reproducible Analysis",
   "",
   "Companion notebook to the PRISMA systematic review *When Can a Machine Judge Like a Human?*",
   "It (re)builds the extracted-studies corpus, regenerates every figure and table, and reproduces",
   "the headline analyses. All numbers in the paper derive from `data/included_studies.csv`.",
   "",
   "**Pipeline:** `build_corpus.py` → `make_figures.py` → `generate_bib.py`.  ",
   "**Sources:** OpenAlex API + Semantic Scholar API + structured web deep-search (see `data/raw_search/`)."),
md("## 0. Setup"),
code("import os, json, subprocess",
     "from pathlib import Path",
     "import pandas as pd, numpy as np",
     "import matplotlib.pyplot as plt",
     "from IPython.display import Image, display, Markdown",
     "",
     "ROOT = Path.cwd()",
     "if ROOT.name == 'notebooks': ROOT = ROOT.parent",
     "os.chdir(ROOT)",
     "print('project root:', ROOT)"),
md("## 1. Build the corpus and regenerate all artefacts",
   "We call the pipeline scripts so the notebook is a faithful driver of the released code."),
code("for script in ['data/build_corpus.py','analysis/make_figures.py','data/generate_bib.py']:",
     "    r = subprocess.run(['python3', script], capture_output=True, text=True)",
     "    print('### ', script, '(rc=%d)' % r.returncode)",
     "    print(r.stdout.strip().splitlines()[-3:] if r.stdout else r.stderr[-500:])"),
md("## 2. The included-studies corpus"),
code("df = pd.read_csv('data/included_studies.csv')",
     "PR = json.load(open('data/prisma_counts.json'))",
     "print('included studies :', len(df))",
     "print('domains          :', df.domain.nunique())",
     "print('with numeric value:', df.agreement.notna().sum())",
     "df.head(6)[['id','authors','year','domain','task','metric','agreement','verdict_label']]"),
md("### PRISMA 2020 flow counts (two-arm identification)"),
code("display(Markdown('\\n'.join(f'- **{k}**: {v}' for k,v in PR.items())))",
     "display(Image(filename='figures/fig01_prisma_flow.png'))"),
md("## 3. Overall reliability verdict"),
code("vc = df.verdict_label.value_counts()",
     "print(vc.to_string())",
     "print('\\nreliable (verdict 1-2): %.0f%%' % (df.verdict.isin([1,2]).mean()*100))",
     "print('problematic (4-5)    : %.0f%%' % (df.verdict.isin([4,5]).mean()*100))",
     "display(Image(filename='figures/fig02_verdict_overall.png'))"),
md("## 4. RQ1 — reliability by domain (the green/amber/red landscape)"),
code("g = pd.read_csv('data/summary_by_domain.csv').sort_values('rel', ascending=False)",
     "display(g.style.background_gradient(subset=['rel'], cmap='RdYlGn', vmin=1, vmax=5).format({'rel':'{:.2f}'}))",
     "display(Image(filename='figures/fig05_reliability_by_domain.png'))"),
md("## 5. RQ2 — domain × task reliability landscape, subjectivity & expertise"),
code("for f in ['fig06_reliability_landscape','fig07_subjectivity','fig08_expertise','fig12_verdict_by_task']:",
     "    display(Image(filename=f'figures/{f}.png'))"),
md("## 6. RQ3 — agreement vs the human ceiling"),
code("display(Image(filename='figures/fig09_agreement_forest.png'))",
     "display(Image(filename='figures/fig10_parity_scatter.png'))",
     "anchors = df.dropna(subset=['agreement'])[['authors','year','domain','metric','agreement','human_baseline','verdict_label']]",
     "anchors.sort_values(['domain','agreement'])"),
md("## 7. RQ4 — failure modes / biases"),
code("from collections import Counter",
     "c = Counter()",
     "for b in df.biases.dropna():",
     "    for t in str(b).split(';'):",
     "        t=t.strip()",
     "        if t and t!='—': c[t]+=1",
     "print(pd.Series(c).sort_values(ascending=False).head(15).to_string())",
     "display(Image(filename='figures/fig11_bias_frequency.png'))"),
md("## 8. RQ5 / RQ6 — moderators, families, and the deployment decision map"),
code("display(Image(filename='figures/fig13_family_reliability.png'))",
     "display(Image(filename='figures/fig14_temporal_trend.png'))",
     "display(Image(filename='figures/fig15_decision_map.png'))"),
md("## 9. Takeaways",
   "",
   "- **Green zone (validate → automate):** objective, verifiable, low-expertise tasks judged at the aggregate",
   "  level — pairwise preference, system-level MT, relevance, verifiable instruction-following, factuality,",
   "  many classification/annotation tasks. Here LLM judges often *match or exceed* the human–human ceiling.",
   "- **Amber zone (human-in-the-loop + per-task validation):** summarization, RAG, grading, social-science",
   "  annotation, peer review, and *structured* clinical evaluation.",
   "- **Red zone (keep humans):** subjective/expert/low-resource judgments — creative quality, low-resource",
   "  multilingual evaluation, affective/safety ratings, and any unprotected absolute-scoring setting.",
   "",
   "The operative question is never *“can an LLM judge?”* but *“can this judge replace a second human on this",
   "task, at this granularity, under these controls?”* — answerable domain-by-domain from the released corpus.")
]

for i,c in enumerate(cells): c["id"]=f"cell{i:02d}"
nb={"cells":cells,
    "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                "language_info":{"name":"python","version":"3.12"}},
    "nbformat":4,"nbformat_minor":5}
(HERE/"llm_judge_review.ipynb").write_text(json.dumps(nb,indent=1))
print("wrote notebooks/llm_judge_review.ipynb with", len(cells), "cells")
