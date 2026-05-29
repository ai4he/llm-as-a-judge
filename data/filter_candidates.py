#!/usr/bin/env python3
"""Filter OpenAlex results to on-topic 'LLM-judge vs human' studies genuinely
missing from the corpus, so they can be reviewed for inclusion."""
import re, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
od = pd.read_csv(HERE/"raw_search"/"openalex_results.csv").fillna("")
inc = pd.read_csv(HERE/"included_studies.csv").fillna("")

have_ax = set(str(a).split("v")[0].strip() for a in inc["arxiv"] if a)
# author-surname + year combos already represented (reduces false 'missed')
have_ay = set()
for _,r in inc.iterrows():
    sur = re.split(r"[ ,&]", str(r["authors"]))[0].lower()
    if sur and sur not in ("authors",): have_ay.add((sur, int(r["year"]) if str(r["year"]).isdigit() else 0))

LLM  = re.compile(r"\b(llm|large language model|gpt|chatgpt|claude|gemini|llama|mistral|qwen)\b", re.I)
METH = re.compile(r"(judge|evaluator|evaluat|annotat|assessor|rater|grad(e|ing)|scoring|label|as-a-judge)", re.I)
HUM  = re.compile(r"(human|expert|crowd|annotator|inter-?annotator|agreement|correlat|physician|clinician|gold|preference|kappa)", re.I)
OFF  = re.compile(r"(technical report|survey of large|encode clinical knowledge|usmle|opinion paper|swot|"
                  r"policy framework|academic integrity|autonomous agents|visual instruction tuning|"
                  r"lost in the middle|protein|sparks of artificial|five priorities|simulacra)", re.I)

def topical(r):
    blob = (str(r["title"]) + " " + str(r["abstract"]))
    if OFF.search(str(r["title"])): return False
    return bool(LLM.search(blob) and METH.search(blob) and HUM.search(blob))

def already(r):
    ax = str(r["arxiv"]).split("v")[0].strip()
    if ax and ax in have_ax: return True
    sur = re.split(r"[ ,&]", str(r["authors"]))[0].lower()
    if (sur, int(r["year"]) if str(r["year"]).isdigit() else 0) in have_ay: return True
    return False

od["topical"] = od.apply(topical, axis=1)
od["already"] = od.apply(already, axis=1)
cand = od[od["topical"] & (~od["already"]) & (od["cited_by"]>=3)].sort_values("cited_by", ascending=False)
cand.to_csv(HERE/"raw_search"/"openalex_topical_candidates.csv", index=False)
print(f"topical works total            : {int(od['topical'].sum())}")
print(f"  of which already in corpus   : {int((od['topical']&od['already']).sum())}")
print(f"  topical & MISSING (cited>=3) : {len(cand)}")
print("\n=== topical candidates missing from corpus (top 45 by citations) ===")
for _,r in cand.head(45).iterrows():
    print(f"[{int(r['cited_by']):>4}] {r['year']} {str(r['arxiv'] or r['doi'])[:24]:24s} | {r['title'][:82]}")
