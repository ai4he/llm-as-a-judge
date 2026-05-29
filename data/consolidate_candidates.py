#!/usr/bin/env python3
"""Union the on-topic candidate lists from all sources (OpenAlex + both Semantic Scholar
runs), de-duplicate, and list those genuinely missing from the corpus for curation."""
import re, pandas as pd
from pathlib import Path
HERE=Path(__file__).resolve().parent; RAW=HERE/"raw_search"

def load(fn, src):
    p=RAW/fn
    if not p.exists(): return pd.DataFrame()
    d=pd.read_csv(p).fillna("")
    keep=["title","year","venue","cited_by","arxiv","doi","authors"]
    for k in keep:
        if k not in d.columns: d[k]=""
    d=d[keep].copy(); d["source"]=src; return d

frames=[load("openalex_topical_candidates.csv","OA"),
        load("s2_topical_candidates.csv","S2"),
        load("s2_topical_candidates_run12.csv","S2b")]
cand=pd.concat([f for f in frames if len(f)],ignore_index=True)

def norm_title(t): return re.sub(r"[^a-z0-9]","",str(t).lower())[:55]
cand["axk"]=cand["arxiv"].apply(lambda a:str(a).split("v")[0].strip())
cand["doik"]=cand["doi"].apply(lambda d:str(d).lower().replace("https://doi.org/","").strip())
cand["tk"]=cand["title"].apply(norm_title)
cand["cited_by"]=pd.to_numeric(cand["cited_by"],errors="coerce").fillna(0).astype(int)

# unify duplicates across sources -> keep richest row (max cited_by) per title-key
cand["key"]=cand.apply(lambda r: r["axk"] or r["doik"] or r["tk"], axis=1)
cand=cand.sort_values("cited_by",ascending=False).drop_duplicates("tk").drop_duplicates("key")

# dedup vs corpus
inc=pd.read_csv(HERE/"included_studies.csv").fillna("")
have_ax=set(str(a).split("v")[0].strip() for a in inc["arxiv"] if a)
have_ay=set()
for _,r in inc.iterrows():
    sur=re.split(r"[ ,&]",str(r["authors"]))[0].lower()
    if sur and sur!="authors": have_ay.add((sur,int(r["year"]) if str(r["year"]).isdigit() else 0))
# also dedup vs titles already used (from generate_bib TITLES)
import importlib.util
spec=importlib.util.spec_from_file_location("gb",HERE/"generate_bib.py")
have_titles=set()
try:
    gb=importlib.util.module_from_spec(spec); spec.loader.exec_module(gb)
    have_titles=set(norm_title(t) for t in gb.TITLES.values())
except Exception as e:
    print("note: could not import TITLES:",e)

def in_corpus(r):
    if r["axk"] and r["axk"] in have_ax: return True
    if r["tk"] in have_titles: return True
    sur=re.split(r"[ ,&]",str(r["authors"]))[0].lower()
    return (sur,int(r["year"]) if str(r["year"]).isdigit() else 0) in have_ay
cand["in_corpus"]=cand.apply(in_corpus,axis=1)
miss=cand[~cand["in_corpus"]].sort_values("cited_by",ascending=False)
miss.to_csv(RAW/"consolidated_missing_candidates.csv",index=False)

print(f"union candidates (deduped) : {len(cand)}")
print(f"already in corpus           : {int(cand['in_corpus'].sum())}")
print(f"MISSING from corpus         : {len(miss)}")
print("\n=== consolidated missing candidates (top 55 by citations) ===")
for _,r in miss.head(55).iterrows():
    idr=r["arxiv"] or r["doik"]
    print(f"[{r['cited_by']:>4}|{r['source']:>3}] {r['year']} {str(idr)[:22]:22s} | {str(r['title'])[:74]}")
