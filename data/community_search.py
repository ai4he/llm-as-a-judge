#!/usr/bin/env python3
"""Venue/community-scoped identification for HCI, CSCW, and ML, to verify full coverage
of LLM-as-a-judge-vs-human studies in those communities. OpenAlex API."""
import json, time, re, urllib.parse, urllib.request
from pathlib import Path
import pandas as pd
HERE=Path(__file__).resolve().parent; RAW=HERE/"raw_search"
API="https://api.openalex.org/works"; KEY=__import__("os").environ.get("OPENALEX_API_KEY",""); MAIL="ai4helab@gmail.com"

QUERIES=[
 # HCI / CSCW oriented
 "large language model evaluation human agreement CHI human factors computing",
 "LLM qualitative coding CSCW computer supported cooperative work inter-rater",
 "human-AI LLM annotation collaboration user study IUI intelligent interfaces",
 "ChatGPT thematic analysis inter-rater reliability qualitative HCI",
 "human-LLM collaborative annotation verification CHI",
 "LLM evaluator design study UIST human preference",
 "crowdsourcing large language model annotation quality HCOMP human computation",
 "LLM replicate crowdsourcing pipeline human computation workers",
 "LLM-as-a-judge user trust human-centered evaluation interface",
 # ML evaluator models / meta-eval
 "fine-tuned LLM judge scalable JudgeLM PandaLM human agreement",
 "generative judge evaluating alignment Auto-J critique human",
 "self-rewarding meta-rewarding language model LLM judge",
 "reward model benchmark RM-Bench preference proxy evaluation human",
 "LLM-as-a-judge bias benchmark EvalBiasBench position verbosity",
 "meta-evaluation LLM evaluators human correlation benchmark NeurIPS ICLR",
 "Arena-Hard automatic benchmark human preference agreement crowdsourced",
 "provable guarantees selective LLM judge human agreement cascade",
 "critique model Shepherd CritiqueLLM Themis informative critique evaluation",
 "empirical study LLM-as-a-judge for LLM evaluation reliability",
 "multi-agent LLM debate evaluation human agreement",
 "open-source evaluator LLM fine-grained rubric human correlation",
 "LLM annotation replace crowd workers data labeling agreement",
]

def fetch(q,per=50):
    p={"search":q,"per-page":per,"sort":"relevance_score:desc",
       "filter":"from_publication_date:2022-09-01",
       "select":"id,doi,title,display_name,publication_year,cited_by_count,type,authorships,primary_location,locations,abstract_inverted_index",
       "mailto":MAIL,"api_key":KEY}
    url=API+"?"+urllib.parse.urlencode(p)
    req=urllib.request.Request(url,headers={"User-Agent":f"prisma-review ({MAIL})"})
    for a in range(3):
        try:
            with urllib.request.urlopen(req,timeout=40) as r: return json.load(r).get("results",[])
        except Exception as e:
            if a==2: print("  ! fail",q[:36],repr(e)[:50]); return []
            time.sleep(1.2)
    return []

def arxiv_of(w):
    for loc in (w.get("locations") or [])+[w.get("primary_location") or {}]:
        u=(loc or {}).get("landing_page_url") or ""
        m=re.search(r"arxiv\.org/abs/(\d{4}\.\d{4,5})",u)
        if m: return m.group(1)
    return ""
def venue_of(w): return ((w.get("primary_location") or {}).get("source") or {}).get("display_name") or ""
def auth_of(w,k=3):
    a=[x["author"]["display_name"] for x in (w.get("authorships") or [])[:k]]
    s=", ".join(a)
    if len(w.get("authorships") or [])>k: s+=" et al."
    return s
def abs_of(w):
    inv=w.get("abstract_inverted_index")
    if not inv: return ""
    pos={}
    for t,idxs in inv.items():
        for i in idxs: pos[i]=t
    return " ".join(pos[i] for i in sorted(pos))[:500]

seen,rows={},[]
for q in QUERIES:
    for w in fetch(q):
        wid=w["id"].split("/")[-1]
        if wid in seen: continue
        seen[wid]=True
        rows.append(dict(title=(w.get("title") or "").strip(),year=w.get("publication_year"),
            venue=venue_of(w),cited_by=w.get("cited_by_count") or 0,type=w.get("type"),
            arxiv=arxiv_of(w),doi=(w.get("doi") or "").replace("https://doi.org/","").lower(),
            authors=auth_of(w),abstract=abs_of(w)))
    print(f"  q='{q[:44]:44s}' cum_unique={len(rows)}"); time.sleep(0.15)
cm=pd.DataFrame(rows).drop_duplicates("title")
cm.to_csv(RAW/"community_results.csv",index=False)

# topical + dedupe vs corpus
inc=pd.read_csv(HERE/"included_studies.csv").fillna("")
have_ax=set(str(a).split("v")[0].strip() for a in inc["arxiv"] if a)
have_ay=set((re.split(r"[ ,&]",str(r.authors))[0].lower(), int(r.year) if str(r.year).isdigit() else 0)
            for r in inc.itertuples() if str(r.authors)!="authors")
import importlib.util
spec=importlib.util.spec_from_file_location("gb",HERE/"generate_bib.py")
gb=importlib.util.module_from_spec(spec); spec.loader.exec_module(gb)
have_titles=set(re.sub(r"[^a-z0-9]","",t.lower())[:55] for t in gb.TITLES.values())
LLM=re.compile(r"\b(llm|large language model|gpt|chatgpt|claude|gemini|llama)\b",re.I)
METH=re.compile(r"(judge|evaluat|annotat|assessor|rater|critique|grad(e|ing)|scoring|label|reward model)",re.I)
HUM=re.compile(r"(human|expert|crowd|annotator|agreement|correlat|preference|kappa|inter-?rater|inter-?annotator)",re.I)
def tk(t): return re.sub(r"[^a-z0-9]","",str(t).lower())[:55]
def topical(r): return bool(LLM.search(str(r.title)+" "+str(r.abstract)) and METH.search(str(r.title)+" "+str(r.abstract)) and HUM.search(str(r.title)+" "+str(r.abstract)))
def have(r):
    ax=str(r.arxiv).split("v")[0].strip()
    if ax and ax in have_ax: return True
    if tk(r.title) in have_titles: return True
    sur=re.split(r"[ ,&]",str(r.authors))[0].lower()
    return (sur,int(r.year) if str(r.year).isdigit() else 0) in have_ay
cm["topical"]=cm.apply(topical,axis=1); cm["have"]=cm.apply(have,axis=1)
miss=cm[cm.topical & ~cm.have & (cm.cited_by>=2)].sort_values("cited_by",ascending=False)
miss.to_csv(RAW/"community_missing.csv",index=False)
print(f"\ncommunity unique works: {len(cm)} | topical: {int(cm.topical.sum())} | MISSING in-scope: {len(miss)}")
print("\n=== community-scoped candidates MISSING from corpus (top 45) ===")
for r in miss.head(45).itertuples():
    print(f"[{int(r.cited_by):>4}] {r.year} {str(r.arxiv or r.doi)[:22]:22s} | {str(r.venue)[:22]:22s} | {str(r.title)[:60]}")
