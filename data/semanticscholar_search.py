#!/usr/bin/env python3
"""semanticscholar_search.py -- third independent identification source (Semantic Scholar
Graph API) to confirm search saturation and catch any remaining on-topic papers."""
import json, time, re, urllib.parse, urllib.request
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent; RAW = HERE/"raw_search"; RAW.mkdir(exist_ok=True)
EP = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "title,year,abstract,venue,citationCount,externalIds,authors,publicationTypes"

# NOTE: identical 30-query concept set to data/openalex_search.py, for cross-platform consistency.
QUERIES = [
 "LLM-as-a-judge human agreement","large language model evaluator human judgment correlation",
 "ChatGPT annotation human agreement crowdworkers","GPT-4 evaluation human evaluation alternative",
 "LLM judge summarization human correlation","LLM machine translation evaluation human GEMBA",
 "LLM judge code generation evaluation human","large language model relevance judgment assessor",
 "LLM annotation social science human coder validation","LLM hate speech toxicity annotation human",
 "LLM medical clinical evaluation physician agreement","LLM mental health counseling evaluation expert",
 "LLM legal evaluation expert agreement","LLM automated essay scoring human rater agreement",
 "LLM creative writing evaluation human judgment","LLM peer review scientific human reviewer",
 "multimodal LLM judge vision language human agreement","multilingual LLM-as-a-judge reliability",
 "reward model human preference benchmark RewardBench","LLM fact-checking claim human annotator",
 "LLM dialogue evaluation human correlation","LLM qualitative coding thematic analysis human",
 "self-preference bias LLM judge","position bias verbosity bias LLM evaluator",
 "alternative annotator test LLM replace human","inter-annotator agreement LLM human",
 "LLM-as-a-judge survey bias reliability","panel of LLM evaluators jury",
 "instruction following evaluation LLM judge","retrieval augmented generation evaluation LLM judge faithfulness",
]

def fetch(q, limit=100):
    url = EP + "?" + urllib.parse.urlencode({"query": q, "limit": limit, "fields": FIELDS,
                                             "year": "2022-2026"})
    req = urllib.request.Request(url, headers={"User-Agent": "prisma-review (ai4helab@gmail.com)"})
    for attempt in range(6):          # patient: we have time, prioritise completeness over speed
        try:
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.load(r).get("data", []) or []
        except urllib.error.HTTPError as e:
            if e.code == 429: time.sleep(6 + 3*attempt); continue   # exponential-ish backoff on throttle
            if attempt == 5: print("  ! HTTP", e.code, q[:40], flush=True); return []
            time.sleep(3)
        except Exception as e:
            if attempt == 5: print("  ! err", repr(e)[:60], q[:40], flush=True); return []
            time.sleep(3)
    return []

def authors_of(p, k=3):
    a = [x.get("name","") for x in (p.get("authors") or [])[:k]]
    s = ", ".join(a)
    if len(p.get("authors") or []) > k: s += " et al."
    return s

def main():
    seen, rows, raw = {}, [], 0
    for q in QUERIES:
        d = fetch(q); raw += len(d)
        for p in d:
            pid = p.get("paperId")
            if not pid or pid in seen: continue
            seen[pid] = True
            ext = p.get("externalIds") or {}
            rows.append(dict(paperId=pid, title=(p.get("title") or "").strip(),
                year=p.get("year"), venue=p.get("venue") or "", cited_by=p.get("citationCount") or 0,
                arxiv=ext.get("ArXiv","") or "", doi=ext.get("DOI","") or "",
                ptypes=";".join(p.get("publicationTypes") or []),
                authors=authors_of(p), abstract=(p.get("abstract") or "")[:600]))
        print(f"  q='{q[:46]:46s}' fetched={len(d)}", flush=True); time.sleep(1.2)
    s2 = pd.DataFrame(rows).drop_duplicates("paperId")
    s2.to_csv(RAW/"s2_results.csv", index=False)

    inc = pd.read_csv(HERE/"included_studies.csv").fillna("")
    have_ax = set(str(a).split("v")[0].strip() for a in inc["arxiv"] if a)
    have_ay = set()
    for _,r in inc.iterrows():
        sur = re.split(r"[ ,&]", str(r["authors"]))[0].lower()
        if sur and sur != "authors": have_ay.add((sur, int(r["year"]) if str(r["year"]).isdigit() else 0))
    try:
        oa = pd.read_csv(RAW/"openalex_results.csv").fillna("")
        oa_ax = set(str(a).split("v")[0].strip() for a in oa["arxiv"] if a)
        oa_doi = set(str(d).lower() for d in oa["doi"] if d)
    except Exception:
        oa_ax, oa_doi = set(), set()

    LLM=re.compile(r"\b(llm|large language model|gpt|chatgpt|claude|gemini|llama|mistral|qwen)\b",re.I)
    METH=re.compile(r"(judge|evaluator|evaluat|annotat|assessor|rater|grad(e|ing)|scoring|label|as-a-judge)",re.I)
    HUM=re.compile(r"(human|expert|crowd|annotator|inter-?annotator|agreement|correlat|physician|clinician|gold|preference|kappa)",re.I)
    OFF=re.compile(r"(technical report|survey of large|encode clinical knowledge|usmle|opinion paper|swot|simulacra|sparks of artificial)",re.I)
    def topical(r):
        blob=str(r["title"])+" "+str(r["abstract"])
        if OFF.search(str(r["title"])): return False
        return bool(LLM.search(blob) and METH.search(blob) and HUM.search(blob))
    def in_corpus(r):
        ax=str(r["arxiv"]).split("v")[0].strip()
        if ax and ax in have_ax: return True
        sur=re.split(r"[ ,&]",str(r["authors"]))[0].lower()
        return (sur, int(r["year"]) if str(r["year"]).isdigit() else 0) in have_ay
    def in_openalex(r):
        ax=str(r["arxiv"]).split("v")[0].strip(); doi=str(r["doi"]).lower()
        return (ax and ax in oa_ax) or (doi and doi in oa_doi)

    s2["topical"]=s2.apply(topical,axis=1); s2["in_corpus"]=s2.apply(in_corpus,axis=1)
    s2["in_openalex"]=s2.apply(in_openalex,axis=1)
    cand=s2[s2["topical"] & (~s2["in_corpus"]) & (s2["cited_by"]>=3)].sort_values("cited_by",ascending=False)
    cand.to_csv(RAW/"s2_topical_candidates.csv",index=False)
    counts=dict(queries=len(QUERIES),raw_hits=raw,unique=len(s2),
                topical=int(s2["topical"].sum()),
                topical_in_corpus=int((s2["topical"]&s2["in_corpus"]).sum()),
                topical_missing=len(cand),
                topical_missing_also_absent_from_openalex=int((cand["in_openalex"]==False).sum()))
    json.dump(counts,open(RAW/"s2_counts.json","w"),indent=2)
    print("\n=== Semantic Scholar identification summary ==="); [print(f"  {k:42s}: {v}") for k,v in counts.items()]
    print("\n=== topical & missing-from-corpus (top 40 by citations; * = also absent from OpenAlex pull) ===")
    for _,r in cand.head(40).iterrows():
        star="*" if not r["in_openalex"] else " "
        print(f"{star}[{int(r['cited_by']):>4}] {r['year']} {str(r['arxiv'] or r['doi'])[:24]:24s} | {r['title'][:80]}")

if __name__=="__main__": main()
