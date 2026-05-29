#!/usr/bin/env python3
"""
openalex_search.py -- systematic bibliographic identification via the OpenAlex API.
Upgrades the PRISMA 'identification' step from web-search-only to a real database query.

- Runs a concept-block query set against api.openalex.org/works
- Reconstructs abstracts, extracts arXiv id / DOI / venue / citations
- De-duplicates by OpenAlex work id
- Flags candidate works NOT already represented in data/included_studies.csv
- Writes data/raw_search/openalex_results.csv + openalex_candidates.csv
- Writes data/raw_search/openalex_counts.json (real identification counts)
"""
import json, time, re, csv, urllib.parse, urllib.request
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW  = HERE/"raw_search"; RAW.mkdir(exist_ok=True)
API  = "https://api.openalex.org/works"
API_KEY = __import__("os").environ.get("OPENALEX_API_KEY","")
MAILTO  = "ai4helab@gmail.com"

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

def fetch(query, per_page=50):
    params = {"search": query, "per-page": per_page, "sort": "relevance_score:desc",
              "filter": "from_publication_date:2022-09-01",
              "select": "id,doi,title,display_name,publication_year,cited_by_count,type,"
                        "authorships,primary_location,locations,abstract_inverted_index",
              "mailto": MAILTO, "api_key": API_KEY}
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": f"prisma-review ({MAILTO})"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.load(r)
        except Exception as e:
            if attempt == 2:
                print("  ! failed:", query[:40], repr(e)[:80]); return {"results": [], "meta": {}}
            time.sleep(1.5)

def arxiv_of(w):
    for loc in (w.get("locations") or []) + [w.get("primary_location") or {}]:
        u = (loc or {}).get("landing_page_url") or ""
        m = re.search(r"arxiv\.org/abs/(\d{4}\.\d{4,5})", u)
        if m: return m.group(1)
    doi = (w.get("doi") or "")
    m = re.search(r"arxiv\.(\d{4}\.\d{4,5})", doi.lower())
    return m.group(1) if m else ""

def venue_of(w):
    pl = w.get("primary_location") or {}
    src = (pl.get("source") or {})
    return src.get("display_name") or ""

def authors_of(w, k=3):
    a = [au["author"]["display_name"] for au in (w.get("authorships") or [])[:k]]
    s = ", ".join(a)
    if len(w.get("authorships") or []) > k: s += " et al."
    return s

def abstract_of(w):
    inv = w.get("abstract_inverted_index")
    if not inv: return ""
    pos = {}
    for tok, idxs in inv.items():
        for i in idxs: pos[i] = tok
    return " ".join(pos[i] for i in sorted(pos))[:600]

def main():
    seen, rows, raw_total = {}, [], 0
    per_query_meta = []
    for q in QUERIES:
        d = fetch(q); res = d.get("results", [])
        cnt = d.get("meta", {}).get("count", len(res))
        per_query_meta.append((q, cnt, len(res)))
        raw_total += len(res)
        for w in res:
            wid = w["id"].split("/")[-1]
            if wid in seen: continue
            seen[wid] = True
            rows.append(dict(
                openalex_id=wid, title=(w.get("title") or w.get("display_name") or "").strip(),
                year=w.get("publication_year"), venue=venue_of(w),
                cited_by=w.get("cited_by_count"), type=w.get("type"),
                arxiv=arxiv_of(w), doi=(w.get("doi") or "").replace("https://doi.org/",""),
                authors=authors_of(w), abstract=abstract_of(w)))
        print(f"  q='{q[:46]:46s}' total_hits={cnt:>7}  fetched={len(res)}")
        time.sleep(0.15)

    od = pd.DataFrame(rows).drop_duplicates("openalex_id")
    od.to_csv(RAW/"openalex_results.csv", index=False)

    # compare against current corpus by arxiv id (version-stripped) + fuzzy title
    inc = pd.read_csv(HERE/"included_studies.csv")
    have_ax = set(str(a).split("v")[0].strip() for a in inc["arxiv"].dropna())
    def norm(t): return re.sub(r"[^a-z0-9]","", str(t).lower())[:60]
    have_titles = set(norm(t) for t in inc["finding"]) | set()  # weak; arxiv is the strong key
    od["in_corpus"] = od["arxiv"].apply(lambda a: str(a).split("v")[0] in have_ax and str(a)!="")
    cand = od[(~od["in_corpus"]) & (od["type"].isin(["article","preprint","review"]))]
    cand = cand.sort_values("cited_by", ascending=False)
    cand.to_csv(RAW/"openalex_candidates.csv", index=False)

    counts = dict(queries=len(QUERIES), raw_hits=raw_total,
                  unique_works=len(od), already_in_corpus=int(od["in_corpus"].sum()),
                  candidate_not_in_corpus=len(cand))
    json.dump(counts, open(RAW/"openalex_counts.json","w"), indent=2)
    json.dump([{"query":q,"openalex_total":c,"fetched":f} for q,c,f in per_query_meta],
              open(RAW/"openalex_per_query.json","w"), indent=2)

    print("\n=== OpenAlex identification summary ===")
    for k,v in counts.items(): print(f"  {k:28s}: {v}")
    print("\n=== Top 35 candidate works NOT yet in corpus (by citations) ===")
    for _,r in cand.head(35).iterrows():
        print(f"  [{r['cited_by']:>5}] {r['year']} {str(r['arxiv']):>11} | {r['title'][:86]}")

if __name__ == "__main__":
    main()
