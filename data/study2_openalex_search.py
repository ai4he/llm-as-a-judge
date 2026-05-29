#!/usr/bin/env python3
"""Study 2 identification (OpenAlex API): LLMs vs human input in Human-Centered Computing,
HCI, and qualitative-research methods that historically require human judgment."""
import json, time, re, urllib.parse, urllib.request
from pathlib import Path
import pandas as pd
HERE=Path(__file__).resolve().parent; RAW=HERE/"raw_search"; RAW.mkdir(exist_ok=True)
API="https://api.openalex.org/works"; KEY=__import__("os").environ.get("OPENALEX_API_KEY",""); MAIL="ai4helab@gmail.com"

QUERIES=[
 "large language model qualitative coding human coders agreement",
 "GPT thematic analysis qualitative research human researchers reliability",
 "LLM deductive coding codebook inter-rater reliability human",
 "LLM inductive coding open coding qualitative human agreement",
 "large language model content analysis human coders agreement",
 "LLM collaborative qualitative analysis CSCW CHI coding",
 "human-AI collaboration qualitative data analysis sensemaking LLM",
 "large language model replace human participants user study HCI",
 "large language model simulate survey respondents silicon sampling",
 "LLM synthetic users personas user research HCI evaluation",
 "LLM usability evaluation heuristic evaluation expert comparison",
 "LLM UX design critique feedback human designers",
 "LLM open-ended survey response coding human annotators",
 "LLM interview transcript analysis qualitative human",
 "LLM crowdsourcing annotation replace crowd workers human computation",
 "LLM social media annotation human coders computational social science",
 "LLM sentiment emotion annotation user generated human agreement",
 "generative AI qualitative research reflexivity interpretivist epistemology",
 "LLM accessibility evaluation human experts",
 "LLM grounded theory qualitative analysis human",
 "LLM data annotation HCI research data work human",
 "ChatGPT qualitative research validity human comparison coding",
 "large language model theme development affinity clustering qualitative",
 "human-centered evaluation LLM annotation human input replace",
 "LLM persona bias synthetic participant identity flatten",
 "LLM as research tool HCI human evaluation annotation",
]

def fetch(q,per=50):
    p={"search":q,"per-page":per,"sort":"relevance_score:desc","filter":"from_publication_date:2022-09-01",
       "select":"id,doi,title,display_name,publication_year,cited_by_count,type,authorships,primary_location,locations,abstract_inverted_index",
       "mailto":MAIL,"api_key":KEY}
    req=urllib.request.Request(API+"?"+urllib.parse.urlencode(p),headers={"User-Agent":f"prisma ({MAIL})"})
    for a in range(3):
        try:
            with urllib.request.urlopen(req,timeout=40) as r: return json.load(r).get("results",[])
        except Exception as e:
            if a==2: print("  ! fail",q[:34],repr(e)[:46]); return []
            time.sleep(1.2)
    return []

def arxiv_of(w):
    for loc in (w.get("locations") or [])+[w.get("primary_location") or {}]:
        m=re.search(r"arxiv\.org/abs/(\d{4}\.\d{4,5})",(loc or {}).get("landing_page_url") or "")
        if m: return m.group(1)
    return ""
def venue_of(w): return ((w.get("primary_location") or {}).get("source") or {}).get("display_name") or ""
def auth_of(w,k=3):
    a=[x["author"]["display_name"] for x in (w.get("authorships") or [])[:k]]
    return ", ".join(a)+(" et al." if len(w.get("authorships") or [])>k else "")
def abs_of(w):
    inv=w.get("abstract_inverted_index")
    if not inv: return ""
    pos={}
    for t,ix in inv.items():
        for i in ix: pos[i]=t
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
    print(f"  q='{q[:42]:42s}' cum={len(rows)}"); time.sleep(0.15)
cm=pd.DataFrame(rows).drop_duplicates("title")
cm.to_csv(RAW/"study2_results.csv",index=False)

LLM=re.compile(r"\b(llm|large language model|gpt|chatgpt|claude|gemini|llama|generative ai)\b",re.I)
METH=re.compile(r"(qualitative|coding|thematic|content analysis|annotat|usability|heuristic|persona|participant|crowd|interview|survey|grounded|sensemaking|reflexiv|design critique|user stud|card sort|affinity)",re.I)
HUM=re.compile(r"(human|coder|annotator|researcher|expert|participant|inter-?rater|inter-?coder|agreement|kappa|crowd)",re.I)
OFF=re.compile(r"(survey of large|technical report|systematic review of (deep|machine))",re.I)
def topical(r):
    b=str(r.title)+" "+str(r.abstract)
    return bool(LLM.search(b) and METH.search(b) and HUM.search(b)) and not OFF.search(str(r.title))
cm["topical"]=cm.apply(topical,axis=1)
cand=cm[cm.topical & (cm.cited_by>=1)].sort_values("cited_by",ascending=False)
cand.to_csv(RAW/"study2_candidates.csv",index=False)
json.dump(dict(queries=len(QUERIES),unique=len(cm),topical=int(cm.topical.sum()),candidates=len(cand)),
          open(RAW/"study2_counts.json","w"),indent=2)
print(f"\nstudy2 unique={len(cm)} topical={int(cm.topical.sum())} candidates={len(cand)}")
print("\n=== top 55 Study-2 candidates (HCC/HCI/qualitative; by citations) ===")
for r in cand.head(55).itertuples():
    print(f"[{int(r.cited_by):>4}] {r.year} {str(r.arxiv or r.doi)[:24]:24s} | {str(r.venue)[:20]:20s} | {str(r.title)[:58]}")
