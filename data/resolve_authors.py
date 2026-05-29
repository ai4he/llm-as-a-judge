#!/usr/bin/env python3
"""Resolve real lead-author names for corpus records that carry the 'authors' placeholder,
using the saved result CSVs first, then the OpenAlex API. Prints a paste-ready dict."""
import json, re, time, urllib.request, urllib.parse
from pathlib import Path
import pandas as pd
HERE=Path(__file__).resolve().parent; RAW=HERE/"raw_search"

# id -> identifier (arxiv id or DOI) from build_corpus placeholders
IDS={
 "overcorrect2026":"2603.00539","biasloop2026":"2604.16790","codespec2025":"2508.12358",
 "mathrobust2026":"2604.22597","hatespeechanno2025":"2512.09662","icwsm2025hatebias":"2410.07991",
 "medjudge2026scoping":"2604.25933","llmevalmed2025":"2506.04078","brain2024radgpt":"10.1007/s00330-024-11032-8",
 "clindialog2026":"2603.00314","frenchmedqa2026":"2603.04033","mhtrust2025":"2510.19032",
 "counselbench2025":"2506.08584","mhsupport2026":"2601.18630","lemaj2025":"2510.07243",
 "legaldocrec2025":"2509.12382","bizfinbench2025":"2505.19457","k12shortanswer2024":"2405.02985",
 "sefl2025feedback":"2502.12927","k12sci2026":"2602.13243","litbench2025":"2507.00769",
 "creativeevalmdpi2025":"10.3390/app15062971","murad2024quality":"10.1186/s12874-024-02372-6",
 "multiljudge2025":"2505.12201","mmeval2024":"2410.17578","reliablemulti2026":"2605.28710",
 "ifreward2026":"2603.04738","politicaltruths2024":"2411.05775","factsopinions2025":"2506.03655",
 "ifcritic2025":"2511.01014","thematic2025charity":"10.1007/s00146-025-02487-4","ner2024augment":"2404.01334",
 "nli2024labeldist":"2412.13942","temperature2026":"2603.28304","judgesverdict2025":"2510.09738",
 "imrit2026iaa":"2603.06865",
}
# records with non-resolvable PII/PMC identifiers -> author known from prior search results
KNOWN={
 "clinstruct2025":"Brake","writingmultidim2024":"Steiss","asagfair2025":"Henkel",
 "studentsjudge2025":"Banihashem","chatgptreviewers2024":"Ho","prismascreen2024":"Guo",
 "qualcoding2025jla":"Xiao",
}

def load_idx():
    ax,doi={},{}
    for fn in ["openalex_results.csv","s2_results.csv","community_results.csv","s2_results_run12.csv"]:
        p=RAW/fn
        if not p.exists(): continue
        d=pd.read_csv(p).fillna("")
        for _,r in d.iterrows():
            a=str(r.get("arxiv","")).split("v")[0].strip(); dd=str(r.get("doi","")).lower().replace("https://doi.org/","").strip()
            au=str(r.get("authors","")).strip()
            if a and au and a not in ax: ax[a]=au
            if dd and au and dd not in doi: doi[dd]=au
    return ax,doi

def surname(full):
    full=re.sub(r"\bet al\.?$","",str(full)).strip().strip(",")
    first=full.split(",")[0].strip()
    parts=first.split()
    return parts[-1] if parts else first

def api_author(ident):
    doi = ident if ident.startswith("10.") else f"10.48550/arXiv.{ident}"
    url=f"https://api.openalex.org/works/doi:{urllib.parse.quote(doi)}?select=authorships&mailto=ai4helab@gmail.com&api_key={__import__('os').environ.get('OPENALEX_API_KEY','')}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"prisma"}),timeout=25) as r:
            d=json.load(r); a=d.get("authorships") or []
            return a[0]["author"]["display_name"] if a else ""
    except Exception: return ""

ax,doi=load_idx(); out={}
for sid,ident in IDS.items():
    key=ident.split("v")[0].strip(); au=""
    if re.match(r"\d{4}\.\d{4,5}",key): au=ax.get(key,"")
    if not au: au=doi.get(ident.lower(),"")
    if not au: au=api_author(ident); time.sleep(0.15)
    out[sid]= (surname(au)+" et al.") if au else "MISS"
out.update({k:v+" et al." for k,v in KNOWN.items()})
print("AUTHORS_OVERRIDE = {")
for k,v in sorted(out.items()): print(f'  "{k}": "{v}",')
print("}")
print("\nMISSES:", [k for k,v in out.items() if v=="MISS"])
