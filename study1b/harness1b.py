#!/usr/bin/env python3
"""Study 1b harness: do the MOST POWERFUL open models (Clemson RCD) surpass the
LLM-as-a-judge numbers REPORTED for 'promising' domains in Study 1a?

Flagship dataset = SummEval (Fabbri 2021): 100 CNN/DM source docs x 16 machine
summaries, each with expert means for coherence/consistency/fluency/relevance.
We replicate the G-Eval (Liu 2023) summary-level evaluation: rate each summary 1-5
per aspect with the G-Eval rubric, then compare summary-level Spearman to the
reported G-Eval/GPT-4 correlations.  Reusable for additional promising-domain sets.

Reuses the Study-3 OpenAI-compatible async client (call_chat).  Append-only JSONL
checkpoint keyed by (dataset,item_id,aspect,model,run_id): resumable, never double-calls.
Usage:
  source secrets.sh
  python3 study1b/harness1b.py --dataset summeval --models glm-5.1-fp8 deepseek-v4-pro gptoss-120b qwen3.6-35b-a3b-fp8 --run powerful
  python3 study1b/harness1b.py --smoke
"""
import os, json, re, asyncio, argparse, sys, time
from pathlib import Path
import aiohttp
ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT/"study1b"/"outputs"; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT/"study3"))
from harness import BASE, KEY, call_chat, MODEL_CONCURRENCY   # reuse client + concurrency map

# the 'most powerful' tier of the Clemson catalogue (large dense / large MoE flagships)
POWERFUL_PANEL = ["glm-5.1-fp8","deepseek-v4-pro","gptoss-120b","qwen3.6-35b-a3b-fp8"]

# ---- G-Eval aspect rubric (verbatim-style definitions from Liu et al. 2023) ----
ASPECTS = {
 "coherence":  ("Coherence (1-5) - the collective quality of all sentences. The summary should be "
                "well-structured and well-organized, building from sentence to sentence into a coherent "
                "body of information about a topic, not just a heap of related information."),
 "consistency":("Consistency (1-5) - the factual alignment between the summary and the source. A factually "
                "consistent summary contains only statements that are entailed by the source document; "
                "penalize summaries that contain hallucinated or unsupported facts."),
 "fluency":    ("Fluency (1-5) - the quality of individual sentences. Sentences should have no formatting "
                "problems, capitalization errors, or obviously ungrammatical/awkward constructions that make "
                "the text difficult to read."),
 "relevance":  ("Relevance (1-5) - selection of the most important content from the source. The summary "
                "should include only important information; penalize redundancies and excess/unimportant content."),
}
SYS = ("You are an expert evaluator of text summarization. You rate machine-generated summaries against the "
       "source article on a single quality dimension using a 1-5 integer scale, following the given criterion strictly.")
def user_prompt(aspect, source, summary):
    return (f"You will be given one summary written for a news article. Your task is to rate the summary on ONE "
            f"metric. Please read and understand the instructions carefully.\n\n"
            f"Evaluation Criterion:\n{ASPECTS[aspect]}\n\n"
            f"Source Article:\n{source}\n\n"
            f"Summary:\n{summary}\n\n"
            f"Give ONLY an integer from 1 to 5 for {aspect.capitalize()}. "
            f'Answer with ONLY a JSON object on the last line: {{"score": <1-5>}}')

def parse_score(text):
    if not text: return None
    m = re.findall(r'"score"\s*:\s*([1-5])(?:\.0)?', text)
    if m: return int(m[-1])
    # fallback: last standalone 1-5 in the tail
    m = re.findall(r'\b([1-5])\b', text.strip().splitlines()[-1] if text.strip() else "")
    return int(m[-1]) if m else None

# ---------------- loaders ----------------
def load_summeval(n_docs=None):
    from datasets import load_dataset
    ds = load_dataset("mteb/summeval", split="test")
    items=[]
    for di,row in enumerate(ds):
        if n_docs and di>=n_docs: break
        src=row["text"]
        for si,summ in enumerate(row["machine_summaries"]):
            items.append({"id":f"{di:03d}:{si:02d}","docid":di,"sysidx":si,
                          "source":src,"summary":summ,
                          "human":{a:row[a][si] for a in ASPECTS}})
    return items
LOADERS={"summeval":load_summeval}
REPORTED={  # G-Eval-4 summary-level Spearman (Liu et al. 2023, Table 3) -- the bar to beat
 "summeval":{"coherence":0.582,"consistency":0.507,"fluency":0.455,"relevance":0.547,
             "source":"G-Eval-4 (GPT-4), Liu et al. 2023"}}

def done_keys(path):
    s=set()
    if path.exists():
        for l in open(path):
            try: d=json.loads(l); s.add((d["id"],d["aspect"],d["model"]))
            except: pass
    return s

async def one(session, sem, model, item, aspect, run_id, fh, lock):
    msgs=[{"role":"system","content":SYS},
          {"role":"user","content":user_prompt(aspect,item["source"],item["summary"])}]
    async with sem:
        res=await call_chat(session,model,msgs,1200,0.0)
    text=res.get("content") or res.get("reasoning") or ""
    sc=parse_score(text)
    if sc is None and res.get("finish")=="length":
        async with sem:
            res=await call_chat(session,model,msgs,3000,0.0)
        text=res.get("content") or res.get("reasoning") or ""; sc=parse_score(text)
    rec={"id":item["id"],"docid":item["docid"],"sysidx":item["sysidx"],"aspect":aspect,
         "model":model,"run_id":run_id,"score":sc,"human":item["human"][aspect],
         "error":res.get("error"),"finish":res.get("finish")}
    async with lock:
        fh.write(json.dumps(rec)+"\n"); fh.flush()
    return rec

async def run(dataset, models, run_id, n_docs, aspects):
    assert KEY,"source secrets.sh"
    items=LOADERS[dataset](n_docs)
    path=OUT/f"{dataset}.{run_id}.judgments.jsonl"
    done=done_keys(path)
    sems={m:asyncio.Semaphore(min(48,MODEL_CONCURRENCY.get(m,16))) for m in models}
    tasks_planned=[(m,it,a) for it in items for a in aspects for m in models
                   if (it["id"],a,m) not in done]
    print(f"[{dataset}] items={len(items)} aspects={aspects} models={models} "
          f"planned={len(tasks_planned)} (resumed {len(done)})")
    lock=asyncio.Lock()
    with open(path,"a") as fh:
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=128)) as s:
            tasks=[one(s,sems[m],m,it,a,run_id,fh,lock) for (m,it,a) in tasks_planned]
            t0=time.time(); done_n=0
            for f in asyncio.as_completed(tasks):
                await f; done_n+=1
                if done_n%500==0:
                    r=done_n/(time.time()-t0)
                    print(f"  {done_n}/{len(tasks)}  {r:.1f}/s  ETA {int((len(tasks)-done_n)/max(r,1e-9))}s",flush=True)
    print(f"[{dataset}] done -> {path}")

async def smoke():
    items=load_summeval(1)[:1]; it=items[0]
    print("loaded",len(load_summeval(2)),"items (2 docs); sample human:",it["human"])
    async with aiohttp.ClientSession() as s:
        for m in ["glm-5.1-fp8"]:
            r=await one(s,asyncio.Semaphore(1),m,it,"coherence","smoke",open(os.devnull,"w"),asyncio.Lock())
            print(m,"coherence score=",r["score"],"(human",r["human"],") finish",r["finish"])

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",default="summeval"); ap.add_argument("--run",default="powerful")
    ap.add_argument("--models",nargs="*",default=POWERFUL_PANEL)
    ap.add_argument("--n-docs",type=int,default=None)
    ap.add_argument("--aspects",nargs="*",default=list(ASPECTS))
    ap.add_argument("--smoke",action="store_true")
    a=ap.parse_args()
    if a.smoke: asyncio.run(smoke())
    else: asyncio.run(run(a.dataset,a.models,a.run,a.n_docs,a.aspects))
