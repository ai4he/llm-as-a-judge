#!/usr/bin/env python3
"""Study 3 LLM-as-a-judge harness (parallel, OpenAI-compatible Clemson RCD API).

- Per-model concurrency = the documented limit (catalogue) via asyncio semaphores.
- Reasoning-model aware: large max_tokens + parse content, fall back to scanning the
  `reasoning` field, retry once with a bigger budget on truncation.
- Idempotent append-only JSONL checkpoint keyed by (dataset,item_id,model,prompt_id,run_id):
  resumable, never double-calls.
Reads RCD_LLM_API_KEY / RCD_LLM_BASE_URL from env (source secrets.sh).
Usage:  python3 study3/harness.py --dataset chaosnli_snli --n 200 --models qwen3.5-9b gemma-4-31b gptoss-120b --run r0
        python3 study3/harness.py --smoke
"""
import os, json, re, zipfile, asyncio, argparse, time, random
from pathlib import Path
import aiohttp

ROOT = Path(__file__).resolve().parent.parent
RAW  = ROOT/"study3"/"data_raw"
OUT  = ROOT/"study3"/"outputs"; OUT.mkdir(parents=True, exist_ok=True)
BASE = os.environ.get("RCD_LLM_BASE_URL","https://llm.rcd.clemson.edu/v1")
KEY  = os.environ.get("RCD_LLM_API_KEY","")

# model id -> documented max concurrency (from the Clemson catalogue)
MODEL_CONCURRENCY = {
    "gptoss-120b":128, "qwen3.5-9b":128, "glm-5.1-fp8":64, "qwen3.6-27b-fp8":64,
    "qwen3.6-35b-a3b-fp8":64, "deepseek-v4-pro":48, "gemma-4-31b":16, "qwen3-omni-30b-a3b":16,
}
DEFAULT_PANEL = ["glm-5.1-fp8","gptoss-120b","deepseek-v4-pro","gemma-4-31b","qwen3.6-27b-fp8","qwen3.5-9b"]

# ---------------- prompts (versioned) ----------------
PROMPTS = {
 "nli_v1": {
   "system":"You are an expert linguistic annotator performing Natural Language Inference (NLI). "
            "You decide the relationship between a premise and a hypothesis.",
   "user":"Premise: {premise}\nHypothesis: {hypothesis}\n\n"
          "Does the premise ENTAIL the hypothesis, CONTRADICT it, or is it NEUTRAL (neither)?\n"
          "Answer with ONLY a JSON object on the last line: {{\"label\": \"entailment\"|\"neutral\"|\"contradiction\"}}",
   "labels":{"entailment":"e","neutral":"n","contradiction":"c"},
 },
}

def parse_label(text, labelmap):
    if not text: return None
    # prefer a JSON object
    for m in re.finditer(r'\{[^{}]*"label"\s*:\s*"([a-zA-Z]+)"[^{}]*\}', text):
        v=m.group(1).lower()
        if v in labelmap: return labelmap[v]
    # else last mention of a label word
    last=None
    for w,code in labelmap.items():
        for mm in re.finditer(r'\b'+re.escape(w)+r'\b', text, re.I): last=(mm.start(),code)
    if last: return last[1]
    # single-letter fallback
    m=re.search(r'\b([enc])\b', text.strip().lower())
    return m.group(1) if m else None

# ---------------- dataset loaders ----------------
def load_chaosnli(subset="snli", n=None, seed=20260529):
    z=zipfile.ZipFile(RAW/"chaosNLI_v1.0.zip")
    fn={"snli":"chaosNLI_snli.jsonl","mnli":"chaosNLI_mnli_m.jsonl"}[subset]
    path=[x for x in z.namelist() if x.endswith(fn)][0]
    items=[]
    for line in z.open(path):
        d=json.loads(line); ex=d.get("example",{})
        p=ex.get("premise"); h=ex.get("hypothesis")
        if not p or not h: continue
        items.append({"id":d["uid"],"premise":p,"hypothesis":h,
                      "gold":d["majority_label"],            # 'e'/'n'/'c'
                      "label_count":d["label_count"],         # [e,n,c]
                      "n_annot":sum(d["label_count"])})
    random.Random(seed).shuffle(items)
    return items[:n] if n else items

LOADERS={"chaosnli_snli":lambda n: load_chaosnli("snli",n),
         "chaosnli_mnli":lambda n: load_chaosnli("mnli",n)}
TASK_PROMPT={"chaosnli_snli":"nli_v1","chaosnli_mnli":"nli_v1"}

# ---------------- async judging ----------------
async def call_chat(session, model, messages, max_tokens, temperature):
    body={"model":model,"messages":messages,"max_tokens":max_tokens,"temperature":temperature}
    for attempt in range(5):
        try:
            async with session.post(BASE+"/chat/completions",
                    headers={"Authorization":"Bearer "+KEY,"Content-Type":"application/json"},
                    json=body, timeout=aiohttp.ClientTimeout(total=180)) as r:
                if r.status in (429,500,502,503,504):
                    await asyncio.sleep(2*(attempt+1)+random.random()); continue
                d=await r.json()
                ch=(d.get("choices") or [{}])[0]; msg=ch.get("message",{}) or {}
                return {"content":msg.get("content"),"reasoning":msg.get("reasoning"),
                        "finish":ch.get("finish_reason"),"served":d.get("model")}
        except Exception as e:
            if attempt==4: return {"error":repr(e)[:120]}
            await asyncio.sleep(2*(attempt+1)+random.random())
    return {"error":"retries_exhausted"}

async def judge(session, sem, model, item, prompt_id, run_id, temperature):
    pr=PROMPTS[prompt_id]
    messages=[{"role":"system","content":pr["system"]},
              {"role":"user","content":pr["user"].format(premise=item["premise"],hypothesis=item["hypothesis"])}]
    async with sem:
        res=await call_chat(session,model,messages,1500,temperature)
        text=res.get("content") or res.get("reasoning") or ""
        label=parse_label(text,pr["labels"])
        if label is None and res.get("finish")=="length":           # truncated mid-reasoning -> retry bigger
            res=await call_chat(session,model,messages,4096,temperature)
            text=res.get("content") or res.get("reasoning") or ""; label=parse_label(text,pr["labels"])
    return {"model":model,"item_id":item["id"],"prompt_id":prompt_id,"run_id":run_id,
            "pred":label,"gold":item["gold"],"finish":res.get("finish"),
            "served":res.get("served"),"error":res.get("error")}

def done_keys(path):
    keys=set()
    if path.exists():
        for line in open(path):
            try: d=json.loads(line); keys.add((d["item_id"],d["model"],d["prompt_id"],d["run_id"]))
            except: pass
    return keys

async def run(dataset, n, models, run_id, temperature):
    items=LOADERS[dataset](n); prompt_id=TASK_PROMPT[dataset]
    out=OUT/f"{dataset}.judgments.jsonl"; done=done_keys(out)
    sems={m:asyncio.Semaphore(MODEL_CONCURRENCY.get(m,16)) for m in models}
    conn=aiohttp.TCPConnector(limit=sum(MODEL_CONCURRENCY.get(m,16) for m in models)+8)
    tasks=[]
    async with aiohttp.ClientSession(connector=conn) as session:
        for it in items:
            for m in models:
                if (it["id"],m,prompt_id,run_id) in done: continue
                tasks.append(judge(session,sems[m],m,it,prompt_id,run_id,temperature))
        print(f"[{dataset}] items={len(items)} models={len(models)} new_calls={len(tasks)} (skipped {len(items)*len(models)-len(tasks)})")
        t0=time.time(); n_done=0
        with open(out,"a") as f:
            for fut in asyncio.as_completed(tasks):
                rec=await fut; f.write(json.dumps(rec)+"\n"); f.flush()
                n_done+=1
                if n_done%50==0: print(f"   {n_done}/{len(tasks)} in {time.time()-t0:.0f}s")
        print(f"[{dataset}] wrote {n_done} judgments in {time.time()-t0:.0f}s -> {out}")
    # quick parse-rate diagnostic
    recs=[json.loads(l) for l in open(out)]
    by={}
    for r in recs:
        by.setdefault(r["model"],[0,0,0]); by[r["model"]][0]+=1
        if r["pred"] is None: by[r["model"]][1]+=1
        if r.get("error"): by[r["model"]][2]+=1
    print("model           n   unparsed errors")
    for m,(tot,un,er) in by.items(): print(f"  {m:18s} {tot:4d} {un:5d} {er:5d}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",default="chaosnli_snli"); ap.add_argument("--n",type=int,default=200)
    ap.add_argument("--models",nargs="*",default=None); ap.add_argument("--run",default="r0")
    ap.add_argument("--temperature",type=float,default=0.0); ap.add_argument("--smoke",action="store_true")
    a=ap.parse_args()
    assert KEY, "set RCD_LLM_API_KEY (source secrets.sh)"
    if a.smoke:
        asyncio.run(run("chaosnli_snli",3,["qwen3.5-9b","gemma-4-31b"],"smoke",0.0)); return
    models=a.models or DEFAULT_PANEL
    asyncio.run(run(a.dataset,a.n,models,a.run,a.temperature))

if __name__=="__main__": main()
