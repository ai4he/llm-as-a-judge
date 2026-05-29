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
 "hate_v1": {
   "system":"You are an expert content-moderation annotator. You label social-media posts as hateful, "
            "offensive, or normal following standard hate-speech annotation guidelines. Content may be "
            "offensive; classify it objectively.",
   "user":"Post: {text}\n\nClassify the post: HATE SPEECH (attacks/dehumanizes a group based on identity), "
          "OFFENSIVE (abusive/offensive but not group-targeted hate), or NORMAL.\n"
          "Answer with ONLY a JSON object on the last line: {{\"label\": \"hate speech\"|\"offensive\"|\"normal\"}}",
   "labels":{"hate speech":"hate","hatespeech":"hate","hateful":"hate","hate":"hate",
             "offensive":"off","abusive":"off","normal":"norm","neither":"norm","none":"norm"},
 },
 "emotion_v1": {
   "system":"You are an expert emotion/sentiment annotator labeling short social-media (Reddit) comments.",
   "user":"Comment: {text}\n\nWhat is the overall sentiment of the comment? POSITIVE, NEGATIVE, AMBIGUOUS "
          "(mixed/uncertain such as surprise or curiosity), or NEUTRAL.\n"
          "Answer with ONLY a JSON object on the last line: {{\"label\": \"positive\"|\"negative\"|\"ambiguous\"|\"neutral\"}}",
   "labels":{"positive":"pos","negative":"neg","ambiguous":"amb","mixed":"amb","neutral":"neu"},
 },
 "offensive_v1": {
   "system":"You are a content-moderation annotator. You judge whether social-media posts are offensive "
            "(rude, disrespectful, toxic, or demeaning toward a person or group). Content may be offensive; judge objectively.",
   "user":"Post: {text}\n\nIs this post offensive (including potentially offensive / 'maybe')?\n"
          "Answer with ONLY a JSON object on the last line: {{\"label\": \"offensive\"|\"not offensive\"}}",
   "labels":{"offensive":"off","maybe":"off","yes":"off","not offensive":"notoff","inoffensive":"notoff","no":"notoff","none":"notoff"},
 },
 "irony_v1": {
   "system":"You are an expert annotator labeling whether a social-media reply is ironic or sarcastic in the "
            "context of the post it replies to.",
   "user":"{text}\n\nIs the REPLY ironic/sarcastic (it conveys a meaning different from or opposite to its literal "
          "content), or not ironic (literal/sincere)?\n"
          "Answer with ONLY a JSON object on the last line: {{\"label\": \"ironic\"|\"not ironic\"}}",
   "labels":{"ironic":"iro","irony":"iro","sarcastic":"iro","sarcasm":"iro","yes":"iro",
             "not ironic":"noiro","notironic":"noiro","literal":"noiro","sincere":"noiro","no":"noiro","not":"noiro"},
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
                      "gold":d["majority_label"],"classes":["e","n","c"],
                      "label_count":d["label_count"],"n_annot":sum(d["label_count"])})
    random.Random(seed).shuffle(items)
    return items[:n] if n else items

def load_hatexplain(n=None, seed=20260529):
    "HateXplain (3 annotators/post; hate/offensive/normal). Sensitive content; only ids/labels are stored."
    import urllib.request
    from collections import Counter
    cache=RAW/"hatexplain_dataset.json"
    if not cache.exists():
        urllib.request.urlretrieve("https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/dataset.json", cache)
    d=json.load(open(cache)); LM={"hatespeech":"hate","offensive":"off","normal":"norm"}; classes=["hate","off","norm"]
    items=[]
    for pid,rec in d.items():
        labs=[LM[a["label"]] for a in rec.get("annotators",[]) if a.get("label") in LM]
        if len(labs)<3: continue
        cnt=Counter(labs); top,tc=cnt.most_common(1)[0]
        if list(cnt.values()).count(tc)>1: continue            # no majority -> drop (HateXplain convention)
        items.append({"id":pid,"text":" ".join(rec.get("post_tokens",[])),"gold":top,
                      "classes":classes,"label_count":[cnt.get(c,0) for c in classes],"n_annot":len(labs)})
    random.Random(seed).shuffle(items)
    return items[:n] if n else items

_GE_POS={"admiration","amusement","approval","caring","desire","excitement","gratitude","joy","love","optimism","pride","relief"}
_GE_NEG={"anger","annoyance","disappointment","disapproval","disgust","embarrassment","fear","grief","nervousness","remorse","sadness"}
_GE_AMB={"confusion","curiosity","realization","surprise"}
_GE_EMOS=sorted(_GE_POS|_GE_NEG|_GE_AMB|{"neutral"})
def _ge_sent(flags):
    from collections import Counter
    g=Counter()
    for e in flags:
        g["pos" if e in _GE_POS else "neg" if e in _GE_NEG else "amb" if e in _GE_AMB else "neu" if e=="neutral" else "x"]+=1
    g.pop("x",None)
    if not g: return "neu"
    top=g.most_common()
    return "amb" if len(top)>1 and top[0][1]==top[1][1] else top[0][0]
def load_goemotions(n=None, seed=20260529):
    "GoEmotions collapsed to 4-way sentiment (pos/neg/amb/neu); multi-rater majority gold."
    from datasets import load_dataset
    from collections import Counter
    ds=load_dataset("google-research-datasets/go_emotions","raw",split="train")
    byid={}
    for r in ds:
        flags=[e for e in _GE_EMOS if r.get(e)==1]
        d=byid.setdefault(r["id"],{"text":r["text"],"s":[]}); d["s"].append(_ge_sent(flags))
    classes=["pos","neg","amb","neu"]; items=[]
    for rid,d in byid.items():
        if len(d["s"])<3: continue
        cnt=Counter(d["s"]); top,tc=cnt.most_common(1)[0]
        if list(cnt.values()).count(tc)>1: continue
        items.append({"id":rid,"text":d["text"],"gold":top,"classes":classes,
                      "label_count":[cnt.get(c,0) for c in classes],"n_annot":len(d["s"])})
    random.Random(seed).shuffle(items); return items[:n] if n else items

def load_sbic(n=None, seed=20260529):
    "Social Bias Frames: binary offensiveness (off/notoff) from per-annotator offensiveYN; majority gold. Sensitive."
    import urllib.request, tarfile, csv as _csv
    from collections import defaultdict, Counter
    tgz=RAW/"SBIC.v2.tgz"
    if not tgz.exists(): urllib.request.urlretrieve("https://maartensap.com/social-bias-frames/SBIC.v2.tgz", tgz)
    tf=tarfile.open(tgz)
    name=[m.name for m in tf.getmembers() if m.name.endswith("trn.csv") and "agg" not in m.name.lower()][0]
    rows=list(_csv.DictReader((l.decode("utf-8","ignore") for l in tf.extractfile(name))))
    byp=defaultdict(lambda:{"text":None,"o":[]})
    for r in rows:
        post=r.get("post"); off=r.get("offensiveYN")
        if not post or off in (None,""): continue
        try: lab="off" if float(off)>=0.5 else "notoff"
        except: continue
        d=byp[post]; d["text"]=post; d["o"].append(lab)
    classes=["off","notoff"]; items=[]; i=0
    for post,d in byp.items():
        if len(d["o"])<3: continue
        cnt=Counter(d["o"]); top,tc=cnt.most_common(1)[0]
        if list(cnt.values()).count(tc)>1: continue
        items.append({"id":f"sbic_{i}","text":post,"gold":top,"classes":classes,
                      "label_count":[cnt.get(c,0) for c in classes],"n_annot":len(d["o"])}); i+=1
    random.Random(seed).shuffle(items); return items[:n] if n else items

def load_multipico(n=500, lang="en", seed=20260529):
    "MultiPICo (2024): subjective irony detection, binary. Multi-annotator + demographics; recent release"
    " -> low-contamination CLEAN condition vs the older sensitive sets."
    from datasets import load_dataset
    from collections import defaultdict, Counter
    ds=load_dataset("Multilingual-Perspectivist-NLU/MultiPICo","default",split="train",streaming=True)
    byk=defaultdict(lambda:{"text":None,"labs":[]})
    for ex in ds:
        if ex.get("language")!=lang: continue
        try: lab="iro" if int(ex["label"])==1 else "noiro"
        except: continue
        d=byk[(ex.get("post_id"),ex.get("reply_id"))]; d["labs"].append(lab)
        if d["text"] is None:
            d["text"]=f"Post: {(ex.get('post') or '').strip()}\nReply: {(ex.get('reply') or '').strip()}"
    classes=["iro","noiro"]; items=[]; i=0
    for k,d in byk.items():
        if len(d["labs"])<3: continue
        cnt=Counter(d["labs"]); top,tc=cnt.most_common(1)[0]
        if list(cnt.values()).count(tc)>1: continue          # drop ties (no clear majority)
        items.append({"id":f"mpc_{i}","text":d["text"],"gold":top,"classes":classes,
                      "label_count":[cnt.get(c,0) for c in classes],"n_annot":len(d["labs"])}); i+=1
    random.Random(seed).shuffle(items); return items[:n] if n else items

LOADERS={"chaosnli_snli":lambda n: load_chaosnli("snli",n),
         "chaosnli_mnli":lambda n: load_chaosnli("mnli",n),
         "hatexplain":lambda n: load_hatexplain(n),
         "go_emotions":lambda n: load_goemotions(n),
         "social_bias_frames":lambda n: load_sbic(n),
         "multipico":lambda n: load_multipico(n)}
TASK_PROMPT={"chaosnli_snli":"nli_v1","chaosnli_mnli":"nli_v1","hatexplain":"hate_v1",
             "go_emotions":"emotion_v1","social_bias_frames":"offensive_v1","multipico":"irony_v1"}

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
              {"role":"user","content":pr["user"].format(**item)}]
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
