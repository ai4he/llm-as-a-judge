#!/usr/bin/env python3
"""Cycle-1 revision API experiments:
 1. Prompt-robustness: 3 paraphrased instruction variants per task x 3 models x 6 datasets (n=120);
    report per-cell accuracy mean/SD across prompts and whether the reliability verdict flips.
 2. Contamination clean-control: run the verbatim-recall probe on a freshly authored post-cutoff
    holdout (should be LOW) and contrast with the in-study datasets, to show the probe discriminates.
Writes data/rev_prompt_robustness.json and data/rev_contamination_control.json.
"""
import os, json, asyncio, sys, re, difflib
from pathlib import Path
import aiohttp
ROOT=Path(__file__).resolve().parent.parent; OUT=ROOT/"study3"/"outputs"; DATA=ROOT/"data"
sys.path.insert(0,str(ROOT/"study3"))
from harness import BASE,KEY,call_chat,LOADERS,MODEL_CONCURRENCY,parse_label,PROMPTS,TASK_PROMPT
MODELS=["glm-5.1-fp8","gemma-4-31b","qwen3.5-9b"]
DSETS=["chaosnli_snli","go_emotions","social_bias_frames","hatexplain","multipico","chaosnli_mnli"]
N=120
# 3 paraphrased USER templates per task (semantically equivalent rewordings of the v1 prompt)
PARA={
 "nli_v1":[
  "Read the premise and hypothesis. Premise: {premise}\nHypothesis: {hypothesis}\nDecide the logical relation. Reply with ONLY a JSON object on the last line: {{\"label\": \"entailment\"|\"neutral\"|\"contradiction\"}}",
  "Given Premise: {premise}\nand Hypothesis: {hypothesis}\nis the hypothesis entailed by, contradicted by, or unrelated to (neutral) the premise? End with ONLY: {{\"label\": \"entailment\"|\"neutral\"|\"contradiction\"}}",
  "Premise: {premise}\nHypothesis: {hypothesis}\nClassify their relationship as entailment, contradiction, or neutral. Final line ONLY a JSON object: {{\"label\": \"entailment\"|\"neutral\"|\"contradiction\"}}"],
 "hate_v1":[
  "Read this social-media post and decide its category. Post: {text}\nCategories: HATE SPEECH (targets a group by identity), OFFENSIVE (abusive but not group-targeted), NORMAL. Reply ONLY: {{\"label\": \"hate speech\"|\"offensive\"|\"normal\"}}",
  "Post: {text}\nDoes this post constitute hate speech (identity-based attack), offensive language (not group-targeted), or normal speech? End with ONLY a JSON object: {{\"label\": \"hate speech\"|\"offensive\"|\"normal\"}}",
  "Classify the following post. Post: {text}\nIs it hate speech, offensive, or normal? Answer ONLY with: {{\"label\": \"hate speech\"|\"offensive\"|\"normal\"}}"],
 "emotion_v1":[
  "Comment: {text}\nWhat overall sentiment does it express: positive, negative, ambiguous (mixed/uncertain), or neutral? Reply ONLY: {{\"label\": \"positive\"|\"negative\"|\"ambiguous\"|\"neutral\"}}",
  "Read the Reddit comment. Comment: {text}\nChoose its sentiment from positive / negative / ambiguous / neutral. Final line ONLY: {{\"label\": \"positive\"|\"negative\"|\"ambiguous\"|\"neutral\"}}",
  "Comment: {text}\nLabel the comment's sentiment as positive, negative, ambiguous, or neutral. Answer ONLY with a JSON object: {{\"label\": \"positive\"|\"negative\"|\"ambiguous\"|\"neutral\"}}"],
 "offensive_v1":[
  "Post: {text}\nWould a reasonable reader consider this post offensive (rude/toxic/demeaning), including borderline cases? Reply ONLY: {{\"label\": \"offensive\"|\"not offensive\"}}",
  "Decide whether the post is offensive. Post: {text}\nEnd with ONLY a JSON object: {{\"label\": \"offensive\"|\"not offensive\"}}",
  "Is the following post offensive or not offensive? Post: {text}\nAnswer ONLY with: {{\"label\": \"offensive\"|\"not offensive\"}}"],
 "irony_v1":[
  "{text}\nIs the reply ironic/sarcastic (meaning differs from the literal words) or not? Reply ONLY: {{\"label\": \"ironic\"|\"not ironic\"}}",
  "Consider the reply in context. {text}\nDoes the reply express irony/sarcasm, or is it literal? End with ONLY: {{\"label\": \"ironic\"|\"not ironic\"}}",
  "{text}\nLabel the reply as ironic or not ironic. Answer ONLY with a JSON object: {{\"label\": \"ironic\"|\"not ironic\"}}"],
}
def verdict(acc,ceil):
    g=acc-ceil
    return ("Validated" if g>=-0.02 else "Promising" if g>=-0.08 else "Mixed" if g>=-0.18 else "Caution" if g>=-0.30 else "Unreliable")

async def judge_variant(session,sem,model,item,pid,tmpl):
    pr=PROMPTS[pid]
    safe={"text":item.get("text",""),"premise":item.get("premise",""),"hypothesis":item.get("hypothesis",""),
          "post":item.get("post",item.get("text","")),"reply":item.get("reply","")}
    try: user=tmpl.format(**{**safe,**item})
    except Exception: user=tmpl.format_map(__import__("collections").defaultdict(str,{**safe,**item}))
    msgs=[{"role":"system","content":pr["system"]},{"role":"user","content":user}]
    async with sem:
        r=await call_chat(session,model,msgs,1500,0.0)
    t=r.get("content") or r.get("reasoning") or ""
    return parse_label(t,pr["labels"])

async def prompt_robustness():
    res={}
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=64)) as s:
        for ds in DSETS:
            items=LOADERS[ds](N); pid=TASK_PROMPT[ds]; ceil=None
            try: ceil=json.load(open(OUT/f"{ds}.full.scores.json"))["human_ceiling_acc"]
            except: pass
            for m in MODELS:
                sem=asyncio.Semaphore(min(32,MODEL_CONCURRENCY.get(m,16)))
                accs=[]; verds=[]
                for vi,tmpl in enumerate(PARA[pid]):
                    tasks=[judge_variant(s,sem,m,it,pid,tmpl) for it in items]
                    preds=await asyncio.gather(*tasks)
                    acc=sum(1 for p,it in zip(preds,items) if p==it["gold"])/len(items)
                    accs.append(round(acc,3));
                    if ceil is not None: verds.append(verdict(acc,ceil))
                import statistics
                res[f"{ds}|{m}"]={"acc_per_prompt":accs,"acc_mean":round(statistics.mean(accs),3),
                    "acc_sd":round(statistics.pstdev(accs),3),"verdicts":verds,
                    "verdict_stable":len(set(verds))<=1 if verds else None}
                print(f"  {ds:18s} {m:16s} acc={accs} sd={res[f'{ds}|{m}']['acc_sd']:.3f} verdicts={verds} stable={res[f'{ds}|{m}']['verdict_stable']}")
    json.dump(res,open(DATA/"rev_prompt_robustness.json","w"),indent=2)
    nstab=sum(1 for v in res.values() if v["verdict_stable"]); print(f"verdict-stable cells: {nstab}/{len(res)}")

# ---- contamination clean control: freshly authored post-cutoff short texts ----
CLEAN=[ "The violet drone hummed over the quartz market at dawn selling pickled starlight.",
 "Marisol taught her left-handed parrot to file quarterly taxes in three currencies.",
 "Our office moved the coffee machine onto the roof so meetings would end faster.",
 "The committee voted to rename Tuesday after a beetle nobody had ever photographed.",
 "He knit a sweater for the bridge so it would stop complaining about the wind.",
 "A small comet applied for a library card under the name Gerald Frostwhistle.",
 "The recipe called for two cups of patience and a pinch of borrowed thunder.",
 "She debugged the garden by replacing every third tomato with a polite rumor.",
 "The mayor of the elevator announced a surprise inspection of all the buttons.",
 "We trained the umbrella to predict gossip with ninety-one percent recall." ]
def norm(s): return re.sub(r"\s+"," ",re.sub(r"[^a-z0-9 ]","",str(s).lower())).strip()
async def clean_control(models=("gemma-4-31b","qwen3.5-9b")):
    async def probe(s,sem,m,txt):
        words=txt.split(); prefix=" ".join(words[:max(3,len(words)//3)])
        msgs=[{"role":"system","content":"You are being tested on verbatim recall of a known public dataset. If you recognize the text, reproduce it exactly."},
              {"role":"user","content":f"Reproduce the COMPLETE original text verbatim (only the text):\n\n\"{prefix} ...\""}]
        async with sem:
            r=await call_chat(s,m,msgs,400,0.0)
        out=r.get("content") or r.get("reasoning") or ""
        return difflib.SequenceMatcher(None,norm(out),norm(txt)).ratio()
    rep={}
    async with aiohttp.ClientSession() as s:
        for m in models:
            sem=asyncio.Semaphore(min(16,MODEL_CONCURRENCY.get(m,16)))
            ov=await asyncio.gather(*[probe(s,sem,m,t) for t in CLEAN])
            mean=sum(ov)/len(ov)
            rep[m]={"n":len(ov),"mean_overlap":round(mean,3),"flag":"HIGH" if mean>0.6 else "MEDIUM" if mean>0.35 else "LOW"}
            print(f"  CLEAN-CONTROL {m}: mean_overlap={mean:.3f} ({rep[m]['flag']})")
    json.dump(rep,open(DATA/"rev_contamination_control.json","w"),indent=2)

async def main():
    assert KEY,"source secrets.sh"
    print("=== prompt robustness ==="); await prompt_robustness()
    print("=== contamination clean control ==="); await clean_control()
if __name__=="__main__": asyncio.run(main())
