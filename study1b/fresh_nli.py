#!/usr/bin/env python3
"""Cycle-7 adjudicating experiment (Paper A): a FRESHLY AUTHORED NLI holdout that no model can have
seen, to disentangle genuine NLI competence from memorization of SNLI/MultiNLI. The items are novel,
unambiguous (clear gold), and were written for this paper. We run the same six-model open panel and
report per-model accuracy + jury vs the authored gold. High accuracy => the NLI result is competence,
not contamination. Writes data/fresh_nli.json + paper/tables/freshnli_autostats.tex."""
import json, asyncio, sys
from pathlib import Path
import aiohttp
ROOT=Path(__file__).resolve().parent.parent; DATA=ROOT/"data"; TAB=ROOT/"paper"/"tables"
sys.path.insert(0,str(ROOT/"study3")); from harness import BASE,KEY,call_chat,PROMPTS,parse_label,MODEL_CONCURRENCY,DEFAULT_PANEL
# 45 freshly authored, unambiguous items (e=entailment, n=neutral, c=contradiction)
ITEMS=[
("A maintenance crew repainted the old lighthouse a bright shade of green last Tuesday.","The lighthouse is now green.","e"),
("Priya finished assembling the bookshelf before her guests arrived for dinner.","The bookshelf was assembled.","e"),
("Every member of the rowing team wore a yellow cap during the morning race.","At least one rower wore a yellow cap.","e"),
("The bakery sold out of cinnamon rolls within an hour of opening.","The bakery had cinnamon rolls for sale that day.","e"),
("Marcus carried the heavy oak table up three flights of stairs by himself.","Marcus moved the table upstairs.","e"),
("The orchestra rehearsed the new symphony for six hours without a single break.","The orchestra practiced the symphony.","e"),
("A sudden hailstorm shattered two greenhouse windows on the Henderson farm.","Some greenhouse windows were broken.","e"),
("The librarian stamped each returned book and shelved them in alphabetical order.","The returned books were placed on shelves.","e"),
("Lena planted forty tulip bulbs along the eastern edge of the courtyard.","Tulip bulbs were planted in the courtyard.","e"),
("The committee unanimously approved the budget after a brief discussion.","Every committee member voted to approve the budget.","e"),
("A pod of dolphins followed the ferry for almost twenty minutes near the harbor.","Dolphins swam near the ferry.","e"),
("The chef garnished every plate with a sprig of fresh basil before serving.","The dishes were garnished with basil.","e"),
("Two electricians rewired the entire warehouse over the long weekend.","The warehouse was rewired.","e"),
("The children built a sandcastle with four tall towers near the tide line.","The children made a sandcastle.","e"),
("Aunt Rosa knitted a wool scarf for each of her seven grandchildren.","Rosa knitted scarves.","e"),
("A delivery driver dropped a package at the blue house on Carver Street.","The driver delivered a parcel to a house.","n"),
("The students gathered in the auditorium before the assembly began.","The students were excited about the assembly.","n"),
("Theo bought a secondhand bicycle from a shop downtown on Saturday.","Theo plans to ride the bicycle to work.","n"),
("A photographer set up her tripod at the edge of the canyon at dawn.","The photographer was taking pictures of wildlife.","n"),
("The cafe replaced its paper straws with a new compostable brand.","Customers complained about the old straws.","n"),
("Nadia received a handwritten letter from an address she did not recognize.","The letter contained good news.","n"),
("The museum extended its hours during the summer festival.","The museum was crowded every evening.","n"),
("A gardener trimmed the hedges in the park early in the morning.","The gardener was paid by the city.","n"),
("The startup moved into a larger office on the fourth floor.","The startup had hired ten new employees.","n"),
("Owen left his umbrella on the train during the evening commute.","Owen was traveling home from work.","n"),
("The choir traveled to a neighboring town for a weekend performance.","The choir won a competition in the town.","n"),
("A scientist labeled each sample jar with a different colored sticker.","The samples were collected from a river.","n"),
("The tailor measured the fabric twice before making the first cut.","The tailor was making a wedding dress.","n"),
("A street vendor sold roasted chestnuts near the train station.","The chestnuts were grown locally.","n"),
("The teacher handed back the graded essays at the end of class.","Every student passed the essay.","n"),
("The entire shipment of glass vases arrived intact at the gallery.","Several of the vases arrived broken.","c"),
("Diego stayed awake through the whole overnight flight to Lisbon.","Diego slept soundly during the flight.","c"),
("The pond froze solid enough for skating by the second week of December.","The pond remained completely unfrozen all winter.","c"),
("Every ticket for the concert sold out within minutes of release.","Plenty of concert tickets were still available a week later.","c"),
("The new policy requires all visitors to sign in at the front desk.","Visitors may enter the building without signing in.","c"),
("Hana watered the ferns every single morning while her neighbor was away.","Hana never watered the ferns.","c"),
("The volcano has been completely dormant for over four hundred years.","The volcano erupted last month.","c"),
("All of the puppies in the litter were born with short black fur.","One of the puppies had long white fur.","c"),
("The bridge was closed to all traffic during the inspection.","Cars crossed the bridge freely during the inspection.","c"),
("The recipe calls for exactly three cups of flour and no sugar.","The recipe requires a large amount of sugar.","c"),
("Grandfather has never once traveled outside his home country.","Grandfather frequently vacations on other continents.","c"),
("The warehouse was empty after the final truck pulled away.","The warehouse was still full of crates.","c"),
("The marathon route avoided every major highway in the city.","The marathon was run entirely along the highways.","c"),
("Sofia returned the borrowed ladder to her neighbor the next day.","Sofia kept the ladder and never gave it back.","c"),
("The thermostat kept the greenhouse at a steady warm temperature all night.","The greenhouse was freezing cold throughout the night.","c"),
]
LABELMAP=PROMPTS["nli_v1"]["labels"]  # entailment->e etc.
async def judge(session,sem,model,prem,hyp):
    msgs=[{"role":"system","content":PROMPTS["nli_v1"]["system"]},
          {"role":"user","content":PROMPTS["nli_v1"]["user"].format(premise=prem,hypothesis=hyp)}]
    async with sem:
        r=await call_chat(session,model,msgs,1500,0.0)
    t=r.get("content") or r.get("reasoning") or ""
    return parse_label(t,LABELMAP)
async def main():
    assert KEY; panel=DEFAULT_PANEL
    res={"n_items":len(ITEMS),"models":{}}
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=64)) as s:
        for m in panel:
            sem=asyncio.Semaphore(min(32,MODEL_CONCURRENCY.get(m,16)))
            preds=await asyncio.gather(*[judge(s,sem,m,p,h) for p,h,g in ITEMS])
            acc=sum(1 for pr,(p,h,g) in zip(preds,ITEMS) if pr==g)/len(ITEMS)
            res["models"][m]={"accuracy":round(acc,3),"parsed":sum(1 for x in preds if x)}
            res["models"][m]["_preds"]=preds
            print(f"  {m:18s} acc={acc:.3f} parsed={res['models'][m]['parsed']}/{len(ITEMS)}")
    # jury (majority vote per item)
    from collections import Counter
    jc=0
    for i,(p,h,g) in enumerate(ITEMS):
        votes=[res["models"][m]["_preds"][i] for m in panel if res["models"][m]["_preds"][i]]
        if votes and Counter(votes).most_common(1)[0][0]==g: jc+=1
    res["jury_accuracy"]=round(jc/len(ITEMS),3)
    for m in panel: res["models"][m].pop("_preds",None)
    best=max(res["models"],key=lambda m:res["models"][m]["accuracy"])
    res["best_model"]=best; res["best_acc"]=res["models"][best]["accuracy"]
    print(f"  JURY acc={res['jury_accuracy']}  best={best} {res['best_acc']}")
    DATA.mkdir(exist_ok=True); json.dump(res,open(DATA/"fresh_nli.json","w"),indent=2)
    g=lambda x:f"{x:.2f}"
    (TAB/"freshnli_autostats.tex").write_text("% auto-generated by fresh_nli.py\n"+
        f"\\newcommand{{\\FNn}}{{{len(ITEMS)}}}\n\\newcommand{{\\FNbest}}{{{g(res['best_acc'])}}}\n"
        f"\\newcommand{{\\FNjury}}{{{g(res['jury_accuracy'])}}}\n\\newcommand{{\\FNbestmodel}}{{\\texttt{{{best.replace('-fp8','')}}}}}\n")
    print("wrote data/fresh_nli.json + freshnli_autostats.tex")
if __name__=="__main__": asyncio.run(main())
