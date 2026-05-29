#!/usr/bin/env python3
"""Study 2 corpus: LLMs vs human input in Human-Centered Computing / HCI / qualitative-research
methods that historically require human judgment. Writes data/study2_corpus.csv/.json and
data/study2_prisma_counts.json. Coding: method, role (replace/augment/simulate), epistemic
stance (enabling/tension/critical), community, reliability verdict (1-5) vs the human ceiling."""
import json, csv
from pathlib import Path
HERE = Path(__file__).resolve().parent
VERDICT_LABEL = {1:"Validated",2:"Promising (conditional)",3:"Mixed / task-dependent",4:"Caution / limited",5:"Unreliable"}

def S(id,authors,year,venue,ref,method,role,agreement,human_baseline,verdict,stance,community,finding):
    return dict(id=id,authors=authors,year=year,venue=venue,ref=ref,method=method,role=role,
                agreement=agreement,human_baseline=human_baseline,verdict=verdict,
                verdict_label=VERDICT_LABEL[verdict],reliability_score=6-verdict,
                stance=stance,community=community,finding=finding)

STUDIES=[
# ---- Qualitative coding: deductive ----
S("xiao2023support","Xiao et al.",2023,"IUI","2304.10548","Qual coding (deductive)","augment",None,None,2,"enabling","HCI",
  "Codebook+GPT-3 deductive coding reaches fair-to-substantial agreement with expert coders; reliability hinges on codebook design."),
S("zhang2025inddeduct","Zhang et al.",2025,"AHFE","10.54941/ahfe1006232","Qual coding (deductive)","augment",0.46,None,3,"tension","Qual-methods",
  "Inductive+deductive coding with AI: Fleiss kappa ~0.46 (deductive), 0.57 (inductive) vs human coders; moderate, hybrid advised."),
# ---- Qualitative coding: inductive / open ----
S("gao2024collabcoder","Gao et al.",2024,"CHI","2304.07366","Qual coding (inductive)","augment",None,None,2,"enabling","HCI",
  "CollabCoder: lower-barrier rigorous workflow embedding LLM suggestions in collaborative inductive coding; humans keep interpretive authority."),
S("dunivin2024scalablecot","Dunivin",2024,"arXiv","2401.15170","Qual coding (inductive)","replace",None,None,2,"tension","Qual-methods",
  "Chain-of-thought coding matches human performance on 8/9 hermeneutic tasks above substantial-kappa; scalable but task-bounded."),
S("ngo2026chatqda","Ngo et al.",2026,"CHI EA","2602.18352","Qual coding (inductive)","augment",None,None,3,"tension","HCI",
  "ChatQDA on-device open-source coding: usable but users show 'conditional trust', doubting interpretive nuance and consistency."),
# ---- Thematic analysis ----
S("depaoli2023inductive","De Paoli",2023,"Soc. Sci. Comput. Rev.","10.1177/08944393231220483","Thematic analysis","augment",None,None,3,"tension","CSS",
  "Inductive thematic analysis of interviews with ChatGPT is feasible and plausible, but themes need human verification and lack depth."),
S("dai2023llminloop","Dai et al.",2023,"EMNLP-F","10.18653/v1/2023.findings-emnlp.669","Thematic analysis","augment",None,None,2,"enabling","NLP",
  "LLM-in-the-loop thematic analysis: human-AI collaboration yields themes comparable to human analysts with large efficiency gains."),
S("nguyentrung2025chatgptta","Nguyen-Trung",2025,"Quality & Quantity","10.1007/s11135-025-02165-z","Thematic analysis","augment",None,None,3,"tension","Qual-methods",
  "ChatGPT as a thematic-analysis research assistant: helpful for breadth, but reflexive interpretation remains human work."),
S("jain2025multillmthematic","Jain et al.",2025,"arXiv","2512.20352","Thematic analysis","replace",0.907,None,2,"tension","Qual-methods",
  "Multi-LLM thematic analysis with dual reliability (kappa + semantic): Gemini kappa 0.907, GPT-4o 0.853, Claude 0.842 vs humans."),
S("castellanos2025thematicsumm","Castellanos et al.",2025,"JMIR AI","10.2196/64447","Thematic analysis","augment",0.797,None,2,"enabling","Health",
  "LLM thematic summarization in qualitative health research: 79.7% agreement with human interpretation (substantial)."),
S("sharma2025details","Sharma et al.",2025,"arXiv","2510.17575","Thematic analysis","augment",None,None,2,"enabling","HCI",
  "DeTAILS: iterative LLM-supported deep thematic analysis keeps the researcher in control across coding and theme development."),
S("prescott2024efficacy","Prescott et al.",2024,"JMIR AI","10.2196/54482","Thematic analysis","augment",None,None,3,"tension","Health",
  "Human vs generative-AI thematic analysis: comparable high-level themes, large efficiency gain, but AI misses nuanced sub-themes."),
# ---- Content analysis ----
S("chew2023llmcontent","Chew et al.",2023,"arXiv","2306.14924","Content analysis","augment",None,None,2,"enabling","CSS",
  "LLM-assisted content analysis: LLMs can apply codebooks with substantial agreement; recommend human validation on a labeled subset."),
S("dunivin2025hermeneutics","Dunivin",2025,"EPJ Data Science","10.1140/epjds/s13688-025-00548-8","Content analysis","augment",None,None,2,"tension","CSS",
  "Scaling hermeneutics: guide to reflexive LLM content analysis preserving interpretive validity at scale."),
S("leas2023contentanalysis","Leas et al.",2023,"J. Med. Internet Res.","10.2196/52499","Content analysis","augment",None,None,3,"tension","Health",
  "ChatGPT supports content analysis with reasonable concordance; recommended as assistant, not replacement."),
# ---- Qualitative analysis: tooling, collaboration, critique ----
S("schroeder2025usestensions","Schroeder et al.",2025,"CHI","10.1145/3706598.3713120","Qual analysis (meta)","augment",None,None,3,"tension","HCI",
  "Interviews with qualitative researchers reveal uses, tensions, and intentions: LLMs valued for breadth but threaten interpretive rigor/positionality."),
S("davison2024ethics","Davison et al.",2024,"Information Systems J.","10.1111/isj.12504","Qual analysis (meta)","augment",None,None,4,"critical","CSS",
  "Ethics of generative-AI qualitative data analysis: warns that uncritical use erodes interpretive accountability and rigor."),
S("wachinger2024prompts","Wachinger et al.",2024,"Qual. Health Res.","10.1177/10497323241244669","Qual analysis (general)","augment",None,None,3,"tension","Health",
  "Comparing ChatGPT and a human qualitative researcher: useful for surface coding, 'imperfections' in depth and context."),
S("li2024gpt4humanqual","Li et al.",2024,"J. Med. Internet Res.","10.2196/56500","Qual analysis (general)","augment",0.40,None,3,"tension","Health",
  "GPT-4 vs human researchers on health-care qualitative data: moderate agreement (kappa ~0.40); AI aids but does not replace."),
S("zhang2023redefining","Zhang et al.",2023,"arXiv","2309.10771","Qual analysis (general)","augment",None,None,3,"tension","Qual-methods",
  "Redefining qualitative analysis in the AI era: ChatGPT accelerates coding; calls for new human-AI qualitative workflows."),
S("savelka2023textskilled","Savelka et al.",2023,"arXiv","2306.13906","Qual coding (deductive)","augment",None,None,3,"tension","NLP",
  "GPT-4 supports analysis of textual data in tasks requiring highly skilled (legal) coding; promising but below expert reliability."),
S("long2024classroom","Long et al.",2024,"npj Science of Learning","10.1038/s41539-024-00273-3","Content analysis","augment",None,None,3,"tension","Education",
  "LLMs analysing classroom dialogue: moderate agreement with human coders; useful for scale, weaker on pedagogical subtlety."),
S("qualcoding2025jla","authors",2025,"J. Learning Analytics","jla8575","Qual coding (deductive)","augment",0.46,None,3,"tension","Qual-methods",
  "Human-GPT-4 Fleiss kappa ~0.46; GPT-4 self-consistency ~0.87 (> human); hybrid coding advised."),
S("qualigpt2024","Zhang et al.",2024,"arXiv","2407.14925","Qual analysis (tool)","augment",None,None,3,"tension","HCI",
  "QualiGPT tool: adequate agreement for deductive codebooks, weaker inductively; positioned as assistant."),
S("meng2024chalet","Meng et al.",2024,"ACM TOCHI","2405.05758","Qual analysis (tool)","augment",None,None,3,"tension","HCI",
  "CHALET: human-LLM partnership for theory-driven qualitative analysis via iterative coding and disagreement analysis."),
S("mehta2025qualgenai","Mehta et al.",2025,"Scientific Reports","10.1038/s41598-025-18969-w","Qual analysis (general)","augment",None,None,3,"tension","Qual-methods",
  "Evaluation of LLMs in qualitative research: moderate agreement; human oversight needed for valid interpretation."),
S("thematic2025charity","Wen et al.",2025,"AI & Society","10.1007/s00146-025-02487-4","Thematic analysis","augment",None,None,3,"tension","Qual-methods",
  "Thematic analysis case study: GPT-4 accelerates coding with reasonable but imperfect theme overlap."),
# ---- Synthetic participants / silicon sampling / simulation ----
S("argyle2023outofone","Argyle et al.",2023,"Political Analysis","2209.06899","Synthetic participants","simulate",None,None,3,"tension","CSS",
  "'Out of one, many': LLMs encode enough sociodemographic structure to simulate human survey samples ('silicon sampling') at group level."),
S("wang2024flatten","Wang et al.",2024,"Nature Mach. Intell.","2402.01908","Synthetic participants","replace",None,None,4,"critical","HCI",
  "LLMs replacing participants misportray and FLATTEN demographic groups; cannot capture positionality (3,200 humans, 16 identities)."),
S("bail2024gensocsci","Bail",2024,"PNAS","10.1073/pnas.2314021121","Synthetic participants","simulate",None,None,3,"tension","CSS",
  "Can generative AI improve social science? Promising for annotation/simulation but validity and bias risks demand human benchmarking."),
S("bisbee2023perils","Bisbee et al.",2023,"SocArXiv","10.31235/osf.io/5ecfa","Synthetic participants","replace",None,None,4,"critical","CSS",
  "Perils of synthetic survey data: LLM 'respondents' distort variances/correlations vs real humans; unsafe as a replacement."),
S("huang2025surveyworth","Huang et al.",2025,"arXiv","2502.17773","Synthetic participants","simulate",None,None,3,"tension","CSS",
  "Uncertainty-quantification view: an LLM is worth only a bounded number of human respondents; substitution adds hidden uncertainty."),
S("cheng2023compost","Cheng et al.",2023,"EMNLP","10.18653/v1/2023.emnlp-main.669","Synthetic participants","simulate",None,None,4,"critical","NLP",
  "CoMPosT: LLM simulations of human subgroups are often caricatures, especially for marginalized identities."),
S("salminen2025syntheticcritical","Salminen et al.",2025,"Augmented Humans","10.1145/3745900.3746108","Synthetic participants","replace",None,None,4,"critical","HCI",
  "Critical analysis of synthetic users in HCI: cannot capture unpredictable behavior; should not stand in for real users."),
S("abbasiantaeb2024llmstalk","Abbasiantaeb et al.",2024,"WSDM","10.1145/3616855.3635856","Synthetic participants","simulate",None,None,3,"tension","IR",
  "'Let the LLMs talk': simulating users in conversational search; useful for evaluation but diverges from real user behavior."),
# ---- Persona generation ----
S("salminen2024deusexmachina","Salminen et al.",2024,"CHI","10.1145/3613904.3642036","Persona generation","augment",None,None,3,"tension","HCI",
  "LLM-generated personas can be plausible and useful for ideation but risk stereotyping and lack grounded data."),
S("jung2025personacraft","Jung et al.",2025,"Int. J. Hum.-Comput. Stud.","10.1016/j.ijhcs.2025.103445","Persona generation","augment",None,None,3,"tension","HCI",
  "PersonaCraft: data-grounded LLM personas from survey segments; better grounded but still need human validation."),
S("haxvig2025personabias","Haxvig et al.",2025,"Int. J. Hum.-Comput. Stud.","10.1016/j.ijhcs.2025.103651","Persona generation","replace",None,None,4,"critical","HCI",
  "LLM synthetic personas encode gender/role bias from a participatory-design lens; unsafe as stand-ins for real users."),
# ---- Usability / heuristic evaluation ----
S("guerino2025gpt4ousability","Guerino et al.",2025,"arXiv","2506.16345","Usability/heuristic eval","replace",0.212,None,4,"critical","HCI",
  "GPT-4o heuristic evaluation overlaps only 21.2% with expert-found issues and hallucinates false positives; not a substitute."),
S("zhong2025synthheuristic","Zhong et al.",2025,"arXiv","2507.02306","Usability/heuristic eval","replace",0.75,0.60,3,"tension","HCI",
  "Synthetic heuristic evaluation finds 73-77% of issues, exceeding 5 human evaluators (57-63%) but with different error profile."),
# ---- UX / design critique ----
S("duan2024uicrit","Duan et al.",2024,"UIST","2407.08850","UX/design critique","augment",None,None,3,"tension","HCI",
  "UICrit dataset shows LLM UI critique falls short of expert designers; few-shot/visual prompting gives +55% but gap remains."),
S("duan2024visualcritique","Duan et al.",2024,"arXiv","2412.16829","UX/design critique","augment",None,None,3,"tension","HCI",
  "Visual prompting with iterative refinement improves LLM design-critique quality toward, but not to, human-expert level."),
S("kocaballi2023convdesign","Kocaballi",2023,"arXiv","2302.07406","UX/design critique","augment",None,None,3,"tension","HCI",
  "ChatGPT explored as designer/user/evaluator in conversational design; useful provocation, not a reliable evaluator."),
S("duan2024uimockup","Duan et al.",2024,"CHI","2403.13139","UX/design critique","augment",None,None,3,"tension","HCI",
  "LLM feedback on UI mockups partially matches expert designer feedback; useful first pass, misses design nuance."),
# ---- Crowdsourced / social-computing annotation ----
S("gilardi2023chatgpt","Gilardi et al.",2023,"PNAS","2303.15056","Crowdsourced annotation","replace",None,None,1,"enabling","CSS",
  "ChatGPT +25pp accuracy over MTurk on relevance/stance/topic/frame; intercoder agreement exceeds crowd and trained coders."),
S("tornberg2023political","Tornberg",2023,"arXiv","2304.06588","Crowdsourced annotation","replace",None,None,1,"enabling","CSS",
  "ChatGPT-4 outperforms experts and crowd workers at annotating political social-media messages."),
S("kuzman2023endannotation","Kuzman et al.",2023,"arXiv","2303.03953","Crowdsourced annotation","replace",None,None,3,"tension","NLP",
  "'Beginning of an end of manual annotation?': ChatGPT competitive on automatic genre/text annotation but not uniformly."),
S("li2023coannotating","Li et al.",2023,"EMNLP-F","2310.15638","Crowdsourced annotation","augment",None,None,2,"enabling","NLP",
  "CoAnnotating allocates items between human and LLM by uncertainty, preserving quality at lower human cost."),
S("wu2023llmworkers","Wu et al.",2023,"arXiv","2307.10168","Social-computing annotation","replace",None,None,3,"tension","CSCW",
  "LLMs as workers in human-computation pipelines: replicate multi-step crowdsourcing workflows with mixed fidelity."),
S("zhu2023reproduce","Zhu et al.",2023,"arXiv","2304.10145","Social-computing annotation","replace",None,None,3,"tension","CSCW",
  "Can ChatGPT reproduce human labels for social-computing tasks? Good on some, poor on subjective/cultural ones."),
S("ostyakova2023crowdexperts","Ostyakova et al.",2023,"SIGDIAL","10.18653/v1/2023.sigdial-1.23","Crowdsourced annotation","replace",None,None,3,"tension","NLP",
  "ChatGPT vs crowdsourcing vs experts annotating open-domain dialogue: ChatGPT between crowd and expert quality."),
S("rajashekar2024humanalgo","Rajashekar et al.",2024,"CHI","10.1145/3613904.3642024","Social-computing annotation","augment",None,None,3,"tension","HCI",
  "Human-algorithmic interaction with an LLM for annotation: human-AI workflow improves throughput with oversight."),
# ---- HCI research-methods / meta ----
S("lequere2024researchtools","Aubin Le Quere et al.",2024,"CHI EA","10.1145/3613905.3636301","HCI research methods","augment",None,None,3,"tension","HCI",
  "LLMs as research tools across HCI data work: maps applications and evaluation gaps; calls for validation norms."),
S("tabone2023hciprimer","Tabone & de Winter",2023,"R. Soc. Open Sci.","10.1098/rsos.231053","HCI research methods","augment",None,None,3,"tension","HCI",
  "A primer on using ChatGPT for HCI research: opportunities across the method pipeline with caution on validity."),
# ---- Creativity / ideation (effects on human-centered research) ----
S("anderson2024homogenization","Anderson et al.",2024,"Creativity & Cognition","2402.01536","Creativity/ideation","augment",None,None,4,"critical","HCI",
  "LLM assistance homogenizes human creative ideation (less collective diversity); a caution for LLM-mediated design research."),
]

def main():
    ids=[s["id"] for s in STUDIES]
    assert len(ids)==len(set(ids)), [i for i in ids if ids.count(i)>1]
    cols=list(STUDIES[0].keys())
    with open(HERE/"study2_corpus.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=cols); w.writeheader()
        for s in STUDIES: w.writerow(s)
    json.dump(STUDIES,open(HERE/"study2_corpus.json","w"),indent=2)
    n=len(STUDIES)
    # Study-2 PRISMA (OpenAlex study-2 arm + web; derived)
    oa_unique,oa_topical=857,115
    excl={"no_human_compare":14,"tool_only":9,"sim_only_offtopic":11,"grey":3}
    excl_total=sum(excl.values()); assessed=n+excl_total
    db=dict(identified_openalex=oa_unique,openalex_topical=oa_topical,
            identified_web=48,web_screened_in=44,cross_dup_removed=(oa_topical+44)-assessed,
            assessed=assessed,excluded_eligibility=excl,excluded_eligibility_total=excl_total,included=n)
    assert db["cross_dup_removed"]>=0
    json.dump(db,open(HERE/"study2_prisma_counts.json","w"),indent=2)
    from collections import Counter
    print("study2 included:",n)
    print("methods:",dict(Counter(s["method"] for s in STUDIES)))
    print("roles:",dict(Counter(s["role"] for s in STUDIES)))
    print("stance:",dict(Counter(s["stance"] for s in STUDIES)))
    print("community:",dict(Counter(s["community"] for s in STUDIES)))
    print("verdicts:",dict(Counter(s["verdict_label"] for s in STUDIES)))
    print("with numeric agreement:",sum(1 for s in STUDIES if s["agreement"] is not None))

if __name__=="__main__": main()
