# Pre-registered Review Protocol

## LLM-as-a-Judge versus Human Annotation: A PRISMA Systematic Review of Domain- and Task-Level Reliability

**Protocol version:** 1.0 — 2026-05-28
**Reporting standard:** PRISMA 2020 (Page et al., *BMJ* 2021) + PRISMA-S (search reporting) + elements of PRISMA-ScR.
**Review type:** Systematic review with quantitative meta-synthesis of reported human–machine agreement and qualitative thematic synthesis.

---

### 1. Background and rationale
Large Language Models (LLMs) are increasingly used as automatic *judges* / *annotators* ("LLM-as-a-judge") to score, rank, compare, or label text, code, and multimodal outputs in place of human annotators. Adoption has outpaced validation. The central open question is **not** "do LLM judges work?" but **"in which domains, tasks, and conditions do LLM judgments agree with human judgments well enough to substitute for them — and where do they not?"** This review systematically maps the empirical evidence comparing LLM-as-a-judge to human annotation across the full breadth of domains and tasks, and codes each study for the conditions under which substitution is validated, conditional, or unsafe.

### 2. Objectives and research questions
- **RQ1.** Across which **domains** (e.g., general NLG, summarization, MT, dialogue, QA/factuality, code, reasoning/math, IR/relevance, social-science annotation, sentiment, content moderation, medicine, mental health, legal, finance, education, creative writing, peer review, multimodal) is LLM-as-a-judge **validated** (agreement ≈ or > human–human) versus **under discussion / unreliable**?
- **RQ2.** Across which **task formulations** (pointwise scoring, pairwise comparison, classification/annotation, ranking, error/fact verification) and **construct types** (objective vs subjective; low vs high domain-expertise) does reliability hold?
- **RQ3.** What **agreement levels** (Pearson r, Spearman ρ, Kendall τ, Cohen's/Fleiss' κ, Krippendorff's α, ICC, QWK, Gwet's AC2, % agreement, accuracy/F1, win-rate correlation) are reported, and how do they compare to the **human–human ceiling** for the same task?
- **RQ4.** What **systematic biases and failure modes** (position, verbosity/length, self-preference/self-enhancement, style-over-substance, bandwagon/authority, familiarity, anchoring, non-determinism, adversarial manipulability, expertise gaps) recur, and how do they modulate reliability?
- **RQ5.** What **moderators** (judge model family/size, prompting strategy, single-judge vs jury/panel, human- vs model-generated text being judged, language resource level) change the conclusion?
- **RQ6.** What **decision guidance** can be synthesized (a domain × task "reliability landscape" with conditions for safe substitution)?

### 3. Eligibility criteria
**Adapted PICO/PECO frame.** *Population:* texts/code/multimodal artifacts requiring quality or label judgments. *Index method:* LLM-as-a-judge / LLM-as-annotator. *Comparator:* human annotation/judgment (crowd, trained, or expert). *Outcome:* a quantitative agreement/correlation/accuracy metric **or** a structured qualitative concordance comparison against humans.

**Inclusion.**
1. Empirical study, benchmark, or meta-evaluation that **directly compares** LLM judgments/annotations to human judgments/annotations.
2. Reports at least one agreement/correlation/accuracy outcome **or** a systematic qualitative comparison.
3. Any domain or task; any judging format (pointwise/pairwise/listwise/classification).
4. Published 2023-01-01 → 2026-05-28 (LLM-as-judge era; foundational neural-metric precursors noted as context only).
5. English-language report.

**Exclusion.**
1. Uses an LLM judge **only as a tool** with no human comparison/validation.
2. Pure model/method proposal with no human-agreement evaluation.
3. Non-peer-reviewed marketing/blog/grey literature (retained as *contextual* sources only; excluded from quantitative synthesis).
4. Duplicates / superseded preprint versions (most recent retained).
5. Pre-2023 classical automatic-metric studies (BLEU/ROUGE/BERTScore/BLEURT) except as background.

### 4. Information sources (two-arm identification)
- **Arm A — bibliographic database (primary).** The **OpenAlex** scholarly index was queried programmatically through its REST API (30 concept-block queries, 50 top-relevance records each, `from_publication_date 2022-09-01`), yielding 1,500 retrieved records and 957 unique works after de-duplication. OpenAlex aggregates Crossref, PubMed, arXiv, and ~250M works, giving database-grade recall with exact counts.
- **Arm B — web deep-search + citation chasing (supplementary).** A structured "deep-research" web protocol (59 queries) surfacing **arXiv**, **ACL Anthology**, **OpenReview**, **Semantic Scholar**, **PubMed/PMC**, **Springer/Nature**, **ScienceDirect/Elsevier**, **ACM DL**, **IEEE Xplore**, **PNAS**, **medRxiv**; plus backward/forward citation chasing via three surveys (JUDGE-BENCH; *From Generation to Judgment*; the LLMs-as-Judges survey) and two healthcare scoping reviews.

Arm A doubles as an independent **completeness / saturation check** on Arm B: the high-citation on-topic core was already present in Arm B, and OpenAlex contributed a bounded set of net-new on-topic studies. Searches executed 2026-05-28/29.

> **Limitation (declared a priori):** screening and extraction were single-reviewer (LLM-assisted) rather than dual independent human reviewers; OpenAlex `search` recall depends on indexing and query phrasing. Both are noted in § Limitations of the paper.

### 5. Search strategy (concept blocks)
Core concept × method × domain blocks, combined per query (59 query strings executed; full list logged in `data/raw_search/search_log.csv`):
- **Method block:** "LLM-as-a-judge", "LLM evaluator", "LLM annotator", "GPT-4 evaluation", "automatic evaluation", "model-as-judge".
- **Comparison block:** "human agreement", "human annotation", "inter-annotator agreement", "correlation with human", "vs human raters/experts/crowdworkers", "Cohen's kappa / Spearman / QWK / Krippendorff".
- **Domain block:** summarization, machine translation, dialogue, QA, RAG, code, math/reasoning, relevance/IR, social-science/political annotation, sentiment/emotion, toxicity/hate-speech/content-moderation, medical/clinical, mental-health, legal, finance, education/essay/grading, creative writing, peer review, multimodal/vision, text-to-image, multilingual, reward modeling, fact-checking, instruction-following, qualitative coding, NER/IE, software-engineering.
- **Reliability/bias block:** position/verbosity/self-preference/style/cognitive bias, robustness/adversarial, reproducibility/temperature, expertise effect, panel/jury, alt-test.

### 6. Selection process
Two-stage screening (title/abstract → full-text/abstract eligibility), single-reviewer (LLM-assisted) with explicit, logged criteria. Records were classified Included / Excluded-with-reason. Counts reported in the PRISMA 2020 flow diagram (`figures/fig01_prisma_flow.*`).

### 7. Data items (extraction schema)
For each included study: citation key; authors/year/venue; arXiv-id/DOI; **domain**; **task formulation**; **construct subjectivity** (objective/mixed/subjective); **domain-expertise required** (low/medium/high); judge model(s); comparator type (crowd/trained/expert; human–human baseline available?); **metric type**; **headline agreement value**; **human–human baseline value** (if reported); **meets human parity** (yes/approx/no); **reliability verdict** (1–5, see §8); reported **biases/failure modes**; sample size; one-line finding; notes. Stored in `data/included_studies.csv` / `.json`.

### 8. Reliability-verdict coding rubric (reproducible)
Each study is assigned an ordinal **verdict** synthesizing its own conclusion + reported numbers vs the task's human–human ceiling:
- **1 — Validated / reliable substitute.** LLM–human agreement ≈ or exceeds human–human ceiling on the task; authors endorse substitution (often with light QA). *e.g., MT-Bench pairwise (no ties), system-level MT, structured/objective annotation, relevance labelling.*
- **2 — Promising, condition-dependent.** Strong agreement achievable but contingent on prompt/rubric/model/reference; endorsed for screening or with human-in-the-loop. *e.g., summarization w/ G-Eval, RAG faithfulness, short-answer grading with rubric.*
- **3 — Mixed / highly task-dependent.** Agreement varies widely within the domain; no blanket endorsement. *e.g., dialogue, essay scoring across dimensions, social-science annotation across constructs.*
- **4 — Caution / limited.** Agreement materially below human ceiling or only on easy slices; substitution discouraged without strong oversight. *e.g., complex clinical reasoning, segment-level MT, expert-knowledge tasks, low-resource languages.*
- **5 — Unreliable / not recommended.** Agreement near chance/negative, or dominated by bias; human judgment required. *e.g., expert creativity judgments, highly subjective safety/affective ratings, adversarially manipulable scoring.*

Mapping numeric thresholds (guidance, not mechanical): κ/α ≥ 0.8 or ≥ human–human → lean 1; 0.6-0.8 → 2; 0.4-0.6 mixed → 3; 0.2-0.4 → 4; <0.2/negative → 5; for r/ρ, ≥0.8→1, 0.6-0.8→2, 0.4-0.6→3, 0.2-0.4→4, <0.2→5; for % agreement, ≥ human–human → 1, within 5pp → 2. Verdicts always reconciled with the authors' stated conclusion and the human-baseline comparison; ties broken conservatively (toward higher caution).

### 9. Risk-of-bias / quality appraisal
Study-level appraisal flags: (a) human-baseline reported? (b) expert vs non-expert comparator? (c) multiple human annotators / IAA reported? (d) bias controls applied (position swap, length control)? (e) sample size adequate? (f) single vs multiple judge models? Aggregated into a qualitative confidence rating per domain.

### 10. Synthesis methods
- **Quantitative:** descriptive distributions of agreement by domain/task/metric; LLM–human vs human–human parity scatter; verdict distributions (overall, by domain, by subjectivity, by expertise, by year); moderator breakdowns (judge family, single vs jury); a domain × task **reliability-landscape heatmap**. Because metrics are heterogeneous, we synthesize within metric families and report a harmonized **reliability score** (1=verdict-5 … 5=verdict-1, or normalized agreement) for cross-domain visualization rather than a pooled effect size; no random-effects meta-analytic pooling across incommensurable metrics (declared limitation).
- **Qualitative:** thematic synthesis of failure modes, biases, and moderators; a "traffic-light" decision table.

### 11. Deliverables
LaTeX manuscript (`paper/main.tex`) + BibTeX (`paper/references.bib`); reproducible analysis notebook (`notebooks/`) + scripts; extracted dataset (`data/`); figure set (`figures/`). All counts/figures regenerated from `data/included_studies.csv`.
