# When Can a Machine Judge Like a Human?
### A PRISMA Systematic Review of LLM-as-a-Judge versus Human Annotation across Domains and Tasks

This repository contains a **fully reproducible** PRISMA-2020 systematic review of the empirical evidence
comparing **LLM-as-a-judge / LLM-as-annotator** against **human annotation**, coded to answer the question that
matters in practice: *in which domains, tasks, and conditions is it safe to replace human annotation with an LLM
judge — and where is it not?*

The paper (LaTeX), the BibTeX bibliography, the extracted-studies dataset, every figure, and a runnable notebook
all regenerate from a single source of truth: `data/included_studies.csv`.

---

## Headline findings

The corpus codes each study on a 5-level reliability rubric (1 = unreliable … 5 = validated), benchmarked against
the **human–human agreement ceiling** for the same task. Exact counts are auto-generated in
`paper/tables/autostats.tex` and `data/prisma_counts.json`.

| Zone | Where | Recommendation |
|------|-------|----------------|
| 🟢 **Green — validated** | machine translation (system-level), relevance/IR, instruction-following (verifiable), QA/factuality, general-NLG pairwise preference, information extraction, text-to-image alignment, sentiment & much text classification | **Automate** with light human audit; LLM judges often *match or exceed* the human ceiling |
| 🟡 **Amber — conditional** | summarization, RAG, education/grading, social-science annotation, peer review, finance, multimodal, and *structured* clinical evaluation | **Human-in-the-loop**; validate per task on a labelled subset, fix the rubric, then automate with sampling audits |
| 🔴 **Red — keep humans** | low-resource multilingual evaluation, creative-writing quality, affective/safety ratings (mental health), adversarial/absolute-scoring settings | **Do not substitute**; human judgment required |

**Cross-cutting principles:** objectivity + tractable verification + a low human floor predict reliability;
subjectivity, high required expertise, fine granularity, and presentation/self-preference biases predict failure;
*comparative* scoring and *juries* beat single absolute judges; always *validate, then automate*, and *pin* model/
temperature/version for reproducibility.

---

## Repository structure

```
judge/
├── README.md                     ← this file
├── Makefile                      ← `make all` rebuilds data → figures → bib → paper
├── protocol/
│   └── prisma_protocol.md         ← pre-registered protocol (RQs, eligibility, rubric, synthesis)
├── data/
│   ├── build_corpus.py            ← defines the coded corpus → included_studies.csv/.json
│   ├── included_studies.csv/.json ← THE extracted-studies dataset (one row per study)
│   ├── prisma_counts.json         ← two-arm PRISMA flow counts
│   ├── summary_by_domain.csv      ← per-domain reliability + traffic-light signal
│   ├── summary_by_family.csv      ← per-family reliability
│   ├── openalex_search.py         ← Arm A: OpenAlex API systematic search
│   ├── semanticscholar_search.py  ← Semantic Scholar API cross-check
│   ├── filter_candidates.py       ← topicality filter → on-topic candidates
│   ├── generate_bib.py            ← builds paper/references.bib
│   ├── validate_latex.py          ← static LaTeX checker (citations/figs/inputs/refs/envs)
│   └── raw_search/                ← search logs + raw API results (audit trail)
├── analysis/
│   └── make_figures.py            ← all 16 figures (PNG+PDF) + data-driven LaTeX tables + autostats
├── figures/                       ← fig01–fig16 (.png and .pdf)
├── notebooks/
│   ├── build_notebook.py          ← generates the notebook
│   └── llm_judge_review.ipynb     ← runnable, end-to-end analysis with inline figures
└── paper/
    ├── main.tex                   ← the manuscript (LaTeX)
    ├── references.bib             ← BibTeX (auto-generated; 139 studies + PRISMA)
    └── tables/                    ← auto-generated tables + autostats macros
```

## Reproduce

```bash
make data       # rebuild the corpus + PRISMA counts from build_corpus.py
make figures    # regenerate all figures (PNG+PDF) and the LaTeX tables
make bib        # regenerate references.bib
make paper      # compile the PDF (needs a TeX engine: latexmk/pdflatex+bibtex)
# or everything:
make all
# re-run the systematic searches (needs network; OpenAlex key + Semantic Scholar):
make search
# execute the notebook in place:
make notebook
```

Python deps: `numpy pandas matplotlib seaborn scipy networkx statsmodels adjustText` (all used by the analysis).
Figures render to PDF/PNG with **no LaTeX dependency**.

## Methods at a glance

- **Reporting standard:** PRISMA 2020 (+ PRISMA-S search reporting).
- **Identification (two arms):** (A) **OpenAlex** REST API — 30 concept-block queries; (B) structured web
  **deep-research** (59 queries) + citation chasing; **Semantic Scholar** API as an independent cross-check /
  saturation test. Full logs in `data/raw_search/`.
- **Eligibility:** empirical studies (2023–2026) that directly compare LLM judgments to human judgments and report
  an agreement metric or structured concordance; any domain/task.
- **Extraction & coding:** domain, task formulation, subjectivity, required expertise, judge model(s), comparator,
  metric, headline agreement, human baseline, recurring biases, and a 5-level reliability verdict.
- **Synthesis:** verdict/agreement distributions; domain×task reliability landscape; subjectivity/expertise
  gradients; failure-mode catalogue; a four-quadrant deployment decision map.

## Limitations

Single-reviewer (LLM-assisted) screening/extraction rather than dual human reviewers; heterogeneous metrics
synthesised within families and via an ordinal verdict rather than pooled effect sizes; several domains rest on
small *n* (flagged in the paper and mitigated with family-level aggregates); database recall depends on indexing
and query phrasing; findings reflect models available through mid-2026. See the paper's *Limitations* section.

## Citation

If you use this review or its dataset, please cite the manuscript (`paper/main.tex`). The dataset
(`data/included_studies.csv`) is released for reuse and extension.

*This review was produced with an AI-assisted systematic-search and extraction pipeline; all code, logs, and the
coded dataset are released for transparency and independent verification.*
