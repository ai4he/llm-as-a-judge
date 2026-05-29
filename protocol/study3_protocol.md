# Study 3 — Pre-Registration Protocol
## A standardized, contamination-aware LLM-as-a-Judge evaluation on Human-Centered Computing datasets with human ground truth

**Status:** planning complete; **execution gated** on the decisions in §0. Companion to Studies 1–2 of
*"When Can a Machine Judge Like a Human? Evaluating LLM-as-a-Judge Approaches for Human-Centered Computing
Research."* Protocol version 1.0, 2026-05-29.

---

## 0. Decision gates (must be confirmed before execution)
1. **Final dataset set** (§4) — especially which *sensitive* datasets (hate speech, social bias, mental-health
   empathy) are in scope, given each dataset's **license / data-use agreement** and our ethics posture (§9).
2. **Ethics / data use** — confirm we may download and run inference on each chosen dataset; some (e.g.,
   TalkLife portion of EPITOME) require an agreement and would be **excluded** absent approval.
3. **Model panel** (§3) — confirm the 6-model judge panel and the embedding/rerank use for relevance tasks.
4. **Run scope** — pilot first (one low-contamination, non-sensitive dataset, full pipeline) → then full run.

API is validated (base `https://llm.rcd.clemson.edu/v1`, Bearer auth, `/chat/completions`, `/embeddings`,
`/rerank`; 14 models live). The key lives only in the git-ignored `secrets.sh` (Clemson docs explicitly warn
against committing it).

---

## 1. Rationale and gap identification
Studies 1–2 established *where* LLM-as-a-judge is reliable and *how HCC positions it*. Three gaps remain that a
rigorous empirical study can close:

- **G1 — Qualitative coding/content analysis at scale, multi-model, contamination-aware.** Study 2 found
  qualitative coding is "augment-only / mixed," but the evidence is mostly *single-model* (ChatGPT/GPT-4), on
  *private* interview data (not reproducible), with *small* samples and *no contamination control*. No study runs a
  **panel of open models** against **public, multi-annotator** coding data with a **human ceiling** and a formal
  replacement test.
- **G2 — Subjective social-computing annotation positioned against the human ceiling.** Hate speech, social bias,
  emotion, irony/sarcasm, politeness, empathy — HCC/CSCW staples with **low but measurable human IAA**. Prior LLM
  work reports accuracy vs a majority gold but rarely vs the *human–human ceiling* with the **alt-test**, rarely
  across *model families/sizes*, and rarely checks **contamination**.
- **G3 — Contamination confound.** Almost no HCC LLM-annotation study asks whether the model has *seen the
  dataset (and its labels)* in pretraining — yet most canonical datasets are on GitHub/HuggingFace and predate the
  models. Inflated agreement may be memorization, not judgment.

**Impact on the field.** If LLM judges are validated for some HCC tasks, the field gains scalable, cheaper
annotation and evaluation; if they are not (or only with humans-in-the-loop), uncritical adoption threatens the
**interpretive validity, reflexivity, and demographic fidelity** that define human-centered research (Study 2:
flattening of identity groups, caricature, hollowed-out rigor). Study 3 gives HCC researchers *dataset-level,
contamination-adjusted, statistically-powered* evidence and a reusable harness — converting Study 2's qualitative
verdicts into quantitative guidance.

---

## 2. Objectives, research questions, pre-registered hypotheses
- **RQ3.1 (agreement vs ceiling).** For each HCC task, how closely does each model's agreement with humans approach
  the *human–human* ceiling? **H1:** agreement ≈/> ceiling for objective/low-IAA-by-difficulty tasks; well below
  for interpretive/high-expertise tasks (replicating Studies 1–2 *within* fresh data).
- **RQ3.2 (replacement test).** On which datasets does the **alt-test** justify replacing a human annotator?
  **H2:** justified for objective annotation; not justified for interpretive coding / subjective affect.
- **RQ3.3 (contamination).** Does measured agreement decline on **post-cutoff / low-contamination** datasets vs
  **pre-cutoff / high-contamination** ones, holding task constant? **H3:** a non-trivial portion of high-contamination
  agreement is attributable to exposure; the clean condition is the honest estimate.
- **RQ3.4 (moderators).** How do **model family, size, jury aggregation, and prompt** affect agreement? **H4:**
  juries > best single model; larger > smaller within family; agreement is prompt-sensitive (Study 1).

All hypotheses, the analysis code, and the prompts are fixed *before* the full run (this document + the harness).

---

## 3. Models (judge panel, comparability, parallelization)
Selected from Clemson RCD to span **families, sizes, and capability**, prioritizing **Active/Active-LTS** lifecycle
for durability and **comparability with prior LLM-judge work** (which used GPT-4/Claude/Llama/Qwen/Gemini/DeepSeek/
Prometheus). Deprecated models (`gptoss-20b`, `qwen3-30b-a3b-instruct`) and the domain-specialized `leanstral-2603`
are **excluded**.

| Model (request id) | Family | Size | Lifecycle | Concurrency | Role / comparability |
|---|---|---|---|---|---|
| `glm-5.1-fp8` | Zhipu/GLM | 754B | Active-LTS | **64** | frontier-class (↔ GPT-4-class judges) |
| `gptoss-120b` | OpenAI-open | 120B MoE | Active | **128** | frontier-class, GPT lineage |
| `deepseek-v4-pro` | DeepSeek | large | Experimental | **48** | frontier-class, widely used in judge work |
| `gemma-4-31b` | Google/Gemma | 31B | Active | **16** | mid-size; multimodal (UI/visual tasks) |
| `qwen3.6-27b-fp8` | Qwen (dense) | 27B | Experimental | **64** | mid-size; Qwen size-ladder |
| `qwen3.5-9b` | Qwen | 9B | Active | **128** | small; Qwen size-ladder (scale effect) |

- **Scale analysis** holds family constant (Qwen 9B → 27B) and varies across families (9B → 31B → 120B → 754B).
- **Jury / Panel-of-LLMs (PoLL):** majority vote (categorical) / mean (ordinal) over the panel; reported vs the
  best single model (Studies 1–2 show juries reduce bias).
- **Relevance/IR tasks:** `qwen3-embedding-4b` + `qwen3-rerank-4b` as an embedding/rerank judge for ranking-style
  HCC tasks (if included).
- **Reasoning-model handling (validated):** Qwen3.x / gptoss / deepseek / glm emit a `reasoning` field; we set
  `max_tokens` ≥ 1024, request a **terminal structured answer** (a fenced JSON with the label), and parse
  `content` (falling back to the last JSON object in `reasoning` if `content` is null with `finish_reason=length`,
  then retry with a larger budget). Temperature 0 for the primary run.

**Parallelization.** Async client (one `aiohttp`/`asyncio` worker pool) with a **per-model semaphore set to the
documented concurrency** (above) and a global cap (e.g., 256) so we never exceed a model's limit while keeping all
models busy. Exponential backoff on 429/5xx; idempotent, **append-only JSONL checkpointing** keyed by
(dataset, item_id, model, prompt_id, run_id) so the job is resumable and never double-charges. Estimated primary
run: ~6 datasets × 6 models × ~500 items × ~1 prompt ≈ 18k calls; at these concurrencies, minutes–low hours.

---

## 4. Datasets (ground truth, contamination, sample size)
**Inclusion criteria.** (a) public or accessible under a license we honor; (b) **≥3 human annotators per item**
(for the human ceiling and alt-test); (c) ground truth derivable (majority vote + Dawid–Skene); (d) ≥ the
power-based sample size (§4.3); (e) an HCC-relevant construct in a Study-1/2 gap; (f) documented provenance (for
contamination tiering).

### 4.1 Candidate set (comprehensive across gap constructs and contamination tiers)
| Dataset | HCC construct / method | Items (≈) | Annotators | Human ceiling | Source / format | Released | Contam. tier | Access |
|---|---|---|---|---|---|---|---|---|
| **HateXplain** | hate/offensive (social comp.) | 20k | 3 | α≈0.46 | Twitter/Gab, HF/GitHub | 2021 | **HIGH** | open |
| **Social Bias Frames (SBIC)** | offensiveness/bias, disaggregated | 45k posts | 3+ | low–mod | HF, CC-BY | 2020 | **HIGH** | open |
| **GoEmotions** | 27-way emotion | 58k | 3–5 | mod | Reddit, HF/Apache | 2020 | **HIGH** | open |
| **ChaosNLI** | NLI w/ label distribution | 4.5k | **100/item** | very high ceiling | HF/GitHub | 2020 | **HIGH** | open |
| **Stanford Politeness** | politeness (interpersonal) | 4.3k | ≥5 | mod | Wikipedia/SE, ConvoKit | 2013 | **HIGH** | open |
| **GAQCorpus** | argument quality (4 dims) | 5.3k | 3 expert + crowd | mod | forums, GitHub | 2020 | **MED-HIGH** | open |
| **LeWiDi-2025** | irony/sarcasm/paraphrase/NLI (disagreement) | per task | multiple | task-varying | shared task | **2025** | **MED** | open |
| **RedditESS** | mental-health social support | — | multiple | — | Reddit | **2025** | **MED-LOW** | open (sensitive) |
| **"Just Put a Human in the Loop" codebooks** | **qualitative coding** (2 codebooks) | — | human + LLM-assisted | reported | released w/ paper | **2025** | **LOW-MED** | open |
| **"Wisdom of the LLM Crowd" 2026** | election-related harmful content (multi-label) | — | multiple | reported | released w/ paper | **2026** | **LOW** (post-cutoff) | open (sensitive) |
| **EPITOME** | empathy in MH support | 10k pairs | multiple | reported | Reddit + TalkLife | 2020 | HIGH (Reddit) | **TalkLife restricted → exclude unless approved** |

**Core proposal (≈6–8):** one **qualitative-coding** set (G1), 3–4 **subjective social-computing** sets spanning
constructs (hate/bias, emotion, irony/sarcasm, argument quality), **ChaosNLI** (gold-standard human ceiling for
calibration), and **≥2 low-contamination post-cutoff** sets (LeWiDi-2025, "Wisdom of the LLM Crowd" 2026) as the
**clean condition**. Final set fixed at the §0 gate.

### 4.2 Ground-truth construction
Per item: **majority vote** as the primary gold *and* **Dawid–Skene** model-based consensus as a robustness check;
**retain disaggregated per-annotator labels** for (i) the human–human ceiling (mean pairwise κ / Krippendorff α)
and (ii) the alt-test. Report precision/recall/F1 vs the majority gold *and* agreement vs the ceiling — never
agreement alone.

### 4.3 Sample size & power (from `data/study3_power.py`)
- **n = 500 labelable items per dataset** (use the **full** set if ≤1000; else a **stratified random sample** by
  class label and annotator-disagreement stratum so rare classes and contested items are represented).
- Justification: n=500 → 95% CI half-width **≤ ±4.4%** on an agreement proportion (worst case p=0.5; tighter at
  p≥0.8), and **≥80% power** to detect a **0.10** gap vs the human ceiling on the same items (paired/McNemar
  n≈391; independent n≈330). For ordinal tasks, n=500 gives ±0.05 CI on ρ≈0.7 (needs ~403). 
- **≥3 annotators/item** required; **≥5 preferred** for stable alt-test leave-one-out winning rates.

### 4.4 Contamination assessment (RQ3.3)
1. **Recency tiering** (table above): the Clemson models are 2026-era (cutoffs ~2024–2025); classify each dataset
   HIGH/MED/LOW and **always include a LOW (post-cutoff) clean condition**.
2. **Membership/exposure probes** on a held-out slice per dataset: (a) **guided instruction / quote-completion** —
   give the dataset name + a truncated item and ask the model to complete it (verbatim completion ⇒ exposure);
   (b) **TS-Guessing** style — mask the gold label and see if the model "recalls" the canonical label beyond
   task-inference; (c) compare agreement on **HIGH vs LOW** tiers for the *same construct*. Report a per-dataset
   **contamination flag** and treat the LOW-tier estimate as the headline.
3. Interpretation: a model that reproduces held-out items/labels verbatim has its agreement reported **with a
   contamination caveat**; cross-tier deltas estimate the inflation.

---

## 5. Standardized evaluation methodology (comparable to Studies 1–2 and prior work)
- **LLM as virtual annotator.** Each model receives the **same instructions/codebook** the humans used, item by
  item, producing the task's native label (category, scale, or span). Pointwise for labeling; **pairwise** with
  **position-swap** for any preference/quality ranking.
- **Metrics (native to each task, matching prior work):**
  - categorical → **Cohen's κ** (LLM vs each human), **Krippendorff's α / Fleiss' κ** (LLM added as an annotator),
    **% agreement**, **accuracy**, **macro-F1**, **precision/recall** vs majority gold;
  - ordinal/continuous → **Spearman ρ**, **Kendall τ**, **ICC**, **QWK**.
- **Human ceiling & replacement:** report **LLM–human vs human–human** agreement; apply the **alt-test**
  (Calderon et al., leave-one-annotator-out; winning rate; ε≈0.15–0.20 decision rule) as the formal
  replacement criterion, for direct comparability with Study-1/2 evidence.
- **Robustness/bias controls:** position-swap (pairwise); length normalization (scoring); **≥3 prompt variants**
  (Study 1 design-sensitivity); **temperature 0 primary + 3 replicates at T=0.7** for self-consistency; pin model
  version, seed, and request params; log raw responses.
- **Verdict mapping:** every (dataset, model, jury) result is mapped onto the **same 5-level reliability verdict**
  so Study-3 points drop into the Study-1/2 landscape figures.

---

## 6. Statistical analysis plan
- **Estimation:** bootstrap 95% CIs (≥5,000 resamples) for every agreement metric (per dataset × model × prompt).
- **Inference:** paired bootstrap / McNemar to test LLM–human vs human–human; **equivalence testing (TOST)** with
  margin δ to support "as good as a second human" claims (not just non-rejection of difference).
- **Modeling:** mixed-effects regression `agreement ~ family + size + subjectivity + contamination_tier + prompt
  + (1|dataset) + (1|item)`; report fixed-effect estimates with CIs.
- **Jury vs single, scale, family:** pre-planned contrasts.
- **Multiplicity:** Benjamini–Hochberg FDR across the (dataset × model × metric) grid; report q-values.
- **Reproducibility:** all numbers regenerate from released per-judgment JSONL via a single script.

---

## 7. Engineering & reproducibility
- Async harness `study3/run_judges.py` with per-model concurrency semaphores (§3), backoff, and append-only JSONL
  checkpoints; `study3/score.py` computes all metrics/CIs; `study3/contamination.py` runs the probes.
- Pinned: model ids, prompts (versioned), temperature/seed, harness commit hash. Note **model-retirement risk**
  (some models are Experimental) — we record served `model` strings and dates; results are conditioned on the
  2026-05 snapshot.
- Raw dataset text is **not** redistributed in the repo (license/ethics); only **aggregate metrics, prompts,
  per-judgment labels (not source text), and code** are released.

---

## 8. Threats to validity & mitigations
- **Contamination** → recency tiering + probes + clean post-cutoff condition (§4.4).
- **Prompt sensitivity** → ≥3 prompts, report variance.
- **Ground-truth quality** → multi-annotator datasets, majority + Dawid–Skene, human ceiling reported.
- **Construct validity of "replacement"** → alt-test + TOST, not bare accuracy.
- **Reasoning-model artifacts** → token budget + structured parsing (validated).
- **Generalizability** → multiple families/sizes/constructs/datasets.
- **Multiplicity** → FDR control.
- **Durability** → prefer Active/Active-LTS; record snapshot; harness re-runnable.

## 9. Ethics, data use, sensitive content
Hate-speech, social-bias, and mental-health datasets contain harmful and sensitive content. We will: honor each
dataset's **license/DUA** (exclude TalkLife/EPITOME-restricted unless approved); add content warnings; **not
redistribute raw sensitive text**; report only aggregate results; avoid any individual re-identification; and treat
mental-health data with extra care (no generation of advice, evaluation only). Confirm institutional data-use/IRB
posture at the §0 gate before downloading sensitive datasets.

## 10. Outputs & integration
A Study-3 results section + figures (agreement-vs-ceiling per dataset/model, contamination deltas, alt-test
pass/fail grid, family/size/jury effects), a Study-3 corpus row set mapped to the 5-level verdict, and an updated
**unified decision framework** (Studies 1–3) for HCC. All committed and tagged.
