#!/usr/bin/env python3
"""generate_bib.py -- build paper/references.bib from included_studies.csv, resolving
titles/authors/venues from the OpenAlex + Semantic Scholar metadata pulls (by arXiv id /
DOI), with a manual TITLES/AUTHORS override for any records not covered by those pulls."""
import re, csv
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
RAW = HERE/"raw_search"
inc = pd.read_csv(HERE/"included_studies.csv").fillna("")
# union Study 2 corpus (HCC/HCI/qualitative) so references.bib covers both studies
try:
    _s2 = pd.read_csv(HERE/"study2_corpus.csv").fillna("").rename(columns={"ref":"arxiv"})
    for _c in ["arxiv","authors","year","venue","finding"]:
        if _c not in _s2.columns: _s2[_c] = ""
    inc = pd.concat([inc, _s2[["id","authors","year","venue","arxiv","finding"]]],
                    ignore_index=True).drop_duplicates("id", keep="first").fillna("")
except FileNotFoundError:
    pass

# ---- build metadata index from the database pulls -------------------------------------
idx_ax, idx_doi = {}, {}
for fn in ("openalex_results.csv","s2_results.csv"):
    p = RAW/fn
    if not p.exists(): continue
    t = pd.read_csv(p).fillna("")
    for _,r in t.iterrows():
        rec = dict(title=str(r.get("title","")).strip(), authors=str(r.get("authors","")).strip(),
                   year=r.get("year",""), venue=str(r.get("venue","")).strip())
        ax = str(r.get("arxiv","")).split("v")[0].strip()
        doi = str(r.get("doi","")).lower().replace("https://doi.org/","").strip()
        if ax and re.match(r"\d{4}\.\d{4,5}", ax): idx_ax.setdefault(ax, rec)
        if doi: idx_doi.setdefault(doi, rec)

# ---- manual title overrides for records not in the pulls (filled after dry-run) -------
TITLES = {
 "zheng2023mtbench":"Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena",
 "chiang2023alt":"Can Large Language Models Be an Alternative to Human Evaluations?",
 "liu2023geval":"G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment",
 "dubois2024lc":"Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators",
 "verga2024poll":"Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models",
 "kim2023prometheus":"Prometheus: Inducing Fine-grained Evaluation Capability in Language Models",
 "kim2024prometheus2":"Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models",
 "zeng2024llmbar":"Evaluating Large Language Models at Evaluating Instruction Following",
 "gu2024survey":"A Survey on LLM-as-a-Judge",
 "li2024gen2judge":"From Generation to Judgment: Opportunities and Challenges of LLM-as-a-judge",
 "li2024compsurvey":"LLMs-as-Judges: A Comprehensive Survey on LLM-based Evaluation Methods",
 "tang2024tofueval":"TofuEval: Evaluating Hallucinations of LLMs on Topic-Focused Dialogue Summarization",
 "lee2024unisumeval":"UniSumEval: Towards Unified, Fine-Grained, Multi-Dimensional Summarization Evaluation for LLMs",
 "kocmi2023gemba":"Large Language Models Are State-of-the-Art Evaluators of Translation Quality",
 "kocmi2023gembamqm":"GEMBA-MQM: Detecting Translation Quality Error Spans with GPT-4",
 "mendonca2024dialogue":"Leveraging LLMs for Dialogue Quality Measurement",
 "wei2024safe":"Long-form Factuality in Large Language Models",
 "min2023factscore":"FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation",
 "madaan2024refverdict":"Reference-Guided Verdict: LLMs-as-Judges in Automatic Evaluation of Free-Form Text",
 "es2023ragas":"RAGAS: Automated Evaluation of Retrieval Augmented Generation",
 "saadfalcon2023ares":"ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems",
 "jin2024ragreward":"RAG-RewardBench: Benchmarking Reward Models in Retrieval Augmented Generation for Preference Alignment",
 "tong2024codejudge":"CodeJudge: Evaluating Code Generation with Large Language Models",
 "zhuo2025codejudgeeff":"On the Effectiveness of LLM-as-a-judge for Code Generation and Summarization",
 "watson2024aicodereview":"AI-powered Code Review with LLMs: Early Results",
 "overcorrect2026":"Are LLMs Reliable Code Reviewers? Systematic Overcorrection in Requirement Conformance Judgement",
 "biasloop2026":"Bias in the Loop: Auditing LLM-as-a-Judge for Software Engineering",
 "codespec2025":"Uncovering Systematic Failures of LLMs in Verifying Code Against Natural Language Specifications",
 "tan2024judgebench":"JudgeBench: A Benchmark for Evaluating LLM-based Judges",
 "raju2024calc2adj":"From Calculation to Adjudication: Examining LLM Judges on Mathematical Reasoning Tasks",
 "mathrobust2026":"Rethinking Math Reasoning Evaluation: A Robust LLM-as-a-Judge Framework Beyond Symbolic Rigidity",
 "thomas2024relevance":"Large Language Models can Accurately Predict Searcher Preferences",
 "upadhyay2024largescale":"A Large-Scale Study of Relevance Assessments with Large Language Models: An Initial Look",
 "faggioli2023perspectives":"Perspectives on Large Language Models for Relevance Judgment",
 "gilardi2023chatgpt":"ChatGPT Outperforms Crowd Workers for Text-Annotation Tasks",
 "tornberg2025political":"Large Language Models Outperform Expert Coders and Supervised Classifiers at Annotating Political Social Media Messages",
 "pangakis2023validation":"Automated Annotation with Generative AI Requires Validation",
 "ziems2024css":"Can Large Language Models Transform Computational Social Science?",
 "reiss2023testing":"Testing the Reliability of ChatGPT for Text Annotation and Classification: A Cautionary Remark",
 "pangakis2024keephumans":"Keeping Humans in the Loop: Human-Centered Automated Annotation with Generative AI",
 "zhang2023sentiment":"Sentiment Analysis in the Era of Large Language Models: A Reality Check",
 "belal2023sentiment":"Leveraging ChatGPT As Text Annotation Tool For Sentiment Analysis",
 "hatespeechanno2025":"Can LLMs Evaluate What They Cannot Annotate? Revisiting LLM Reliability in Hate Speech Detection",
 "icwsm2025hatebias":"Human and LLM Biases in Hate Speech Annotations: A Socio-Demographic Analysis of Annotators and Targets",
 "hatespeechprobe2023":"Probing LLMs for hate speech detection: strengths and vulnerabilities",
 "szymanski2025limits":"Limitations of the LLM-as-a-Judge Approach for Evaluating LLM Outputs in Expert Knowledge Tasks",
 "medjudge2026scoping":"A Scoping Review of LLM-as-a-Judge in Healthcare and the MedJUDGE Framework",
 "clinstruct2025":"Automated Evaluation of Large Language Model Outputs Against Clinical Documentation Standards",
 "llmevalmed2025":"LLMEval-Med: A Real-world Clinical Benchmark for Medical LLMs with Physician Validation",
 "sun2023radimpressions":"Evaluating GPT-4 on Impressions Generation in Radiology Reports",
 "brain2024radgpt":"Comparative analysis of GPT-4-based ChatGPT's diagnostic performance with radiologists using real-world radiology reports of brain tumors",
 "healthbench2025":"HealthBench: Evaluating Large Language Models Towards Improved Human Health",
 "clindialog2026":"When Metrics Disagree: Automatic Similarity vs. LLM-as-a-Judge for Clinical Dialogue Evaluation",
 "frenchmedqa2026":"Who Judges the Judge? Evaluating LLM-as-a-Judge for French Medical Open-Ended QA",
 "mhtrust2025":"When Can We Trust LLMs in Mental Health? Large-Scale Benchmarks for Reliable LLM Evaluation",
 "counselbench2025":"CounselBench: A Large-Scale Expert Evaluation and Adversarial Benchmark of Large Language Models in Mental Health Counseling",
 "mhsupport2026":"Assessing the Quality of Mental Health Support in LLM Responses through Multi-Attribute Human Evaluation",
 "lemaj2025":"LeMAJ (Legal LLM-as-a-Judge): Bridging Legal Reasoning and LLM Evaluation",
 "legaldocrec2025":"LLM-as-a-Judge: Rapid Evaluation of Legal Document Recommendation for Retrieval-Augmented Generation",
 "trident2025":"TRIDENT: Benchmarking LLM Safety in Finance, Medicine, and Law",
 "bizfinbench2025":"BizFinBench: A Business-Driven Real-World Financial Benchmark for Evaluating LLMs",
 "mizumoto2023aes":"Exploring the Potential of Using an AI Language Model for Automated Essay Scoring",
 "aescomparative2024":"Is GPT-4 Alone Sufficient for Automated Essay Scoring? A Comparative Judgment Approach Based on Rater Cognition",
 "k12shortanswer2024":"Can Large Language Models Make the Grade? An Empirical Study Evaluating LLMs' Ability to Mark Short Answer Questions in K-12 Education",
 "asagfair2025":"Is GPT-4 fair? An empirical analysis in automatic short answer grading",
 "writingmultidim2024":"Harnessing LLMs for multi-dimensional writing assessment: Reliability and alignment with human judgments",
 "sefl2025feedback":"SEFL: A Framework for Generating Synthetic Educational Assignment Feedback with LLM Agents",
 "k12sci2026":"Judging the Judges: Human Validation of Multi-LLM Evaluation for High-Quality K-12 Science Instructional Materials",
 "studentsjudge2025":"Can students judge like experts? A large-scale study on the pedagogical quality of AI and human personalized formative feedback",
 "chakrabarty2024artartifice":"Art or Artifice? Large Language Models and the False Promise of Creativity",
 "litbench2025":"LitBench: A Benchmark and Dataset for Reliable Evaluation of Creative Writing",
 "ismayilzada2024creative":"Evaluating Creative Short Story Generation in Humans and Large Language Models",
 "creativeevalmdpi2025":"Evaluating Creativity: Can LLMs Be Good Evaluators in Creative Writing Tasks?",
 "liang2024feedback":"Can Large Language Models Provide Useful Feedback on Research Papers? A Large-Scale Empirical Analysis",
 "chatgptreviewers2024":"Assessing ChatGPT's ability to emulate human reviewers in scientific research: A descriptive and qualitative approach",
 "murad2024quality":"Concordance between humans and GPT-4 in appraising the methodological quality of case reports and case series using the Murad tool",
 "prismascreen2024":"Implementation and evaluation of an additional GPT-4-based reviewer in PRISMA-based medical systematic literature reviews",
 "chen2024mllmjudge":"MLLM-as-a-Judge: Assessing Multimodal LLM-as-a-Judge with Vision-Language Benchmark",
 "lee2024promvision":"Prometheus-Vision: Vision-Language Model as a Judge for Fine-Grained Evaluation",
 "lu2024wildvision":"WildVision: Evaluating Vision-Language Models in the Wild with Human Preferences",
 "yasunaga2025mmrewardbench":"Multimodal RewardBench: Holistic Evaluation of Reward Models for Vision Language Models",
 "lin2024vqascore":"Evaluating Text-to-Visual Generation with Image-to-Text Generation",
 "wu2023hpsv2":"Human Preference Score v2: A Solid Benchmark for Evaluating Human Preferences of Text-to-Image Synthesis",
 "multiljudge2025":"How Reliable is Multilingual LLM-as-a-Judge?",
 "mmeval2024":"MM-Eval: A Multilingual Meta-Evaluation Benchmark for LLM-as-a-Judge and Reward Models",
 "reliablemulti2026":"Towards Reliable Multilingual LLMs-as-a-Judge: An Empirical Study",
 "lambert2024rewardbench":"RewardBench: Evaluating Reward Models for Language Modeling",
 "ifreward2026":"IF-RewardBench: Benchmarking Judge Models for Instruction-Following Evaluation",
 "politicaltruths2024":"Fact or Fiction? Can LLMs be Reliable Annotators for Political Truths?",
 "factsopinions2025":"Facts are Harder Than Opinions -- A Multilingual, Comparative Analysis of LLM-Based Fact-Checking Reliability",
 "claimmatch2023":"Automated Claim Matching with Large Language Models: Empowering Fact-Checkers in the Fight Against Misinformation",
 "zhou2023ifeval":"Instruction-Following Evaluation for Large Language Models",
 "ifcritic2025":"IF-CRITIC: Towards a Fine-Grained LLM Critic for Instruction-Following Evaluation",
 "qualcoding2025jla":"Qualitative Coding with GPT-4: Where it Works Better",
 "qualigpt2024":"When Qualitative Research Meets Large Language Model: Exploring the Potential of QualiGPT as a Tool for Qualitative Coding",
 "thematic2025charity":"Leveraging large language models for thematic analysis: a case study in the charity sector",
 "ner2024augment":"Augmenting NER Datasets with LLMs: Towards Automated and Refined Annotation",
 "nli2024labeldist":"A Rose by Any Other Name: LLM-Generated Explanations Are Good Proxies for Human Explanations to Collect Label Distributions on NLI",
 "humanllmcollab2024":"Human-LLM Collaborative Annotation Through Effective Verification of LLM Labels",
 "mirzakhmedova2024argument":"Are Large Language Models Reliable Argument Quality Annotators?",
 "panickssery2024selfpref":"LLM Evaluators Recognize and Favor Their Own Generations",
 "stureborg2024inconsistent":"Large Language Models are Inconsistent and Biased Evaluators",
 "koo2024cobbler":"Benchmarking Cognitive Biases in Large Language Models as Evaluators",
 "chen2024humansorllms":"Humans or LLMs as the Judge? A Study on Judgement Biases",
 "feuer2024style":"Style Outweighs Substance: Failure Modes of LLM Judges in Alignment Benchmarking",
 "raina2024robust":"Is LLM-as-a-Judge Robust? Investigating Universal Adversarial Attacks on Zero-shot LLM Assessment",
 "shankar2024validators":"Who Validates the Validators? Aligning LLM-Assisted Evaluation of LLM Outputs with Human Preferences",
 "temperature2026":"The Necessity of Setting Temperature in LLM-as-a-Judge",
 "justice2024prejudice":"Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge",
 "calderon2025alttest":"The Alternative Annotator Test for LLM-as-a-Judge: How to Statistically Justify Replacing Human Annotators with LLMs",
 "bavaresco2024judgebench":"LLMs instead of Human Judges? A Large Scale Empirical Study across 20 NLP Evaluation Tasks",
 "judgesverdict2025":"Judge's Verdict: A Comprehensive Analysis of LLM Judge Capability Through Human Agreement",
 "imrit2026iaa":"Counting on Consensus: Selecting the Right Inter-annotator Agreement Metric for NLP Annotation and Evaluation",
 # OpenAlex top-up
 "wang2023chatgptnlg":"Is ChatGPT a Good NLG Evaluator? A Preliminary Study",
 "shen2023notyet":"Large Language Models are Not Yet Human-Level Evaluators for Abstractive Summarization",
 "xu2023instructscore":"INSTRUCTSCORE: Towards Explainable Text Generation Evaluation with Automatic Feedback",
 "zhang2024newssumm":"Benchmarking Large Language Models for News Summarization",
 "tang2023medevidence":"Evaluating large language models on medical evidence summarization",
 "hackl2023reliablerater":"Is GPT-4 a reliable rater? Evaluating consistency in GPT-4's text ratings",
 "grevisse2024asagmed":"LLM-based automatic short answer grading in undergraduate medical education",
 "yancey2023cefr":"Rating Short L2 Essays on the CEFR Scale with GPT-4",
 "floden2024gradingexams":"Grading exams using large language models: A comparison between human and AI grading",
 "kortemeyer2023physics":"Toward AI grading of student problem solutions in introductory physics: A feasibility study",
 "dubois2023alpacafarm":"AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback",
 "quah2024dentalaes":"Reliability of ChatGPT in automated essay scoring for dental undergraduate examinations",
 "gao2025nlgsurvey":"LLM-based NLG Evaluation: Current Status and Challenges",
 "khondaker2023gptaraeval":"GPTAraEval: A Comprehensive Evaluation of ChatGPT on Arabic NLP",
 "khademi2023bardassessment":"Can ChatGPT and Bard Generate Aligned Assessment Items? A Reliability Analysis against Human Performance",
 "zhuang2024beyondyesno":"Beyond Yes and No: Improving Zero-Shot LLM Rankers via Scoring Fine-Grained Relevance Labels",
 "alaofi2024fooledrelevant":"LLMs can be Fooled into Labelling a Document as Relevant",
 "wang2025setesting":"Can LLMs Replace Human Evaluators? An Empirical Study of LLM-as-a-Judge in Software Engineering",
 "ollion2023mindhype":"ChatGPT for Text Annotation? Mind the Hype!",
 "cegin2023paraphrases":"ChatGPT to Replace Crowdsourcing of Paraphrases for Intent Classification: Higher Diversity and Comparable Model Robustness",
 "song2024finesure":"FineSurE: Fine-grained Summarization Evaluation using LLMs",
 "li2023hotchatgpt":"'HOT' ChatGPT: The Promise of ChatGPT in Detecting and Discriminating Hateful, Offensive, and Toxic Comments on Social Media",
 "wang2024clinicalevent":"Validation of GPT-4 for clinical event classification: A comparative analysis with ICD codes and human reviewers",
 "croxford2025clinsummjudge":"Evaluating clinical AI summaries with large language models as judges",
 "pan2024hcdjudge":"Human-Centered Design Recommendations for LLM-as-a-judge",
 "hua2025mhscoping":"A scoping review of large language models for generative tasks in mental health care",
 # consolidation top-up
 "gao2023humanlike":"Human-like Summarization Evaluation with ChatGPT",
 "luo2023factinconsistency":"ChatGPT as a Factual Inconsistency Evaluator for Text Summarization",
 "li2023coannotating":"CoAnnotating: Uncertainty-Guided Work Allocation between Human and Large Language Models for Data Annotation",
 "nasution2024chatgptlabel":"ChatGPT Label: Comparing the Quality of Human-Generated and LLM-Generated Annotations in Low-Resource Language NLP Tasks",
 "aldeen2023chatgptvshuman":"ChatGPT vs. Human Annotators: A Comprehensive Analysis of ChatGPT for Text Annotation",
 "ronningstad2024gptannotator":"A GPT among Annotators: LLM-based Entity-Level Sentiment Annotation",
 "qiu2025labelensemble":"Labeling Free-text Data using Language Model Ensembles",
 "niu2024text2emotion":"From Text to Emotion: Unveiling the Emotion Annotation Capabilities of LLMs",
 "koptyra2023clarinemo":"CLARIN-Emo: Training Emotion Recognition Models Using Human Annotation and ChatGPT",
 "li2024stance":"Advancing Annotation of Stance in Social Media Posts: A Comparative Analysis of Large Language Models and Crowd Sourcing",
 "nguyen2024crisismisinfo":"Human vs ChatGPT: Effect of Data Annotation in Interpretable Crisis-Related Microblog Classification",
 "leas2023contentanalysis":"Using Large Language Models to Support Content Analysis: A Case Study of ChatGPT for Adverse Event Detection",
 "kaikaus2023financecorpora":"Humans vs. ChatGPT: Evaluating Annotation Methods for Financial Corpora",
 "clarke2024cantreplace":"LLM-based Relevance Assessment Still Can't Replace Human Relevance Assessment",
 "arabzadeh2025benchrelevance":"Benchmarking LLM-based Relevance Judgment Methods",
 "lu2023erroranalysis":"Error Analysis Prompting Enables Human-Like Translation Evaluation in Large Language Models",
 "huang2024nlexpl":"ChatGPT Rates Natural Language Explanation Quality Like Humans: But on Which Scales?",
 "guo2023howclose":"How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection",
 "doostmohammadi2024autoeval":"How Reliable Are Automatic Evaluation Methods for Instruction-Tuned LLMs?",
 "zhang2023widerdeeper":"Wider and Deeper LLM Networks are Fairer LLM Evaluators",
 "zhu2024nationalitybias":"Quite Good, but Not Enough: Nationality Bias in Large Language Models -- a Case Study of ChatGPT",
 "saito2023verbositybias":"Verbosity Bias in Preference Labeling by Large Language Models",
 "li2024splitmerge":"Split and Merge: Aligning Position Biases in LLM-based Evaluators",
 "liu2024pairwisepref":"Aligning with Human Judgement: The Role of Pairwise Preference in Large Language Model Evaluators",
 "liu2023trustworthy":"Trustworthy LLMs: a Survey and Guideline for Evaluating Large Language Models' Alignment",
 "voutsa2025biaseddesign":"Biased by Design? Evaluating Bias and Behavioral Diversity in LLM Annotation of Social Media Content",
 "chung2025verifact":"VeriFact: Verifying Facts in LLM-Generated Clinical Text with Electronic Health Records",
 "kim2025emulate":"Large Language Models' Accuracy in Emulating Human Experts' Evaluation of Public Sentiments about Health Interventions",
 "croxford2025autoeval":"Automating Evaluation of AI Text Generation in Healthcare with a Large Language Model (LLM)-as-a-Judge",
 "hou2024classroom":"Automated Assessment of Encouragement and Warmth in Classrooms Leveraging Multimodal Emotional Features and ChatGPT",
 "koraishi2024langassess":"The Intersection of AI and Language Assessment: A Study on the Reliability of ChatGPT in Scoring",
 "haase2025creativitypeaked":"Has the Creativity of Large-Language Models Peaked? An Analysis of Inter- and Intra-LLM Variability",
 "rose2025rob":"Using a Large Language Model (ChatGPT) to Assess Risk of Bias in Randomized Controlled Trials of Medical Interventions",
 # second consolidation pass
 "thakur2024judgingjudges":"Judging the Judges: Evaluating Alignment and Vulnerabilities in LLMs-as-Judges",
 "hsu2023figurecaption":"GPT-4 as an Effective Zero-Shot Evaluator for Scientific Figure Captions",
 "lai2023styletransfer":"Multidimensional Evaluation for Text Style Transfer Using ChatGPT",
 "wang2024beyondagreement":"Beyond Agreement: Diagnosing the Rationale Alignment of Automated Essay Scoring Methods based on Large Language Models",
 "wataoka2024selfpref":"Self-Preference Bias in LLM-as-a-Judge",
 "frobe2025assessorsagree":"Large Language Model Relevance Assessors Agree With One Another More Than With Human Assessors",
 "arabzadeh2025promptsens":"A Human-AI Comparative Analysis of Prompt Sensitivity in LLM-Based Relevance Judgments",
 "lin2024wildbench":"WildBench: Benchmarking LLMs with Challenging Tasks from Real Users in the Wild",
 "brake2024clinicalnote":"Comparing Two Model Designs for Clinical Note Generation; Is an LLM a Useful Evaluator of Consistency?",
 "mehta2025qualgenai":"Evaluation of Large Language Models within Generative AI in Qualitative Research",
 "ahmad2025moralmachine":"Large-Scale Moral Machine Experiment on Large Language Models",
 "hu2025trainjudge":"Training an LLM-as-a-Judge Model: Pipeline, Insights, and Practical Lessons",
 "kholodna2024activelearning":"LLMs in the Loop: Leveraging Large Language Model Annotations for Active Learning in Low-Resource Settings",
 "li2024pedants":"PEDANTS: Cheap but Effective and Interpretable Answer Equivalence",
 "shopovski2025peerreview":"Revolutionizing Peer Review: A Comparative Analysis of ChatGPT and Human Reviewers",
 # community-depth pass: HCI / CSCW / ML
 "zhu2023judgelm":"JudgeLM: Fine-tuned Large Language Models are Scalable Judges",
 "wang2023pandalm":"PandaLM: An Automatic Evaluation Benchmark for LLM Instruction Tuning Optimization",
 "li2023autoj":"Generative Judge for Evaluating Alignment",
 "wang2023shepherd":"Shepherd: A Critic for Language Model Generation",
 "ke2024critiquellm":"CritiqueLLM: Towards an Informative Critique Generation Model for Evaluation of Large Language Model Generation",
 "yuan2024selfrewarding":"Self-Rewarding Language Models",
 "wu2024metarewarding":"Meta-Rewarding Language Models: Self-Improving Alignment with LLM-as-a-Meta-Judge",
 "li2024arenahard":"From Crowdsourced Data to High-Quality Benchmarks: Arena-Hard and BenchBuilder Pipeline",
 "chiang2024arena":"Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference",
 "meng2024chalet":"Exploring the Human-LLM Synergy in Advancing Theory-driven Qualitative Analysis (CHALET)",
 "wu2023llmworkers":"LLMs as Workers in Human-Computational Algorithms? Replicating Crowdsourcing Pipelines with LLMs",
 "zhu2023reproduce":"Can ChatGPT Reproduce Human-Generated Labels? A Study of Social Computing Tasks",
 "desmond2024evalullm":"EvaluLLM: LLM Assisted Evaluation of Generative Outputs",
 "kim2024meganno":"MEGAnno+: A Human-LLM Collaborative Annotation System",
 "duan2024uimockup":"Generating Automatic Feedback on UI Mockups with Large Language Models",
 "lin2023llmeval":"LLM-Eval: Unified Multi-Dimensional Automatic Evaluation for Open-Domain Conversations with Large Language Models",
 "chan2023chateval":"ChatEval: Towards Better LLM-based Evaluators through Multi-Agent Debate",
 "hashemi2024llmrubric":"LLM-Rubric: A Multidimensional, Calibrated Approach to Automated Evaluation of Natural Language Texts",
 "ostyakova2023crowdexperts":"ChatGPT vs. Crowdsourcing vs. Experts: Annotating Open-Domain Conversations with Large Language Models",
 "xie2023nextchapter":"The Next Chapter: A Study of Large Language Models in Storytelling",
 "ni2024afacta":"AFaCTA: Assisting the Annotation of Factual Claim Detection with Reliable LLM Annotators",
 "lee2023rlaif":"RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback",
 "jung2024trustescalate":"Trust or Escalate: LLM Judges with Provable Guarantees for Human Agreement",
 "li2025prefleakage":"Preference Leakage: A Contamination Problem in LLM-as-a-Judge",
 "murugadoss2025evaltheeval":"Evaluating the Evaluator: Measuring LLMs' Adherence to Task Evaluation Instructions",
 "fan2024sedareval":"SedarEval: Automated Evaluation using Self-Adaptive Rubrics",
 "horych2025promisespitfalls":"The Promises and Pitfalls of LLM Annotations in Dataset Labeling",
 "gao2025reranking":"Re-evaluating Automatic LLM System Ranking for Alignment with Human Preference",
 "yamauchi2025designchoices":"An Empirical Study of LLM-as-a-Judge: How Design Choices Impact Evaluation Reliability",
 "huang2025empiricaljudge":"An Empirical Study of LLM-as-a-Judge for LLM Evaluation: Fine-tuned Judge Model is not a General Substitute for GPT-4",
 # ===== Study 2: HCC / HCI / qualitative-research methods =====
 "xiao2023support":"Supporting Qualitative Analysis with Large Language Models: Combining Codebook with GPT-3 for Deductive Coding",
 "zhang2025inddeduct":"Exploring Inductive and Deductive Qualitative Coding with AI: Investigating Inter-Rater Reliability between Large Language Model and Human Coders",
 "gao2024collabcoder":"CollabCoder: A Lower-barrier, Rigorous Workflow for Inductive Collaborative Qualitative Analysis with Large Language Models",
 "dunivin2024scalablecot":"Scalable Qualitative Coding with LLMs: Chain-of-Thought Reasoning Matches Human Performance in Some Hermeneutic Tasks",
 "ngo2026chatqda":"Qualitative Coding Analysis through Open-Source Large Language Models: A User Study and Design Recommendations",
 "depaoli2023inductive":"Performing an Inductive Thematic Analysis of Semi-Structured Interviews with a Large Language Model",
 "dai2023llminloop":"LLM-in-the-loop: Leveraging Large Language Model for Thematic Analysis",
 "nguyentrung2025chatgptta":"ChatGPT in Thematic Analysis: Can AI Become a Research Assistant in Qualitative Research?",
 "jain2025multillmthematic":"Multi-LLM Thematic Analysis with Dual Reliability Metrics: Combining Cohen's Kappa and Semantic Similarity for Qualitative Research Validation",
 "castellanos2025thematicsumm":"Large Language Models for Thematic Summarization in Qualitative Health Care Research: Comparative Analysis of Model and Human Performance",
 "sharma2025details":"DeTAILS: Deep Thematic Analysis with Iterative LLM Support",
 "prescott2024efficacy":"Comparing the Efficacy and Efficiency of Human and Generative AI Qualitative Thematic Analyses",
 "chew2023llmcontent":"LLM-Assisted Content Analysis: Using Large Language Models to Support Deductive Coding",
 "dunivin2025hermeneutics":"Scaling Hermeneutics: A Guide to Qualitative Coding with LLMs for Reflexive Content Analysis",
 "schroeder2025usestensions":"Large Language Models in Qualitative Research: Uses, Tensions, and Intentions",
 "davison2024ethics":"The Ethics of Using Generative AI for Qualitative Data Analysis",
 "wachinger2024prompts":"Prompts, Pearls, Imperfections: Comparing ChatGPT and a Human Researcher in Qualitative Data Analysis",
 "li2024gpt4humanqual":"Comparing GPT-4 and Human Researchers in Health Care Data Analysis: Qualitative Description Study",
 "zhang2023redefining":"Redefining Qualitative Analysis in the AI Era: Utilizing ChatGPT for Efficient Thematic Analysis",
 "savelka2023textskilled":"Can GPT-4 Support Analysis of Textual Data in Tasks Requiring Highly Skilled Domain Knowledge?",
 "long2024classroom":"Evaluating Large Language Models in Analysing Classroom Dialogue",
 "argyle2023outofone":"Out of One, Many: Using Language Models to Simulate Human Samples",
 "wang2024flatten":"Large Language Models that Replace Human Participants can Harmfully Misportray and Flatten Identity Groups",
 "bail2024gensocsci":"Can Generative AI Improve Social Science?",
 "bisbee2023perils":"Synthetic Replacements for Human Survey Data? The Perils of Large Language Models",
 "huang2025surveyworth":"How Many Human Survey Respondents is a Large Language Model Worth? An Uncertainty Quantification Perspective",
 "cheng2023compost":"CoMPosT: Characterizing and Evaluating Caricature in LLM Simulations",
 "salminen2025syntheticcritical":"The Use of Large Language Models in HCI: A Critical Analysis of Synthetic Users",
 "abbasiantaeb2024llmstalk":"Let the LLMs Talk: Simulating Human-to-Human Conversational Search Sessions",
 "salminen2024deusexmachina":"Deus Ex Machina and Personas from Large Language Models: Investigating the Composition of AI-Generated Persona Descriptions",
 "jung2025personacraft":"PersonaCraft: Leveraging Language Models for Data-Driven Persona Development",
 "haxvig2025personabias":"I've Never Seen a Glass Ceiling Better Represented: Bias and Gendering in LLM-Generated Synthetic Personas from a Participatory Design Perspective",
 "guerino2025gpt4ousability":"Can GPT-4o Evaluate Usability Like Human Experts? A Comparative Study on Issue Identification in Heuristic Evaluation",
 "zhong2025synthheuristic":"Synthetic Heuristic Evaluation: A Comparison between AI- and Human-Powered Usability Evaluation",
 "duan2024uicrit":"UICrit: Enhancing Automated Design Evaluation with a UI Critique Dataset",
 "duan2024visualcritique":"Visual Prompting with Iterative Refinement for Design Critique Generation",
 "kocaballi2023convdesign":"Conversational AI-Powered Design: ChatGPT as Designer, User, and Evaluator",
 "kuzman2023endannotation":"ChatGPT: Beginning of an End of Manual Linguistic Data Annotation? Use Case of Automatic Genre Identification",
 "rajashekar2024humanalgo":"Human-Algorithmic Interaction Using a Large Language Model-Augmented Artificial Intelligence Clinical Decision Support System",
 "lequere2024researchtools":"LLMs as Research Tools: Applications and Evaluations in HCI Data Work",
 "tabone2023hciprimer":"Using ChatGPT for Human-Computer Interaction Research: A Primer",
 "anderson2024homogenization":"Homogenization Effects of Large Language Models on Human Creative Ideation",
}

CONF = re.compile(r"(neurips|icml|iclr|acl|emnlp|naacl|eacl|coling|sigir|cvpr|eccv|iccv|aaai|"
                  r"chi|uist|kdd|www|findings|workshop|conference|proceedings|recsys|comma|bea|ictir|icwsm|iui|wmt|newsum|hucllm)", re.I)

def fmt_authors(a):
    a = str(a).strip()
    if not a or a == "authors": return "{LLM-as-a-Judge Review Corpus}"
    a = a.replace(" & ", " and ").replace(" et al.", " and others").replace("et al.", "and others")
    parts = [p.strip() for p in a.split(",") if p.strip()]
    parts = ["others" if p=="and others" else p for p in parts]
    return " and ".join(parts)

def resolve(study):
    sid, ax = study["id"], str(study["arxiv"]).strip()
    is_arxiv = bool(re.match(r"\d{4}\.\d{4,5}", ax)); axk = ax.split("v")[0]
    is_doi = ax.startswith("10.")
    meta = {}
    if is_arxiv and axk in idx_ax: meta = idx_ax[axk]
    elif is_doi and ax.lower() in idx_doi: meta = idx_doi[ax.lower()]
    title = TITLES.get(sid) or meta.get("title") or study["finding"][:80]
    authors = study["authors"] if str(study["authors"]) not in ("","authors") else meta.get("authors","")
    venue = study["venue"] or meta.get("venue","")
    return title, fmt_authors(authors), venue, is_arxiv, axk, is_doi, ax

EXTRA_BIB = r"""@article{page2021prisma,
  author = {Page, Matthew J. and McKenzie, Joanne E. and Bossuyt, Patrick M. and Boutron, Isabelle and Hoffmann, Tammy C. and Mulrow, Cynthia D. and others},
  title = {{The PRISMA 2020 statement: an updated guideline for reporting systematic reviews}},
  journal = {BMJ},
  volume = {372},
  pages = {n71},
  year = {2021},
  doi = {10.1136/bmj.n71}
}"""

def main():
    out = ["% references.bib -- auto-generated from data/included_studies.csv",
           f"% PRISMA review: LLM-as-a-Judge vs Human Annotation.  {len(inc)} references.\n",
           EXTRA_BIB]
    for _,s in inc.iterrows():
        title, authors, venue, is_arxiv, axk, is_doi, doi = resolve(s)
        key, year = s["id"], int(s["year"]) if str(s["year"]).isdigit() else s["year"]
        esc = lambda s: str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")
        title_b = "{" + esc(title) + "}"
        venue_e = esc(venue)
        fields = [f"  author = {{{authors}}}", f"  title = {{{title_b}}}", f"  year = {{{year}}}"]
        if is_arxiv:
            etype = "@article"
            fields += [f"  journal = {{arXiv preprint arXiv:{axk}}}", f"  eprint = {{{axk}}}",
                       "  archivePrefix = {arXiv}"]
        elif CONF.search(str(venue)):
            etype = "@inproceedings"; fields += [f"  booktitle = {{{venue_e}}}"]
            if is_doi: fields += [f"  doi = {{{doi}}}"]
        else:
            etype = "@article"; fields += [f"  journal = {{{venue_e or 'Preprint'}}}"]
            if is_doi: fields += [f"  doi = {{{doi}}}"]
        out.append(f"{etype}{{{key},\n" + ",\n".join(fields) + "\n}")
    (ROOT/"paper"/"references.bib").write_text("\n\n".join(out)+"\n")

    # report any records that fell back to a finding-derived title (i.e., no real title)
    miss = [s["id"] for _,s in inc.iterrows()
            if not (TITLES.get(s["id"]) or resolve(s)[0]!=s["finding"][:80])]
    print(f"wrote paper/references.bib with {len(inc)} entries")
    print(f"records with NO resolved title (using finding fallback): {len(miss)}")
    for m in miss: print("   -", m)

if __name__ == "__main__": main()
