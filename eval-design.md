# UPSC SLM Evaluation Design v2

**Owner:** Yeeshan (data scientist, prayas.ai)
**Status:** Revision 2 — 2026-05-14 (rewritten to lead with quantitative metrics; model shortlist refreshed)
**Companion:** [project-context.md](project-context.md)

This document specifies *what we measure*, *how it is computed*, and *which library implements it*. **Primary metrics are deterministic given the inputs and a fixed scorer-model checkpoint** — they reproduce on rerun. LLM-as-judge metrics are explicitly relegated to a secondary/diagnostic tier (§4) and labeled as such.

---

## 1. The four conditions under test

| ID | Condition | Notes |
|---|---|---|
| **C1a** | FT-SLM: `google/gemma-4-E4B-it` + LoRA multi-task adapter (tasks A+B+C+E) | Local via MLX-LM on M5 16GB |
| **C1b** | FT-SLM: `Qwen/Qwen3.5-4B` + LoRA multi-task adapter (same recipe as C1a) | Local via MLX-LM on M5 16GB; native MLX build `mlx-community/Qwen3.5-4B-MLX-4bit` |
| **C2** | `gemini-3.5-flash`, zero-shot | Google API, single prompt with task instruction only. Pricing: $0.50/M input, $3.00/M output (May 2026 published rate). Pre-registration named `gemini-3-flash`; the actual generateContent API exposes `gemini-3.5-flash` as the live current-default flash-tier model and `gemini-3-flash-preview` as the still-preview variant. Switched at v1 inference time and recorded in `model_version` per prediction. The model name is env-configurable via `GEMINI_MODEL`. |
| **C3** | `gemini-3.5-flash`, few-shot | Google API, with 3 task-matched UPSC exemplars in the prompt. Same model swap as C2. |

---

## 2. FT base models — two candidates

### 2.0a Candidate 1 — `google/gemma-4-E4B-it`

| Field | Value |
|---|---|
| **HuggingFace ID** | `google/gemma-4-E4B-it` (instruction-tuned). Base variant: `google/gemma-4-E4B`. |
| **Architecture** | MatFormer (Matryoshka) + Per-Layer Embeddings (PLE). 42 layers, 262K vocab, hybrid attention (local sliding-window 512 + global). |
| **Parameters** | 4.5B *effective* / 8B *total*. Vision encoder ~150M; audio encoder ~300M. |
| **License** | Apache 2.0 |
| **Context window** | 128K tokens |
| **Multilingual** | 140+ languages pretrained; 35+ languages native instruction-tuned. Hindi is in the pretrained tier (not explicitly named in the 35+ "native" list, but in the 140+ pretrained pool). |
| **Multimodality** | text + image (variable aspect ratio) + audio (ASR + speech-to-translated-text, max 30s clips). Video is supported only by Gemma 4 31B Dense, not by E4B. |
| **Training cutoff** | January 2025 |
| **MLX path** | `mlx-community/gemma-4-e4b-it-4bit` (~5 GB MLX-quantized; multimodal weights present but mlx-lm's `gemma4` class properly ignores `vision_tower` / `audio_tower` / `embed_*` / `multi_modal_projector` and loads only `language_model.*`). Requires **mlx-lm post-commit `df1d3f3c`** (2026-05-04, *"Fix Gemma 4 sanitize() not stripping KV projections for shared layers"*) — at the 0.31.3 PyPI release the loader rejected 126 extra K/V projections from Gemma 4's hybrid-attention shared layers. Pinned as `git+https://github.com/ml-explore/mlx-lm.git@df1d3f3c…` in `requirements.txt`; replace with the next tagged release that contains the fix when published. FT framework: `mlx_lm.lora`. |
| **Memory footprint** | ~3 GB RAM @ Q4 inference on M5 (PLE keeps embeddings CPU-side); FT ≈ 7-9 GB peak. |

### 2.0b Candidate 2 — `Qwen/Qwen3.5-4B`

| Field | Value |
|---|---|
| **HuggingFace ID** | `Qwen/Qwen3.5-4B` (instruction-tuned). Base variant: `Qwen/Qwen3.5-4B-Base`. |
| **Architecture** | Dense decoder. Unified vision-language foundation with early-fusion training. Hybrid reasoning (`enable_thinking=True` toggles CoT mode). |
| **Parameters** | 4.66B dense. |
| **License** | Apache 2.0 |
| **Context window** | 262,144 tokens native; extensible to 1,010,000. |
| **Multilingual** | **201 languages** — Hindi explicitly enumerated. |
| **Multimodality** | text + image (early-fusion vision-language). |
| **Training cutoff** | March 2026 release; data through early 2026. |
| **MLX path** | Native MLX builds: `mlx-community/Qwen3.5-4B-MLX-4bit`, `mlx-community/Qwen3.5-4B-MLX-8bit`, `mlx-community/Qwen3.5-4B-MLX-bf16`. |
| **Memory footprint** | ~3 GB RAM @ Q4 inference on M5; FT ≈ 6-8 GB peak. |

### 2.0c Why two candidates

Same LoRA recipe, same FT corpus, same eval set — only the base model differs. This isolates the architecture/pretraining variable. The Hindi-stratum delta `C1a − C1b` is a direct read on "Indic-via-FT (Gemma's pretraining-pool Hindi) vs Indic-via-pretraining (Qwen's explicit Hindi)". Other-task deltas test whether v1 results are portable across base SLM families or sensitive to the choice.

### 2.1 Hindi coverage — verification protocol (applies to both candidates)

Hindi appears in Gemma-4's 140+ pretrained-languages pool but not in the 35+ native instruction-tuned tier. Qwen3.5-4B explicitly lists Hindi in its 201-language coverage. We measure rather than assume — **§10 A2** locks the pre-FT Hindi-capability triage. Pass criterion is a one-sided binomial test against random-chance (`H0: accuracy = 0.25`, `H1: accuracy > 0.25`, α = 0.05) on 50 Hindi MCQs.

**Result (base models, pre-FT, executed 2026-05-21, chat-templated + `enable_thinking=False` + `max_tokens=24`):**

| Base model | Correct / 50 | Accuracy | p-value | Verdict |
|---|---:|---:|---:|---|
| `google/gemma-4-E4B-it` (via `mlx-community/gemma-4-e4b-it-4bit`) | **26 / 50** | **52.0 %** | < 0.00001 | **PASS** |
| `Qwen/Qwen3.5-4B` (via `mlx-community/Qwen3.5-4B-MLX-4bit`) | 15 / 50 | 30.0 % | 0.252 | **FAIL** — cannot reject H₀ |

**Protocol note.** The probe must wrap the prompt in the model's chat template (`tokenizer.apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)`) and allocate enough `max_tokens` to clear any `<think>` block. Earlier un-templated runs hit measurement artefacts (Gemma 4-IT emitted EOS on the first token → 0 / 50; Qwen 3.5 truncated mid-thinking 18 / 50 times). Direction is the **inverse** of the pre-registered prediction in `experiment-report.md §5.1` — Gemma's 140-language pretrained pool delivers stronger usable Hindi than Qwen's explicit 201-language enumeration, at least on this protocol. Failing the gate **does not block FT**; it routes the Hindi-stratum results post-FT to a separate finding rather than folding them into the bilingual aggregate.

### 2.2 Models considered and not used in v1

- `microsoft/Phi-4-mini-instruct` — Hindi not in MS's enumerated language list; deferred.
- `mistralai/Ministral-3-3B-Instruct-2512` — Hindi not in Mistral's enumerated language list; deferred.
- `google/gemma-3n-E2B-it` — superseded by gemma-4-E4B (newer, larger effective params, Apache 2.0 vs Gemma terms, 128K vs 32K context).
- `Sarvam-1 (2B)` — Indic-native but smaller; reserved for v2 if v1 results suggest Indic specialization matters beyond what Qwen3.5-4B already provides.
- All earlier candidates (Qwen3-8B, AryaBhatta-GemmaGenZ-Vikas, Qwen3.5-9B, Aya Expanse 8B) — superseded.

---

## 3. Eval set construction

### 3.1 Stratified sample (2,000 items)

| Task | Surface | Count | Stratification | Source tables (read from `data/prayas_local.sqlite`) |
|---|---|---:|---|---|
| **A** | Prelims MCQ | 800 | Paper (**GS-I, CSAT**) × Subject × `silly_mistake_prone` × `language` (en / hi) | `prelims_pyq_questions` (454 in eval) + `upsc_prelims_ai_generated_que` (256) + `mcqs ⨝ learning_items` (90 — GS1 = 70 + CSAT = 20) |
| **B** | Mains generation | 400 | Paper (GS1, GS2, GS3, GS4, Essay) × Subject × word-count band (150 / 250 / essay) × language | `pyqs` (has bilingual `model_answer`) |
| **C** | Mains rubric grading | 500 | Subject × score-band (low ≤ 30 %, mid 30-60 %, high > 60 % of `max_score`) | `evaluation_questions` |
| **E** | Current Affairs synthesis | 300 | `newsThemeId` × month, news article date ≤ 2026-04-30 | `news_articles` |

Total: **2,000 unique items**. Tasks F (Prelims Explanation Generation, §4.6) and G (Mains Model-Answer Generation, §4.7) **reuse** the 800 Task-A and 400 Task-B items respectively; no new eval-set rows are added for the production-prompt capability tests. Per-condition prediction count: 2 000 (core) + 1 200 (F + G) = 3 200 rows × 4 conditions = **12 800 predictions**.

**Note on the CSAT stratum.** `paper` lives on `learning_items` (not directly on `mcqs`); the snapshot SQL `LEFT JOIN`s the two so each prod MCQ carries its paper tag, then `freeze_eval_set.py` filters mcqs to `paper ∈ {gs1, csat}`. CSAT becomes a first-class stratum (`CSAT|UNTAGGED|silly=0|en`) with 20 items held out for the gate slice.

### 3.2 Freezing the eval set

A single Python script (`scripts/freeze_eval_set.py`) selects IDs deterministically from the local SQLite snapshot using `random.Random(seed=20260514)` and writes one Parquet file: `data/eval_set.parquet` with a SHA-256 sidecar. **Current artifact:** SHA-256 `e2b62a3f…`. The freezer reads from `data/prayas_local.sqlite` (a read-only mirror written by the single auditable script `scripts/snapshot_to_local.py`) — no downstream script connects to remote Postgres.

Schema:

```
question_id  str   -- source PK, prefixed by source (e.g. 'pyq:42:en', 'prod_mcq:abc-123:en')
task         str   -- 'A' | 'B' | 'C' | 'E'
source_db    str   -- 'upscdev' | 'prod-prayas-db'
source_table str
paper        str   -- 'GS1'..'GS4' | 'Essay' | 'CSAT' | 'UNTAGGED'
subject      str
language     str   -- 'en' | 'hi'
gold_payload str   -- JSON-serialized full gold record needed at scoring time
stratum_key  str   -- compact stratum label for grouping
```

**FT-corpus current artifact:** `data/ft_corpus.parquet`, **41 749 supervised pairs**, SHA-256 `d57be52c…`. Breakdown: A = 26 638 (incl. CSAT = 2 361 + GS1 = 8 548 from prod.mcqs), B = 2 608, C = 9 600, E = 2 903. The build asserts `eval ∩ ft = ∅` and fails hard if any overlap is detected (the CI guard you'd expect).

### 3.3 Current Affairs cutoff

Task E items use only articles with `prod.news_articles.date <= '2026-04-30'`. All three model conditions see the same article text as input — Gemini's training cutoff is irrelevant because we hand it the article.

---

## 4. Metric tiers

> **Tier 1 — Primary** (deterministic given inputs + fixed scorer-model checkpoint; reported in headline results).
> **Tier 2 — Secondary** (LLM-as-judge or other model-dependent assessments). **Deferred for v1.** All Tier-2 metrics described below are documented for design completeness but are not computed in the v1 pipeline — the quantitative-first directive plus a 45-metric Tier-1 inventory cover the headline science. The Anthropic API key is unused; `claude-sonnet-4-6` judge inference is not executed. Tier-2 can be re-enabled in v2 if a teaching-quality view on top of Tier-1 is wanted.
> Every metric below names the exact Python package and the call.

**v1-deferred Tier-1 metrics** (re-enable in v2 by installing the git-only deps in a separate `requirements-scoring-git.txt`):

- **BLEURT-20** (Task B) — `bleurt-pytorch` is git-only; signal dominated by BERTScore-F1 at our scoring scale.
- **SummaC-ZS / AlignScore** (Task E faithfulness) — git-only; the deterministic `hallucination_rate` + UPSC fact-lookup precision metrics cover the same axis at lower fidelity.
- **FactScore** (Task E) — git-only; decomposes into atomic claims and NLI-checks each, using an LLM in the loop (effectively Tier 2).
- **Generation perplexity** (Task B) — requires loading the Gemma-4-E4B base into `transformers` alongside scoring; heavy memory cost dominated by other signal.
- **Glossary term recall** (Task E) — requires `prod.glossary` (7 475 keywords) in the local snapshot; not yet imported.
- **METEOR** (Task B) — NLTK wordnet corpus dep adds runtime download friction and the signal is redundant with BERTScore + ROUGE-L.

### 4.0 Shared utility — `data/upsc_facts.json` (static UPSC knowledge lookup)

A curated, committed-to-repo static JSON file used by Tier-1 fact-lookup metrics across Tasks A, B, and E. Built once from public sources; hashed in `manifest.json` per run.

```
{
  "articles": {
    "21": {"title": "Protection of life and personal liberty",
           "part": "III", "topic_tokens": ["life", "liberty", "due process", "personal"]},
    "32": {"title": "Right to constitutional remedies", ...},
    ...
  },
  "schedules": { "9": {"title": "Land reforms validation", ...}, ... },
  "acts": {
    "RTI 2005":    {"year": 2005, "topic_tokens": ["information", "transparency", ...]},
    "RPA 1951":    {"year": 1951, ...},
    ...
  },
  "five_year_plans": { "1": {"start": 1951, "end": 1956}, ... },
  "schemes": {
    "PMGSY": {"start_year": 2000, "ministry": "Rural Development", ...},
    "MGNREGA": {"start_year": 2005, ...},
    ...
  },
  "office_holders": {
    "president": [{"name": "Droupadi Murmu", "start": "2022-07-25", "end": null}, ...],
    "pm":        [{"name": "Narendra Modi", "start": "2014-05-26", "end": null}, ...],
    ...
  }
}
```

**Build source:** public Constitution of India text + India.gov.in scheme pages + Lok Sabha official records. Build script `scripts/build_upsc_facts.py` is deterministic and seeded.

**Used by:**
- §4.1 Task A: *Article/scheme citation accuracy*
- §4.2 Task B: *UPSC fact-lookup precision*
- §4.4 Task E: *UPSC fact-lookup precision*

**Why static and not LLM-as-judge:** facts about the Constitution and major Acts are not opinion. A deterministic lookup gives reproducible scoring and catches the exact failure mode (wrong Article number / wrong year / wrong scheme name) that UPSC graders punish.

### 4.1 Task A — Prelims MCQ

Two sub-categories: **correctness/calibration metrics** (on the answer letter) and **explanation-quality / pedagogical-clarity metrics** (on the explanation text). All Tier 1 are deterministic; pedagogical clarity gets Tier 1 *quantitative proxies* and a Tier 2 *LLM-judge rubric* that cross-checks them.

#### Tier 1 — Correctness & calibration (8 metrics)

| Metric | Definition | Library / call |
|---|---|---|
| **Accuracy** | `is_correct.mean()` overall + per (paper, subject, difficulty, silly_mistake_prone, language) | `sklearn.metrics.accuracy_score` |
| **UPSC Negative-marking Score** | GS-I MCQs: `+2` correct, `−2/3` wrong, `0` abstain. CSAT: `+2.5` correct, `−2.5/3` wrong. Mean per 100 questions. | Custom function |
| **Expected Calibration Error (ECE)** | 15-bin reliability-diagram ECE on (stated_confidence/100 vs is_correct) | `torchmetrics.classification.BinaryCalibrationError(n_bins=15, norm='l1')` |
| **Brier Score** | `mean((confidence_prob − is_correct)**2)` | `sklearn.metrics.brier_score_loss` |
| **Brier Skill Score** | `1 − Brier / Brier_of_baserate_predictor` | Custom |
| **Refusal / format-fail rate** | % where parser cannot extract a single option in {A,B,C,D} | Custom regex |
| **Bilingual accuracy delta** | `acc(en) − acc(hi)` on paired bilingual items | Custom — paired t-test |
| **Silly-mistake breakdown** | accuracy on subset where `silly_mistake_prone=True` vs `False` | Group-by |

#### Tier 1 — Explanation quality (8 metrics; reference-based vs `prelims_pyq_questions.explanation` + static lookup)

| Metric | Definition | Library / call |
|---|---|---|
| **Explanation BERTScore-F1** | Semantic similarity of generated explanation vs gold `explanation.english` / `explanation.hindi` | `bert_score` w/ `deberta-xlarge-mnli` (en) or `xlm-roberta-large` (hi) |
| **Explanation ROUGE-L F1** | LCS overlap vs gold | `evaluate.load('rouge')` |
| **Explanation Entity-F1** | NER + UPSC-glossary-term overlap vs gold (catches Article-number / scheme-name precision) | `spacy` + `prod-prayas-db.glossary` keyword match |
| **Distractor coverage** | Fraction of wrong options (A/B/C/D minus correct) explicitly addressed in the explanation. Detection: option letter token + ≥1 distinctive content word from that option appearing in the explanation. | Custom (string-match function) |
| **Reasoning-step density** | Count of discourse markers (`because`, `therefore`, `however`, `first`, `second`, `if`/`then`) per 100 words | Custom regex over a fixed marker list |
| **Article/scheme citation accuracy** | When the explanation mentions `Article N` / `Schedule N` / a named Act, look up N in `data/upsc_facts.json` (see §4.0) and verify the surrounding context tokens overlap ≥0.3 Jaccard with the lookup's expected-context tokens. Aggregate: fraction of citations that pass. | Custom: regex extraction + lookup |
| **Answer position bias** | χ² test of the model's predicted A/B/C/D distribution vs the gold distribution over the 800 Task-A items. p-value reported; |Δ| ≤ 0.05 per option preferred. Catches "always picks C" pathology. | `scipy.stats.chisquare` |
| **Sentence-length variance** | Variance of token-counts per sentence in the explanation. Too-uniform variance (< 5 tokens²) flags AI-fingerprint / template-style; too-high (> 80) flags incoherence. | `nltk.sent_tokenize` + `np.var` |

#### Tier 2 — Pedagogical Clarity rubric (LLM-judge, diagnostic only)

5-axis rubric, each scored 1–5 by `claude-sonnet-4-6`. Total = sum (5–25).

| Axis | What the judge looks for |
|---|---|
| 1. **Step-by-step reasoning** | Does the explanation walk through *how* to arrive at the answer, not just state the conclusion? |
| 2. **Distractor addressing** | Does it explain why the wrong options are wrong, not only why the right option is right? |
| 3. **Conceptual grounding** | Does it tie the answer to a named UPSC syllabus concept (e.g. "Article 368 procedure", "Demographic Dividend") rather than restating the question? |
| 4. **Specificity** | Does it cite specific Article numbers, years, scheme names, places — not vague references? |
| 5. **Accessibility** | Is the explanation pitched at a UPSC aspirant level — neither over-jargonized nor over-simplified? |

The Tier 1 proxies (Distractor coverage + Reasoning-step density + Entity-F1 vs gold) are designed to **track** the Tier 2 rubric axes; we report Kendall's τ between the Tier 1 composite and the Tier 2 total per condition. If τ > 0.5, the Tier 2 rubric primarily adds nuance over Tier 1; if τ < 0.3, the Tier 2 metric is capturing something Tier 1 misses and we flag both.

#### Inference protocol — three passes (used identically across all four conditions)

```
PASS 1: "You are taking the UPSC Prelims. Question: {q}. Options: A) ... D) ...
         Answer with ONLY the letter."
PASS 2: "On a scale of 0 to 100, how confident are you that {predicted_letter} is correct?
         Respond with ONLY the integer."
PASS 3: "Now explain why option {predicted_letter} is correct. For each other option, briefly
         explain why it is wrong. Write the explanation in {language}."
```

All three passes use temperature 0 / top_p 1. Pass 3 is what feeds the explanation-quality + pedagogical-clarity metrics.

### 4.2 Task B — Mains generation (Tier 1 quantitative + Tier 2 LLM-judge diagnostic)

#### Tier 1 (primary)

| Metric | Definition | Library / call |
|---|---|---|
| **BERTScore-F1** | F1 of contextual embedding similarity between generation and `pyqs.model_answer`; use `microsoft/deberta-xlarge-mnli` rescaled baseline | `bert_score.score(cands, refs, model_type='microsoft/deberta-xlarge-mnli', rescale_with_baseline=True, lang='en')` for English; multilingual XLM-R for Hindi |
| **BLEURT-20** | learned regression metric on (gen, ref) pairs | `bleurt-pytorch` package, checkpoint `lucadiliello/BLEURT-20` |
| **ROUGE-L F1** | longest-common-subsequence F1 | `evaluate.load('rouge')` then `.compute(predictions=..., references=...)` |
| **chrF++** | character n-gram F1 (more robust for Devanagari than BLEU) | `sacrebleu.corpus_chrf(generations, [references], word_order=2)` |
| **METEOR** | unigram alignment with synonymy + stemming | `evaluate.load('meteor')` |
| **Word-count adherence** | `1 − abs(words(gen) − words_target) / words_target`, clipped to [0,1]; target ∈ {150, 250, 1200} | Custom one-liner using `len(text.split())` |
| **Sentence-count adherence** | absolute delta in sentence count vs reference | `nltk.tokenize.sent_tokenize` |
| **Entity-F1 (recall of gold entities)** | NER on `gen` and `gold`, F1 over entity surface forms | `spacy` model `en_core_web_trf` (English), `xx_ent_wiki_sm` (multilingual); custom F1 |
| **Date / number exact-match F1** | regex-extracted dates, years, percentages, numerical claims; F1 on the multisets | Custom — regex `\b(19|20|21)\d{2}\b` + `\d+(?:\.\d+)?%?` |
| **Hindi code-mixing rate** | fraction of tokens whose Unicode block is *not* Devanagari, in a Hindi-prompted answer | Custom using `unicodedata` |
| **Generation perplexity** | mean per-token NLL under a fixed held-out scorer (`google/gemma-4-E4B` base); lower = more fluent | `transformers.AutoModelForCausalLM` with `output.loss` |
| **Type-Token Ratio (MATTR)** | Moving-Average TTR with window 100 — measures lexical diversity stable across answer length. Low values = repetitive vocabulary. | Custom (NumPy rolling window) |
| **Flesch-Kincaid Grade Level** | Readability formula: `0.39 × (words/sentences) + 11.8 × (syllables/words) − 15.59`. UPSC answers typically land at grade 12-15; below 10 = too simple, above 18 = jargon-dense. | `textstat.flesch_kincaid_grade` |
| **Paragraph count adherence** | `1 − |paragraphs(gen) − paragraphs_norm(target_word_count)| / paragraphs_norm`, clipped [0,1]. Norms: 150-word → 1-2 paragraphs; 250-word → 3-5; 1200-word essay → 8-12. | Custom: split on `\n\n+` |
| **4-gram repetition rate** | Number of 4-grams repeated more than once / total 4-grams. Pathological repetition is a known SLM failure; UPSC graders penalize. | Custom: `nltk.ngrams` |
| **UPSC fact-lookup precision** | For every regex-extracted `Article N` / `Section N` / `Year ####` / named-scheme reference in the generated answer, look up correctness against `data/upsc_facts.json` (§4.0). Report precision = correct / total-extracted. | Custom: regex extraction + JSON lookup |

All Tier-1 metrics are deterministic given the same scorer checkpoint. We pin checkpoints in `requirements.txt` and check them into `models/lockfile.json`.

#### Tier 2 (diagnostic — clearly labeled, never primary)

| Metric | Why secondary |
|---|---|
| **G-Eval rubric scores** (Content / Contextual / Analytical / Structural / Directive compliance — 1-5 each) | Uses Gemini-2.5-Pro as judge; subjective despite numeric output; reported only as diagnostic alongside Tier 1 |
| **LLM-judge "Directive Compliance" binary** | Same caveat |
| **LLM-judge factual-error count** | Same caveat |

We additionally compute Krippendorff's α between G-Eval and BERTScore-F1 rank ordering; if α > 0.7 we note that Tier 2 reinforces Tier 1; if < 0.3 we surface the discrepancy as a finding rather than a problem.

### 4.3 Task C — Mains rubric grading (Tier 1 only)

Predict `(score, strengths[], improvements[])` given `(question_text, answer_text, max_score)`. Gold = existing rows in `upscdev.evaluation_questions`.

| Metric | Definition | Library / call |
|---|---|---|
| **Quadratic Weighted Kappa (QWK)** vs `evaluation_questions.score` (binned to integer points) | The standard for ordinal automated essay scoring (ASAP/Kaggle precedent) | `sklearn.metrics.cohen_kappa_score(y_true, y_pred, weights='quadratic')` |
| **Score MAE** | `mean(abs(predicted_score − gold_score))` | `sklearn.metrics.mean_absolute_error` |
| **Score Spearman ρ** | rank correlation between predicted and gold scores | `scipy.stats.spearmanr` |
| **Score Pearson r** | linear correlation | `scipy.stats.pearsonr` |
| **Confusion matrix on score bands** | bin scores into {low ≤30%, mid 30-60%, high >60% of max_score}; report 3×3 confusion | `sklearn.metrics.confusion_matrix` |
| **Strengths token-F1** | tokenize predicted vs gold `strengths` JSON, F1 over lemma multiset | `spacy` lemmatizer + custom F1 |
| **Improvements token-F1** | same for `improvements` (nested by section — body/intro/conclusion in the schema) | Same |
| **Strengths sentence-level BERTScore-F1** | per-strength sentence semantic match | `bert_score` |
| **Score variance ratio** | `var(predicted_score) / var(gold_score)`. Detects mean-collapsed predictions where the model returns ~the dataset mean for every input regardless of student answer quality. 1.0 = matched dispersion; < 0.5 = collapsed. | `numpy.var` |
| **JSON schema validity rate** | Fraction of predictions that parse against the strict schema `{score: float, strengths: [str], improvements: {body: [str], intro: [str], conclusion: [str]}}`. | `jsonschema.validate` |
| **Strengths/Improvements item-count adherence** | `1 − abs(items(gen) − items(gold)) / items(gold)`, computed per list (strengths and improvements separately). UPSC mentor feedback has typical bullet counts (strengths 2-4, improvements 3-6); collapse to one item or balloon to ten is a failure mode. | Custom (list length) |

#### Tier 2 — Feedback Pedagogical Clarity rubric (LLM-judge, diagnostic only)

The above Tier 1 metrics measure *agreement with the gold rubric*. They don't measure whether the predicted feedback is **useful to a student**. We add a separate Tier 2 rubric scored on the predicted (strengths, improvements) JSON, 1–5 per axis:

| Axis | What the judge looks for |
|---|---|
| 1. **Actionability** | Can the student act on the feedback? (e.g. "add more examples from current affairs" beats "improve content depth") |
| 2. **Specificity** | Does the feedback cite *the specific part of the student's answer* it refers to, not generic platitudes? |
| 3. **Constructiveness** | Is the criticism framed as a path to improvement, not just a negative judgment? |
| 4. **UPSC-rubric fidelity** | Does the feedback use UPSC-specific terminology (directive words, paper-specific expectations, marker-rewarded structures)? |
| 5. **Coverage proportionality** | Does the feedback weight strengths and improvements roughly proportional to the student's actual gaps (not a one-size template)? |

This rubric is **diagnostic only** — never replaces the Tier-1 statistical comparison vs gold scores. It surfaces *whether the model is teaching well*, which is the prayas product question that pure rubric agreement cannot answer.

### 4.3a Why Task B does not get its own pedagogical-clarity axis

Mains generation is a *student-mimicking* task (write what an aspirant would write), not a *teaching* task. The existing G-Eval Content / Contextual / Analytical / Structural / Directive axes already cover what a mark-rewarded UPSC answer looks like. A "pedagogical clarity" axis on top would be either redundant with these or measure something UPSC graders don't actually reward.

### 4.4 Task E — Current Affairs synthesis (Tier 1 + bounded Tier 2)

Given `news_articles.text` and `date`, produce `prelims_info` and `mains_info`. Compared to gold columns in `news_articles`.

#### Tier 1

| Metric | Definition | Library / call |
|---|---|---|
| **ROUGE-L F1** | vs gold `prelimsInfo` and `mainsInfo` separately | `evaluate.load('rouge')` |
| **BERTScore-F1** | semantic overlap with gold | `bert_score` |
| **chrF++** | character-level overlap (catches Hindi gracefully) | `sacrebleu.corpus_chrf` |
| **Entity-F1 (named entities)** | NER over (gen ∪ source ∪ gold); report (a) gen-vs-gold F1, (b) **hallucination rate** = entities in gen not in source / entities in gen | `spacy` `en_core_web_trf` + `xx_ent_wiki_sm` |
| **Date exact-match F1** | regex date-extraction; F1 on date multisets between gen and source | Custom |
| **Numeric exact-match F1** | regex on `\d+(?:\.\d+)?%?\s?(crore|lakh|billion|million|%)?` | Custom |
| **Subject-tag exact-match** | does generated `prelims_subject` / `mains_subject` match gold? | `sklearn.metrics.accuracy_score` |
| **Topic-tag Jaccard** | Jaccard similarity between generated `mains_topics` set and gold set | Custom set-overlap |
| **SummaC-ZS faithfulness** | NLI-based consistency score — segments source into sentences and aggregates entailment scores | `from summac.model_summac import SummaCZS; SummaCZS(granularity='sentence', model_name='vitc').score(sources, generations)` |
| **AlignScore** (alternative NLI-faithfulness, ACL 2023) | sentence-pair alignment score against source | `from alignscore import AlignScore; scorer.score(...)` |
| **Coverage** | recall of source entities that appear in gen | Same NER pipeline as Entity-F1 |
| **Compression ratio compliance** | `len(gen_tokens) / len(source_tokens)`. UPSC mains-info gold typical range 0.25-0.45. Score = `1` if in [0.20, 0.50], else linear decay outside. | Custom: token count + clipping |
| **Glossary term recall** | Fraction of UPSC glossary terms (`prod-prayas-db.glossary.keyword`, 7,475 entries) appearing in gold mainsInfo that also appear in the generation. Uses prayas's own UPSC vocabulary. | Custom: set intersection vs glossary table |
| **Source citation density** | (named-entities + dates + numbers) per 100 generated words. UPSC-mainsInfo norm: ≥ 4 per 100. Below 2 = under-grounded; above 10 = entity-stuffing. | spaCy NER + custom counter |
| **Lead-100-word entity recall** | Recall of the source article's *headline-entity set* (people, organizations, dates in the first paragraph of the source) within the first 100 words of the generation. Catches "did the synthesis lead with the key facts?" | spaCy NER + first-100-word slice |
| **UPSC fact-lookup precision** | Same metric as Task B: for every `Article N` / `Section N` / `Year ####` / named-scheme reference in the generation, verify against `data/upsc_facts.json` (§4.0). | Custom: regex + JSON lookup |

#### Tier 2

| Metric | Why secondary | Library / call |
|---|---|---|
| **FactScore** | Decomposes generation into atomic claims via LLM, then NLI-checks each — uses an LLM in the pipeline | `factscore` package (Min et al. 2023) |

#### Tier 2 — Pedagogical Clarity rubric (LLM-judge, diagnostic only)

Task E output (`prelims_info`, `mains_info`) is **study material consumed by aspirants**, not student-mimicking like Task B — so the same teaching-vs-mimicking framing that excluded Task B from a pedagogical rubric *includes* Task E. 5-axis rubric, each scored 1–5 by `claude-sonnet-4-6`. Total = sum (5–25).

| Axis | What the judge looks for |
|---|---|
| 1. **Syllabus grounding** | Does the synthesis tie news facts to specific UPSC syllabus locations (e.g. "GS3 → Economy → Inclusive Growth" rather than free-floating commentary)? |
| 2. **Static-Dynamic bridge** | Does the dynamic news content connect to relevant *static* concepts (Acts, Articles, schemes, doctrines, predecessor events) — the thing UPSC mentors call "linkage"? |
| 3. **Multi-dimensional framing** | Does the Mains-info cover multiple dimensions (political / economic / social / environmental / international / ethical) appropriate to the topic? Single-dimension framings lose Mains marks. |
| 4. **Specificity** | Concrete named entities — committees, dates, scheme names, court cases, exact amounts — over vague references. |
| 5. **Mains-utility framing** | Does the synthesis surface *why this matters for UPSC* and what likely question angles look like? The `news_articles.mainsInfo` gold style explicitly does this; we measure whether candidates match. |

This rubric is **diagnostic only** — Tier-1 ROUGE/BERTScore/SummaC/Entity-F1 against gold `mainsInfo` are the headline metrics. Pedagogical Clarity tells us *whether the generated synthesis would actually help an aspirant* in a way surface metrics cannot.

Kendall's τ between Pedagogical-Clarity total and the Tier-1 BERTScore-F1 ranking is reported per condition. τ < 0.3 means "clarity" is capturing something faithfulness/overlap metrics miss — itself a finding for ed-tech use.

### 4.6 Task F — Prelims Explanation Generation (prayas production prompt)

**Capability test, not new science.** Same trained model checkpoints (C1a, C1b) and same comparator (C2, C3) — only the prompt scaffold changes to prayas's production "Prelims explanation generation" prompt (received 2026-05-26, stored at [`configs/prompts/prelims_explanation.md`](configs/prompts/prelims_explanation.md), loaded via `scripts/runners.get_production_prompt("F")`). Input includes the gold correct-option letter, so unlike Task A the model is not asked to *pick* the answer — it is asked to *write the explanation* given the answer. This is post-hoc rationalization (Wiegreffe & Marasovic, 2021), evaluated against a reference explanation. Bilingual (en + hi). Reuses the 800 Task-A eval items; no new eval set.

#### Tier 1 (primary)

The shape of this task is "long-form constrained generation vs a reference of 50-300 words" — close to Task A's explanation-quality subset. Metric inventory was [vetted against the 2024-2026 BEA / NLP4Edu literature](#references-research-vetting) (BEA 2025 survey on automated distractor evaluation; BEA-2025 didactic-clarity papers); no deterministic metric specific to "explanation-of-distractor completeness" or "didactic clarity" exists beyond the string-match heuristics already in our stack.

| Metric | Definition | Library / call |
|---|---|---|
| **BERTScore-F1** *(headline)* | Semantic similarity of generated explanation vs gold `explanation.english` / `explanation.hindi` | `bert_score` with `roberta-large` (en) / multilingual fallback (hi) |
| **ROUGE-L F1** | Longest-common-subsequence overlap with gold | `rouge_score.RougeScorer(['rougeL'])` |
| **chrF++** | Character n-gram F1 (essential for Devanagari/Hindi) | `sacrebleu.sentence_chrf(... word_order=2)` |
| **Entity-F1** | spaCy NER entity-set F1 between generated and gold explanation | `en_core_web_sm` |
| **Distractor coverage** | Fraction of the three wrong options each explicitly addressed: option letter present + ≥1 distinctive content token from that option's text | Custom string-match (lemma + entity overlap is a v2 refinement) |
| **Reasoning-step density** | Discourse-marker count per 100 generated words; bilingual marker list (`because`, `therefore`, `however`, `if/then`, `क्योंकि`, `इसलिए`, …) | Custom regex |
| **Article/scheme citation accuracy** | For every `Article N` / `Schedule N` / named-Act/scheme reference, fraction resolved in [`data/upsc_facts.json` §4.0](#40-shared-utility-) | Custom — same logic as Task A |
| **UPSC fact-lookup precision** | Same lookup function, applied as precision across all factual references in the explanation | Custom |
| **Word-count adherence** | `1 − abs(words(gen) − target) / target`, clipped [0,1]. Target derived per-question from gold-explanation length (mean per stratum) until prayas's production prompt specifies a target | Custom |
| **Hindi code-mixing rate** | For Hindi rows: fraction of letter chars NOT in the Devanagari Unicode block | `unicodedata` |

#### Metrics **not** included for Task F (with reasoning)

- **Accuracy / UPSC negative-marking score / Brier / ECE / Position bias** — N/A: the model is not asked to pick an answer in this task.
- **Sentence-length variance** — *dropped on research grounds.* The BEA-2025 didactic-clarity literature does not use it; it is noisy at 50-300 word lengths and is dominated by MATTR + Reasoning-step density.
- **BARTScore / BLEURT-Extended / MAUVE / QAFactEval** — checked and rejected. BARTScore is redundant with BERTScore at this length; BLEURT-Extended is WMT-domain; MAUVE is distributional (no per-row score); QA-style faithfulness metrics need a source document Task F does not have.

#### Tier 2 (deferred — Path C)

Pedagogical-Clarity LLM-judge rubric — the same 5-axis rubric specified for Task A in §4.1 applies *more naturally* here because the model isn't co-confounded with answer correctness. Deferred for v1 alongside all other Tier-2 metrics.

### 4.7 Task G — Mains Model-Answer Generation (prayas production prompt)

**Capability test, not new science.** Same trained model checkpoints — only the prompt scaffold changes to prayas's production "Mains model-answer generation" prompt (received 2026-05-26, stored at [`configs/prompts/mains_model_answer.md`](configs/prompts/mains_model_answer.md), ~21 KB / ~5400 tokens — DSL L1-L4 layers + banned-word/phrase lists + mandatory diagram + R-D-S-C protocol; loaded via `scripts/runners.get_production_prompt("G")`). Reuses the 400 Task-B eval items; no new eval set. The headline question is: *does the production prompt close the gap to gold model-answer style vs the generic Task-B prompt?* Tested as a paired comparison between Task B and Task G output **on the same model + same eval items**, with the prompt as the lone variable. Note: the production prompt is markedly more constrained than the simpler Task-B training scaffold; format-compliance is therefore a meaningful Tier-1 metric on Task G — the FT model was not trained on this DSL.

#### Tier 1 (primary)

All 14 of [§4.2 Task B's Tier-1 metrics](#42-task-b--mains-generation-tier-1-quantitative--tier-2-llm-judge-diagnostic) carry over identically — BERTScore-F1 (headline), ROUGE-L, chrF++, word/sentence/paragraph-count adherence, Entity-F1, date/number F1, Hindi code-mixing, MATTR-100, Flesch-Kincaid grade, 4-gram repetition, UPSC fact-lookup precision. **Plus** one addition vetted against [NAACL-Short 2024](https://aclanthology.org/2024.naacl-short.9/) for long-form structural coherence, and one engineered metric for UPSC multi-dimensional framing:

| Metric | Definition | Library / call |
|---|---|---|
| **Dimension-keyword coverage** | UPSC Mains rewards covering multiple "dimensions" (political / economic / social / environmental / ethical / international). A static PESEE-style lexicon (committed to repo as `data/dimension_keywords.json`) maps each of these six dimensions to ~30 keywords. Metric = count of distinct dimensions touched in the generation / count touched in the gold. | Custom — set lookup against lexicon. Engineered metric, not from published literature; documented as such. |
| **Directive-conditioned discourse density** *(exploratory)* | Per question's directive verb (`analyze` / `examine` / `evaluate` / `discuss` / `comment` / `critically`), count of *expected* discourse markers per 100 generated words. `analyze` and `evaluate` should produce higher causal/contrastive marker density (`because`, `however`, `whereas`); `describe` should not. Score = density(generated) / density(gold) per question. | Custom regex; engineered proxy, not a published metric. Reported alongside Tier-1 but flagged as *exploratory*. |

#### Metrics **deferred to v2** (with reasoning)

- **PDD (Positional Discourse Divergence)**, NAACL-Short 2024 (arXiv:2402.10175) — the strongest published deterministic long-form-coherence metric, beats DiscoScore and BARTScore by ~10 correlation points at system level. Adding it requires a discourse parser (RST or PDTB) which is non-trivial dep work; defer to v2 once headline metrics confirm the comparison is interesting.
- **DiscoScore** (EACL 2023) — superseded by PDD for our case.

#### Metrics **not** included for Task G (with reasoning)

- **Tier-2 G-Eval rubric** — deferred Path C alongside Tasks A/B/C/E.
- **BARTScore** — research-vetted as redundant with BERTScore at our scoring scale.
- **METEOR** — wordnet dep adds friction and the signal is dominated by BERTScore + ROUGE-L.

### References — research vetting

Metric choices in §4.6 and §4.7 were vetted against the 2024-2026 educational-NLP literature on 2026-05-19. Key sources:

- [BEA 2025 Proceedings](https://aclanthology.org/volumes/2025.bea-1/) — confirmed no deterministic alternative to string-match for distractor-explanation completeness or didactic clarity.
- [Bitew et al., "A Survey on Automated Distractor Evaluation in MCQs", BEA 2025](https://aclanthology.org/2025.bea-1.5.pdf) — surveys distractor *generation* metrics; nothing applicable to *explanation-of-distractor* completeness beyond what we have.
- [Wang et al., "PDD: Positional Discourse Coherence", NAACL-Short 2024](https://aclanthology.org/2024.naacl-short.9/) — informs the §4.7 v2 deferral.
- [Zhao & Strube, "DiscoScore", EACL 2023](https://arxiv.org/pdf/2201.11176) — informs the §4.7 v2 deferral.
- Confirmed-rejected for v1: BARTScore (NeurIPS 2021), BLEURT-Extended, COMET, MAUVE, QAFactEval, FactCC, AUF — none clear the cost/benefit bar given the Path-C constraint and the task structures of F and G.

### 4.8 Universal metrics (every condition, every task)

| Metric | What |
|---|---|
| **Latency p50 / p95 / p99** | Wall-clock from request to last token |
| **TTFT** | Time-to-first-token |
| **Tokens-per-second** | Throughput |
| **Input / output token counts** | For cost comparison |
| **$ cost per query** | Gemini: published per-1k-token rate × tokens. FT-SLM (local): zero marginal $; report energy proxy = (wall-clock × M5 sustained-power-draw) |
| **Format-validity rate** | did the model produce parseable structured output for the task? |

---

## 5. Implementation contract

### 5.1 Python dependencies (pinned)

Authoritative list lives in `requirements.txt`. Snapshot at the time of this revision (v1 PyPI-installable subset only — git-only deferrals listed in §4):

```
# Scoring (Tier 1)
bert-score==0.3.13            # semantic similarity (BERTScore-F1)
rouge-score==0.1.2            # ROUGE-L
sacrebleu==2.5.1              # chrF++
spacy==3.8.2 + en_core_web_sm==3.8.0   # NER for Entity-F1, lemmas for Task-C token-F1
nltk==3.9.4                   # ngram counting, fallback sent_tokenize (not used yet)
textstat==0.7.13              # Flesch-Kincaid; 0.7.4 had a pkg_resources import broken under setuptools 70+
torchmetrics==1.5.2           # ECE / Brier helpers
scikit-learn==1.6.0           # QWK, MAE, confusion matrix
scipy==1.15.0                 # paired tests, χ², binomtest, Pearson, Spearman
statsmodels==0.14.4           # BH-FDR multiple-comparison correction (§6.2)
jsonschema==4.23.0            # strict schema validation (Task C)
evaluate==0.4.5               # ROUGE alt loader, kept for parity with prior runs

# Inference
mlx-lm==0.31.3                # Apple Silicon LLM inference + LoRA FT — bumped from 0.21.5 to add gemma4 + qwen3_5 model classes
google-genai==1.18.0          # Gemini 3 family client
anthropic==0.45.0             # currently unused (Tier-2 deferred, Path C)
tenacity==9.0.0               # retry-with-backoff decorator (inference plane)

# Data plane
psycopg2-binary==2.9.11       # only used inside scripts/snapshot_to_local.py
pandas==2.2.3
pyarrow==18.1.0
numpy==2.1.3                  # variance / TTR / ngram counters

# Dashboard (Phase 7 — not yet built)
streamlit==1.42.0

# Test runner
pytest==8.3.4
```

**Git-only deps not installed for v1** (re-enable in v2): `bleurt-pytorch`, `summac==0.0.6`, `alignscore==0.1.3`, `factscore==0.1.7`. Their respective metrics are listed as deferred in §4.

### 5.2 Scorer-model checkpoints (pinned)

| Purpose | Model | Status |
|---|---|---|
| BERTScore (English) | `roberta-large` (bert-score `lang='en'` default) | **active** — first-run weights cached locally |
| BERTScore (Hindi) | `bert-base-multilingual-cased` (bert-score `lang='hi'` default) | **active** |
| spaCy NER + lemmatizer | `en_core_web_sm` 3.8.0 | **active** — installed via wheel URL |
| Base-model loader (LoRA FT + inference) | `mlx-community/Qwen3.5-4B-MLX-4bit`, plus a text-only Gemma-4-E4B MLX repo (TBD — see §2.0a) | Qwen active under mlx-lm 0.31.3; Gemma blocked |
| BLEURT-20 / SummaC NLI / AlignScore / FactScore / `en_core_web_trf` / `xx_ent_wiki_sm` | (per original design) | **v2** — deferred with their metric families |
| Perplexity scorer (`google/gemma-4-E4B` base) | (per original design) | **v2** — deferred with the perplexity metric |
| LLM-judge | `claude-sonnet-4-6` (Anthropic) | **v2** — Tier-2 deferred under Path C |

Reproducibility for v1's active scorers comes from the pinned library versions (bert-score 0.3.13, spaCy 3.8.2 + en_core_web_sm 3.8.0) plus a seeded RNG (`numpy.random.default_rng(20260514)`) on the bootstrap. A formal `models/lockfile.json` will be added in v2 alongside the BLEURT / SummaC / AlignScore checkpoints, which need explicit version pinning to remain comparable across reruns.

### 5.3 Per-row result schema (Parquet)

The scoring pipeline writes three Parquet files under `results/`, each with its own schema. (Aggregations are computed offline in `scripts/aggregate.py` and `scripts/test_hypotheses.py`; the Streamlit dashboard — Phase 7 — reads the aggregated outputs.)

**`results/predictions.parquet`** — one row per `(condition, question_id)`, written by `scripts/run_inference.py`:

```
run_id              str   -- e.g. '20260519'
condition           str   -- 'C1a' | 'C1b' | 'C2' | 'C3'
model_version       str   -- e.g. 'gemma-FT@gemma4-e4b-upsc-v1', 'gemini-zs@gemini-3-flash'
task                str   -- 'A' | 'B' | 'C' | 'E' | 'F' | 'G'
question_id         str
language            str
paper               str
subject             str
stratum_key         str
input_text          str
gold_payload        str   -- JSON-serialized
prediction          str   -- JSON-serialized: {answer, explanation, confidence, ...} for A; task-specific shape otherwise
raw_output          str
latency_ms          float
ttft_ms             float
input_tokens        int
output_tokens       int
created_at          str   -- ISO-8601 UTC
```

**`results/scores_tier1.parquet`** — one row per `(condition, question_id)`, written by `scripts/score_tier1.py`. Identifier columns mirror `predictions.parquet`; per-task metric columns are populated only for rows where the task matches. Each Tier-1 scalar is its own column (~55 columns total). Aggregation-level metrics (ECE, Brier-skill, QWK, position-bias, bootstrap CIs) are NOT here — they're in `aggregate.parquet`.

**`results/aggregate.parquet`** — one row per `(condition, task, language, metric)`, written by `scripts/aggregate.py`:

```
condition  str
task       str    -- 'A' | 'B' | 'C' | 'E' | 'F' | 'G'
language   str    -- 'en' | 'hi' | 'all'
metric     str
n          int    -- non-NaN count behind the mean
mean       float
ci_lo      float  -- percentile bootstrap, N = 1000, seed 20260514
ci_hi      float
```

Plus `results/aggregate.extras.json` for non-tabular breakdowns — bilingual-delta paired t, silly-mistake breakdown, Task-C rank correlations + confusion matrix.

**`results/hypothesis_tests.parquet`** — one row per `(task, metric, condition_a, condition_b)` from `scripts/test_hypotheses.py`, with paired-t / Wilcoxon / bootstrap-diff-CI / BH-FDR-adjusted p-value / `significant_fdr` boolean.

---

## 6. Statistical methodology

### 6.1 Headline comparisons

For each (task, metric) we compare conditions pairwise across the four conditions (C1a, C1b, C2, C3) — six pairs per (task, metric).

| Metric type | Test |
|---|---|
| Binary outcome (accuracy, format-validity, refusal) | **Paired t-test** on the (0/1) per-row indicator + **Wilcoxon signed-rank** as the non-parametric companion |
| Continuous, paired (BERTScore, ROUGE, chrF, MAE, …) | **Paired t-test** + **Wilcoxon signed-rank**, plus **percentile bootstrap** (`numpy.random.default_rng(20260514)`, `n_resamples=1000`) for the 95 % CI on the mean difference |
| Calibration (ECE, Brier Skill Score) | Computed at the (condition, task) aggregate; not paired per-row |
| Rank (Spearman ρ, Pearson r, QWK) | Computed at the (condition, task) aggregate |

We do **not** use Welch's t-test on per-row metrics — long-form scoring distributions are heavily skewed and not Gaussian; the paired-t reports a moment-statistic with Wilcoxon as the distribution-free check.

**Bootstrap N revised from 10 000 → 1 000.** At our scoring scale (≤ 800 paired observations per task, ≥ 6 pairs × 6 tasks × ~50 metrics = ~1 800 tests) the additional CI precision from 10 000 resamples is negligible while wall-clock grows linearly. 1 000 resamples is the standard for screening across a large test family.

### 6.2 Multiple-comparison correction

Across the ~1 800 (task × metric × condition-pair) cells we apply **Benjamini–Hochberg FDR at q = 0.05** (`statsmodels.stats.multitest.multipletests(method='fdr_bh')`). BH is the standard for screening many genuinely-distinct hypotheses; Bonferroni at this N would demand p ≤ 0.0000278 per test and obliterate real effects. The output column `significant_fdr` in `results/hypothesis_tests.parquet` is what report tables read. The orientation is flipped before testing for metrics where smaller is better (`brier_loss`, `format_fail`, `score_abs_err`, `hallucination_rate`, `ngram4_repetition_rate`) so the sign of the reported difference always means "ca beats cb."

### 6.3 Per-stratum reporting

Headline tables report the overall metric and one row per stratum:

- Per UPSC paper (GS1 / GS2 / GS3 / GS4 / Essay / **CSAT** — added as a first-class stratum, see §3.1)
- Per subject (top-5 by volume)
- Per language (en vs hi) — Tasks A, B, F
- Per difficulty band (Task A) or score band (Task C)
- Per silly-mistake flag (Task A `silly_mistake_prone=True` subset)

---

## 7. External anchors

Alongside the in-house eval, we report all four conditions on:

- **MILU** (AI4Bharat, NAACL 2025) — 500-Q sample, English-only stratum to keep cost down; provides an external Indic baseline.
- **MMLU-Pro Indian-Polity / Economy / Geography subsets** — links our numbers to global LLM literature.

These external numbers are not used for primary claims; they situate the in-house metric movements.

---

## 8. Limitations (disclosed in any external write-up)

1. **Task C gold was itself LLM-generated.** Mitigation: 50 rows hand-spot-checked by a prayas mentor; we report κ between our predictions and the mentor for that subset alongside κ-vs-original-gold.
2. **Scorer-model dependence.** BERTScore / BLEURT / SummaC numbers depend on the underlying checkpoint. Pinning + reporting checkpoint SHAs makes the comparison reproducible but not "absolute."
3. **Eval set is internal.** Leakage with prior Sarvam / Qwen / Gemini training data is unknown. The held-out invariant in §3.2 protects against our own FT-data leakage only.
4. **Stratum sizes are modest** (often ≈50 items per cell). Per-subject claims are suggestive; the overall and per-paper claims are well-powered.
5. **Verbal confidence is a proxy** for true model uncertainty (Gemini's logits are unavailable via API). Both conditions are tested the same way — comparison is fair, but absolute ECE is approximate.

---

## 9. Out of scope for v1 (revisit in v2)

- Personalized tutoring (T2 — Q + student_memory → A)
- Interview / DAF (Task D)
- Multi-turn conversational eval
- Cost-adjusted quality Pareto front
- Live A/B with real prayas students
- IRT-based judge calibration

---

## 10. Acceptance criteria for this design

This design is "good enough to build against" if all of these hold:

- [ ] Every metric in §4 has a library + call cited (§5.1).
- [ ] Every scorer model is pinned (§5.2).
- [ ] No metric in Tier 1 depends on an LLM-judge's free-text decision.
- [ ] Eval-set construction is deterministic (§3.2 seed).
- [ ] Stratification is concrete (no "TBD" cells in §3.1).
- [ ] Statistical tests are named per metric type (§6.1).
- [ ] **A2 — Pre-FT Hindi-capability triage:** before FT begins, run a 50-item Hindi MCQ probe (sampled from `upscdev.upsc_prelims_ai_generated_que.question_hindi`, seed 20260514) on **both** `google/gemma-4-E4B-it` and `Qwen/Qwen3.5-4B`. Pass criterion: **one-sided binomial test** against the random-chance baseline (H0: accuracy = 0.25, H1: accuracy > 0.25) at α = 0.05. Model passes iff p-value < 0.05. At n=50 the critical value is **k = 18 (36% accuracy)** — i.e. P(X ≥ 18 \| n=50, p=0.25) ≈ 0.045. For any model that fails (cannot reject H0), the Hindi-stratum results are reported only post-FT, as a separate finding, never folded into the bilingual aggregate. Output: `results/pre_ft_hindi_probe.parquet` with one row per (model, item); the gate script (`scripts/gate_hindi.py`) computes the p-value via `scipy.stats.binomtest`.

If any box is unchecked, the design is not ready and the build does not start.
