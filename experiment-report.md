# Experiment Report — UPSC SLM vs Frontier-API Baseline

| Field | Value |
|---|---|
| **Title** | Quantifying the performance of a UPSC-fine-tuned 4.5B-effective open-source SLM against `gemini-3-flash` on bilingual Indian Civil Services Examination tasks |
| **Pre-registration date** | 2026-05-14 |
| **Principal investigator** | Yeeshan — Data Scientist, prayas.ai (irshad@prayas.ai) |
| **Status** | Pre-registered. Setup and procedure are finalized; outcomes and results are blanks awaiting execution. |
| **Version** | v1 (a v2 — personalized tutoring + interview task — is out of scope; see [§11](#11-out-of-scope)). |
| **Document hash committed at FT-start time** | (filled at run time; SHA-256 of this document and `eval-design.md`) |

Pre-registration format: hypotheses and analysis plans committed *before* execution. Without pre-registration, p-hacking across ~40 (task × metric × stratum) cells is essentially unbounded.

For non-technical context, see [`project-brief.md`](project-brief.md).

---

## Revisions since pre-registration (2026-05-18 — 2026-05-19)

This block records decisions taken **after** the original pre-registration. Hypotheses, verdict criteria, and primary statistical methodology are unchanged; the changes below affect implementation, scope, and metric inventory.

1. **Path A2 — unified JSON I/O end-to-end.** Both the FT corpus and the inference prompts now use a single JSON schema per task: A → `{answer, explanation}`, B → `{answer}`, C → `{score, strengths, improvements}`, E → `{prelims_info, mains_info}`. The pre-registered Task-A "three-pass" protocol (Pass 1 letter, Pass 2 confidence, Pass 3 explanation) collapses to two passes — Pass 1 now returns answer + explanation in a single JSON object; Pass 2 (verbal confidence elicitation) is unchanged. The model now sees the same instruction string at train and inference time, eliminating a train-test distribution shift in the prompt.

2. **CSAT added as a first-class Task-A stratum.** `paper` lives on `prod.learning_items`, not on `mcqs`; the snapshot SQL was updated to LEFT JOIN them. `mcqs` is now a third Task-A FT source alongside `prelims_pyq_questions` and `upsc_prelims_ai_generated_que`, filtered to `paper ∈ {gs1, csat}`. CSAT (number puzzles / reading comprehension / deduction) is content-distinct from GS-I and is stratified accordingly.

3. **Tier-2 LLM-judge deferred for v1 (Path C).** All LLM-judge / G-Eval / Pedagogical-Clarity rubric metrics are excluded from v1. Tier-1 has ~45 deterministic metrics across the four tasks and is sufficient to evaluate the headline hypotheses. The Anthropic key is unused; `claude-sonnet-4-6` judge inference is not executed. The Tier-2 tables in [§6.5](#65-tier-2-llm-judge-diagnostic-only--not-headline) are retained for reference but not populated.

4. **Tier-1 metric inventory finalized.** Five originally listed Tier-1 metrics are deferred — `BLEURT-20` (`bleurt-pytorch` is git-only), `SummaC-ZS` / `AlignScore` / `FactScore` (Task E faithfulness — all git-only), generation perplexity (heavy base-model load, signal dominated by BERTScore), `METEOR` (NLTK wordnet corpus dep, redundant with BERTScore + ROUGE-L), and Glossary-term recall (would require `prod.glossary` in the local snapshot). The remaining ~45 implemented metrics are listed and explained in [§6A — Tier-1 metric glossary](#6a-tier-1-metric-glossary).

5. **`mlx-lm` bumped 0.21.5 → 0.31.3.** The original pin predated both `gemma4` and `qwen3_5` model classes. 0.31.3 (April 2026) is the smallest tested version that loads both candidates. The `load / generate / stream_generate / make_sampler / TokenizerWrapper.apply_chat_template` API surface used by `runners.py`, `validate_adapter.py`, and `run_hindi_probe.py` is unchanged across the bump.

6. **A2 Hindi probe — first run, base models pre-FT.** `Qwen/Qwen3.5-4B` answered 14/50 = **28.0 %**, one-sided binomial p = 0.363 — **fails** the α=0.05 gate. `deadbydawn101/gemma-4-E4B-mlx-4bit` raised `ValueError: Received 126 parameters not in model`; the community 4-bit checkpoint is a *multimodal* gemma4 variant and is structurally incompatible with mlx-lm 0.31.3's text-only `gemma4` class. A text-only MLX-format Gemma 4 E4B repo must be located before C1a can run.

7. **Local SQLite snapshot enforces a single prod-DB boundary.** All scripts that previously connected to Postgres now read from `data/prayas_local.sqlite`; only `scripts/snapshot_to_local.py` uses the remote DSN, and it does so via read-only `SELECT` statements. This is the per-instance-approval standing rule made architectural.

---

## 1. Aim

### 1.1 Research question

Does task-specific fine-tuning of a ~4B-parameter open-source language model produce equivalent or better performance on UPSC Civil Services Examination tasks compared to a frontier closed model accessed via API, while running entirely on a 16 GB consumer laptop? Does this finding depend on the choice of base SLM family?

### 1.2 Hypotheses

Four conditions in v1: **C1a** = `google/gemma-4-E4B-it` + LoRA (UPSC multi-task adapter), **C1b** = `Qwen/Qwen3.5-4B` + LoRA (same recipe), **C2** = `gemini-3-flash` zero-shot, **C3** = `gemini-3-flash` few-shot.

Six tasks split into two groups:
- **Core comparison tasks (4):** A Prelims MCQ, B Mains generation, C Mains rubric grading, E Current Affairs synthesis — the hypothesis-test family.
- **Production-capability tests (2):** F Prelims Explanation Generation, G Mains Model-Answer Generation — same model checkpoints, prayas's production prompt scaffold. Reuse Task A and Task B eval items respectively; no new eval set. Reported as a separate verdict on "does the FT-SLM drop into prayas's existing prompts?"

For each task × Tier-1 metric in [`eval-design.md §4`](eval-design.md):

| Hypothesis | H0 | H1 |
|---|---|---|
| H1: Gemma-FT vs Qwen-FT | C1a = C1b | C1a ≠ C1b (direction reported) |
| H2: Gemma-FT vs zero-shot frontier | C1a = C2 | C1a ≠ C2 |
| H3: Gemma-FT vs few-shot frontier | C1a = C3 | C1a ≠ C3 |
| H4: Qwen-FT vs zero-shot frontier | C1b = C2 | C1b ≠ C2 |
| H5: Qwen-FT vs few-shot frontier | C1b = C3 | C1b ≠ C3 |
| H6: Few-shot vs zero-shot | C2 = C3 | C2 ≠ C3 |

### 1.3 Verdict criteria

The "champion FT-SLM" for each task is whichever of C1a or C1b achieves the better primary-metric value on that task. Verdict is judged on the per-task champion vs C3:

- **Strong win for the FT approach:** the per-task champion beats C3 on ≥3 of 4 tasks at the primary metric, BH-FDR-corrected, with bootstrap 95% CIs excluding zero.
- **Non-inferiority:** the per-task champion is within 5pp (or 0.05 absolute on [0,1] metrics) of C3 on ≥3 of 4 tasks.
- **Loss:** the per-task champion is worse than C3 by > 5pp on ≥3 of 4 tasks at significance.

Secondary verdict — **architecture/family matters**: H1 (C1a vs C1b) is significant on ≥2 of 4 core tasks. Tells us whether the result is portable across base models or sensitive to which one we picked.

Tertiary verdict — **production drop-in viability** (Tasks F + G): the per-task champion's headline metric (BERTScore-F1) on F and G is **within 0.05 absolute** of the corresponding Task A explanation-quality / Task B answer metric. If so, the production prompts integrate cleanly. If F or G *exceeds* their Task A/B counterparts, the prayas-house-style prompt actively improves output. If F or G *trails* by > 0.05, the production prompt would need revising before drop-in.

Per-task directional outcomes are themselves informative regardless of the aggregate.

---

## 2. Background

UPSC content has three properties that pressure general LLMs:

1. **Bilingual at question-paper level** — every paper is officially issued in English and Hindi.
2. **Culturally specific knowledge** — MILU (AI4Bharat, NAACL 2025) shows frontier models score lowest on Arts & Humanities and Law & Governance, the bulk of UPSC GS-I/II/IV.
3. **Rubric-graded long-form answers** with strict word counts, directive-word semantics, and factual specificity (Article numbers, Act dates, scheme names).

No UPSC-specific LLM benchmark exists in the literature as of 2026-05-14. Closest priors: MILU (Indic-broad), JEEBench (engineering reasoning). Neither targets civil-services reasoning.

---

## 3. Setup

### 3.1 Hardware

| Role | Device | Specs |
|---|---|---|
| FT compute (primary) | Apple M-series, **24 GB unified memory** | MLX framework; macOS. Requires `sudo sysctl iogpu.wired_limit_mb=21504` plus the in-process `mx.set_wired_limit / mx.set_cache_limit / mx.clear_cache` block in `scripts/run_ft.py` (open mlx-lm issues #828 + #1185 — val→train transition OOMs on the default Metal cap without these). The original plan called for FT on the 16 GB M5; OOM diagnosis on 2026-05-21 escalated to a 24 GB device for both adapters. ~10-14 h per adapter. |
| FT compute (backup) | Kaggle Notebooks | T4 (16 GB VRAM), 30 GPU-hours/week, 9-hour session limit. Would require switching from MLX-LM to PyTorch + `peft` + `bitsandbytes` (pipeline change is non-trivial; only used if 24 GB M-series unavailable). |
| Inference (FT-SLMs C1a, C1b) | Mac M5 16 GB (default-cap macOS Metal) | MLX-LM, 4-bit quantization; ~5 GB resident for Gemma-4-E4B-it, ~3 GB for Qwen3.5-4B. Inference fits within the default Metal cap; only FT (with backward + activation memory) needed the 24 GB device. |
| Inference (frontier baselines) | Google API | `gemini-3-flash` via Vertex AI / Gemini API |
| Tier-2 LLM-judge inference | (Deferred — Path C, see [Revisions item 3](#revisions-since-pre-registration-2026-05-18--2026-05-19)) | `claude-sonnet-4-6` (Anthropic) — judge inference not executed for v1; tables retained for reference only. |
| Dashboard host | Same Mac M5 | Streamlit (Phase 7 — not yet built). |

### 3.2 Software stack

Pinned in [`requirements.txt`](requirements.txt). Key choices for v1:

- **Base models:** `Qwen/Qwen3.5-4B` (local MLX 4-bit via `mlx-community/Qwen3.5-4B-MLX-4bit`) — **confirmed loadable** under mlx-lm 0.31.3. `google/gemma-4-E4B-it` is **blocked** until a text-only MLX repo replaces the multimodal `deadbydawn101/gemma-4-E4B-mlx-4bit` (see [Revisions item 6](#revisions-since-pre-registration-2026-05-18--2026-05-19)).
- **MLX framework:** `mlx-lm==0.31.3`. The LoRA recipe in `configs/lora.yaml` is invoked via subprocess `python -m mlx_lm.lora`; inference uses `mlx_lm.load(..., adapter_path=...)` + `mlx_lm.stream_generate(...)`.
- **Metric libraries (Tier 1, v1 only):** `bert-score==0.3.13` (semantic similarity), `rouge-score==0.1.2` (ROUGE-L), `sacrebleu==2.5.1` (chrF++), `spacy==3.8.2` + `en_core_web_sm==3.8.0` (NER for Entity-F1, lemmas for Task-C token-F1), `torchmetrics==1.5.2` + `scipy==1.15.0` (ECE / Brier / paired tests), `scikit-learn==1.6.0` (QWK / MAE / confusion matrix), `textstat==0.7.13` (Flesch-Kincaid), `jsonschema==4.23.0` (Task-C schema validity).
- **Deferred metric libraries (v1 → v2):** `bleurt-pytorch` (BLEURT-20), `summac` (Task-E faithfulness), `alignscore`, `factscore` — all distributed git-only; not installed.
- **Statistical analysis:** `statsmodels.stats.multitest.multipletests(method='fdr_bh')` with q = 0.05. Percentile-bootstrap CIs with `numpy.random.default_rng(20260514)` and `n_resamples=1000` (a 10× reduction from the original 10 000 — sufficient for 95 % CI width at the per-row scoring scale, and faster on the ≥50-column scoring matrix).
- **Data plane:** `psycopg2-binary==2.9.11` (only inside `scripts/snapshot_to_local.py`); all downstream scripts read from `data/prayas_local.sqlite` via `scripts/local_db.py`. `pandas==2.2.3` + `pyarrow==18.1.0` for the Parquet result store.
- **Dashboard:** `streamlit==1.42.0` (Phase 7 — not yet built).

### 3.3 Data

#### 3.3.1 Sources (internal)

Three PostgreSQL 17.6 databases on prayas.ai infrastructure, inventoried 2026-05-14:

| Database | Host | Size | Role |
|---|---|---:|---|
| `upscdev` | RDS `prayas-db.cbii0i4yge4n.ap-south-1.rds.amazonaws.com:5432` | 362 MB / 89 tables | Curated UPSC content + rubric data |
| `prod-prayas-db` | Same RDS | 820 MB / 57 tables | Production app DB (read-only SELECTs) |
| `prayas` (app_dev) | `13.203.24.116:6001` | 222 MB / 74 tables | App dev DB with chatbot tables (used for chat exemplars only) |

#### 3.3.2 Eval-set construction

The eval set is **frozen** prior to FT — committed as `data/eval_set.parquet` with a SHA-256 hash recorded in this document. The freezer script (`scripts/freeze_eval_set.py`) uses `random.Random(seed=20260514)` for deterministic sampling.

| Task | n | Source tables | Stratification |
|---|---:|---|---|
| **A — Prelims MCQ** | 800 | `upscdev.prelims_pyq_questions` (454 rows in eval) + `upscdev.upsc_prelims_ai_generated_que` (256) + `prod-prayas-db.mcqs` ⨝ `learning_items` (90; GS1=70 + CSAT=20) | Paper (GS-I, CSAT) × Subject × `silly_mistake_prone` × `language` (en / hi) |
| **B — Mains generation** | 400 | `upscdev.pyqs` | Paper (GS1, GS2, GS3, GS4, Essay) × Subject × word-count band (150 / 250 / essay) × language |
| **C — Mains rubric grading** | 500 | `upscdev.evaluation_questions` | Subject × score-band (low ≤30 % / mid 30-60 % / high >60 % of `max_score`) |
| **E — Current Affairs synthesis** | 300 | `prod-prayas-db.news_articles` (`date ≤ 2026-04-30`) | Month × `newsThemeId` |
| **F — Prelims Explanation Generation** *(prod-prompt capability test)* | 800 *(reused Task-A items)* | Same as Task A; gold correct-letter is now part of the input | Same as Task A |
| **G — Mains Model-Answer Generation** *(prod-prompt capability test)* | 400 *(reused Task-B items)* | Same as Task B; prayas's production prompt scaffold replaces the generic Task-B scaffold | Same as Task B |

Total eval-set size: **2,000 unique items** = 800 (A / F) + 400 (B / G) + 500 (C) + 300 (E). Same items used across all four conditions (paired design). Tasks F and G **re-use** the same 800 + 400 items rather than expanding the set — the variation is the prompt scaffold, not the questions. Total per-condition prediction count: 800 + 400 + 500 + 300 + 800 + 400 = **3,200 rows × 4 conditions = 12,800 predictions** (up from 8,000 before F/G; ~+50 % Gemini-API cost on C2 + C3). Current artifact: `data/eval_set.parquet`, SHA-256 `e2b62a3f…`. 20 CSAT items are held out in the eval set as a first-class gate slice for the new stratum.

#### 3.3.3 FT training corpus

Drawn from the same Postgres tables, **explicitly excluding any `question_id` in `eval_set.parquet`**. The exclusion is enforced as a CI assertion in `scripts/build_ft_corpus.py` that fails the build if any eval-set ID appears in the FT corpus.

| Task adapter | FT examples (actual) | Source |
|---|---:|---|
| Prelims MCQ → answer + explanation | **26 638** | `prelims_pyq_questions` (6 770) + `upsc_prelims_ai_generated_que` (8 959) + `mcqs ⨝ learning_items` (10 909 — GS1 = 8 548 + CSAT = 2 361) − eval IDs |
| Mains generation → model answer | **2 608** | `pyqs` − eval IDs |
| Mains rubric grading → (score, strengths, improvements) | **9 600** | `evaluation_questions` − eval IDs |
| Current Affairs synthesis → (prelims_info, mains_info) | **2 903** | `news_articles` (`date < 2026-04-30`) − eval IDs |

Single multi-task corpus, **41 749 supervised pairs total**. Each pair is a `[TASK=X] {instruction} | {JSON input}` → `{JSON output}` triple — same shape the model sees at inference under Path A2. Current artifact: `data/ft_corpus.parquet`, SHA-256 `d57be52c…`. The CI leakage assertion `eval ∩ ft = ∅` passes on this artifact. The corpus will train both C1a (Gemma) and C1b (Qwen) adapters with an identical recipe — only the base model differs, isolating architecture/pretraining as the variable.

External corpora (NCERTs, Drishti / Vision / Insights compilations, official syllabus PDFs) are **out of scope for v1** — see [§11](#11-out-of-scope) — to keep the experimental signal traceable to internal data.

### 3.4 Bias control

Pre-registered to prevent post-hoc rationalization:

| Risk | Mitigation |
|---|---|
| Eval leakage into FT | `question_id` set-difference enforced in CI; eval-set frozen with SHA-256. |
| Judge intra-family bias | LLM judge is `claude-sonnet-4-6` (Anthropic). Candidate (Gemma) and comparator (Gemini) are both Google; judge is non-Google to break the family. |
| Hindi-stratum unfair comparison | **A2 protocol** ([`eval-design.md §10`](eval-design.md)) — Hindi-MCQ probe on the base model pre-FT. If pre-FT Hindi accuracy < 30%, Hindi stratum is reported as a separate finding, not folded into the bilingual aggregate. |
| Cherry-picked metric on which to declare a win | All Tier-1 metrics in [`eval-design.md §4`](eval-design.md) are pre-registered; BH-FDR correction at q = 0.05 applied across the ~40 (task × metric × stratum) cells. |
| Few-shot exemplar leakage | Few-shot exemplars for C3 are drawn from FT data, never from the eval set. Their `question_id`s are recorded in `prompts/fewshot_exemplars.json`. |
| Temporal Current-Affairs confound | Cutoff date 2026-04-30 fixed for both Gemini conditions and the eval set; both models receive the article text as input — Gemini's training cutoff does not affect Task E because the source text is provided. |
| Prompt-engineering leakage advantage | The same prompt scaffold is used for both C1 and C2/C3 within a task. Prompt files committed to `prompts/`. No condition-specific prompt tuning post-eval-freeze. |

---

## 4. Procedure

This section is the runbook. Anyone with access to the repo + DB credentials should be able to reproduce the experiment.

### Stage 1 — Setup & freeze (must complete before any model touches data)

| Step | Command | Output |
|---|---|---|
| 1.1 Verify env | `make verify-env` | Validates Python 3.14, `mlx-lm` install, library version pins, API keys present, Postgres reachable |
| 1.2 Freeze eval set | `python scripts/freeze_eval_set.py --seed 20260514 --out data/eval_set.parquet` | `data/eval_set.parquet` + `data/eval_set.sha256` |
| 1.3 Build FT corpus | `python scripts/build_ft_corpus.py --eval data/eval_set.parquet --out data/ft_corpus.parquet` | `data/ft_corpus.parquet`; **CI assertion checks zero overlap with eval set** |
| 1.4 Hash + register | `python scripts/register_run.py` | Writes content hashes of this report, `eval-design.md`, `eval_set.parquet`, and `ft_corpus.parquet` into `runs/<timestamp>/manifest.json` |

### Stage 2 — Pre-FT baseline (A2 Hindi probe, both base models)

| Step | Command | Output |
|---|---|---|
| 2.1 Pull 50 Hindi MCQs | (within `scripts/run_hindi_probe.py`) | `data/hindi_probe.parquet` |
| 2.2 Run base models on probe | `python scripts/run_hindi_probe.py --model gemma-4-E4B-it --quant 4bit`<br>`python scripts/run_hindi_probe.py --model Qwen3.5-4B --quant 4bit` | `results/pre_ft_hindi_probe.parquet` (one row per (model, item)) |
| 2.3 Pass-criterion gate | `python scripts/gate_hindi.py` | One-sided binomial test vs random (p=0.25) at α=0.05; per-model verdict. Failing models route Hindi-stratum results to separate post-FT reporting. |

### Stage 3 — Fine-tuning (two adapters)

| Step | Command | Output |
|---|---|---|
| 3.1 LoRA configuration | `configs/lora.yaml`: rank=16, alpha=32, dropout=0.05, target_modules=`["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]`, learning_rate=2e-4, batch_size=1, grad_accumulation=8, max_seq_len=2048, num_epochs=3 | Same config for both adapters |
| 3.2a Train Gemma adapter | `mlx_lm.lora --config configs/lora.yaml --base gemma-4-E4B-it --data data/ft_corpus.parquet --output adapters/gemma4-e4b-upsc-v1` | `adapters/gemma4-e4b-upsc-v1/final.npz` |
| 3.2b Train Qwen adapter | `mlx_lm.lora --config configs/lora.yaml --base Qwen3.5-4B --data data/ft_corpus.parquet --output adapters/qwen35-4b-upsc-v1` | `adapters/qwen35-4b-upsc-v1/final.npz` |
| 3.3 Training logs | Streamed per adapter to `runs/<timestamp>/training_gemma.jsonl` and `training_qwen.jsonl` | One line per step, per adapter |
| 3.4 Validate fits | `python scripts/validate_adapter.py --adapter <path>` (runs once per adapter) | Sanity-check on 50 held-out (not eval) examples per task; flags catastrophic forgetting or NaN losses |

Estimated wall-clock on M5 16 GB: **~5-7 hours per adapter** × 2 adapters = ~10-14 hours total FT. Runs can be sequenced overnight.

### Stage 4 — Inference (all four conditions)

For each of the 2,000 eval items, for each of C1a / C1b / C2 / C3, the same task-specific prompt is sent. Records are accumulated into `results/predictions.parquet`. Total: 8,000 prediction rows.

| Step | Command | Output |
|---|---|---|
| 4.1 C1a — Gemma FT-SLM inference | `python scripts/run_inference.py --condition C1a --adapter adapters/gemma4-e4b-upsc-v1` | 2,000 rows |
| 4.2 C1b — Qwen FT-SLM inference | `python scripts/run_inference.py --condition C1b --adapter adapters/qwen35-4b-upsc-v1` | 2,000 rows |
| 4.3 C2 — zero-shot frontier | `python scripts/run_inference.py --condition C2 --model gemini-3-flash --shots 0` | 2,000 rows |
| 4.4 C3 — few-shot frontier | `python scripts/run_inference.py --condition C3 --model gemini-3-flash --shots 3` | 2,000 rows |
| 4.5 Confidence elicitation (Task A) | Pass 2 per row asking `"0-100 confidence"`; appended to each row | confidence column populated |
| 4.6 Explanation elicitation (Task A) | Pass 3 per row asking for explanation in the row's `language`; feeds explanation-quality + pedagogical-clarity metrics | explanation column populated |

Each condition completes in a single run to keep latency metrics comparable. Each request logs `latency_ms`, `ttft_ms`, `input_tokens`, `output_tokens` to the prediction row.

### Stage 5 — Scoring

| Step | Command | Output |
|---|---|---|
| 5.1 Tier-1 scoring | `python scripts/score_tier1.py` | All deterministic metrics (BERTScore, BLEURT, ROUGE-L, chrF++, METEOR, Entity-F1, ECE, Brier, QWK, MAE, etc.) computed per row; written to `results/scored.parquet` |
| 5.2 Tier-2 LLM-judge | `python scripts/score_tier2.py --judge claude-sonnet-4-6` | G-Eval rubric scores (Tasks B, E only); written as separate columns; **not aggregated into Tier-1 headline metrics** |
| 5.3 Universal metrics | (already in prediction rows from Stage 4) latency p50/p95/p99, tokens/sec, $ cost | Aggregated in Stage 6 |

### Stage 6 — Analysis & write-up

| Step | Command | Output |
|---|---|---|
| 6.1 Aggregate | `python scripts/aggregate.py` | Per-(task, condition, stratum) means + bootstrap 95% CIs |
| 6.2 Paired tests | `python scripts/test_hypotheses.py` | McNemar (binary) / paired bootstrap (continuous) for H1, H2, H3 |
| 6.3 BH-FDR | (within `test_hypotheses.py`) | Multiple-comparison-corrected p-values across all ~40 cells |
| 6.4 Populate Sections 6-8 of this report | `python scripts/render_report.py` | Auto-fills [§6 Actual Outcome](#6-actual-outcome) and [§7 Results](#7-results) tables from `results/aggregate.parquet` |
| 6.5 Render dashboard | `streamlit run dashboard/app.py` | Live dashboard reads `results/scored.parquet` |

### Stage 7 — Release

| Step | Action |
|---|---|
| 7.1 Tag the run | `git tag run-<timestamp>` |
| 7.2 Archive artifacts | Upload `results/` + `adapters/` + `manifest.json` to S3 (read-only). |
| 7.3 Discussion writeup | Human authors fill [§8 Inference](#8-inference) section. |

---

## 5. Expected Outcomes (pre-registered predictions)

These are **predictions made before observing results**, to anchor analysis honesty. They are *not* assertions; they are recorded so any post-hoc agreement or surprise is auditable.

### 5.1 By task

Predictions framed for the per-task champion (max(C1a, C1b)) unless C1a and C1b are predicted to diverge meaningfully.

| Task | Pre-registered prediction | Rationale |
|---|---|---|
| **A — Prelims MCQ (answer + calibration)** | Champion beats C2 by **+8 to +15 pp accuracy** on English. C1b (Qwen, explicit-Indic) **outperforms C1a (Gemma) on the Hindi stratum** by ≥5pp. C3 (few-shot) closes most of the gap on English but stays below the champion on `silly_mistake_prone=True` items. | UPSC Prelims rewards memorizing specific Article numbers / dates / schemes — direct FT encodes this. Few-shot can't carry that volume. Qwen's enumerated Hindi pretraining should beat Gemma's pretraining-pool-only Hindi. |
| **A — Explanation quality + pedagogical clarity** | Champion **beats C2 and C3** on Explanation BERTScore-F1 and Distractor coverage by ≥10pp. Pedagogical Clarity rubric (Tier 2): C3 (few-shot) ≈ champion on Step-by-step + Specificity; champion wins on Conceptual grounding (FT data contains the exact UPSC-syllabus phrasings) and Distractor addressing (FT explanations consistently address each option). | `prelims_pyq_questions.explanation` JSONB carries the exact phrasing UPSC graders reward; FT directly imitates it. Frontier models can sound fluent but don't know the prayas/Drishti house-style. |
| **B — Mains generation** | Champion within **−0.04 to +0.02** of C3 on BERTScore-F1 against `pyqs.model_answer`. Champion **better** on word-count adherence and chrF++ (Hindi). Champion **worse** than C3 on G-Eval rubric (Tier 2) because Gemini-Flash writes more fluent prose. C1a vs C1b roughly tied (predict |Δ| < 0.02). | FT on a small corpus aligns surface form but cannot match a frontier model's fluency. Both 4B-class models should converge to similar Mains output quality. |
| **C — Mains rubric grading** | Champion achieves **QWK ≥ 0.55** against gold scores. C2 starts at **QWK ≤ 0.30** (untrained on rubric). C3 reaches **QWK ≈ 0.45** with three exemplar rubrics. C1a vs C1b roughly tied. | Rubric vocabulary + score distribution are highly domain-specific. Few-shot carries rubric *style* but not score-scale calibration. Both base models can learn the same rubric from the same FT corpus. |
| **E — Current Affairs synthesis** | Champion and C3 roughly tied on Entity-F1 and Date-exact-match. Hallucination rate **lower** for champion (article text in-distribution after FT). SummaC-ZS faithfulness **higher** for champion. C1a vs C1b roughly tied. **Pedagogical Clarity (Tier 2):** champion wins on Syllabus grounding and Static-Dynamic bridge (FT data encodes the prayas/Drishti house-style of linking news to syllabus); C3 wins on Multi-dimensional framing (frontier breadth); roughly tied on Specificity. | Current affairs is synthesis-from-source — Gemini's general capability suffices when the article is provided. The differentiators are faithfulness (favors FT) and *teaching style* (favors whichever was trained on prayas's mainsInfo gold). |
| **H1 — C1a vs C1b** | Significant difference on **Task A Hindi stratum only**. Other tasks: predicted |Δ| < 0.03 on primary metric. | Qwen's explicit-Indic pretraining vs Gemma's pool-Indic should matter for Hindi recall; English and rubric/factual tasks should converge under identical FT. |

### 5.2 By universal metric

| Metric | Pre-registered prediction |
|---|---|
| Latency p50 | C1a/C1b ~600-1200 ms (local MLX, no network); C2/C3 ~1500-3500 ms (API + serialization). C1a/C1b win by 2-3×. C1b slightly faster than C1a (smaller true dense vs MatFormer-with-PLE). |
| TTFT | C1a/C1b < 300 ms; C2/C3 > 800 ms. |
| Tokens/sec generation | C1a/C1b ~30-60 t/s on M5; C2/C3 ~150 t/s server-side, offset by network. |
| Cost / query | C1a/C1b ≈ $0 marginal. C2 ≈ $0.005-0.015 per query. C3 higher (+ few-shot tokens). |
| Format-validity rate | C1a/C1b ≥ 0.97 after FT (formats in training data); C2 ≥ 0.95; C3 ≥ 0.97. |

### 5.3 Aggregate verdict prediction

We expect **non-inferiority for the champion FT-SLM against C3** on 3 of 4 tasks, with a **clear win on Task C (rubric grading)** and a **clear loss on Task B G-Eval (Tier 2 fluency)**. If the champion also beats C3 on Task A, the v1 headline holds. We predict **C1a and C1b converge on most metrics**, with the meaningful divergence on the Hindi stratum of Task A favoring C1b.

---

## 6. Actual Outcome

This section is auto-populated by `scripts/render_report.py` from `results/aggregate.parquet` once the experiment runs. The structure below is final; numbers are blanks.

### 6.1 Run metadata

| Field | Value |
|---|---|
| `run_id` | (to fill) |
| `git_sha` | (to fill) |
| `experiment_report_sha256` | (to fill — hash of this document at run-start) |
| `eval_set_sha256` | (to fill) |
| `ft_corpus_sha256` | (to fill) |
| `gemma_adapter_sha256` | (to fill) |
| `qwen_adapter_sha256` | (to fill) |
| `wall_clock_total_hours` | (to fill) |
| `total_inference_cost_usd` | (to fill) |

### 6.2 A2 Hindi probe outcome (base models, pre-FT)

One-sided binomial test, H0: accuracy = 0.25, H1: accuracy > 0.25, α = 0.05. At n = 50 the critical value is k = 18 (36 % accuracy).

| Base model | Correct / 50 | Accuracy | p-value | Pass (p < 0.05)? |
|---|---:|---:|---:|---|
| `google/gemma-4-E4B-it` (via `mlx-community/gemma-4-e4b-it-4bit`) | **26 / 50** | **52.0 %** | < 0.00001 | **PASS** |
| `Qwen/Qwen3.5-4B` (via `mlx-community/Qwen3.5-4B-MLX-4bit`) | 15 / 50 | 30.0 % | 0.252 | **FAIL** |

**Protocol note:** the probe was re-run after `scripts/run_hindi_probe.py` was patched to wrap the prompt in each model's chat template (`tokenizer.apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)`) and `max_tokens` was raised from 6 → 24. Prior runs hit two measurement artefacts: Gemma 4-IT emitted EOS on the first token when handed an un-templated user message (yielding 0 / 50); Qwen 3.5 entered `<think>` mode by default and 18 / 50 of its responses were truncated mid-thinking by `max_tokens=6`. The values in the table above are the artefact-corrected measurements.

**Implications for v1:** the **direction of the predicted Hindi gap is inverted vs §5.1**. Pre-registered prediction was *"C1b (Qwen, explicit-Indic) outperforms C1a (Gemma) on the Hindi stratum by ≥ 5 pp."* Pre-FT, **Gemma is the strong-Hindi base** (52 % > 30 %); Qwen's Hindi knowledge is indistinguishable from chance. The "Indic-via-pretraining-enumeration" hypothesis fails the empirical test — pretraining-pool inclusion (Gemma 140-language tier) appears to deliver more usable Hindi than explicit instruction-enumeration alone (Qwen 201-language list). Qwen fails the binomial gate → its post-FT Hindi stratum (Task A, 347 Hindi items) will be reported as a separate finding, **not** folded into the bilingual aggregate. FT proceeds for both models regardless. This is precisely the kind of refuted pre-registered prediction the report design was built to surface honestly — recorded here, addressed in [§8.2 Pre-registered prediction vs reality](#82-pre-registered-prediction-vs-reality).

### 6.3 Per-task primary metric values

Per condition × task × primary metric. Tables auto-fill from `aggregate.parquet`.

#### Task A — Prelims MCQ (correctness & calibration)

| Condition | Accuracy (en) | Accuracy (hi) | UPSC neg-mark score | ECE | Brier | Refusal rate |
|---|---:|---:|---:|---:|---:|---:|
| C1a (Gemma-4-E4B-it + LoRA) | — | — | — | — | — | — |
| C1b (Qwen3.5-4B + LoRA) | — | — | — | — | — | — |
| C2 (zero-shot Gemini-3-Flash) | — | — | — | — | — | — |
| C3 (few-shot Gemini-3-Flash) | — | — | — | — | — | — |

#### Task A — Explanation quality (Tier 1)

| Condition | Expl. BERTScore-F1 | Expl. ROUGE-L | Expl. Entity-F1 | Distractor coverage | Reasoning-step density | Article/scheme citation acc. | Position-bias χ² p | Sentence-len variance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C1a | — | — | — | — | — | — | — | — |
| C1b | — | — | — | — | — | — | — | — |
| C2  | — | — | — | — | — | — | — | — |
| C3  | — | — | — | — | — | — | — | — |

#### Task B — Mains generation

| Condition | BERTScore-F1 | BLEURT-20 | ROUGE-L F1 | chrF++ | Word-count adh. | Entity-F1 | Hindi code-mix | MATTR | F-K grade | Paragraph adh. | 4-gram rep. rate | UPSC fact prec. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C1a | — | — | — | — | — | — | — | — | — | — | — | — |
| C1b | — | — | — | — | — | — | — | — | — | — | — | — |
| C2  | — | — | — | — | — | — | — | — | — | — | — | — |
| C3  | — | — | — | — | — | — | — | — | — | — | — | — |

#### Task C — Mains rubric grading

| Condition | QWK vs gold | Score MAE | Spearman ρ | Per-criterion κ | Strengths F1 | Improvements F1 | Score var. ratio | JSON schema valid | Item-count adh. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C1a | — | — | — | — | — | — | — | — | — |
| C1b | — | — | — | — | — | — | — | — | — |
| C2  | — | — | — | — | — | — | — | — | — |
| C3  | — | — | — | — | — | — | — | — | — |

#### Task E — Current Affairs synthesis

| Condition | BERTScore-F1 | Entity-F1 | Halluc. rate | Date F1 | SummaC-ZS | Subject-tag acc | Compression adh. | Glossary recall | Citation density | Lead-100 entity recall | UPSC fact prec. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C1a | — | — | — | — | — | — | — | — | — | — | — |
| C1b | — | — | — | — | — | — | — | — | — | — | — |
| C2  | — | — | — | — | — | — | — | — | — | — | — |
| C3  | — | — | — | — | — | — | — | — | — | — | — |

#### Task F — Prelims Explanation Generation (prayas production prompt)

Paired comparison: each cell is the Task-F metric value; the right-most column reports Δ vs the same model+condition's Task-A explanation metric (positive = prod prompt helps).

| Condition | BERTScore-F1 (en) | BERTScore-F1 (hi) | ROUGE-L F1 | chrF++ | Entity-F1 | Distractor coverage | Reasoning-step density | Article citation acc. | Word-count adh. | Hindi code-mix | Δ BERTScore-F1 vs Task A |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C1a | — | — | — | — | — | — | — | — | — | — | — |
| C1b | — | — | — | — | — | — | — | — | — | — | — |
| C2  | — | — | — | — | — | — | — | — | — | — | — |
| C3  | — | — | — | — | — | — | — | — | — | — | — |

#### Task G — Mains Model-Answer Generation (prayas production prompt)

Same paired-comparison logic against Task B.

| Condition | BERTScore-F1 | ROUGE-L F1 | chrF++ | Word-count adh. | Paragraph adh. | Entity-F1 | Date/Num F1 | MATTR | F-K grade | 4-gram rep. | UPSC fact prec. | Dim-keyword cov. | Δ BERTScore-F1 vs Task B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C1a | — | — | — | — | — | — | — | — | — | — | — | — | — |
| C1b | — | — | — | — | — | — | — | — | — | — | — | — | — |
| C2  | — | — | — | — | — | — | — | — | — | — | — | — | — |
| C3  | — | — | — | — | — | — | — | — | — | — | — | — | — |

### 6.4 Universal metrics

| Condition | Latency p50 (ms) | TTFT (ms) | Tokens/sec | Cost/query (USD) | Format-validity rate |
|---|---:|---:|---:|---:|---:|
| C1a | — | — | — | — | — |
| C1b | — | — | — | — | — |
| C2  | — | — | — | — | — |
| C3  | — | — | — | — | — |

### 6.5 Tier-2 (LLM-judge, diagnostic only — not headline)

Filled separately from Tier 1 to make clear they are not the primary signal.

#### Task B — G-Eval Mains rubric (1–5 per axis)

| Condition | G-Eval Content | G-Eval Contextual | G-Eval Analytical | G-Eval Structural | G-Eval Directive |
|---|---:|---:|---:|---:|---:|
| C1a | — | — | — | — | — |
| C1b | — | — | — | — | — |
| C2  | — | — | — | — | — |
| C3  | — | — | — | — | — |

Cohen's κ between G-Eval ranking and BERTScore-F1 ranking on Task B: **—**. (If < 0.3, Tier 2 disagrees with Tier 1 — note explicitly as a finding rather than a problem.)

#### Task A — Pedagogical Clarity rubric (1–5 per axis, total 5–25)

| Condition | Step-by-step | Distractor addr. | Conceptual grounding | Specificity | Accessibility | Total |
|---|---:|---:|---:|---:|---:|---:|
| C1a | — | — | — | — | — | — |
| C1b | — | — | — | — | — | — |
| C2  | — | — | — | — | — | — |
| C3  | — | — | — | — | — | — |

Kendall's τ between Pedagogical-Clarity total and Tier-1 explanation-composite (BERTScore + Distractor-coverage + Reasoning-step-density, equal-weighted): **—**. If τ > 0.5, Tier 2 reinforces Tier 1; if < 0.3, Tier 2 is capturing something Tier 1 misses (reported as a finding).

#### Task C — Feedback Pedagogical Clarity rubric (1–5 per axis)

| Condition | Actionability | Specificity | Constructiveness | UPSC-rubric fidelity | Coverage proportionality |
|---|---:|---:|---:|---:|---:|
| C1a | — | — | — | — | — |
| C1b | — | — | — | — | — |
| C2  | — | — | — | — | — |
| C3  | — | — | — | — | — |

#### Task E — Pedagogical Clarity rubric (1–5 per axis, total 5–25)

| Condition | Syllabus grounding | Static-Dynamic bridge | Multi-dimensional framing | Specificity | Mains-utility framing | Total |
|---|---:|---:|---:|---:|---:|---:|
| C1a | — | — | — | — | — | — |
| C1b | — | — | — | — | — | — |
| C2  | — | — | — | — | — | — |
| C3  | — | — | — | — | — | — |

Kendall's τ between Pedagogical-Clarity total (Task E) and Tier-1 BERTScore-F1 ranking: **—**. τ < 0.3 means clarity is capturing teaching-quality signal that surface faithfulness misses — itself an ed-tech-relevant finding.

---

## 6A. Tier-1 metric glossary

This section explains, in plain English, what each Tier-1 metric *measures* and how to *read* its value. Use it to make sense of the §6.3 tables when they fill. Direction column: **↑** = higher is better, **↓** = lower is better, **target** = a specific value is best (clipped/penalized on either side). Implemented in [`scripts/score_tier1.py`](scripts/score_tier1.py); aggregations + statistical tests in [`scripts/aggregate.py`](scripts/aggregate.py) and [`scripts/test_hypotheses.py`](scripts/test_hypotheses.py).

### 6A.1 Universal interpretation patterns

- **Range [0, 1]:** treat as a proportion. A 5-percentage-point gap (0.05 absolute) is the smallest gap that's usually meaningful at our n; smaller gaps need CI overlap inspection.
- **F1 metrics:** harmonic mean of precision and recall. F1 = 0.5 means the cand and ref agree on ~half the items in a precision-balanced way; F1 = 0 means no overlap.
- **BERTScore-F1** rescaled-against-baseline range is roughly [0.0, 1.0] in practice; raw BERTScore (what we use, faster) sits in [0.6, 0.95] for sentence-similar text — read deltas across conditions on the *same* metric, not absolute values across metrics.
- **Adherence metrics** (word count, sentence count, paragraph count, item count): formula `1 − |pred − target| / target`, clipped to [0, 1]. 1.0 = exact match; 0.0 = off by 100 % or more.
- **Bootstrap 95 % CI** (from [§7](#7-results--statistical-tests)) is reported alongside means; non-overlapping CIs between conditions are a sufficient (not necessary) condition for the BH-FDR-corrected paired test to find significance.

### 6A.2 Task A — Prelims MCQ

#### Correctness & calibration

| Metric | Dir | What it measures | How to read it |
|---|:-:|---|---|
| **Accuracy** | ↑ | Fraction of questions where the predicted letter equals the gold letter | Headline KPI. UPSC Prelims cutoff is typically ~50 % raw — a model below that is below human-aspirant baseline. |
| **UPSC negative-marking score** | ↑ | Mean per-item score under official rules — GS-I: +2 correct / −2/3 wrong / 0 abstain; CSAT: +2.5 correct / −2.5/3 wrong / 0 abstain | This is what actually shows on a Prelims marksheet. A model can have ≥ 50 % accuracy and still post a negative neg-mark score if its wrong answers outnumber its correct ones with negative-mark weight. |
| **Brier loss** | ↓ | Squared error between the model's stated confidence (Pass-2 elicited 0–100) and the realized correctness (0/1) | 0 = perfect calibration; 0.25 = no-info (constant 0.5 guess); >0.25 = miscalibrated worse than a coin flip. |
| **ECE-15bin** *(aggregate)* | ↓ | Expected Calibration Error across 15 confidence bins — \|mean_acc − mean_conf\| within each bin, weighted by bin population | ECE of 0.05 = on average the model's stated confidence is off the realized accuracy by 5 pp. Critical for tutor use — overconfident wrong answers are worse than calibrated-uncertain ones. |
| **Brier Skill Score** *(aggregate)* | ↑ | 1 − Brier / Brier_of_baserate_predictor (a predictor that always emits the dataset's overall accuracy) | > 0 = beats the no-info baseline; < 0 = worse than just predicting the average. |
| **Format-fail rate** | ↓ | Fraction where the parser cannot extract a letter ∈ {A,B,C,D} from the prediction | Should be near-zero (< 5 %) for a usable model; in-context format-failure is a hard product-blocker. |
| **Position-bias χ² p-value** *(aggregate)* | target ≈ uniform | χ² test of the model's predicted-letter distribution against uniform | Low p (< 0.05) = the model has a position preference (e.g. always picks C). Watch this alongside accuracy — a 50 %-accuracy model that always picks "C" is broken in a different way than a 50 %-accuracy model with uniform output. |
| **Bilingual accuracy delta** *(aggregate)* | target → 0 | `accuracy(en) − accuracy(hi)` on paired English/Hindi versions of the same question stem | Large positive = the model's Hindi is worse than its English. Reported separately for models that failed the A2 gate. |
| **Silly-mistake breakdown** *(aggregate)* | — | Accuracy on the `silly_mistake_prone=True` subset vs the rest | UPSC content tag marking questions specifically engineered to elicit careless reading. Larger drop = the model is "skimming" the question. |

#### Explanation quality (Task A, computed on Pass-1 JSON `explanation` field)

| Metric | Dir | What it measures | How to read it |
|---|:-:|---|---|
| **Explanation BERTScore-F1** | ↑ | Contextual-embedding (roberta-large) semantic similarity vs gold explanation | Insensitive to word-order; captures meaning overlap. 0.85+ on sentence-similar text in our scale. |
| **Explanation ROUGE-L F1** | ↑ | Longest-common-subsequence F1 on tokens | Surface overlap. Low ROUGE with high BERTScore = same content, different words. |
| **Explanation Entity-F1** | ↑ | Set F1 over spaCy NER entities (English only) | Catches whether named entities (Article numbers, schemes, people, places) survive the paraphrase. 0 means no named entities in common. |
| **Distractor coverage** | ↑ | Fraction of wrong options the explanation explicitly addresses (option letter mentioned + ≥1 distinctive token from that option) | A UPSC-aspirant-quality explanation explains *why each wrong option is wrong*. Frontier models often only explain the right one. |
| **Reasoning-step density** | target ≈ 3–8 | Discourse markers per 100 words (`because`, `therefore`, `however`, `first`, `if/then` …; bilingual list) | Tracks structured reasoning. 0 = pure assertion; > 12 = over-marked / templatic. |
| **Article/scheme citation accuracy** | ↑ | For every `Article N` regex match in the explanation, fraction whose `N` exists in `data/upsc_facts.json` | Catches hallucinated Article numbers. 1.0 with non-zero citation count = no fabricated Articles. |
| **Sentence-length variance** | target 5–80 | Variance of token counts per sentence | < 5 = templatic / AI-fingerprint; > 80 = run-ons / incoherent. |

### 6A.3 Task B — Mains generation

| Metric | Dir | What it measures | How to read it |
|---|:-:|---|---|
| **Answer BERTScore-F1** | ↑ | Semantic similarity vs `pyqs.model_answer` | Headline KPI for Mains generation. |
| **Answer ROUGE-L F1** | ↑ | LCS surface overlap | Surface overlap is a poor headline for Mains (many valid answers paraphrase) but useful as a sanity floor. |
| **Answer chrF++** | ↑ | Character n-gram F1, word-order = 2 | Robust for Devanagari (Hindi) where token-level metrics misbehave. |
| **Word-count adherence** | target 1.0 | `1 − \|words(gen) − target\| / target`, clipped [0, 1] | UPSC graders penalize being off the word target. Target ∈ {150, 250, ~1200} from `pyqs.word_count`. |
| **Sentence-count adherence** | target 1.0 | Same formula on sentence counts vs reference | Catches "wrote enough words but as one giant sentence" or "twelve fragments". |
| **Paragraph-count adherence** | target 1.0 | Same idea — 150 w → 1-2 paragraphs, 250 w → 3-5, essay → 8-12 | UPSC Mains structure is rewarded explicitly by markers. |
| **Entity-F1** | ↑ | NER entity-set F1 vs gold (English only) | Same logic as Task A explanation Entity-F1; primary signal for "is the same factual content in here". |
| **Date exact-match F1** | ↑ | F1 over `\b(19\|20\|21)\d{2}\b` regex matches between cand and gold | UPSC Mains rewards date specificity. Penalizes wrong/missing year mentions. |
| **Numeric exact-match F1** | ↑ | F1 over `\d+(?:\.\d+)?%?` matches | Same logic for percentages and figures. |
| **Hindi code-mixing rate** | target → 0 | Fraction of letter characters NOT in the Devanagari Unicode block (Hindi rows only) | Quantifies "the model defaulted to English mid-answer". > 0.20 in a Hindi prompt = systemic code-mixing. |
| **MATTR-100** | ↑ | Moving-Average Type-Token Ratio with window = 100 | Lexical diversity. < 0.5 = repetitive vocabulary; ~0.7-0.8 is typical of human Mains answers. |
| **Flesch-Kincaid grade** | target 12–15 | US-grade-level readability index (`textstat`) | UPSC Mains answers typically land at grade 12-15. Below 10 = too simple; above 18 = jargon-dense / unreadable. |
| **4-gram repetition rate** | ↓ | Fraction of 4-grams that appear more than once / total 4-grams | A known SLM failure mode. >0.10 = pathological repetition; UPSC graders penalize. |
| **UPSC fact-lookup precision** | ↑ | For every Article / Act / scheme reference, fraction recognized in `upsc_facts.json` | Same logic as Task A citation accuracy. Catches fabricated Article numbers / made-up Act names in Mains answers. |

### 6A.4 Task C — Mains rubric grading

This task predicts `(score, strengths[], improvements{intro,body,conclusion})` from `(question, student_answer, max_score)`; gold lives in `evaluation_questions`.

| Metric | Dir | What it measures | How to read it |
|---|:-:|---|---|
| **Score MAE** | ↓ | Mean absolute error between predicted and gold scores | Same units as the rubric (typically 0-15). MAE of 1.0 = on average off by 1 mark. |
| **Score abs-error per row** | ↓ | Per-row \|pred − gold\| (input to MAE; here to drive distribution plots) | Examine the distribution, not just the mean — a fat right tail means occasional large misses. |
| **QWK** *(aggregate)* | ↑ | Quadratic Weighted Kappa on rounded integer scores | The ASAP-Kaggle standard for automated essay scoring. > 0.6 = substantial agreement; > 0.8 = near-perfect. |
| **Spearman ρ** *(aggregate)* | ↑ | Rank correlation between predicted and gold | Tolerates scale shifts (model that consistently grades 1 mark low can still have ρ ≈ 1). |
| **Pearson r** *(aggregate)* | ↑ | Linear correlation | Should track Spearman closely; large divergence = nonlinear or saturating predictions. |
| **Confusion matrix (low / mid / high bands)** *(aggregate)* | — | 3×3 confusion on band assignments (low ≤30 %, mid 30-60 %, high >60 % of `max_score`) | Reveals systematic floor / ceiling collapse — e.g. a model that grades everything "mid" produces a fat middle column. |
| **Score-variance ratio** *(aggregate)* | target ≈ 1.0 | `var(predicted_score) / var(gold_score)` | < 0.5 = mean-collapsed (model returns ~dataset mean for every input); ≈ 1 = matched dispersion; ≫ 1 = noisy. |
| **JSON schema validity rate** | ↑ | Fraction of predictions parsing the strict `{score, strengths[], improvements{intro,body,conclusion}}` schema | The orchestrator skips per-row metric computation on invalid JSON — schema validity is a precondition for the rest of Task C to score at all. |
| **Strengths token-F1** | ↑ | spaCy-lemma-set F1 between predicted and gold `strengths` bullets | Measures content overlap of the bullets, not phrasing — captures whether the model identifies the same strengths. |
| **Improvements token-F1** | ↑ | Same on `improvements` flattened across intro/body/conclusion | Same logic. |
| **Strengths sentence-level BERTScore-F1** | ↑ | Per-strength BERTScore-F1 between pred and gold bullets | Lemma-F1 misses paraphrase; sentence BERTScore catches it. |
| **Strengths / Improvements item-count adherence** | target 1.0 | `1 − \|pred_count − gold_count\| / gold_count`, clipped [0, 1] | Mentor feedback has typical bullet counts (strengths 2-4, improvements 3-6). Collapse-to-one or balloon-to-ten are both failure modes. |

### 6A.5 Task E — Current Affairs synthesis

This task produces `(prelims_info, mains_info)` from `(date, title, article_text)`; gold lives in `news_articles.prelimsInfo / mainsInfo`.

| Metric | Dir | What it measures | How to read it |
|---|:-:|---|---|
| **Prelims-info BERTScore-F1** | ↑ | Semantic similarity of generated `prelims_info` vs gold | Headline KPI for the "key facts" output. |
| **Mains-info BERTScore-F1** | ↑ | Semantic similarity of generated `mains_info` vs gold | Headline KPI for the "multi-dimensional analysis" output. |
| **Prelims-info / Mains-info ROUGE-L F1** | ↑ | LCS overlap on tokens | Surface overlap floor. |
| **Mains-info chrF++** | ↑ | Character n-gram F1 (Hindi-robust) | Hindi support; English values overlap with ROUGE/BERTScore signal. |
| **Entity-F1 vs gold (mains_info)** | ↑ | NER entity-set F1 between gen and gold `mains_info` | Whether the same named entities (committees, court cases, scheme names) survive. |
| **Hallucination rate** | ↓ | Fraction of entities in the generation that do NOT appear in the source article | Faithfulness floor. > 0.15 = noticeable invented content; this is the metric SummaC-ZS would have refined (deferred). |
| **Coverage of source entities** | ↑ | Recall of source-article entities into the generation | Catches "skipped the important entities" failures. |
| **Date F1 vs source** | ↑ | F1 on regex-extracted years between generation and source | UPSC current-affairs grading rewards specific dates. |
| **Numeric F1 vs source** | ↑ | F1 on regex-extracted numbers/percentages | Same logic for figures. |
| **Compression ratio score** | target [0.20, 0.50] | `gen_tokens / source_tokens` scored 1.0 inside [0.20, 0.50], linear decay outside | Mains-info should be a *synthesis*, not a verbatim copy and not a one-liner. UPSC prayas gold typical range. |
| **Citation density per 100 w** | target ≥ 4 / 100 w | (named-entities + dates + numbers) per 100 generated words | < 2 = under-grounded; > 10 = entity-stuffing. Tracks "is the synthesis actually anchored in facts?" |
| **Lead-100w entity recall** | ↑ | Recall of the source article's headline-paragraph entities within the first 100 words of the generation | Catches "did the synthesis lead with the key facts?" — UPSC mainsInfo style explicitly opens with the news anchor. |
| **UPSC fact-lookup precision** | ↑ | Same metric as Tasks A and B — fraction of `Article N` / Act / scheme references that resolve in `upsc_facts.json` | Same hallucination guardrail. |

### 6A.6 Task F — Prelims Explanation Generation (prayas production prompt)

Capability test of "given a question and the correct letter, write a high-quality explanation under prayas's production prompt." Same model checkpoints as Tasks A-E; only the prompt differs. Reuses the 800 Task-A items. Metric set is the [§4.6 vetted inventory](eval-design.md) — the Task-A explanation-quality subset minus *sentence-length variance* (research-flagged as noisy at the 50-300 w lengths typical here). Bilingual.

| Metric | Dir | What it measures | How to read it |
|---|:-:|---|---|
| **Explanation BERTScore-F1** *(headline)* | ↑ | Semantic similarity vs gold explanation, per language | Read paired against the same item's Task-A explanation BERTScore-F1 — positive Δ means the production prompt improves output. |
| **Explanation ROUGE-L F1** | ↑ | Surface LCS overlap with gold | Surface floor; mostly tracks BERTScore for in-distribution gold style. |
| **Explanation chrF++** | ↑ | Character n-gram F1 | Hindi-robust; preferred over ROUGE for Devanagari. |
| **Explanation Entity-F1** | ↑ | spaCy-NER entity overlap with gold (English only) | Catches named-entity coverage (Articles, schemes, people, places). |
| **Distractor coverage** | ↑ | Fraction of the three wrong options each explicitly addressed (letter present + ≥1 distinctive token) | The headline pedagogy axis — UPSC explanations that don't address why each wrong option is wrong are sub-par. |
| **Reasoning-step density** | target 3–8 / 100 w | Discourse markers per 100 words | Same interpretation as Task A. |
| **Article/scheme citation accuracy** | ↑ | Fraction of `Article N` / scheme refs that resolve in `upsc_facts.json` | Hallucination guardrail — particularly important when the model is asked to *justify* with specific cites. |
| **UPSC fact-lookup precision** | ↑ | Same lookup applied as precision across all factual references | See §6A.2. |
| **Word-count adherence** | target 1.0 | `1 − \|words(gen) − target\| / target`, clipped [0, 1]; target = per-stratum mean of gold-explanation length until prayas's prompt specifies one | Explanations that are too short skip distractors; too long bury the point. |
| **Hindi code-mixing rate** | target → 0 | Fraction of letter chars NOT in Devanagari (Hindi rows only) | Same interpretation as Task B. |

### 6A.7 Task G — Mains Model-Answer Generation (prayas production prompt)

Capability test of "given a Mains question, write a model answer under prayas's production prompt." Reuses the 400 Task-B items. The metric set is **Task B's 14 Tier-1 metrics carried over** ([§6A.3](#6a3-task-b--mains-generation)) **plus two structural additions** vetted against [NAACL-Short 2024](https://aclanthology.org/2024.naacl-short.9/) (PDD on long-form coherence is deferred to v2 — see [eval-design.md §4.7](eval-design.md)):

| Metric | Dir | What it measures | How to read it |
|---|:-:|---|---|
| *All Task-B Tier-1 metrics* | (per §6A.3) | (per §6A.3) | The headline BERTScore-F1 (Δ vs Task B paired) is the production-drop-in signal. |
| **Dimension-keyword coverage** | ↑ | Count of distinct UPSC-Mains dimensions (political / economic / social / environmental / ethical / international) touched in the generation, divided by the count touched in the gold | UPSC Mains rewards multi-dimensional framing. < 0.5 = single-axis answer; ≥ 1.0 = matches gold breadth. Engineered metric, documented as such; not from published literature. |
| **Directive-conditioned discourse density** *(exploratory)* | target ≈ 1.0 | For the question's directive verb (`analyze`/`evaluate`/`discuss`/…), ratio of discourse-marker density in the generation to that in the gold | Catches "described instead of analyzed" failures. **Exploratory** — flagged as such because no published prior art validates the proxy. |
| ↳ deferred: **PDD coherence** | ↑ | Positional Discourse Divergence (NAACL 2024) — discourse-parser-based long-form coherence | Strongest published deterministic coherence metric for long-form; requires an RST/PDTB parser dep. **Deferred to v2** — re-enable if v1 headline metrics show F/G is close to Task A/B but PDD-style structural signal would help disambiguate. |

### 6A.8 Universal metrics (every condition)

| Metric | Dir | What it measures | How to read it |
|---|:-:|---|---|
| **Latency p50 / p95 / p99** | ↓ | Wall-clock per request — 50th / 95th / 99th percentile | Local MLX (C1a/C1b) is expected to win by 2-3× on p50 vs network APIs (C2/C3). |
| **TTFT** | ↓ | Time to first token | The latency the *user* experiences before output starts streaming. Local FT-SLMs typically < 300 ms; APIs > 800 ms. |
| **Tokens/sec generation** | ↑ | Output tokens divided by generation wall-clock | Sustainable throughput. On M5: ~30–60 t/s expected. |
| **Input / output token counts** | — | Per-row, for cost arithmetic | Identifying metric, not scored — used to compute $/query. |
| **$ cost per query** | ↓ | Gemini: published per-1k-token rate × tokens. FT-SLM (local): $0 marginal; energy proxy reported separately | The headline economic comparison. |
| **Format-validity rate** | ↑ | Did the prediction parse as the required JSON schema for the task? | A sub-90 % rate makes downstream metrics unreliable — surfaced as a top-line health indicator. |

---

## 7. Results — Statistical Tests

Auto-filled from `scripts/test_hypotheses.py`.

### 7.1 Pairwise hypothesis tests (BH-FDR-corrected)

For each task × metric × pairwise comparison, the table reports the point estimate of the delta, paired-bootstrap 95% CI on the delta, raw p-value (10K-resample), and BH-FDR-corrected p-value (q = 0.05).

| Task | Metric | Comparison | Δ (mean) | 95% CI | p (raw) | p (BH-FDR) | Significant? |
|---|---|---|---:|---|---:|---:|---|
| A | accuracy_en | C1a − C1b | — | (—, —) | — | — | — |
| A | accuracy_en | C1a − C2 | — | (—, —) | — | — | — |
| A | accuracy_en | C1a − C3 | — | (—, —) | — | — | — |
| A | accuracy_en | C1b − C2 | — | (—, —) | — | — | — |
| A | accuracy_en | C1b − C3 | — | (—, —) | — | — | — |
| A | accuracy_en | C2 − C3 | — | (—, —) | — | — | — |
| … (≈60 rows total — 6 pairwise comparisons × ~10 metrics) | | | | | | | |

### 7.2 Per-stratum heatmap data

Each cell: C1 − C3 delta on the primary metric for that (task, stratum).

| Task | Stratum | Δ (C1 − C3) | 95% CI | Verdict |
|---|---|---:|---|---|
| A | en × GS-I × silly_mistake_prone=True | — | (—, —) | — |
| A | hi × GS-I × silly_mistake_prone=True | — | (—, —) | — |
| A | en × GS-I × silly_mistake_prone=False | — | (—, —) | — |
| B | GS4 × 250-word | — | (—, —) | — |
| C | Ethics-Case-Studies × mid-band | — | (—, —) | — |
| E | December × `Geopolitics` theme | — | (—, —) | — |
| … | | | | |

Note: cell `delta` is `champion_metric − C3_metric`, where champion = argmax over (C1a, C1b) on that stratum.

### 7.3 Effect sizes

For each significant comparison, Cohen's d (continuous) or Cohen's h (proportions).

| Task | Metric | Comparison | Effect size | Interpretation |
|---|---|---|---:|---|
| (auto-filled) | | | | |

---

## 8. Inference (Discussion)

To be authored by humans after results land. The structure is fixed in advance:

### 8.1 Summary of findings

One paragraph answering: did C1 beat / tie / lose to C3 on the headline metric per task?

### 8.2 Pre-registered prediction vs reality

Walk through [§5](#5-expected-outcomes-pre-registered-predictions). For each prediction, mark Confirmed / Refuted / Partially confirmed. Where refuted, hypothesize why.

### 8.3 What the per-stratum view tells us

UPSC paper-level and subject-level breakdowns. Examples of questions to answer:
- Is C1's advantage concentrated in subjects where the FT data is densest (Polity, Economy)?
- Does C1 fail on the long-tail subjects (Art & Culture, World History) where FT data is thin?
- How does C1 perform on the bilingual delta — is Hindi a leveler or a differentiator?
- Does the `silly_mistake_prone` flag predict C1's advantage?

### 8.4 Mistakes and limitations actually observed

Catalogue specific failure modes (with `question_id`s).

### 8.5 Implications for prayas.ai's product

If C1 is non-inferior or better, what does it mean for the production roadmap? Specifically:
- Per-query cost at projected scale → annual savings
- Latency improvement → UX implications for the tutor chat
- Hosting requirements → instructor-side or cloud
- Versioning and continual-FT pipeline implications

### 8.6 What v2 should add

Concrete proposals informed by what v1 surfaced.

---

## 9. Limitations (also reproduced from [`eval-design.md §8`](eval-design.md))

1. **Task-C gold is itself LLM-generated.** We acknowledge but do not separately human-verify in v1.
2. **Scorer-model dependence.** BERTScore / BLEURT / SummaC depend on their underlying checkpoints; pinning makes the run reproducible but not architecture-free.
3. **Eval-set leakage with model-vendor training data is unknown.** Our `eval_set` ↛ `ft_corpus` discipline only protects us from our own leakage, not Google's.
4. **Sub-stratum sample sizes are modest** (often ≈50 items). Sub-claims should be read as suggestive; the overall and per-paper claims are well-powered.
5. **Verbal confidence as proxy for true logits** is fair-but-noisy; both conditions tested the same way.
6. **No human-mentor calibration on Task C in v1.** A 50-row mentor calibration is recommended for v2.
7. **Hindi reporting is conditional on the A2 probe.** If Gemma's Hindi capability is below threshold, Hindi findings are post-FT only — they cannot inform pre-FT-to-post-FT comparisons.

---

## 10. Reproducibility checklist

Before declaring the experiment complete:

- [ ] `experiment-report.md` and `eval-design.md` SHAs at run-start committed to `manifest.json`
- [ ] `data/eval_set.parquet` SHA-256 matches `data/eval_set.sha256`
- [ ] CI assertion confirms `eval_set ∩ ft_corpus = ∅`
- [ ] `models/lockfile.json` pins all scorer-model checkpoints used
- [ ] `requirements.txt` exact-version-pinned and `pip-compile --generate-hashes` lockfile committed
- [ ] All four condition runs use the same prompt files from `prompts/` (recorded in `manifest.json`)
- [ ] `claude-sonnet-4-6` judge snapshot ID recorded
- [ ] `gemini-3-flash` model ID + cutoff date recorded
- [ ] Adapter SHA-256 in `manifest.json` matches the deployed weights
- [ ] All `scripts/*.py` files have deterministic seeds (`20260514` standard)
- [ ] [§6 Actual Outcome](#6-actual-outcome) and [§7 Results](#7-results) populated from `aggregate.parquet`, not hand-edited
- [ ] Streamlit dashboard launches and renders without error against `results/scored.parquet`

---

## 11. Out of scope (deferred to v2 or later)

- **T2 — Personalized tutoring** (Q + student state → A). Requires student-memory retrieval; introduces a confounding variable.
- **Task D — Interview / DAF question generation.** Postponed.
- **External corpora** (NCERTs, Drishti / Vision / Insights compilations, official syllabus PDFs). Restricted to internal data in v1 for traceability.
- **Live A/B with real prayas students.** Production roll-in is a separate decision.
- **IRT-based item difficulty estimation.** v2 may introduce IRT to weight per-item gains by question difficulty.
- **Multi-turn conversational evaluation.**
- **Cost-adjusted utility (quality × $/query Pareto front).**
- **Human-mentor calibration on Task C.** Strongly recommended for v2.
- **Robustness to adversarial prompting.** Out of scope.
- **Mains 2024 / 2025 PYQ generalization holdout.** v1 mixes years; v2 should hold out the most recent year as an OOD test.

---

