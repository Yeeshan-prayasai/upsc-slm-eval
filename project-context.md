# Project Context — Prayas.ai SLM vs Gemini 3-Flash on UPSC

**Owner:** Yeeshan (irshad@prayas.ai), Data Scientist, prayas.ai
**Started:** 2026-05-14
**Working dir:** `/Users/yeeshan/PrayasAI/Code/SLM`
**Status:** Scoping / discovery
**Final deliverable:** Streamlit dashboard supporting (a) side-by-side query execution against all three model conditions and (b) aggregate metric dashboard across the eval set.

---

## 1. Goal

Quantify the performance gap between:
1. A **fine-tuned open-source SLM** (model TBD — pending May-2026 market research)
2. **Gemini 3-Flash** in **zero-shot** mode
3. **Gemini 3-Flash** in **few-shot** (prompted with UPSC exemplars) mode

…on UPSC (Indian Civil Services Exam) tasks. Comparison must use **research-backed, ed-tech-centric metrics** specialized for UPSC.

## 2. Scope (confirmed with user)

UPSC task surfaces in scope:
- [ ] Prelims (objective MCQ — GS I + CSAT)
- [ ] Mains (descriptive — GS I-IV, Essay, Optional)
- [ ] Current Affairs Q&A
- [ ] Interview / Personality Test (open-ended ethics & opinion)

All four are in scope per user.

## 3. Data sources

### 3.1 Internal (Postgres) — inventoried 2026-05-14
Credentials: `db-creds.txt` (DO NOT COMMIT).
- `upscdev` (RDS, 89 tables, 362 MB) — **content/curriculum DB**; authorized for SELECT.
- `prod-prayas-db` (RDS, 57 tables, 820 MB) — production **app/student DB**; **read-only SELECT** only.
- `app_dev` @ `13.203.24.116:6001` — out-of-scope for now (user did not include it).

#### High-value tables by task surface

**Prelims (MCQ) — abundant (~37K questions)**
| DB | Table | Rows | Notable cols |
|---|---|---:|---|
| upscdev | `prelims_pyq_questions` | 3,615 | year, paper, subject, topics, model answer, embeddings, **silly_mistake_prone**, **question_pattern** |
| upscdev | `prelims_quiz_questions` | 16,441 | internal quiz Qs, difficulty, options |
| upscdev | `upsc_prelims_ai_generated_que` | 4,615 | **bilingual (English+Hindi)**, quality_pass_flag — already AI-generated baseline |
| prod | `mcqs` | 10,656 | production active set, options, isMultiSelect, explanation jsonb |
| prod | `paper_analysis_questions` | 2,373 | with **subjectIds/topicIds**, answerKey + aiProposedOptionIds (model already has predictions!) |

**Mains (descriptive) — moderate (~12K with answers/rubrics)**
| DB | Table | Rows | Notable cols |
|---|---|---:|---|
| upscdev | `pyqs` | 1,646 | Mains PYQs with `model_answer`, hints, word_count |
| upscdev | `evaluation_questions` | 10,709 | **student answer + score + strengths + improvements + model_answer** — this is rubric-grade gold |
| upscdev | `prayas_test_series_questions` | 484 | test-series Mains Qs with `model_answer` |
| upscdev | `pyq_evals` | 146 | student answers + scores on PYQs |
| upscdev | `dpq_questions` + `dpq_evals` | 35 + 85 | bilingual Daily Practice Qs with student evals |

**Current Affairs — modest but structured**
| DB | Table | Rows | Notable cols |
|---|---|---:|---|
| prod | `news_articles` | 3,573 | full text + `prelimsInfo` + `mainsInfo` + date + theme |
| prod | `current_affairs` | 1,544 | pointed_analysis, mains_analysis, prelims_info, prelims_topics, embedding |
| upscdev | `current_affairs` | 236 | older / superset; same shape |

**Interview (Personality Test) — surprisingly rich**
| DB | Table | Rows | Notable cols |
|---|---|---:|---|
| upscdev | `daf_questions` | 38,685 | DAF-keyword-driven seed + follow-up questions |
| upscdev | `daf_standard_questions` | 86 | curated standard interview Qs |
| upscdev | `probing_questions` | 210 | follow-up probes tied to `flag_id` |

**Reference / structure**
- `upscdev.upsc_syllabus` (18 rows): paper × subject × topics_subtopics (JSONB)
- `prod.learning_items` (38,404 rows): canonical taxonomy with type/difficulty/paper/subject
- `prod.glossary` (7,475 rows): UPSC vocabulary

**Chatbot data — found in app_dev (PG 17.6, 222 MB, 13.203.24.116:6001)**

App_dev largely mirrors prod's schema (users/mcqs/news_articles/learning_items/paper_analysis_*/...) but adds the chat layer:

| Table | Rows | Purpose |
|---|---:|---|
| `chat_sessions` | 58 | One row per tutoring session. `staticContext` field is the **student's full evaluation report** stitched into the prompt. Other cols: `archetype`, `archetypeConfidence`, `arcStage`, `activeDrillState`, `closingSummary` — a state machine for tutoring. |
| `chat_messages` | 258 | role∈{user,assistant}, `content`, `tokenCount`, `langfuseTraceId`, `messageKind`, `actions` jsonb. Langfuse is used for observability. |
| `student_memory` | 3 | Persistent per-student profile: `competencyScores`, `subjectPerformance`, `paperPerformance`, `writingPatterns`, `factualErrors[]`, `interactionHistory`, `examMetadata`. |
| `subjective_questions` | 0 | Empty placeholder. |

**Key finding:** the existing prayas.ai chatbot is a **personalized tutor**, not a generic UPSC Q&A bot. It is grounded with student-specific evaluation reports and tracks long-running competency state. Comparison framing must specify whether we're benchmarking on:
- (T1) **Standalone UPSC capability** — Q→A without student context (most generalizable, easiest comparison)
- (T2) **Personalized tutoring** — Q+student_context→A (closer to prayas.ai's actual product)

For the first pass, **T1 is the cleaner experimental design**; T2 can be a follow-up once T1 metrics are stable.

`chat_messages` (258 rows) is too small to be a primary FT source but is useful for:
- **Query-pattern probing** (what do users actually ask?)
- **Few-shot exemplar selection** for the Gemini "few-shot" condition
- Style tokens for the FT corpus

### Compute path (revised 2026-05-14 — local FT primary)
- **Fine-tuning (PRIMARY):** **Mac M5 16 GB** via MLX-LM. QLoRA-4bit FT of `google/gemma-4-E4B-it` peaks ~7-9 GB (model 5 GB + activations 1.5-3 GB + LoRA + optimizer overhead). Estimated wall-clock for 3 epochs on ~23K examples: **5-7 hours**.
- **Fine-tuning (BACKUP):** Kaggle Notebooks (free, 30 GPU-hours/week, T4 16 GB, 9-hour sessions). Used only if local FT hits an unexpected blocker.
- **Inference (FT-SLM):** local Mac M5 + MLX-LM with `deadbydawn101/gemma-4-E4B-mlx-4bit` (~5 GB resident). MLX 10-20% faster than Ollama for Gemma 4 on Apple Silicon (direct Metal runtime).
- **Inference (Gemini comparators):** `gemini-3-flash` via Google API.
- **Inference (Tier-2 judge):** `claude-sonnet-4-6` via Anthropic API.
- **Streamlit + dashboard:** local Mac M5.
- **Result store:** local Parquet + private S3 mirror.

**Rationale for local-FT-primary:** zero data egress for sensitive student answers (`evaluation_questions`); no 30h/wk session cap; tighter iteration loop; pipeline runs entirely on instructor-side hardware which prefigures the production deployment model (architecture.md §9). Tradeoff: M5 is slightly slower per step than T4, and AC power is required for sustained runs.

### 3.2 External (to be sourced)
NCERTs (Class 6-12), official UPSC syllabus PDFs, PYQ papers from UPSC.gov.in, Drishti IAS / Vision IAS / Insights compilations, government PIB releases, Yojana / Kurukshetra magazines.

## Research findings (2026-05-14)

### Open-source SLM landscape (May 2026)
Active families (per BentoML / open-llm leaderboards): **Llama 3.x**, **Qwen3.5** (200+ languages, multimodal), **Phi-4-mini-instruct** (strong reasoning), **Ministral-3-3B** (edge-focused), **Gemma 3** (multimodal, 128K context). On focused tasks, Phi-3 / Gemma 2 / Mistral 7B deliver 80–90% of GPT-4 quality.

**Verified May-2026 landscape:**
- **Qwen3.5** (Alibaba): MoE 397B total / 17B active — too big for our setup. Smaller Qwen3 variants (~1.7B, ~8B) exist and run on M5.
- **Qwen3.6 27B**: too big for Kaggle T4 FT.
- **DeepSeek-V3.2** (MIT, 671B MoE / 37B active): best overall OSS but far too big.
- **Mistral Medium 3.5** (April 2026, 128B dense): too big.
- **Llama 4 Scout** (10M context): big variant, Indic strength unverified.
- **Gemma 3 4B**: 4.2GB RAM footprint, strong multilingual, Google-quality. **Sweet spot for M5.**
- **Phi-4-reasoning-plus**: strong reasoning, Indic strength unverified.
- **Sarvam-105B / "Indus"** (Feb 2026, India-built, MoE 9B-active): API only — 105B params won't fit local.
- **Sarvam-30B** (Feb 2026, MoE): borderline — too big for FT on free GPU.
- **Airavata** (7B Hindi-instruction-tuned, OpenHathi base, AI4Bharat): Indic-native, well-documented.
- **AryaBhatta-GemmaGenZ-Vikas** (Indic-FT of Gemma): per MILU paper, **the standout among Indic-FT'd open models** — outperforms its Indic peers.

### Final model shortlist — two FT candidates (user-locked 2026-05-15)

#### Candidate 1 — Gemma 4 E4B (instruction-tuned)

| Field | Value |
|---|---|
| HF id | `google/gemma-4-E4B-it` (base: `google/gemma-4-E4B`) |
| Params | 4.5B effective / 8B total (MatFormer + PLE) |
| License | Apache 2.0 |
| Context | 128K |
| Multilingual | 140+ pretrained / 35+ native instruction-tuned (Hindi in pretrained tier only) |
| Multimodal | text + image + audio ≤30s (not video) |
| Training cutoff | Jan 2025 |
| MLX path | `deadbydawn101/gemma-4-E4B-mlx-4bit` (direct 4-bit) |
| Memory | ~3 GB Q4 inference; ~7-9 GB peak FT |

#### Candidate 2 — Qwen3.5-4B (instruction-tuned)

| Field | Value |
|---|---|
| HF id | `Qwen/Qwen3.5-4B` (base: `Qwen/Qwen3.5-4B-Base`) |
| Params | 4.66B dense |
| License | Apache 2.0 |
| Context | 262K native, extensible to 1M |
| Multilingual | **201 languages, Hindi explicitly enumerated** |
| Multimodal | text + image (early-fusion VL) |
| Training cutoff | Not explicitly published; March 2026 release |
| MLX path | `mlx-community/Qwen3.5-4B-MLX-4bit` (native MLX) |
| Memory | ~3 GB Q4 inference; ~6-8 GB peak FT |

**Why two candidates:** isolates the architecture/pretraining variable. Both trained with the same LoRA recipe on the same FT corpus; only the base model differs. The Hindi-stratum delta (C1a − C1b) is a direct read on "Indic-via-FT vs Indic-via-pretraining". Other-task deltas test whether the result is portable across base SLM families or sensitive to which one we picked.

**Hindi coverage:** Gemma's Hindi is in the 140+ pretrained pool, not the 35+ native tier. Qwen lists Hindi explicitly. Per-model A2 Hindi probe locked in [eval-design.md §10 A2](eval-design.md) — both base models run a 200-Q Hindi MCQ probe pre-FT with pass criterion ≥ 30% accuracy (random + 5pp).

**Superseded shortlists (audit trail):**
- 2026-05-14 v1: Qwen3-8B + AryaBhatta-GemmaGenZ-Vikas + Gemma-3-4B
- 2026-05-14 v2: Sarvam-1 + Qwen3.5-9B + Aya Expanse 8B
- 2026-05-14 v3: gemma-3n-E2B-it + Phi-4-mini-instruct + Ministral-3-3B-Instruct-2512
- 2026-05-14 v4: gemma-4-E4B-it as single FT candidate.
- 2026-05-15 v5 **(current):** gemma-4-E4B-it + Qwen3.5-4B as two FT candidates.

### Closest prior-art benchmarks
- **MILU** (AI4Bharat, NAACL 2025) — 85K MCQs, 11 Indic languages, 8 domains, 41 subjects, **includes regional/state exam content**. GPT-4o tops at 74%. **Models worst on Arts & Humanities, Law & Governance — exactly UPSC's strong areas.** → MILU is the natural Indic baseline to also report alongside our UPSC eval.
- **JEEBench** — Indian entrance exam reasoning benchmark.
- **MedQA / MedR-Bench / HealthBench** — high-stakes professional-exam analog: multi-turn, physician-authored rubric, weighted scoring across safety/accuracy/communication. **Methodologically the closest analog for UPSC Mains.**
- No UPSC-specific LLM benchmark exists in the literature → designing one is novel.

### Eval methodology consensus (from 2025–2026 lit)
- For descriptive answers in high-stakes domains: **rubric-based LLM-as-judge** is the standard, *with* human spot-check oversight.
- Decomposed rubrics ("Does the answer mention X? Y? Z?") consistently outperform holistic scoring.
- Judge-model bias is a known confound — must (a) use a different family judge from candidate models, (b) calibrate against human gold, (c) report agreement metrics (Cohen's κ).

Companion code in workspace context (`core-backend/src/chatbot/...`) hints at chatbot tables in app DBs; will confirm via schema discovery.

### 3.2 External
Textbooks, reference books, PYQs, NCERTs, syllabus — to be sourced via web research (PIB, Drishti IAS, Insights, ClearIAS, official UPSC syllabus PDFs, NCERT PDFs).

## 4. Decisions log

| Date | Decision | Rationale |
|---|---|---|
| 2026-05-14 | Eval covers all four UPSC surfaces | User-confirmed |
| 2026-05-14 | 3-way comparison: FT-SLM / zero-shot Gemini-3-Flash / few-shot Gemini-3-Flash | User-confirmed |
| 2026-05-14 | Skip Mixpanel unless eval weighting requires real query distribution | Tangential to substance |
| 2026-05-14 | Final deliverable is a Streamlit dashboard (side-by-side query + aggregate metrics) | User-confirmed |

## 4a. Target architecture (Streamlit deliverable)

```
┌─────────────────────────────────────────────────────────┐
│ Streamlit app (single repo)                             │
│                                                         │
│ ┌──────────────┐    ┌──────────────────────────────┐    │
│ │ Page: Query  │    │ Page: Metrics Dashboard      │    │
│ │ - prompt box │    │ - per-task accuracy, ROUGE,  │    │
│ │ - 3 columns: │    │   BERTScore, LLM-judge, etc. │    │
│ │   FT-SLM /   │    │ - filter by task surface     │    │
│ │   zero-shot/ │    │ - calibration plots, etc.    │    │
│ │   few-shot   │    │                              │    │
│ └──────┬───────┘    └──────────┬───────────────────┘    │
│        │ live calls            │ reads cached results   │
│        ▼                       ▼                        │
│ ┌──────────────┐    ┌──────────────────────────────┐    │
│ │ Model layer  │    │ Eval results store           │    │
│ │ - FT-SLM     │    │ (parquet / sqlite locally,   │    │
│ │   (local)    │    │  one row per (question,     │    │
│ │ - Gemini API │    │  condition) with all metrics)│    │
│ └──────────────┘    └──────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

Implications:
- Eval-set responses are **pre-computed offline** and stored; dashboard reads cached results (fast, reproducible).
- "Side-by-side query" page issues **live** calls (for ad-hoc inspection, not for metric computation).
- FT-SLM inference must be hostable somewhere reachable from Streamlit — local GPU via vLLM/Ollama is simplest if hardware is available. **Open question: compute budget/hardware**.

## 5. Open questions / blockers

- **SLM choice** — pending research on May-2026 open-source landscape (Llama-4? Qwen-3? Phi-4-mini? IndicLLM?). Indic capability is a hard requirement; UPSC content is bilingual (English + Hindi often).
- **Eval set construction** — must source PYQs (Prev Year Questions) for Prelims + Mains; need authoritative answer keys.
- **Compute budget** — unspecified. Affects model size cap (1B vs 7B vs 14B).
- **Deployment target** — unspecified. Affects whether quantization matters for the comparison.
- **Mains/Interview grading** — humans? LLM-as-judge (with which judge)? Rubric-based?
- **FT-SLM hosting for Streamlit** — local GPU (vLLM/Ollama), or a hosted endpoint (Modal, Together, Fireworks)?
- **Live-query latency budget** for the dashboard's interactive page (affects model size we can ship).
- **FT task framing** — single multi-task SLM, or N task-specific LoRA adapters? (Recommended: single multi-task LoRA model; adapters are a fallback if one task degrades the others.)
- **Indic-native challenger model** — include Sarvam / Airavata / Indus as a third candidate alongside Qwen2.5? (Recommended: yes — if a domain-native model wins with less FT, that's a more interesting finding.)
- **Date cutoff for Current Affairs eval** — Current Affairs is time-bounded. Must fix a "knowledge frontier" date (e.g. "all questions about events before 2025-12-31") so both models are evaluated on the same temporal slice.

## 5a. Next steps (locked 2026-05-16)

Design is complete. Implementation is the gate to a publishable result. Phased plan:

### Phase 0 — Reconcile (≤1 h, blocking)
- **Resolve 1 vs 2 SLM scope.** `eval-design.md §2` reverted to single-Gemma; `experiment-report.md`, `architecture.md`, `project-brief.md` still treat 2 (Gemma + Qwen). User decision needed before any inference code is written.
- Confirm `git init` for the repo (the working dir is not yet a git repo per session start; needed for `manifest.json` SHAs).

### Phase 1 — Repo bootstrap (~1-2 h)
- `git init`, `git remote add` (optional)
- Create scaffold dirs: `scripts/`, `configs/`, `configs/prompts/`, `data/`, `results/`, `tests/`, `dashboard/`, `models/`, `adapters/`, `prompts/`
- Write `requirements.txt` from [eval-design.md §5.1](eval-design.md); pip-compile to lockfile with hashes
- Write `Makefile` with targets: `verify-env`, `freeze`, `build-ft-corpus`, `probe-hindi`, `ft-gemma`, `ft-qwen`, `infer`, `score-tier1`, `score-tier2`, `aggregate`, `test-hypotheses`, `dashboard`
- Write `scripts/verify_env.py` (Stage 1.1) — checks Python version, deps installed, API keys present, Postgres reachable, MLX-LM operational
- Run `pip install -r requirements.txt` and verify everything imports cleanly on M5

### Phase 2 — Data plane (~3-5 h)
- `scripts/freeze_eval_set.py` (Stage 1.2) → `data/eval_set.parquet` + `data/eval_set.sha256`
- `scripts/build_ft_corpus.py` (Stage 1.3) → `data/ft_corpus.parquet` + `data/ft_corpus.sha256`
- `tests/test_freeze_determinism.py` (same seed → same SHA, byte-for-byte)
- `tests/test_leakage_assertion.py` (eval_ids ∩ ft_ids = ∅ — hard stop)
- **Parallel track:** `scripts/build_upsc_facts.py` → `data/upsc_facts.json` (Constitution Articles 1-395, Schedules 1-12, major Acts, Five-Year Plans, schemes, office-holders). Sourced from public domain (constitution-of-india.net, india.gov.in, Lok Sabha records).

### Phase 3 — Pre-FT triage (~1 h)
- `scripts/run_hindi_probe.py` (Stage 2.2) — 200 Hindi MCQs against base Gemma-4-E4B-it and Qwen3.5-4B
- `scripts/gate_hindi.py` (Stage 2.3) — pass criterion ≥ 30%; per-model gate
- Output: `results/pre_ft_hindi_probe.parquet`

### Phase 4 — Fine-tuning (~1 evening setup + overnight run)
- `configs/lora.yaml` (rank=16, alpha=32, dropout=0.05, lr=2e-4, batch=1, grad_accum=8, max_seq_len=2048, epochs=3)
- `scripts/run_ft.py` wrapping `mlx_lm.lora` with checkpoint resume + training-log streaming
- Run #1: train Gemma adapter → `adapters/gemma4-e4b-upsc-v1/`
- Run #2: train Qwen adapter → `adapters/qwen35-4b-upsc-v1/` (if 2-SLM scope confirmed in Phase 0)
- `scripts/validate_adapter.py` (Stage 3.4) — 50 held-out samples per task per adapter

### Phase 5 — Inference (~1 day)
- Prompt templates in `configs/prompts/` (Jinja2)
- `scripts/run_inference.py` with `GemmaFTRunner`, `QwenFTRunner`, `GeminiZeroShotRunner`, `GeminiFewShotRunner` per [architecture.md §3.1](architecture.md)
- 2,000 items × (3 or 4) conditions = 6,000 or 8,000 prediction rows
- Resumable (append-only `predictions.parquet`), retry with backoff, cost ceiling enforcement

### Phase 6 — Scoring (~1 day)
- `scripts/score_tier1.py` — all 40+ Tier-1 metrics
- `scripts/score_tier2.py` — Anthropic Claude judge, prompt-cached, per-row disk cache
- `scripts/aggregate.py` — per-condition × stratum bootstrap CIs
- `scripts/test_hypotheses.py` — pairwise tests, BH-FDR correction

### Phase 7 — Dashboard (~1-2 days)
- Streamlit app with the 6 pages in [architecture.md §6](architecture.md): Aggregate metrics, Side-by-side query, Per-question drilldown, Calibration plots, Failure modes, Run comparison

### Phase 8 — Write-up (~half day)
- `scripts/render_report.py` auto-fills [`experiment-report.md`](experiment-report.md) §6 and §7 tables from `aggregate.parquet` and `hypothesis_tests.parquet`
- Human authors §8 Inference (discussion)

**Total: ~1.5 weeks of focused work** to publishable v1 result.

### What can run in parallel
- Phase 2 data-plane work and Phase 4 LoRA-config writing
- `data/upsc_facts.json` build while eval-set freezer is being written
- Streamlit dashboard scaffolding while FT is training overnight

## 6. Working artifacts (this directory)

Will live under `/Users/yeeshan/PrayasAI/Code/SLM/`:
- `project-context.md` — this file, the source of truth across the session
- `db-creds.txt` — provided
- `CLAUDE.md` — behavioral guardrails
- `eval-design.md` — metric set + protocol + statistics (drafted 2026-05-14, rev 2)
- `experiment-report.md` — pre-registered scientific report (Aim / Setup / Procedure / Expected outcome / Actual outcome / Results / Inference) — drafted 2026-05-14, trimmed 2026-05-14
- `architecture.md` — testing architecture (4 planes, reliability, failure modes, UPSC-specific design choices) — drafted 2026-05-14, trimmed 2026-05-14
- `project-brief.md` — non-technical one-pager for prayas leadership / mentors / stakeholders — drafted 2026-05-14
- `test-strategy.md` — Phase 1+2+3 test strategy (6 layers: snapshot, pre-flight, smoke, property, data-quality, negative) — drafted 2026-05-16, extended for Phase 3
- `configs/lora.yaml` — LoRA recipe, shared across Gemma + Qwen FT runs — drafted 2026-05-16
- (to be added) `schema/` — DB schema dumps
- (to be added) `data/eval_set.parquet` — frozen eval-set IDs

## 7. Session log

- 2026-05-14: Scoped DB access, model openness, task surfaces, comparison setup. Confirmed `psycopg2` 2.9.11 available; `psql` CLI not installed.
- 2026-05-14: Connected to `upscdev` (89 tables, 362 MB) and `prod-prayas-db` (57 tables, 820 MB), both PG 17.6 on RDS.
- 2026-05-14: Web search confirms no UPSC-specific LLM benchmark in literature → eval design is novel; closest priors are JEEBench (Indian entrance exams) and MedQA-style high-stakes exam evals.
- 2026-05-14: Final deliverable confirmed as Streamlit dashboard (side-by-side + aggregate).
- 2026-05-14: DB schema inventoried — strong content for all four UPSC surfaces; `evaluation_questions` (rubric-graded student answers) is a standout asset.
- 2026-05-14: MILU identified as the Indic reference benchmark; no UPSC-specific LLM benchmark exists in literature.
- 2026-05-14: app_dev inventoried; chatbot is a personalized tutor with rich student state (`staticContext`, `student_memory`). FT framing for v1 = standalone UPSC capability (T1), not personalization (T2).
- 2026-05-14: Compute path locked: Kaggle (free, 30h/wk T4) for FT, Mac M5 16GB + MLX-LM for inference. Ceiling 7B, comfort 3B at Q4.
- 2026-05-14: Eval-set target = ~2,000 Qs stratified across surfaces. Judge = Gemini-2.5-Pro (different family from Qwen candidates).
- 2026-05-14: v1 task scope = A (Prelims MCQ) + B (Mains generation) + C (Mains rubric grading) + E (Current affairs synthesis). Interview deferred to v2.
- 2026-05-14: v1 framing = T1 standalone capability (no per-student context). T2 personalization deferred to v2.
- 2026-05-14: Model shortlist locked: Qwen3-8B (primary), AryaBhatta-GemmaGenZ-Vikas (Indic challenger), Gemma-3-4B (optional small comparison). **Revised same day** after deeper landscape sweep → see below.
- 2026-05-14 (revision): Shortlist updated to **Sarvam-1 (2B) primary, Qwen3.5-9B (March 2026) generalist comparator, Aya Expanse 8B multilingual challenger.** Reason: Qwen3.5 small-model family (0.8B/2B/4B/9B) released March 2026 was missed in first pass; Sarvam-1 is the modern Indic-native 2B that explicitly beats Gemma-2-2B and Llama-3.2-3B on 10 Indic languages; Aya Expanse 8B is Cohere's benchmark-winning multilingual at size class.
- 2026-05-14 (user-locked v3): **Shortlist replaced by user pick: `gemma-3n-E2B-it`, `Phi-4-mini-instruct`, `Ministral-3-3B-Instruct-2512`.** All three edge-class instruction-tuned. Gemma-3n covers 140+ languages incl. Indic; Phi-4-mini (MIT) and Ministral-3-3B (Apache 2.0) do not officially list Hindi → Hindi-capability triage protocol added as eval-design §10 A2.
- 2026-05-14 (user-locked v4): Single FT candidate `google/gemma-4-E4B-it`. 4.5B effective / 8B total, Apache 2.0, 128K context, 140+ pretrained languages, multimodal text+image+audio≤30s (NOT video — that's the 31B variant).
- 2026-05-15 (user-locked v5 — current): **Second FT candidate added: `Qwen/Qwen3.5-4B`.** 4.66B dense, Apache 2.0, 262K context (extensible to 1M), **201 languages with Hindi explicitly enumerated**, multimodal (vision), early-fusion VL training. Released March 2026. Native MLX builds (`mlx-community/Qwen3.5-4B-MLX-4bit`). Adds direct test of "Indic-via-FT (Gemma) vs Indic-via-pretraining (Qwen)" hypothesis and architecture/family comparison. Both adapters trained with identical LoRA recipe on identical FT corpus; only the base model differs.
- 2026-05-14 (verification pass): Comparator updated to `gemini-3-flash` (current default Flash as of May 2026; Gemini 2.5 family is legacy). LLM-judge for Tier 2 switched from `gemini-2.5-pro` to `claude-sonnet-4-6` — both the candidate (Gemma) and the comparator (Gemini) are Google, so the judge must be a different family to avoid intra-family bias. MLX path simplified to direct 4-bit (`deadbydawn101/gemma-4-E4B-mlx-4bit`) plus `mlx_lm.lora` for FT — no GGUF detour needed.
- 2026-05-14 (compute path): Local M5 16GB confirmed feasible for QLoRA on Gemma-4-E4B (peak ~7-9 GB of 12 GB usable). **FT moved to local-primary, Kaggle backup.** Estimated 5-7 hours for 3 epochs on ~23K-example multi-task corpus. Verified against [Antigravity Lab M3 Max Gemma 4 FT guide](https://antigravitylab.net/en/articles/antigravity/gemma-4-finetuning-apple-silicon-mlx-guide), [Unsloth Gemma 4 doc](https://unsloth.ai/docs/models/gemma-4), [gemma4.dev MLX guide](https://gemma4.dev/run-local/gemma-4-mlx).
- 2026-05-14 (deliverables): [experiment-report.md](experiment-report.md) and [architecture.md](architecture.md) drafted. Pre-registered scientific report covers Aim / Setup / Procedure / Expected outcomes (predictions before execution) / Actual outcome (auto-filled by `render_report.py`) / Results / Inference; UPSC + ed-tech specific throughout. Architecture covers four planes (data / inference / scoring / dashboard) with idempotence, retry, leakage CI, S3 mirroring, audit log, DPDPA compliance.
- 2026-05-14 (post-feedback): Bloat removed from `architecture.md` (cut speculative repo layout, full production-deployment-stages section, redundant narrative). `experiment-report.md` trimmed (background compressed, sign-off boilerplate removed). New `project-brief.md` created as non-technical one-pager for stakeholders — covers goal, process, success criteria, timeline, risks in plain language without jargon.
- 2026-05-15 (name): User edited "Irshad" → "Yeeshan" in `architecture.md` and confirmed the propagation. Yeeshan is the preferred first name in all docs going forward; `irshad@prayas.ai` remains the email handle. Memory updated.
- 2026-05-15 (second SLM): Second FT candidate added: `Qwen/Qwen3.5-4B`. v1 now compares **four** conditions: C1a (Gemma-FT), C1b (Qwen-FT), C2 (Gemini-3-Flash zero-shot), C3 (Gemini-3-Flash few-shot). Hypothesis set expanded from H1-H3 (3 comparisons) to H1-H6 (6 pairwise comparisons across the 4 conditions). Verdict criteria reframed around "per-task champion" = max(C1a, C1b). FT corpus is identical between adapters; only the base model differs — isolates architecture/pretraining as the variable. Both adapters trained on M5 overnight (~10-14 h total). All five docs updated (eval-design, experiment-report, architecture, project-context, project-brief).
- 2026-05-16 (pedagogical clarity): Added a **Pedagogical Clarity** metric family. Three scopes: (i) **Task A explanation pass (NEW)** — a third inference pass elicits the model's explanation for its chosen MCQ answer; Tier 1 measures explanation quality vs gold (`prelims_pyq_questions.explanation` JSONB) using BERTScore-F1, ROUGE-L, Entity-F1, **Distractor coverage** (does it address each wrong option?), and **Reasoning-step density**; Tier 2 adds a 5-axis LLM-judge rubric (Step-by-step / Distractor addressing / Conceptual grounding / Specificity / Accessibility) on `claude-sonnet-4-6`. (ii) **Task C feedback Pedagogical Clarity** — Tier 2 only, 5-axis rubric (Actionability / Specificity / Constructiveness / UPSC-rubric fidelity / Coverage proportionality) measuring whether predicted strengths/improvements actually teach the student. (iii) **Task E synthesis Pedagogical Clarity** — Tier 2 only, 5-axis rubric (Syllabus grounding / Static-Dynamic bridge / Multi-dimensional framing / Specificity / Mains-utility framing) — Task E output is study material consumed by aspirants, so the teaching-task framing applies. Task B is NOT included — Mains generation is student-mimicking, not teaching, and existing G-Eval already covers what UPSC graders reward.
- 2026-05-16 (Tier-1 objective expansion): Added 15 more deterministic Tier-1 metrics per user feedback "majorly NON subjective, NUMERICALLY BACKED, scored objectively." **New shared utility:** `data/upsc_facts.json` — static lookup of Constitution Articles, Schedules, Acts, Five-Year Plans, schemes, office-holders; built deterministically from public sources; hashed in `manifest.json`. Powers fact-lookup metrics across Tasks A, B, E. **Task A explanation (5 → 8 metrics):** added Article/scheme citation accuracy (via lookup), Answer position bias (χ² test of A/B/C/D distribution), Sentence-length variance. **Task B Mains (11 → 16 metrics):** added Type-Token Ratio (MATTR), Flesch-Kincaid Grade Level, Paragraph count adherence, 4-gram repetition rate, UPSC fact-lookup precision. **Task C rubric grading (8 → 11 metrics):** added Score variance ratio (detects mean-collapse), JSON schema validity rate, Strengths/Improvements item-count adherence. **Task E synthesis (11 → 15 metrics):** added Compression ratio compliance, Glossary term recall (against `prod.glossary` 7,475 keywords), Source citation density, Lead-100-word entity recall, UPSC fact-lookup precision. New deps: `textstat==0.7.4`, `jsonschema==4.23.0`, `numpy==2.1.3` (explicit pin). All metrics are pure functions of the response + gold/source/static-lookup — no LLM-judge, no human judgment, fully reproducible.
- 2026-05-16 (next steps locked): Implementation plan added as §5a above — Phase 0 (reconcile 1 vs 2 SLM, git init) through Phase 8 (write-up). Total ~1.5 weeks focused work. Two pending design blockers identified: (a) 1-vs-2-SLM scope reconciliation, (b) git-init authorization. Awaiting user direction on Phase 0 before Phase 1 starts. User-standing instruction: project-context.md is to be updated at every significant step — saved as project-memory.
- 2026-05-16 (Phase 0 complete): User confirmed **dual-candidate v1** (Gemma-4-E4B-it + Qwen3.5-4B); `eval-design.md §2` re-expanded to dual spec, aligning with experiment-report.md / architecture.md / project-brief.md. **git initialized** on `main` branch with comprehensive `.gitignore` protecting `db-creds.txt`, `*.env`, `results/`, `adapters/`, `data/*.parquet`, model caches, Python build artifacts. `git check-ignore -v db-creds.txt` confirms protection. No commits made — staged tree left for user to inspect and authorize. Phase 0 done; ready for Phase 1 (repo bootstrap) and Phase 2 (data plane) on user go-ahead.
- 2026-05-16 (Phase 1+2 built): **Working code only — no scaffolding.** Files: `requirements.txt` (26 pinned deps), `Makefile` (6 working targets only: verify-env / build-facts / freeze / build-ft-corpus / test / clean), `scripts/db_creds.py` (29 LOC, parses db-creds.txt → DSN dicts for upscdev + prod only), `scripts/verify_env.py` (57 LOC, precondition gate for Python + deps + DB + API keys), `scripts/freeze_eval_set.py` (312 LOC, 4 pull fns + seeded stratified sampler + SHA-256 sidecar), `scripts/build_ft_corpus.py` (214 LOC, 4 builders + hard-stop leakage assertion), `scripts/build_upsc_facts.py` (69 LOC, schema-validates the facts JSON), `data/upsc_facts.json` (181 lines: 49 Articles, 12 Schedules, 23 Acts, 12 Plans, 20 schemes, office-holders, commissions — curated from public sources), `tests/test_freeze_determinism.py` (integration: same seed → same SHA), `tests/test_leakage_assertion.py` (unit: 4 cases). Total: 1,034 lines across 11 files.
- 2026-05-16 (scaffolding/dead-code removal): Per user "no scaffolding code, no empty/dead code" directive — removed: (a) 6 empty directories (`adapters/`, `dashboard/`, `models/`, `prompts/`, `results/`, `configs/`); (b) 2 empty `__init__.py` files (Python package markers were unneeded — scripts use explicit sys.path manipulation, tests invoke via subprocess); (c) 7 Makefile targets referencing scripts that don't exist yet (`probe-hindi`, `ft-gemma`, `ft-qwen`, `infer`, `score-tier1`, `score-tier2`, `aggregate`, `test-hypotheses`, `dashboard`) — will be re-added in their own phase when the scripts they invoke exist; (d) `app_dev` DB target from `db_creds.py` and `verify_env.py` — app_dev (chatbot personalization) is v2 scope, not v1, so its DSN logic and reachability check were dead code; (e) `@pytest.mark.integration` from `test_freeze_determinism.py` + the pytest-marker registration that would have required `pyproject.toml` — overkill for two tests, removed. Self-review fixes from earlier turn (O(n²) stratify, statsmodels/tenacity/pytest pins in eval-design §5.1) retained.
- 2026-05-16 (placeholder/shadow-code removal): User clarified "scaffolding" = placeholder/not-fully-implemented code. Found one real instance: **`tests/test_leakage_assertion.py` had a shadow `_check` function** re-implementing the leakage-assertion logic instead of testing the production code. Fixed: extracted `assert_no_leakage(eval_ids, ft_ids)` as a named function in `scripts/build_ft_corpus.py` (replacing the inline `assert overlap is empty` in `main()`); rewrote the test to `from build_ft_corpus import assert_no_leakage` so it exercises the real production function — any future bug in the assertion now fails the test. Full sweep then run: zero TODO/FIXME/NotImplementedError markers, zero bare-pass / ellipsis function bodies, zero empty-body functions, zero unused imports (a naive grep flagged `pandas` and `annotations` but `pandas` is aliased to `pd` and `annotations` is a `__future__` directive, both false positives).
- 2026-05-16 (test strategy): **`test-strategy.md` drafted** — covers Phase 1+2 only, 5 layers (pre-flight / smoke / property tests / data-quality spot-checks / negative tests) with concrete commands and binary pass criteria per check. Acceptance checklist (12 boxes) gates Phase 3. Layers 1-3 are automated via `make verify-env`, `make build-facts`, `make freeze`, `make build-ft-corpus`, `make test`. Layer 4 is pasteable Python in a REPL. Layer 5 is five hand-run negative scenarios (different seed differs, leakage trips on injection, verify-env fails on missing keys, build-ft-corpus fails on missing eval-set, determinism survives re-runs). Phase 3-8 each get their own narrow strategy when their code lands — all share the contract that Phase-2 outputs (`eval_set.parquet`, `ft_corpus.parquet`, `upsc_facts.json`) are immutable and verified.
- 2026-05-16 (local snapshot architecture): User imposed standing rule — **zero writes to any prod DB without per-instance explicit approval.** Verified existing code is already read-only (`conn.set_session(readonly=True)` everywhere; zero `INSERT/UPDATE/DELETE/DROP/ALTER/TRUNCATE` SQL anywhere in `scripts/`). Reorganized the data plane to centralize prod access in a single auditable script: **`scripts/snapshot_to_local.py` is the only script that connects to remote Postgres**, and it does so only via read-only SELECTs. Snapshots 8 tables (~44K rows; ~50-200 MB) into local SQLite at `data/prayas_local.sqlite`. New utility `scripts/local_db.py` exposes `read_table(name)` and `write_table(name, df, json_columns)`; JSONB and Postgres-array columns serialize to TEXT in SQLite and round-trip back to dict/list on read. Refactored `freeze_eval_set.py` and `build_ft_corpus.py` to read from local SQLite via `local_db.read_table()` — removed all remote-DB imports/queries from those scripts. `verify_env.py` is now an offline-only gate (no remote-DB ping; that happens during `make snapshot`). `Makefile` adds a `snapshot` target; `freeze` depends on `snapshot`. `.gitignore` adds `data/prayas_local.sqlite`. `test-strategy.md` adds Layer 0 covering the snapshot step. Saved as standing user-feedback memory `feedback_no_prod_writes.md` so future sessions inherit the constraint.
- 2026-05-16 (Phase 3 built): A2 Hindi-capability probe + gate. **`scripts/run_hindi_probe.py`** (95 LOC) — reads `upsc_prelims_ai_generated_que` from local SQLite, filters to `question_hindi` rows with `quality_pass_flag=True`, deterministically samples 200 with `seed=20260514`, loads the base MLX model via `mlx_lm.load(...)`, runs Pass-1 forced-choice prompts at temperature 0 via `make_sampler(temp=0.0)`, parses the model's first A/B/C/D token, writes per-row results to `results/pre_ft_hindi_probe.parquet`. Idempotent at the model level — re-running with the same `--model` replaces that model's rows. **`scripts/gate_hindi.py`** (45 LOC) — groups results by model, computes accuracy, exits 0 if all models ≥ 0.30 (random + 5pp on 4-option MCQs), else exits 1 with a list of failing models. Failure does NOT block Phase 4 FT; it just routes Hindi-stratum results to separate post-FT reporting per [eval-design.md §10 A2](eval-design.md). `Makefile` gains `probe-hindi` and `gate-hindi` targets (probe depends on snapshot). `test-strategy.md` extends Layer 2 with a §2.4 covering Phase 3 expectations. First-run wall-clock: ~15-40 min total (model downloads dominate; both models then cached). No remote-DB access. Same single-script prod boundary preserved.
- 2026-05-16 (Phase 4 built): LoRA FT pipeline. **`configs/lora.yaml`** (29 lines) — pinned recipe matching [experiment-report.md §4 Stage 3.1](experiment-report.md): rank=16, alpha=32, scale=10, dropout=0.05, lr=2e-4, batch=1, max_seq_len=2048, iters=20000 (~1 epoch over the 23K-pair corpus on M5 at ~1s/iter ⇒ ~5-6h wall-clock), `num_layers=16`, LoRA applied to `{q,k,v,o,gate,up,down}_proj`. `grad_checkpoint=true` to keep peak memory in the ~7-9 GB band on the M5 16 GB. **`scripts/run_ft.py`** (95 LOC) — reads `data/ft_corpus.parquet`, deterministic stratified 95/5 train/valid split (seed 20260514), writes `data/ft_split/{train,valid}.jsonl` in chat-message format, subprocess-invokes `python -m mlx_lm.lora` with the YAML config and CLI-injected `--model`/`--adapter-path`. Tees subprocess stdout to `adapters/<adapter-out>/training.log` alongside the terminal stream. **`scripts/validate_adapter.py`** (90 LOC) — Stage 3.4 sanity check. Loads `(base + adapter)` via `mlx_lm.load(adapter_path=...)`, samples 50 items per task from the valid-split JSONL (the same data MLX-LM held out during training, so the assertion checks generalization to data the adapter didn't see), runs each through `tokenizer.apply_chat_template + generate`, asserts task-specific parseability (Task A: letter regex; Task B: non-empty; Tasks C/E: valid JSON), exits 1 if >5% unparseable overall or per task. `Makefile` gains 4 targets: `ft-gemma`, `ft-qwen`, `validate-gemma`, `validate-qwen` — each depends on `build-ft-corpus`. `.gitignore` adds `data/ft_split/` (deterministically rebuildable). No remote DB. Prod boundary preserved.
- 2026-05-16 (probe size reduced 200 → 50): User requested smaller Hindi probe. Updated default `--n` in `scripts/run_hindi_probe.py`, plus references in `eval-design.md §10 A2`, `experiment-report.md §3.4` and `§6.2`, `test-strategy.md §2.4`.
- 2026-05-18 (Phases 0-2+test executed via uv venv, end-to-end green): `uv venv .venv --python 3.12` → `uv pip install -r requirements.txt`. Hit one install issue: `summac==0.0.6`, `alignscore==0.1.3`, `factscore==0.1.7` are git-only packages (not on PyPI) — removed from requirements.txt with note that Phase 6 will install them via git URLs. Hit a second runtime issue: `textstat==0.7.4` imports `pkg_resources` which was removed from setuptools 70+ → bumped to `textstat==0.7.13` (no pkg_resources). `make verify-env` then passes for env (only API-key warnings remain, expected — Phase 5 deps). Actual run results: `make build-facts` → upsc_facts.json validated (61 articles / 12 schedules / 23 acts / 12 plans / 20 schemes; SHA `4107cbd9...`). `make snapshot` → 45,370 rows / 221.1 MB SQLite snapshot of 8 prod tables (read-only confirmed); SHA `88441fda...`. `make freeze` → eval_set.parquet 2,000 rows, by task {A: 800, B: 400, C: 500, E: 300}, by language {A: 453en+347hi, B: 206en+194hi, C: 500en, E: 300en}; SHA `827928c2...`. `make build-ft-corpus` → 30,833 FT pairs {A: 15,721, B: 2,608, C: 9,600, E: 2,904}; leakage assertion PASS (overlap=0); SHA `c836848c...`. `make test` → 5/5 pytest passes (determinism + 4 leakage cases). Independent defense-in-depth leakage cross-check after the build: eval=2,000 ∩ ft=30,833 = 0. Pipeline solid; ready for Phases 3-5 when user runs them.
- 2026-05-16 (proper significance test for A2 gate): User asked for a proper statistical test instead of a hand-picked threshold. **`scripts/gate_hindi.py` now uses `scipy.stats.binomtest`** for a one-sided binomial test against the random-chance baseline (H0: accuracy = 0.25, H1: accuracy > 0.25) at α = 0.05. Model passes iff p-value < α. At n = 50 the critical value is **k = 18 (36% accuracy)** — i.e. `P(X ≥ 18 | n=50, p=0.25) ≈ 0.045`. CLI args `--alpha` and `--p-null` are configurable; defaults match the design. Updated `eval-design.md §10 A2` (pass criterion now references the binomial test + critical value), `experiment-report.md §3.4` (Stage 2.3 command) and `§6.2` (outcome table gains a `p-value` column), `test-strategy.md §2.4` (expected gate log + pass criterion). The probe itself (`scripts/run_hindi_probe.py`) is unchanged — it only writes raw `is_correct` per item; statistical inference is the gate's job. scipy was already pinned in `requirements.txt`.
- 2026-05-16 (Phase 5 built): Inference plane. **`scripts/runners.py`** (~250 LOC) — `EvalItem` and `Prediction` dataclasses; task-specific prompt builders for Pass-1 MCQ + Pass-2 confidence + Pass-3 explanation (Task A), Mains generation (B), rubric grading (C), and current-affairs synthesis (E); robust JSON-extractor for C/E parsing; `MLXLoRARunner` (loads base + LoRA adapter via `mlx_lm.load(adapter_path=...)`, generates with streaming for accurate TTFT); `GeminiZeroShotRunner` + `GeminiFewShotRunner` using `google-genai` SDK with `generate_content_stream` (also for TTFT) and `tenacity` retry-with-backoff on `TimeoutError`/`ConnectionError`; `GeminiFewShotRunner` deterministically picks 3 task-matched exemplars from `data/ft_corpus.parquet` sorted by `pair_id` and prepends them — pair_ids printed at startup for audit; `estimate_gemini_cost(eval_set, few_shot)` returns a conservative pre-run USD estimate. **`scripts/run_inference.py`** (~140 LOC) — orchestrator. Reads `data/eval_set.parquet`, computes per-(run_id, condition, question_id) resume set from existing `results/predictions.parquet`, processes pending rows; for Task A also runs Pass-2 confidence + Pass-3 explanation; checkpoints every N rows (default 50); records `run_id`, `condition`, `model_version`, `task`, `question_id`, `language`, `paper`, `subject`, `stratum_key`, `gold_payload`, `prediction`, `raw_output`, `latency_ms`, `ttft_ms`, `input_tokens`, `output_tokens`, `created_at` per [eval-design.md §5.3](eval-design.md); refuses to start C2/C3 above $25/condition cost estimate unless `--confirm-cost`. **Makefile** gains `infer-c1a`/`c1b`/`c2`/`c3` targets + an aggregate `infer` target; all share `RUN_ID ?= $(shell date +%Y%m%d)` so a single `make infer` invocation pairs the four conditions under one run_id. No remote DB. Prod boundary preserved.
- 2026-05-14 (revision): `eval-design.md` rewritten to **lead with quantitative metrics**. LLM-judge / G-Eval moved to Tier 2 diagnostic. Every metric is now pinned to a specific Python library (BERTScore, BLEURT-20, ROUGE-L, chrF++, METEOR, Entity-F1 via spaCy, SummaC-ZS, AlignScore, QWK via sklearn, ECE via torchmetrics). Scorer-model checkpoints SHA-pinned in `models/lockfile.json`. Per user feedback that primary metrics must be deterministic and reproducible, not subjective.
- 2026-05-14: Mains grading uses `evaluation_questions` as proxy gold (acknowledged caveat: those rubrics were likely LLM-graded; using same judge family creates circularity — see [eval-design.md] limitations).
- 2026-05-14: [eval-design.md] drafted — per-task research-backed metrics (Accuracy/Brier/ECE for A; BERTScore/BLEURT/G-Eval for B; QWK/per-criterion-κ for C; FactScore/SummaC/Entity-F1 for E) + statistical methodology + result-store schema. Awaiting user review.
- 2026-05-18 (Tier-2 vendor question): User asked whether we need two LLM providers and whether deterministic metrics need the Anthropic key. Confirmed: **Tier-1 metrics need NO API keys** — all run from local libraries + local scorer models (BERTScore deberta, BLEURT-20, SummaC, AlignScore, spaCy NER, etc.). The Anthropic dependency exists only for **Tier-2 Pedagogical Clarity / G-Eval / FactScore** rubric judging. Surfaced three honest paths: (A) keep Claude judge; (B) use blinded Gemini-Pro judge (saves a vendor, ~3-7pp self-preference bias per Panickssery et al. 2024 — acceptable for diagnostic-only Tier-2); (C) drop Tier-2 entirely for v1 (Tier-1 has ~45 metrics, sufficient on its own). Recommended **Path C** (drop Tier-2 for v1) given user's quantitative-first emphasis. Awaiting user decision before applying.
- 2026-05-18 (JSONL exporter for human inspection): User wanted to see the FT corpus without binary format. Clarified Parquet is not encrypted (just columnar binary). Added `scripts/export_corpus.py` (32 LOC) — converts any of `data/*.parquet` into pretty JSONL one-pair-per-line. Added `make export-corpus` target — produces `data/ft_corpus.jsonl` (153 MB, 30,833 rows) and `data/eval_set.jsonl` (11 MB, 2,000 rows). Both JSONLs gitignored (deterministically rebuildable from parquet). Confirmed inspection workflow with `head`, `grep`, `jq`. Sample of the first two pairs showed bilingual structure (en + hi pair for the same source question) — confirms language stratification flows correctly through the data plane.
- 2026-05-18 (tasks ABCDE clarified + Path A2 + CSAT added): User asked what tasks A/B/C/D/E are and why D is missing. Explained: D = Interview/Personality Test (DAF-driven), deferred to v2 per [experiment-report.md §11](experiment-report.md) — open-ended ethics/opinion content, hardest to score objectively, would force LLM-judge-only metrics that conflict with quantitative-first design. Letters A/B/C/E kept stable so v2 can drop D in without renumbering. Also surfaced a real train-test prompt mismatch: FT corpus used JSON-input/JSON-output format ([TASK=X] + JSON), but inference prompts in `runners.py` were natural-language single-letter for Task A. Three fix paths offered; user chose **Path A2: unify FT and inference prompts to JSON-format end-to-end**.
- 2026-05-18 (CSAT extracted from app DB / actually prod via learning_items): User noted CSAT MCQs should be in v1. Investigated — `paper` column exists on `learning_items` (in prod and app_dev) but not directly on `mcqs`. Distribution per `prod.learning_items.paper`: gs1=9,753, gs2=11,835, gs3=13,308, gs4=43, csat=3,254, essay=20. Actual MCQs by joined paper: gs1=8,619, csat=2,382. Confirmed CSAT content is genuinely different (number puzzles, reading comprehension, deduction). Modified `scripts/snapshot_to_local.py` SNAPSHOTS to use full-SQL-per-entry form (enables JOINs); the `mcqs` snapshot now `LEFT JOIN`s `learning_items` to attach `paper` + `tags`. Re-ran snapshot: same 11,424 mcqs but now each row carries paper. SHA `aba582c8...` (was `88441fda...`). `scripts/freeze_eval_set.py` Task A: filter mcqs to `paper ∈ {gs1, csat}` and use paper for stratification (CSAT now a first-class stratum). `scripts/build_ft_corpus.py` Task A: ADDED `mcqs` as a third FT source (was only `prelims_pyq_questions` + `upsc_prelims_ai_generated_que`), filtered to gs1+csat with explanations present, with a `_explanation_from_jsonb` helper to flatten the prod.mcqs `explanation` JSONB array of `{content}` blocks. Eval set after rebuild: Task A includes 20 CSAT items (stratum `CSAT|UNTAGGED|silly=0|en`) + 326 GS1 (uppercase, from mcqs) + 454 gs1 (lowercase, from prelims_pyq_questions — paper-tag case inconsistency between source tables is a known cleanup item, doesn't affect correctness).
- 2026-05-18 (Path A2 implementation — unified JSON I/O across all tasks): Updated `TASK_INSTRUCTIONS` in both `scripts/build_ft_corpus.py` and `scripts/runners.py` to identical strings — the model now sees the same instruction text at train and inference time. All tasks now emit JSON output: Task A → `{"answer": "<letter>", "explanation": "..."}`; Task B → `{"answer": "<full Mains essay>"}`; Task C → `{"score": ..., "strengths": [...], "improvements": {...}}`; Task E → `{"prelims_info": "...", "mains_info": "..."}`. `runners.py` consolidates prompt-building into one `build_prompt(item)` function + a `_input_for(item)` helper that mirrors the FT-corpus `input` JSON exactly. `parse_output` now uses `_extract_json` for all tasks; Task A retains a regex fallback for stragglers. **Pass-3 (explanation) call removed** from `run_inference.py` — explanation now comes from the Pass-1 JSON output, saving one inference call per Task-A item (~3,200 fewer model calls across all 4 conditions × 800 Task-A items). Pass-2 (verbal confidence) stays as a separate call. Note: prayas will provide production prompts for Mains model-answer generation and Prelims explanation generation; swap-in is a single edit to `TASK_INSTRUCTIONS` in both files (keeping the same shape).
- 2026-05-18 (vendor question — 2 LLM providers?): User asked whether we need both Google and Anthropic. Confirmed Tier-1 metrics need ZERO API keys (all run from local libraries + local scorer models). Tier-2 LLM-judge is the only thing needing the Anthropic key. Three paths offered: A=keep Claude judge (current), B=blinded Gemini-Pro judge (saves vendor, ~3-7pp self-preference bias per Panickssery et al. 2024), C=drop Tier-2 entirely for v1. Recommendation: **Path C** for v1 given quantitative-first directive. Decision pending.
- 2026-05-18 (Path A2 + CSAT rebuild executed): Fixed `_explanation_from_jsonb` defensive cast — one prod.mcqs row had a dict (not string) in `content`; coerce non-string values to JSON-serialized text instead of aborting the build. Rebuilt eval+FT artifacts end-to-end. **`data/eval_set.parquet`**: 2,000 rows, by task {A: 800, B: 400, C: 500, E: 300}; Task A by source_table {mcqs: 90 (GS1=70, CSAT=20), prelims_pyq_questions: 454, upsc_prelims_ai_generated_que: 256}; SHA `e2b62a3f...`. **`data/ft_corpus.parquet`**: 41,749 pairs (was 30,833 before CSAT/mcqs were added), by task {A: 26,638, B: 2,608, C: 9,600, E: 2,903}; Task A source_table breakdown {mcqs: 10,909 (GS1=8,548, CSAT=2,361), prelims_pyq_questions: 6,770, upsc_prelims_ai_generated_que: 8,959}; leakage assertion PASS (eval ∩ ft = ∅); SHA `d57be52c...`. CSAT now a first-class FT + eval stratum. Phases 3-5 ready to re-run against the larger FT corpus when user pulls the trigger.
- 2026-05-18 (`/fewer-permission-prompts` skill run): Scanned 17 recent transcripts across all prayas projects. Top non-auto-allowed Bash patterns were all interpreters (`python3 <<`, `uv run`, `bash -c`), package runners (`npx jest/tsc`, `npm run`), or mutating commands (`git push/commit/add`, `pkill`, `pip3 install`, `psql -h`) — none safe to wildcard-allowlist. Auto-allowed commands (`grep`, `cat`, `ls`, `git status/diff/log/show`, `find`, `sed -n`, `ps`, `which`, `head`, `tail`, `wc`, `echo`, `sleep`, etc.) already cover everything routine. MCP-tool count below threshold (4 total Atlassian calls). No changes made to `.claude/settings.json`.
- 2026-05-18 (Phase 6 built — scoring + aggregation + hypothesis tests, Path C / Tier-1 only): User said production prompts will arrive later and will be "different tasks either way" — proceeded with code build. Path C confirmed as v1 default (no LLM-judge, no Anthropic key). **Three new scripts (~1,000 LOC total):** `scripts/score_tier1.py` (~480 LOC) — per-row scoring for all 4 tasks. Task A: accuracy, UPSC negative-marking score (paper-aware: GS1 ±2/±0.667, CSAT ±2.5/±0.833), brier loss, format-fail rate, explanation entity-F1 (spaCy NER), distractor coverage, reasoning-step density per 100w, Article citation accuracy (vs `upsc_facts.articles`), sentence-length variance, batched BERTScore-F1 + ROUGE-L on explanation (English + Hindi separately). Task B: BERTScore-F1 + ROUGE-L + chrF (batched), word/sentence/paragraph count adherence, entity-F1, date+number exact-match F1, Hindi code-mixing rate (unicodedata), MATTR-100 lexical diversity, Flesch-Kincaid grade (textstat), 4-gram repetition rate, UPSC fact-lookup precision. Task C: schema validity (jsonschema), strengths/improvements token-F1 (spaCy lemma), strengths-list BERTScore-F1, item-count adherence, score abs-error (per-row; QWK/Spearman/Pearson/confusion in aggregate). Task E: entity-F1 vs gold + hallucination rate (entities not in source) + coverage of source entities, date/numeric F1 vs source, compression-ratio score (target 0.20-0.50), citation density per 100w, lead-100w entity recall, UPSC fact-lookup precision, prelims+mains BERTScore-F1 + ROUGE-L + mains chrF (batched, English). `scripts/aggregate.py` (~200 LOC) — per-(condition, task, language) mean + bootstrap-95% CI for every per-row scalar (seeded RNG, N=1000). Adds aggregation-level metrics: ECE-15bin, Brier Skill Score (vs base-rate predictor), position-bias χ² (Task A); QWK/Spearman/Pearson/confusion-matrix/score-variance ratio (Task C); silly-mistake breakdown + paired bilingual EN-vs-HI delta (Task A, written to `.extras.json` side file). `scripts/test_hypotheses.py` (~120 LOC) — pairwise paired-t + Wilcoxon on every (task, metric, condition-pair) with ≥5 paired observations, percentile-bootstrap CI on the difference, BH-FDR correction across the full test family (statsmodels.multipletests). Orientation flipped for lower-is-better metrics (brier_loss, format_fail, score_abs_err, hallucination_rate, ngram4_repetition). New deps: `bert-score==0.3.13`, `spacy==3.8.2` + `en_core_web_sm==3.8.0` (added to requirements.txt). **Deferred for v1** (documented in score_tier1.py docstring): BLEURT-20 (bleurt-pytorch git-only), SummaC/AlignScore/FactScore (Task E faithfulness — git-only), generation perplexity (heavy base-model load, dominated by BERTScore signal), glossary recall (needs `prod.glossary` in snapshot), METEOR (wordnet corpus dep, dominated by BERTScore + ROUGE-L). Makefile gains `score-tier1`, `aggregate`, `test-hypotheses` targets; aggregate and test-hypotheses depend on score-tier1.
- 2026-05-21 (Gemma unblocked + Hindi probe artefact-corrected — pre-registered prediction refuted): User said "run the experiment, unblock Gemma first." Investigation: `deadbydawn101/gemma-4-E4B-mlx-4bit` raised `ValueError: Received 126 parameters not in model` (multimodal-vs-text-only). `principled-intelligence/gemma-4-E4B-it-text-only` failed conversion too — its `Gemma4TextModel` encoder layout (no `model.` prefix, no `lm_head`) doesn't match what mlx-lm's `gemma4_text` expects. `mlx-community/gemma-4-e4b-it-4bit` (canonical, 2026-05-19) ALSO failed with the same 126-param error. Root cause: mlx-lm 0.31.3's `sanitize()` doesn't strip K/V projections for Gemma 4's shared-attention layers (24-41) — fixed upstream on 2026-05-04 in commit `df1d3f3c` ("Fix Gemma 4 sanitize() not stripping KV projections for shared layers") but not yet in any tagged PyPI release. Pinned `mlx-lm @ git+...@df1d3f3c…` in requirements.txt (replace with next tagged release once it ships). Gemma now loads + generates clean. **Probe re-run also surfaced a measurement-protocol bug:** plain-text prompts produced 0/50 for Gemma (instruction-tuned EOS-on-first-token without chat template) and 18/50-truncated Qwen responses (max_tokens=6 cut off `<think>` block). Patched `scripts/run_hindi_probe.py` to wrap prompts via `tokenizer.apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)` and bumped `max_tokens` 6 → 24. **Final artefact-corrected probe outcome (50 Hindi MCQs, seed 20260514):** Gemma 4 E4B-it 26/50 = **52.0 %** (p < 0.00001 — PASS); Qwen 3.5-4B 15/50 = **30.0 %** (p = 0.252 — FAIL). **Direction inverted vs experiment-report §5.1 pre-registration** which predicted Qwen would beat Gemma on Hindi by ≥ 5 pp. Empirically the opposite: Gemma's 140-language pretrained pool delivers stronger usable Hindi than Qwen's explicit 201-language enumeration. Qwen's Hindi stratum will be reported as a separate post-FT finding per A2 protocol, not folded into the bilingual aggregate. FT proceeds for both models. Updates landed in `experiment-report.md §6.2`, `eval-design.md §2.0a / §2.1`. `data/upsc_facts` artefacts unchanged. Ready for Stage 3 (LoRA FT, ~5-7 h each adapter on M5).

- 2026-05-21 (Stage 3 FT — 16 GB M5 OOM diagnosis + 24 GB handoff): Three `make ft-qwen` attempts OOM'd on the M5 (val pass survives, first training step crashes with `kIOGPUCommandBufferCallbackErrorOutOfMemory`). Reductions tried: max_seq 2048→1536, num_layers 16→8 — each still OOM. Spawned a research subagent: surveyed 2024-2026 LoRA literature + mlx-lm GitHub. Root cause is **two open mlx-lm bugs**: #828 (val-pass cache held when first train step runs) + #1185 (Metal command-buffer leak on Qwen3.5 LoRA), compounding with the macOS default Metal GPU cap (`iogpu.wired_limit_mb=0` → ~12 GB on a 16 GB device). Fix has three layers: (a) macOS `sudo sysctl iogpu.wired_limit_mb=21504` raises the OS cap to 21 GB; (b) in-process `mx.set_wired_limit(20 GiB)` + `mx.set_cache_limit(512 MiB)` keeps MLX inside the cap; (c) explicit `mx.clear_cache()` between val and train flushes the residual. Recipe restored to pre-registration intent (rank=16, num_layers=16, max_seq=2048) with two additions: `grad_accumulation_steps: 8` (effective batch 8, zero extra peak per Dettmers QLoRA NeurIPS 2023) and `val_batches: 25` (halved, reduces val→train residual). `iters` adjusted 20000 → 16000 (≈3 epochs at effective batch 8 over 42,701 pairs). Vanilla LoRA, **not DoRA** (MLX's DoRA materializes the dense norm and eats budget; the ~1-3 pt benefit isn't worth it at this hardware tier). NOT switching to PyTorch+peft+bitsandbytes — published QLoRA-vs-MLX adaptation-quality gap < 2 % per QLoRA paper, not worth breaking the rest of the pipeline. Expected peak ~16-18 GB; wall-clock ~10-14 h per adapter on a 24 GB M-series. Code committed: `configs/lora.yaml` restored + commented with research citations, `scripts/run_ft.py` gains a pre-flight `mx.set_wired_limit/set_cache_limit/clear_cache` block before invoking `mlx_lm.lora`. **Hardware handoff:** the FT runs are passed to a 24 GB M-series device run by another team member. Sources: DoRA (Liu et al., ICML 2024 Oral), LoRA+ (Hayou et al., ICML 2024), QLoRA (Dettmers et al., NeurIPS 2023), "Locating and Editing Factual Associations in GPT" (Meng et al., NeurIPS 2022), mlx-lm issues #828 + #1185, MLX Memory Safety Checklist (dev.to). Stage 4 (inference) blocked until the FT runner ships back `adapters/qwen35-4b-upsc-v1/` and `adapters/gemma4-e4b-upsc-v1/`.

## Runbook for the FT handoff (24 GB M-series device)

Pre-flight (run once per shell session):

```bash
# 1. Confirm hardware
sysctl hw.memsize                            # expect 25769803776 (24 GiB)

# 2. Raise the macOS Metal GPU cap (default is ~18 GB on 24 GB; bump to 21 GB)
sudo sysctl iogpu.wired_limit_mb=21504

# 3. Activate the project virtualenv
cd /path/to/SLM
source .venv/bin/activate                    # or: export PATH=".venv/bin:$PATH"

# 4. Sanity-check mlx-lm + Gemma 4 load
python -c "from mlx_lm import load; load('mlx-community/gemma-4-e4b-it-4bit', lazy=True); print('OK')"
```

Stage 3 — fine-tune both adapters (sequential; ~10-14 h each):

```bash
make ft-qwen     2>&1 | tee /tmp/ft-qwen.log
make ft-gemma    2>&1 | tee /tmp/ft-gemma.log
```

Outputs:
- `adapters/qwen35-4b-upsc-v1/{adapters.safetensors,adapter_config.json,training.log,…}`
- `adapters/gemma4-e4b-upsc-v1/{adapters.safetensors,adapter_config.json,training.log,…}`

Stage 3.4 — validate (each ~5 min):

```bash
make validate-qwen
make validate-gemma
```

Validator exits 1 if > 5 % of held-out generations are unparseable per task.

What to send back:
1. The entire `adapters/` directory (both folders, ~50-100 MB).
2. Both `training.log` files (final loss curves + Iter reports).
3. A line from `sysctl iogpu.wired_limit_mb` and `python -c "import mlx.core as mx; print(mx.get_peak_memory()/1024**3)"` from the end of training (for the experiment-report §6.1 run-metadata section).

If anything OOMs again: send the last 50 lines of the training log and `vm_stat` output; we triage from there.
