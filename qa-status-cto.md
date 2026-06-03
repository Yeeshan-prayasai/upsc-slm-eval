# QA Status — UPSC SLM Evaluation v1

**For:** CTO
**Owner:** Yeeshan
**Last updated:** 2026-06-03

---

## 1. What's been done

- **Eval set frozen:** 2,000 stratified items across 4 core tasks (A Prelims MCQ × 800, B Mains gen × 400, C Mains rubric grading × 500, E Current Affairs × 300) + 2 production-prompt capability tests (F, G — reuse A and B items respectively). 12,800 predictions per condition × 4 conditions = **51,200 total predictions**. SHA-256 hashed.
- **FT corpus:** 41,749 supervised pairs. CI assertion `eval ∩ ft = ∅` blocks training if any eval `question_id` appears in the FT corpus.
- **A2 Hindi probe (pre-FT):** Gemma-4-E4B-it **52 %** (PASS, binomial p < 0.00001 vs random); Qwen-3.5-4B **30 %** (FAIL). Qwen Hindi stratum routed to separate finding per protocol.
- **Two LoRA adapters trained** with identical recipe (rank=16, α=32, dropout=0.05, target_modules={q,k,v,o,gate,up,down}_proj, lr=2e-4, 3 epochs). Same FT corpus, same eval — isolates architecture as the variable.
- **4-condition × 6-task inference** complete.
- **Tier-1 scoring → aggregate → BH-FDR hypothesis testing → auto-renderer** implemented; smoke-tested end-to-end on synthetic data.


## 2. The 6 tasks under test

4 core comparison tasks (hypothesis-test family) + 2 production-prompt capability tests (reuse the A and B eval items; no new questions).

| Task | Surface | Input | Output | Gold reference | n | Source table |
|---|---|---|---|---|---:|---|
| **A** | Prelims MCQ | question + 4 options (paper, subject, language) | `{answer: "A\|B\|C\|D", explanation, confidence}` (3 passes) | `correct_option` + reference `explanation.{en,hi}` | 800 | `prelims_pyq_questions` ∪ `upsc_prelims_ai_generated_que` ∪ `mcqs ⨝ learning_items` |
| **B** | Mains generation | question + paper + subject + target word-count | `{answer: <full Mains text>}` | `pyqs.model_answer` | 400 | `pyqs` |
| **C** | Mains rubric grading | question + student_answer + max_score | `{score, strengths[], improvements{intro,body,conclusion}}` | `evaluation_questions.{score,strengths,improvements}` | 500 | `evaluation_questions` |
| **E** | Current Affairs synthesis | date + title + source_text | `{prelims_info, mains_info}` | `news_articles.{prelimsInfo,mainsInfo}` (cutoff 2026-04-30) | 300 | `news_articles` |
| **F** | Prelims Explanation Generation *(production-prompt capability test)* | Task A item + **gold correct letter** + prayas production prompt (10 KB) | `{english, hindi}` bilingual explanation | Task-A reference `explanation` | 800 *(reuse A)* | reused |
| **G** | Mains Model-Answer Generation *(production-prompt capability test)* | Task B item + prayas DSL production prompt (21 KB) | Mains answer in prayas house style | Task-B `model_answer` | 400 *(reuse B)* | reused |

**Per-condition prediction count:** 800 + 400 + 500 + 300 + 800 + 400 = **3,200**; × 4 conditions = **12,800**. Tasks A, B, F, G evaluated in both English and Hindi where the source provides bilingual content.

**Four conditions under test:**

| ID | What |
|---|---|
| **C1a** | Gemma-4-E4B-it + LoRA adapter (FT on prayas UPSC corpus) |
| **C1b** | Qwen-3.5-4B + LoRA adapter (same recipe) |
| **C2** | `gemini-3.5-flash` zero-shot |
| **C3** | `gemini-3.5-flash` few-shot (3 task-matched exemplars from FT corpus) |

---

## 3. What we measure (~45 Tier-1 deterministic metrics)

| Task | Metric families |
|---|---|
| **A** | Accuracy + UPSC negative-mark score + ECE 15-bin + Brier + Brier-skill + format-fail + position-bias χ² + bilingual-delta paired-t + McNemar on paired EN/HI + silly-mistake breakdown · explanation BERTScore-F1 + ROUGE-L + Entity-F1 + Distractor coverage + Reasoning-step density + Article-citation accuracy + Sentence-length variance |
| **B** | Answer BERTScore-F1 + ROUGE-L + chrF++ + Entity/Date/Number F1 + Word/Sentence/Paragraph adherence (asymmetric: undershoot penalty 1.5×) + Hindi code-mix + MATTR-100 + Flesch-Kincaid + 4-gram repetition + UPSC fact-lookup precision + JSON schema validity |
| **C** | QWK + Score MAE + Spearman ρ + Pearson r + Confusion-matrix (low/mid/high) + Score-variance ratio + Strengths/Improvements token-F1 + per-section (intro/body/conclusion) token-F1 + Strengths sentence-BERTScore + JSON schema validity + Item-count adherence |
| **E** | Mains BERTScore-F1 + ROUGE-L + chrF++ + Entity-F1 vs gold + Hallucination rate (entities not in source) + Source-entity coverage + Date/Numeric F1 + Compression-ratio score + Citation density + Lead-100w entity recall + UPSC fact-lookup precision + Subject-tag PESEE proxy |
| **F** | All Task-A explanation metrics + chrF++ (Devanagari-robust) + Hindi-branch Devanagari purity |
| **G** | All Task-B metrics + Dimension-keyword coverage (PESEE lexicon) + Directive-conditioned discourse density (symmetric log-ratio) |
| **Universal** | Latency p50 / p95 / p99 + TTFT + Tokens/sec + Cost/query + Format-validity rate |
| **Statistical** | Paired t-test + Wilcoxon signed-rank + BH-FDR at q=0.05 across ~1,800 cells + percentile bootstrap 95 % CI N=1000 (seed 20260514) + Cohen's d / h effect sizes + per-stratum heatmap (champion vs C3) |

## 4. QA Plan — 5 layers

| Layer | Failure mode it prevents | Mechanism |
|---|---|---|
| **Methodology** | Post-hoc rationalization; moving-target experiment | Pre-registered hypotheses + verdict criteria locked in `experiment-report.md §5` before any results landed; two-base-model A/B isolates architecture |
| **Leakage** | Eval-set contamination inflating scores | `eval_ids.isdisjoint(ft_ids)` CI assertion blocks FT; eval-set SHA-256 hashed + sidecarred |
| **Computational** | Bugs in scoring code producing wrong values | 20-item bug audit + fix; synthetic-data end-to-end smoke before real-data run; deterministic test for freeze script |
| **Statistical** | False positives from running ~1,800 tests | BH-FDR at q=0.05; dual-test agreement (t AND Wilcoxon must both fire); paired-bootstrap CI; Cohen's d/h alongside p-values |
| **Reproducibility** | Numbers can't be re-derived by anyone else | Seed 20260514 everywhere; pinned `requirements.txt` library versions; pinned scorer-model checkpoints (BERTScore roberta-large, spaCy en_core_web_sm 3.8.0); per-run `manifest.json` with SHAs of code commit, eval set, FT corpus, both adapters, prompt files |

## 5. Acceptance criteria

Production-grade thresholds. Each row is a binary go/no-go. Anything failing means the result is not cleared for publication or production decision.

| Criterion | Threshold | Verification | Status |
|---|---|---|---|
| Eval-set integrity | SHA-256 of `data/eval_set.parquet` matches `data/eval_set.sha256` | `sha256sum -c` | ✓ |
| No FT leakage | `eval ∩ ft = ∅` | CI assertion in `scripts/build_ft_corpus.py` | ✓ |
| All conditions complete | Row count = 3,200 for each of C1a, C1b, C2, C3 | `predictions.parquet` `groupby(condition).size()` | C1a ✓, C2 ✓, C3 ✓, C1b in flight (3,160 / 3,200) |
| Format-validity rate | **≥ 0.90** per (condition, task) — prompt-only JSON benchmark floor | `scores_tier1.format_valid.groupby(condition, task).mean()` | pending scoring |
| Hallucination rate (Task E champion) | **≤ 0.15** — eval-design §4.5 boundary; above this counts as "noticeable invented content" | `aggregate.parquet` row for `(task=E, metric=hallucination_rate)` | pending scoring |
| Calibration (Task A champion ECE-15bin) | **≤ 0.10** — production LLM mid-range acceptable bar | `aggregate.parquet` row for `(task=A, metric=ece_15bin)` | pending scoring |
| Pipeline audit | 0 unresolved critical or important findings | This run: 20/20 resolved (commit `5df55e1`) | ✓ |
| Statistical layer | BH-FDR-corrected p + paired CI + Cohen's d or h for every (task, metric, condition-pair) | `hypothesis_tests.parquet` row count | pending |
| Render coverage | ≥ 95 % of expected (condition × task × metric) cells populated; remainder are documented deferrals | `render_report.py --check` | pending |
| Manifest | SHAs of code commit, eval set, FT corpus, both adapters, scorer-model checkpoints, prompt files | `runs/<id>/manifest.json` | partial — written on finalize |
| Effect-size threshold for "production-relevant" | **Cohen's d ≥ 0.5** (medium) to declare a finding worth acting on; d ≥ 0.2 reported as suggestive | `hypothesis_tests.effect_size` filter | applied at report-render time |
| Pre-registered verdict evaluable | §1.3 of `experiment-report.md` (Strong Win / Non-inferiority / Loss per task) derivable from `aggregate.parquet` | Manual cross-check | pending |

## 6. Expected v2 improvements

Tied to specific deferrals catalogued in `eval-design.md §4` (deferred metrics) and `§9` (out-of-scope tasks).

| Iteration | Adds | Why |
|---|---|---|
| **v2 Tier-1 expansion** | BLEURT-20 (Task B/F faithfulness), SummaC-ZS + AlignScore + FactScore (Task E NLI-based faithfulness), generation perplexity (Task B), glossary-term recall (Task E), METEOR (Task B) | All currently git-only Python deps; deferred from v1 to keep PyPI dep stack clean. Tightens CI on the BERTScore-only signal we currently use. |
| **v2 Tier-2 LLM-judge** | Pedagogical Clarity 5-axis rubric (Tasks A, C, E) + G-Eval 5-axis (Task B) via `claude-sonnet-4-6` | Surface teaching-quality dimension that surface-similarity metrics miss; Kendall's τ vs Tier-1 reported to make the disagreement visible. |
| **v2 long-form coherence** | PDD (Positional Discourse Divergence, NAACL-Short 2024) for Task G | Strongest published deterministic long-form coherence metric — beats DiscoScore + BARTScore by ~10 correlation points at system level. Requires RST/PDTB discourse parser. |
| **v2 Task-C ground truth** | 50-row human-mentor calibration | Removes the "Task C gold was itself LLM-generated" limitation (current `experiment-report.md §9` item 1). |
| **v2 difficulty weighting** | IRT-based per-item difficulty estimates | Weight per-item gains by question difficulty; differentiates models that ace easy questions from those handling hard items. |
| **v2 new tasks** | T2 personalized tutoring (Q + student state → A); Task D interview / DAF question generation; multi-turn conversational eval | UPSC tutor is multi-turn in production; v1 is single-turn. T2 introduces student-state as a variable. |
| **v2 OOD test** | Mains 2024 / 2025 PYQ holdout | Temporal out-of-distribution check; v1 mixes years. |
| **v2 robustness** | Adversarial-prompting evaluation | Currently out of scope. |
| **v2 live validation** | A/B with real prayas students | Final ground-truth signal beyond automated metrics. |
| **v2 economic Pareto** | Cost-adjusted quality Pareto front (quality × $/query) | v1 reports cost and quality separately; v2 combines them for deployment decisions. |
