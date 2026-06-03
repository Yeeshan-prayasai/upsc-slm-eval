# v2 Expert Input Plan

**For:** CTO + UPSC content lead
**Owner:** Yeeshan
**Last updated:** 2026-06-03
**Reads with:** [`experiment-report.md §8.6`](experiment-report.md) (v2 roadmap, 14 items by priority)

This doc answers: *what specifically does each v2 item need from a domain expert, how many hours, and which experts gate the critical path.*

Estimates are grounded in: published rubric-design literature (avg 14.3 criteria per item; 3 raters per item for IRR; Krippendorff α ≥ 0.6 required) [[Autorubric](https://arxiv.org/html/2603.00077v1)], one documented LLM-rubric project at **2,400 expert hours over 12 weeks** [[Appen](https://www.appen.com/llm-evaluation-rubrics)], IRT minimum sample sizes (150/item for 1PL, 500+/item for 3PL) [[IRT sample-size tutorial](https://journals.sagepub.com/doi/10.1177/25152459251314798)], and UPSC mentor-grading throughput from prayas evaluation services (~24 working hours per 5-question evaluation = ~5 h/answer for deep grading) [[Prepp IAS](https://evaluation.prepp.in/), [SuperKalam](https://superkalam.com/upsc-mains-evaluation)].

---

## 1. Expert roles

| Role | Skill profile | Used for |
|---|---|---|
| **UPSC content lead** | Senior mentor, ≥3 yrs UPSC Mains evaluation, bilingual (EN+HI), familiar with prayas rubric/style | Rubric design, gold-answer curation, Hindi corpus, difficulty labels |
| **UPSC mentor (×2)** | Active mentors, EN+HI, currently grading at prayas | Inter-rater reliability validation; Task C human-mentor calibration |
| **SLM/ML methodology expert** | NLP eval, statistics (IRT, BH-FDR, bootstrap), HuggingFace + transformers | Pipeline audit, constrained decoding, calibration refactor, IRT fitting |

Two UPSC mentors are needed for inter-rater reliability — published LLM-rubric work uses **3 raters per item** with **Krippendorff α ≥ 0.6** as the floor for usable agreement [[Encord IRR](https://encord.com/blog/interrater-reliability-krippendorffs-alpha/)].

---

## 2. UPSC content/pedagogy expert tasks

Each row maps to a v2 item in [`experiment-report.md §8.6`](experiment-report.md). Hour estimates are per task per expert (not parallelized).

| # | v2 item | What the UPSC expert produces | Hours | Why this number |
|---|---|---|---:|---|
| A | P1 — Tier-2 Pedagogical Clarity rubric (Tasks A, C, E) | 3 rubrics × ~14 axes each, with prayas-style anchor examples per axis, scored 1-5 | **40-60** | Lit benchmark: ~14 criteria per rubric × 3 tasks × ~1 h per axis to define + anchor; one documented build was 2,400 h for a much larger corpus over 12 weeks |
| A' | P1 — Rubric inter-rater validation | 2 mentors independently score 50 Tier-2 sample rows each task → compute Krippendorff α; reconcile cells with α < 0.6 | **75-100** | 3 rubrics × 50 sample rows × 15-20 min/row × 2 mentors + ~8 h reconciliation per rubric |
| B | P2 — Human-mentor Task C gold calibration | 2 mentors deep-grade 50 student Mains answers each; compute κ vs current LLM-generated gold; flag where gold is wrong | **50-75** | 50 rows × 30-45 min/row × 2 mentors. ~5 h/answer is the prayas "thorough" benchmark; 30 min is calibration-spot-check level (not full grading) |
| C | P1 — Hindi instruction-tuning corpus | Curate + verify 500-1000 high-quality Hindi Task-A explanations + Hindi Task-B model-answers from prayas DB; ensure cultural correctness | **80-150** | 500-1000 items × 10-15 min per item to verify Hindi quality + UPSC alignment |
| D | P2 — Larger per-stratum N | Source + verify ~800 additional Task-A items to lift sub-stratum N from 50-100 → ≥150 (1PL IRT minimum [[Schroeders & Gnambs 2025](https://journals.sagepub.com/doi/10.1177/25152459251314798)]) | **65-80** | 800 items × 5 min per item to verify quality + tag stratum |
| E | P2 — IRT difficulty labels | Rate each of 800 Task-A items on a 1-5 difficulty scale; (alternative: use existing prayas student-response data) | **40** | 800 items × 3 min per item; alternative path requires no new labeling if student data is mineable from prayas DB |
| F | P3 — Mains 2024/2025 PYQ holdout | Pull recent-year Mains PYQs not in v1 eval, author/source reference model answers for ~50 items | **25-50** | 50 items × 30 min/item to author/verify reference answer |
| G | P3 — Multi-turn UPSC tutor sessions | Author 50 realistic multi-turn (3-5 turn) conversations between aspirant and tutor; provide turn-level reference responses | **20-25** | 50 sessions × 25 min/session |
| H | P3 — Live A/B brief | Define the operational metric (retention? CSAT? scores?), aspirant cohort criteria, IRB-equivalent consent flow | **8-16** | Single-pass design doc + alignment with product team |

**UPSC expert total: ~403-596 hours.**

At a single full-time content lead (40 h/week), that's **~10-15 weeks elapsed**. Parallelized between one lead and two mentors (lead drives, mentors validate), the calendar shrinks to **~6-8 weeks elapsed** with critical-path dependencies (rubric design must precede inter-rater validation; mentor calibration can run in parallel with corpus curation).

---

## 3. SLM / ML methodology expert tasks

| # | v2 item | What the SLM expert produces | Hours | Notes |
|---|---|---|---:|---|
| I | P0 — Constrained decoding | Integrate Outlines or XGrammar into inference path; verify format-validity rises from observed 0.61-0.70 → ≥0.99 [[JSONSchemaBench](https://arxiv.org/html/2501.10868v1)] | **16-24** | Schema-per-task wiring + smoke test |
| J | P0 — Logit-based / self-consistency confidence | Replace Pass-2 verbal elicitation; for FT-SLM use top-k probability gap; for Gemini use self-consistency (sample N times, agreement = confidence) | **16-24** | Resolves ECE 0.37-0.89 catastrophe from v1 §6.3 |
| K | P1 — Length-penalty FT loss | Modify SFT loss to add `α × max(0, len(output) − target_len)` term; re-FT both adapters | **8-12** | Direct fix for §6.3 Task B word-count-adherence 0.08 issue |
| L | P1 — SummaC-ZS + AlignScore + FactScore integration | Separate scoring venv (these are git-only deps in transformers 4.x); subprocess driver | **24-32** | Same pattern as the .venv-scorers approach we attempted in v1; needed for separating "added UPSC framing" from real hallucination |
| M | P2 — PDD coherence metric | RST or PDTB discourse parser + integration; ~10 correlation pts over DiscoScore at system level per [NAACL-Short 2024](https://aclanthology.org/2024.naacl-short.9/) | **24-32** | Heaviest dep in v2 metric set |
| N | P2 — IRT model fitting | Fit 1PL Rasch on Task-A response matrix once difficulty labels OR student-response data are available; surface ability + discrimination parameters | **8-16** | Needs ≥150 responses per item per [Schroeders & Gnambs 2025] |
| O | P3 — Cost-adjusted quality Pareto | Compute and plot $/query × quality frontier; identify the routing threshold that maximizes quality at fixed daily-cost budget | **8-12** | Pure analysis — feeds the hybrid-routing decision from §8.5 |
| P | P0/P1 — Independent methodology audit | External SLM/eval expert reviews: pre-registration discipline, BH-FDR family scope, bootstrap protocol, dual-test logic, scorer-checkpoint pinning, the 20 audit bugs we caught | **16-32** | One-shot; happens before v2 publication |

**SLM/ML expert total: ~120-184 hours.**

At a single ML lead (40 h/week, 60 % allocation), that's **~5-8 weeks elapsed**, fully parallelizable with the UPSC expert work since the only cross-blocker is the methodology audit (P) at the end.

---

## 4. Critical-path timeline

Assuming 1 UPSC content lead (full-time), 2 UPSC mentors (part-time for IRR + Task-C calibration), 1 SLM/ML expert (60 % allocation), and 1 external auditor (one-shot):

| Week | UPSC lane | ML lane | Gates |
|---|---|---|---|
| **1-2** | Rubric design kickoff (item A — 40-60 h spread over 2 wk) | P0: constrained decoding + confidence refactor (I + J — 32-48 h) | None |
| **3-4** | Hindi corpus curation (C — 80-150 h, runs to wk 5) | P1: length-penalty FT loss (K — 8-12 h); SummaC/AlignScore wiring (L — 24-32 h) | None |
| **5-6** | Inter-rater validation on rubric (A' — 75-100 h with 2 mentors) | Re-FT Qwen with length-penalty + Hindi corpus (~6 h compute) | Hindi corpus from wk 3-5 |
| **7-8** | Task C human-mentor calibration (B — 50-75 h) + larger-N curation (D — 65-80 h, runs to wk 9) | Tier-2 LLM-judge integration (uses rubric from wk 5-6); PDD coherence (M — 24-32 h) | Rubric IRR-validated from wk 5-6 |
| **9-10** | IRT difficulty labels (E — 40 h) | Re-run full eval with v2 metrics + v2 adapters; IRT fit (N — 8-16 h) | Larger-N curation; new FT adapters |
| **11-12** | Mains 2024/2025 holdout (F — 25-50 h) + multi-turn sessions (G — 20-25 h) | Cost-adjusted Pareto (O — 8-12 h); writeup of v2 paper | All scoring artifacts |
| **13** | Live A/B design brief (H — 8-16 h) | Independent methodology audit (P — 16-32 h, external auditor) | Full v2 results landed |
| **14** | Final review + publication |  |  |

**Elapsed: ~13-14 weeks (≈3-3.5 months)** from v2 kickoff to publication-ready run, gated end-to-end on the rubric IRR validation (week 5-6) and the larger-N curation (week 7-9). All other work is parallelizable.

---

## 5. Total expert input budget

| Lane | Hours | FTE-weeks (40 h/wk) | Calendar weeks (with parallelism) |
|---|---:|---:|---:|
| UPSC content lead | 403-596 | 10-15 | 13-14 |
| UPSC mentors (×2, part-time) | 100-150 each | 5-8 combined | 4-6 |
| SLM/ML expert | 120-184 | 3-5 | 6-8 |
| External methodology auditor | 16-32 | 0.4-0.8 | 1 (one-shot, week 13) |
| **Total expert input** | **639-962 hours** | **18.4-28.8 person-weeks** | **~14 calendar weeks** |

If the UPSC content lead is part-time (50 %), or if you want to compress the timeline by adding a second content lead, the calendar contracts to **~10 calendar weeks** with no change to total hours.

---

## 6. What v1 already gave us — no expert input needed

These v2 items don't require domain experts; they're engineering or pipeline work that lands on the existing v1 code:

- BLEURT-20 / generation perplexity wiring (already partly attempted; needs the sidecar venv approach finished)
- Larger per-stratum N can also be done by re-sampling deeper from `prod.mcqs` without new annotation if existing items have enough volume
- glossary recall (`prod.glossary` 7475 keywords) — already in the local snapshot per `eval-design §4.4`; just needs the lookup wired

These are SLM-expert items already counted in §3.

---

## 7. Minimum-viable v2 (if budget is tight)

If full v2 isn't feasible, the priority-ranked subset that recovers the most decision-relevance:

| Priority | Item | UPSC hours | ML hours | Why this first |
|---|---|---:|---:|---|
| 1 | Constrained decoding (I) | 0 | 16-24 | Lifts format-validity from 0.65 → 0.99; un-blocks every downstream metric |
| 2 | Logit-based confidence (J) | 0 | 16-24 | Fixes the ECE 0.37-0.89 catastrophe; calibration is required before any "tutor-grade" deployment |
| 3 | Hindi corpus + re-FT Qwen (C, K) | 80-150 | 8-12 | Closes the Qwen Hindi gap (currently 42.6 % accuracy) — largest single quality lever the v1 surfaced |
| 4 | Human-mentor Task C calibration (B) | 50-75 | 0 | Removes the "Task-C gold is LLM-generated" limitation; small expert spend, high methodology payoff |
| 5 | Independent methodology audit (P) | 0 | 16-32 | One-shot defensibility check before publishing the v2 result |

**MVP total: 130-225 UPSC hours + 56-92 ML hours, ~5-6 calendar weeks elapsed, one external auditor for 16-32 hours.**
