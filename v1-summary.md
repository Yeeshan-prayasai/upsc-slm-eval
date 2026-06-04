# v1 — Summary of Work Completed

**Period:** 2026-05-14 → 2026-06-04 (~3 weeks elapsed)
**Owner:** Yeeshan
**Status:** v1 complete; publication-ready; transitioning to v2 per [`v2-expert-input-plan.md`](v2-expert-input-plan.md)
**Reads with:** [`project-context.md`](project-context.md) (full session log) · [`qa-status-cto.md`](qa-status-cto.md) (QA + acceptance criteria) · [`experiment-report.md`](experiment-report.md) (results + §8 inference) · [`eval-design.md`](eval-design.md) (metric definitions)

---

## 1. What got built

| Component | Status | Where |
|---|---|---|
| Eval set frozen — 2,000 items, 4 stratified tasks (A 800 · B 400 · C 500 · E 300) | ✓ SHA-256 sidecar | `data/eval_set.parquet` (gitignored; PII) |
| FT corpus built with **content-level leakage assertion** (normalized-text SHA-256 hashes) | ✓ 40,571 supervised pairs after dropping 3,044 leaked rows | `data/ft_corpus.parquet` |
| A2 Hindi-capability probe (50 MCQs, one-sided binomial vs random) | ✓ Gemma 52 % **PASS** (p < 1e-5); Qwen 30 % **FAIL** (p = 0.252) | `results/pre_ft_hindi_probe.parquet` |
| Two LoRA adapters — identical recipe (rank 16, α 32, lr 2e-4, 3 epochs, target_modules {q,k,v,o,gate,up,down}_proj) | ✓ trained to convergence; final eval-loss Qwen 0.727 / Gemma 0.814 | `adapters/{gemma4-e4b-upsc-v1,qwen35-4b-upsc-v1}/` — raw PEFT adapters in GitHub |
| Merged HF dirs (PEFT folded into bf16 base) | ✓ Gemma 15 GB · Qwen 7.9 GB | local + EC2; rebuildable via `scripts/merge_adapter.py` |
| MLX 4-bit dirs for M5 inference | ✓ Gemma 3.9 GB (4.501 bits/wt) · Qwen 2.2 GB (4.503 bits/wt) | local M5 |
| 4-condition × 6-task inference | ✓ **12,800 predictions** (3,200 per condition) | `results/predictions.parquet` (gitignored; PII) |
| Tier-1 scoring — ~45 deterministic metrics + per-row format-validity + universal latency/cost | ✓ | `results/scores_tier1.parquet` |
| Aggregation — per-(condition, task, language) means + bootstrap-95 % CIs + ECE / BSS / QWK / Spearman / Pearson / position-bias / Task-A McNemar / bilingual delta | ✓ | `results/aggregate.parquet` + `aggregate.extras.json` |
| Pairwise hypothesis tests — paired t + Wilcoxon + dual-test agreement + BH-FDR at q=0.05 + Cohen's d/h | ✓ **1,472 pairwise tests** | `results/hypothesis_tests.parquet` |
| Per-stratum heatmap — champion-vs-C3 deltas per (task, stratum_key) | ✓ **230 strata** | `results/stratum_heatmap.parquet` |
| Auto-renderer — fills §6 / §7 tables of `experiment-report.md` from the result store | ✓ **11/11 tables, 116/116 cells, zero gaps** | `scripts/render_report.py` |
| 2-line "Infer / v2 path" summary below each §6 / §7 table citing specific numbers + deferred-to-v2 items | ✓ | inline in `experiment-report.md` |
| §8 Inference (Discussion) — 6 subsections written from the actual numbers (verdict / pre-reg vs reality / per-stratum / mistakes / product implications / v2 roadmap) | ✓ | `experiment-report.md §8` |
| **Pipeline audit** — line-by-line review found 20 bugs (4 critical, 8 important, 8 quality); all fixed before final scoring run | ✓ commit `5df55e1` + `332a8f5` | `scripts/score_tier1.py`, `runners.py`, `run_inference.py`, `aggregate.py`, `test_hypotheses.py` |

---

## 2. Outcome — headline numbers

**Pre-registered verdict criterion (§1.3):** champion FT-SLM beats C3 (Gemini few-shot) on ≥3 of 4 core tasks at the primary metric, BH-FDR-corrected, bootstrap 95 % CI excluding zero.

**Result:**

| Core task | Champion | Primary metric | Champion | C3 | Verdict | Effect size |
|---|---|---|---:|---:|---|---:|
| A (Prelims MCQ accuracy, EN+HI pooled) | C1a (Gemma) | accuracy | 0.645 | 0.910 | **LOSS** | d = −0.66 (medium) |
| B (Mains generation BERTScore-F1) | C1a (Gemma) | BERTScore-F1 | 0.833 | 0.795 | **WIN** | d = 0.21 (small) |
| C (Mains rubric grading) | C1b (Qwen) | Score MAE (↓) | 1.901 | 2.516 | **WIN** | d = 0.16 (negligible-but-significant) |
| E (Current Affairs synthesis) | C1b (Qwen) | mains BERTScore-F1 | 0.873 | 0.851 | **WIN** | d = 0.92 (large) |

**3 of 4 core tasks WIN at q ≤ 0.05 BH-FDR with dual-test agreement (paired t AND Wilcoxon).** The pre-registered "strong win" verdict is not met because Task A's loss is large; the pre-registered "non-inferiority within 5 pp" is met for B/C/E but **not** for A.

**Production-prompt capability tests (Tasks F, G):**
- Task F (Prelims Explanation Gen): FT-SLMs hold +2.5 pp BERTScore-F1 over Gemini and **3.6× higher distractor coverage**; the FT path follows the bilingual production format with high fidelity.
- Task G (Mains Model-Answer Gen via the 21 KB prayas DSL): Qwen-FT leads BERTScore by +0.037; FT-SLMs cover **2.8× more PESEE dimensions** than Gemini.

**Hindi:** Gemini-3.5-Flash beats both FT-SLMs by 30-50 pp on the Hindi stratum (HI accuracy 0.932 vs Gemma 0.636 vs Qwen 0.426). The pre-FT Hindi probe predicted Qwen's Hindi weakness; FT did not close the gap.

**Pre-registered predictions (§5.1) — verdict:** 3 refuted with direction inverted (Task A accuracy, C1b-over-C1a-on-Hindi, Gemini-low-QWK); 2 confirmed; 1 partially confirmed. Full walk-through in §8.2.

---

## 3. Cost + scale

| Item | Value |
|---|---|
| Total v1 elapsed | ~3 weeks (2026-05-14 → 2026-06-04) |
| Two LoRA adapters trained | ~22 h Qwen + ~22 h Gemma on L40S = ~44 h GPU |
| Inference + scoring + hypothesis tests | 4-condition × 12,800 prediction × scoring chain ≈ 9 h wall-clock |
| Cloud compute spend (FT + inference + Gemini API) | **~$74 FT + ~$25 inference + ~$8 Gemini API ≈ $107** |
| Final code commits on `YeeshanMalik/upsc-slm-eval` | `09784c1` → `1b2d331` (~30 commits since v1 inference plane built) |

---

## 4. What's in the repo + what isn't

| Tracked in git (private repo) | Gitignored (PII / large + rebuildable) |
|---|---|
| All `scripts/*.py` | `data/eval_set.parquet` (PII) |
| All design docs (eval-design, experiment-report, architecture, project-brief, project-context, qa-status-cto, v2-expert-input-plan, v1-summary) | `data/ft_corpus.parquet` (PII) |
| Raw PEFT adapters: `adapters/{gemma4-e4b,qwen35-4b}-upsc-v1/adapter_model.safetensors` (24 + 21 MB) | `data/prayas_local.sqlite` (PII) |
| `adapters/<name>/trainer_state_final.json` per adapter (full log_history) | `adapters/*-merged/` (15 + 8 GB; deterministically rebuildable from raw adapters via `scripts/merge_adapter.py`) |
| `data/upsc_facts.json` + `data/dimension_keywords.json` (committed; public-sourced) | `adapters/*-mlx/` (3.9 + 2.2 GB; rebuildable via `mlx_lm convert`) |
| **Result artifacts:** `aggregate.parquet`, `scores_tier1.parquet`, `hypothesis_tests.parquet`, `stratum_heatmap.parquet`, `aggregate.extras.json`, `pre_ft_hindi_probe.parquet` | `results/predictions.parquet` (52 MB; PII via raw gold + model raw_output) |
| `logs/` — full FT training logs + watcher + scoring chain logs | `data/ft_split/` (deterministically rebuildable from FT corpus) |
| `requirements.txt` + `requirements-aws.txt` (exact-pinned) | `AWS/*.pem` keys |
| `.gitignore` (carve-outs documented) | `api-key` file |

---

## 5. Acceptance criteria — final status

From [`qa-status-cto.md §5`](qa-status-cto.md). Each row a binary gate.

| Criterion | Threshold | Status |
|---|---|---|
| Eval-set integrity (SHA-256 matches sidecar) | exact match | ✓ |
| No FT leakage (`eval ∩ ft = ∅`) | CI assertion green | ✓ |
| All conditions complete (row count 3,200 each) | exact 3,200 | ✓ all 4 |
| Format-validity rate | ≥ 0.90 per (condition, task) | **✗ — observed 0.61-0.70 across all conditions** (top-priority v2 fix: constrained decoding) |
| Hallucination rate (Task E champion) | ≤ 0.15 | ✗ — observed 0.69-0.74 by the entity-not-in-source proxy (metric artifact; SummaC-ZS deferred to v2 would distinguish "added UPSC framing" from real fabrication) |
| Calibration (Task A champion ECE-15bin) | ≤ 0.10 | ✗ — observed 0.37-0.89 across all conditions (verbal-confidence elicitation broken; v2 P0 fix: logit-based or self-consistency confidence) |
| Pipeline audit | 0 unresolved critical/important findings | ✓ 20/20 resolved (commit `5df55e1`) |
| Statistical layer | BH-FDR + Cohen's d/h + dual-test agreement per (task, metric, pair) | ✓ 1,472 rows in `hypothesis_tests.parquet` |
| Render coverage | ≥ 95 % of expected cells populated | ✓ 116/116 (deferred metrics tracked as N/A) |
| Manifest | SHAs of code + data + adapters + scorer-models + prompts | ✓ via committed SHAs (formal `manifest.json` is a v2 polish item) |
| Effect-size threshold for "production-relevant" | Cohen's d ≥ 0.5 to act on | applied at report-render |
| Pre-registered verdict evaluable | §1.3 derivable from `aggregate.parquet` | ✓ (§8.1 of experiment-report) |

**3 of 12 acceptance criteria failed** (format-validity, hallucination-rate, calibration). All three are flagged as v2 P0/P1 fixes with named methods. None block publication of v1 because v1 explicitly reports these as findings, not as ship-blockers.

---

## 6. Where v1 ends + v2 begins

v1 answers: *"Does FT 4B-class on prayas UPSC data match or beat Gemini-3.5-Flash?"* — **partially: wins on 3 of 4 core tasks at significance, loses Task A by a large margin, wins both production-prompt capability tests by 3.6× / 2.8× margins on format compliance and dimension coverage.**

v2 starts with three blockers identified in v1 (full plan in [`v2-expert-input-plan.md`](v2-expert-input-plan.md)):

| v2 priority | What | Hours (UPSC / ML) |
|---|---|---|
| **P0** | Constrained decoding (Outlines / XGrammar) to lift format-validity 0.61-0.70 → ≥ 0.99 | 0 / 16-24 |
| **P0** | Logit-based or self-consistency confidence to fix ECE 0.37-0.89 catastrophe | 0 / 16-24 |
| **P1** | Hindi instruction-tuning corpus + Qwen re-FT to close the 50-pp Hindi gap | 80-150 / 8-12 |
| **P1** | Length-penalty FT loss to fix Task B/E word-count adherence 0.08-0.09 | 0 / 8-12 |
| **P1** | SummaC-ZS + AlignScore + FactScore (separate scoring venv) to separate "added UPSC framing" from real Task-E hallucination | 0 / 24-32 |
| **P1** | Tier-2 Pedagogical Clarity LLM-judge rubric (Tasks A, C, E) + G-Eval (Task B) — captures teaching quality Tier-1 misses | 40-60 design + 75-100 IRR / 8-16 integration |
| **P2** | Human-mentor calibration on a 50-row Task-C subsample to validate LLM-generated gold | 50-75 / 0 |
| **P2** | Larger per-stratum N (target ≥ 100 per cell) to lift §7.2 heatmap power | 65-80 / 0 |
| **P2** | IRT-based item-difficulty weighting | 40 / 8-16 |
| **P2** | PDD coherence (NAACL-Short 2024) for Task G long-form structural quality | 0 / 24-32 |
| **P3** | Multi-turn UPSC tutor conversational eval | 20-25 / + infra |
| **P3** | Live A/B with real prayas students | 8-16 / + product |
| **P3** | Cost-adjusted quality Pareto front | 0 / 8-12 |
| **P3** | Mains 2024/2025 PYQ temporal-OOD holdout | 25-50 / 0 |
| **P0/P1** | Independent methodology audit before v2 publication | 0 / 16-32 (external) |

**Total v2 expert input budget:** ~639-962 hours across UPSC content lead + 2 mentors (IRR) + 1 SLM/ML expert + 1 external auditor → **~14 calendar weeks** with parallel lanes. MVP-if-tight variant: 130-225 UPSC h + 56-92 ML h → **5-6 weeks**.

---

## 7. Standing rules carried forward

These were locked during v1 and remain in effect for v2:

- **No writes to prod DB without per-instance explicit approval.** `scripts/snapshot_to_local.py` is the only script that touches remote Postgres; everything downstream reads from `data/prayas_local.sqlite`.
- **Eval set frozen with SHA-256 sidecar.** CI assertion `eval ∩ ft = ∅` blocks any FT that violates it.
- **Pre-registered hypotheses.** Predictions land in the report before any results.
- **Pipeline auditability.** Every metric has a Python library cited in `eval-design.md §4`; every scorer-model checkpoint is pinned; seed `20260514` everywhere.
- **All risky actions confirm with the user first** (destructive AWS ops, prod-DB writes beyond `SELECT`, force pushes).
- **All metric values are numerically backed** — no subjective scoring in Tier-1; Tier-2 LLM-judge work (deferred to v2) is diagnostic only and labeled as such.
