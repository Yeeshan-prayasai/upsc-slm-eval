# v2 Target Metrics — Honest, Literature-Grounded

**Date:** 2026-06-15
**Method:** 3 parallel research passes (CPT-phase + Task A · generation/BERTScore · length/grading/calibration), each web-searched and citation-backed. Targets are **single honest expected values** (modal outcome for *our* budget: QLoRA rank-64 CPT, ~0.19B domain-unique tokens, ~15% exam-format/RC-QA, single L40S), **not aspirational**. Dual-framed: absolute value + delta-over-v1 / %-of-Gemini-gap-closed.

**Framing caveat (applies to everything):** 0.19B unique domain tokens is **1–2 orders of magnitude below every published domain-CPT "win"** (AdaptLLM 1.2–16.7B, SaulLM 30B, PMC-LLaMA multi-B), and LoRA under-acquires knowledge vs full-FT (Biderman 2024). The factual-recall lift comes almost entirely from the exam-format/synthetic-QA slice + SFT, not raw-text exposure.

---

## CPT phase (training-health, intermediate — not pass/fail on a number)

| Signal | Honest target | Basis |
|---|---|---|
| Domain-val CE loss | Monotonic ↓; modest total drop (**< ~0.25 nats** over 1 epoch); **sharp step-down in the decay tail** is the WSD signature. Judge *shape*, not magnitude. | Biderman 2024 (LoRA acquires little); Wen 2024 / Hägele 2024 / MiniCPM (decay tail does the visible work, ~10% of steps) |
| Domain perplexity | Steady ↓ then visible decay-tail step; general/replay PPL flat-to-slightly-down | Wen 2024 |
| **MMLU / general retention** | within **−2 pp** of base; **>3 pp drop ⇒ stop, raise replay / lower LR** | Ibrahim 2024; Biderman 2024 (LoRA forgets less); ~20% replay → ≤1–2pp |
| **NEW — factuality KPI** | Add an entity/number-recall-vs-reference check as the **primary** CPT downstream signal | DACP 2025: CPT gains show on factuality metrics, **invisible to BERTScore** |

---

## Downstream gates — honest expected values vs current methodology gates

| Task / metric | v1 | Gemini | **Old gate** | **Honest expected (v2)** | **Revised gate** | Verdict |
|---|---|---|---|---|---|---|
| **A — Prelims MCQ acc (EN)** | 0.645 | 0.909 | ≥0.75 | **0.70** (band 0.68–0.72) | **≥0.69 floor, 0.70 target, 0.72 stretch** | **0.75 unrealistic** → revise. Closes ~20–25% of gap. (pulse hints 0.76 = above-expectation if real) |
| **A — negative-marking** | 1.06 | — | ≥1.40 | **1.15–1.20** | **≥1.10 no-regression** (1.40 = future, needs calibration fix) | **1.40 unrealistic** without the deferred calibration work → revise. SFT/CPT *worsen* calibration (GPT-4 report) |
| **B — Mains gen BERTScore-F1** | 0.833 | 0.795 | ≥0.825 | **0.833 (flat)** | **≥0.825 keep** | OK as no-regression. BERTScore can't see CPT gains |
| **B — word-count adherence** | 0.086 | 0.30–0.48 | ≥0.40 | **0.40–0.45** (range 0.35–0.55) | **0.40 stretch, ≥0.30 floor** | Borderline. Bug-fix makes a real jump; essay-length + 4B size cap it near the gate |
| **C — rubric grading MAE** | 1.90 | 2.52 | ≤2.20 | **1.7–2.1** | **≤2.20 keep** (target ≤1.90) | Realistic — only gate with genuine headroom. Grading is ~CPT-independent |
| **E — current-affairs BERTScore-F1** | 0.873 | 0.851 | ≥0.865 | **0.875 (flat)** | **≥0.865 keep** | OK as no-regression |
| **F — Prelims-expl BERTScore-F1** | 0.824 | 0.771 | (none) | **0.828 (flat)** | **NEW ≥0.814** | Add guard |
| **G — Mains-DSL BERTScore-F1** | 0.745 | 0.708 | (none) | **0.745 (flat, regression-risk)** | **NEW ≥0.735** | Add guard — DSL format fidelity is the CPT-perturbation risk |

**BERTScore note:** v1 used `rescale_with_baseline=False` (confirmed in `score_tier1.py:297`) → raw scores in the saturated 0.74–0.87 band where ±0.01–0.02 is the noise floor. B/E/F/G are **no-regression guards, not improvement targets** — demanding a BERTScore *gain* from CPT would falsely fail a CPT phase that actually worked. Keep the paired-bootstrap CI for any "beats Gemini" claim (E +0.022 / G +0.037 need it).

---

## What would revise these UP
- Inference-time RAG over the UPSC corpus (Ovadia: RAG ≈ doubles FT gain) → Task A into 0.75–0.80
- Much more / higher-quality synthetic RC-QA with many paraphrases per fact (Ovadia +8pp lever)
- Inference-time calibration (verbalized confidence / tuned abstention) → neg-marking toward 1.40 (≈50% ECE cut)
- Full-FT or ≥10× domain tokens → larger Task-A lift

## What would revise these DOWN
- MMLU drop >3pp during CPT (under-replay) — would force a re-run, and a damaged base caps everything
- Synthetic-QA eval contamination (mitigated: gate is CLEAN) or low-quality synthetic QA
- DSL-format regression on Task G from CPT perturbing instruction-following before SFT recovers it

**Bottom line:** the v2 honest success bar is "**beat v1 on Task A by ~+0.05 (→0.70), hold no-regression everywhere else, show the factual gain on a grounding metric BERTScore can't see**" — not "close the Gemini gap." Closing the full gap to 0.909 is out of scope for single-L40S LoRA-CPT at this token budget; it needs RAG or full-FT.
