# Project Context — Prayas.ai SLM vs Gemini 3-Flash on UPSC

**Owner:** Yeeshan (irshad@prayas.ai), Data Scientist, prayas.ai
**Working dir:** `/Users/yeeshan/PrayasAI/Code/SLM`
**Repo:** `github.com/Yeeshan-prayasai/upsc-slm-eval` (private, transferred from `YeeshanMalik/upsc-slm-eval` on 2026-06-10)
**Status:** v1 results published + dashboard deployed; v2 pipeline audited against literature (2026-06-11, 39 findings — see `v2-audit-findings.md`) and all fixes applied; corpus syncing to EC2; CPT training next.

---

## 1. What this project is

Quantify the performance gap between **fine-tuned open-source SLMs** (Gemma-4-E4B-it + Qwen3.5-4B) and **Gemini 3.5-Flash** (zero-shot + few-shot) on UPSC Civil Services Exam tasks. The deliverable is a Streamlit dashboard reading reproducible eval artifacts.

**v1 outcome (published 2026-06-04):** FT-SLM **wins 3 of 4 core tasks** at q ≤ 0.05 (Mains generation, Rubric grading, Current affairs). Loses Prelims MCQ by 26 pp accuracy. Dashboard live on Streamlit Cloud.

**v2 in progress:** continued pretraining (CPT) on a 1-2 B-token UPSC-domain corpus + SFT with length-penalty loss, targeted at closing the Task A factual-recall gap.

---

## 2. v1 — final results (frozen)

4 conditions × 6 tasks × 2,000 eval items = 12,800 scored predictions in `results/predictions.parquet`.

| Task | Headline metric | Gemma-FT | Qwen-FT | Gemini FS | Δ (Champ−FS) | Sig |
|---|---|---:|---:|---:|---:|---|
| A — Prelims MCQ | is_correct | 0.645 | 0.531 | 0.909 | **−0.264** | ✓ loss |
| B — Mains generation | answer_bertscore_f1 | **0.833** | 0.811 | 0.795 | +0.010 | ✓ |
| C — Rubric grading | score_abs_err (↓ better) | 3.10 | **1.90** | 2.52 | +0.213 | · borderline |
| E — Current affairs | mains_bertscore_f1 | 0.866 | **0.873** | 0.851 | +0.023 | ✓ (large d=0.92) |
| F — Prelims expl. (prod) | explanation_bertscore_f1 | 0.804 | **0.824** | 0.771 | +0.020 | ✓ |
| G — Mains DSL (prod) | answer_bertscore_f1 | 0.716 | **0.745** | 0.708 | +0.037 | ✓ (d=0.63) |

**Reports:** [`experiment-report.md`](experiment-report.md) §6-§8 · [`eval-design.md`](eval-design.md) · [`qa-status-cto.md`](qa-status-cto.md)
**Dashboard:** `dashboard/app.py` (live on Streamlit Cloud)
**v1 adapters:** `adapters/{gemma4-e4b,qwen35-4b}-upsc-v1*/`

---

## 3. v2 production build state

### Code (production-clean, no Phase-N framing, no v2 branding in internal code)

| Component | Path | LOC |
|---|---|---:|
| Acquirers (per-source) | `training/data/acquire/*.py` | ~2,800 |
| Corpus pipeline (OCR → clean → leakage → tokenize) | `training/data/{ocr,clean,leakage,tokenize_pack,build_cpt_corpus,build_sft_corpus}.py` | ~1,400 |
| Trainers (QLoRA + WSD + length-penalty) | `training/trainers/{base,cpt,sft,schedulers,length_penalty_math}.py` | 834 |
| Orchestration (run_cpt, run_sft, run_ablation, smoke_cpt) | `training/orchestration/*.py` | 792 |
| Eval (preflight leakage gate + in-train pulse) | `training/eval/{preflight,pulse,mcq_inference}.py` | 629 |
| Configs (CPT/SFT/runtime YAML per model) | `training/configs/*.yaml` | 182 |
| Tests | `training/tests/test_*.py` | 924 |

**Test suite: 114 passing + 1 skipped** (test_cpt_smoke requires CUDA; runs on EC2).

**Key design choices** (all literature-cited; amended 2026-06-11 per the audit in `v2-audit-findings.md`):
- LoRA rank 64, α=16, **RSLoRA** (Kalajdzievski 2023), all decoder layers × 7 projections — regex covers Gemma-4's nested `model.language_model.layers.*` naming, with an exact 7×n_layers coverage assertion at startup
- QLoRA NF4 + double-quant + paged_adamw_8bit (Dettmers NeurIPS 2023); embeddings re-cast to bf16 post-kbit-prep (saves ~6-12 GB VRAM on Gemma)
- **WSD scheduler**: 1% warmup → 80% stable → 19% cosine decay to 0.1× peak (Wen 2024); resume guard persists the schedule shape
- CPT optimizer: AdamW β₂=0.95, **LR 1e-4** (LoRA-adapter rate per Biderman 2024 — the original 1e-5 was a full-param rate that left adapters in the noise floor); SFT: β₂=0.999, LR 2e-4, cosine_with_min_lr 0.1×
- **Length control in the data, not the loss**: Task-B prompts carry "Answer in approximately N words." (the spec'd penalty term had zero gradient — computed from label lengths); SFT data is chat-templated prompt-completion with completion-only loss, matching inference framing exactly
- **Mix-weighted CPT corpus** (`training/configs/data_mix_cpt.yaml`, enforced at pack time): NCERT/refbooks ×4, notes ×3, govt ×2, ORF/MEA capped 30M, replay capped ~115M (~20%), instruction slice from SFT pairs; max_steps derived as one epoch over the weighted parquet
- Pipeline gates: 50-word document floor (Penedo/Gopher), Devanagari language filter, doc-level MinHash dedup at 0.8 with Jaccard verification + source-priority keep order (Lee 2022), 50-token contiguous leakage check (Carlini 2023) on BOTH CPT and SFT corpora
- Hindi handling: EN-only training; in-training Hindi no-regression pulse hard-stops at −5 pp vs a step-0 same-instrument baseline (HARD_STOP marker → non-zero exit)

---

## 4. Corpus inventory (current state)

**23 sources · 2,160 PDFs + 49,316 markdown docs · 10.6 GB raw**

### Government / authoritative

| Source | Files | Subject |
|---|---:|---|
| NCERT Class 6-12 | 266 PDFs | All GS (base layer) |
| NDMA | 128 PDFs | Disaster Mgmt (GS3) |
| MoEFCC | 252 PDFs | Environment (GS3) |
| MEA | 13,050 .md | International Relations (GS2) |
| Union Budget (5 FYs) | 1,369 PDFs | Economy (GS3) |
| NITI Aayog publications | 86 PDFs | Economy + Social Justice |
| ISRO press releases | 80 .md | Science & Tech (GS3) |
| DRDO (via PIB) | 54 .md | S&T + Defence |
| **PIB (in-flight)** | acquiring | All ministries, top-43 UPSC-relevant |
| Economic Survey | 18 PDFs | Economy |
| 2nd ARC Reports (1-15) | 14 PDFs | Governance + Ethics |
| MHA annual reports | 10 PDFs | Internal Security (GS3) |
| IPCC reports | 7 PDFs | Environment |
| Constitution + Bare Acts | 2 PDFs | Polity (GS2) |

### Standard reference books (cleaned, ~2.59 M tokens)

| Book | Pages | Subject |
|---|---:|---|
| R.S. Sharma *India's Ancient Past* | 422 | GS1 Ancient History (primary text) |
| Spectrum *Modern India* (Rajiv Ahir) | 880 | GS1 Modern History |
| Laxmikanth *Indian Polity 8th ed.* | 1,198 | GS2 Polity (keystone text) |
| Satish Chandra *Medieval India* | 406 | GS1 Medieval History |
| Bipan Chandra *India Since Independence* | 845 | GS1 Post-Independence |
| Poonam Dahiya *Ancient & Medieval India* | 845 | GS1 Ancient + Medieval |
| Karthikeyan *Internal Security (Pearson)* | 337 | GS3 Internal Security |
| PMF Geography 2024 | 273 | GS1 Indian Geography |

Dropped: G.C. Leong (Tesseract on scanned book yielded 24% garbled lines; PMF Geography covers the high-priority Indian Geography content cleanly).

**Still missing (copyright-gated, not sourceable):** Nitin Singhania (Art & Culture), Norman Lowe (World History).

### Think-tank / educational ecosystem

| Source | Files | Subject |
|---|---:|---|
| ORF (research + expert-speak) | 22,700 .md | IR / strategic affairs |
| Mrunal | 2,006 .md (186 cruft filtered) | Economy + current affairs |
| PMF IAS scrape | 1,000 .md | Geo + environment notes |
| PRS Legislative Research | 892 .md | Bill summaries (GS2) |
| Newspapers (Hindu / IE / BL / Mint) | 842 .md | Daily current affairs |

### Replay / general (anti-forgetting)

| Source | Disk | ≈Tokens |
|---|---:|---:|
| FineWeb-Edu (kept name `slimpajama` for compat) | 2.4 GB | ~600 M |
| Wikipedia EN India-subset | 834 MB | ~210 M |
| prayas local DB extracts | 197 MB | ~50 M |

**Token estimates:**
- UPSC-domain (govt + reference books + ecosystem): ~250-300 M tokens after clean
- Replay buffer: ~860 M tokens
- **Total post-clean: ~1.1-1.2 B tokens** (vs methodology §4.5 target of 4 B — ~28-30% of target; the corpus is comprehensive on **named-text coverage** but light on volume)

---

## 5. Current pipeline state

| Step | Status |
|---|---|
| Corpus acquisition | ✅ Done (PIB killed at 92% by choice: 16,723 files / 41 ministries) |
| Reference-book OCR + artifact cleanup | ✅ Done (5.84 M chars, 0 typos remaining) |
| **Pipeline audit vs literature** | ✅ Done 2026-06-11 — 39 findings (6 crash-level, ~10 silent no-ops), ALL fixed; `v2-audit-findings.md` is the record |
| SFT corpus v2 (prompt-completion + gates) | ✅ Rebuilt: 28,993 train / 1,524 valid; 583 cross-language eval siblings + 327 overlap rows excised (v1 shipped with this contamination) |
| Task-pulse holdout | ✅ `data/eval_set_holdout.parquet` (200 MCQs, `training/eval/build_holdout.py`) |
| EC2 environment | ✅ v2 code + deps on the box (18.233.5.108); 136/136 tests pass incl. the 100-step CUDA smoke |
| Corpus build on EC2 | ✅ CLEAN gate; 0.37/0.38B exposures (rebuild pending with growth-sweep sources) |
| Corpus growth sweep (PIB×6 + DTE full + RC/QA + qa_bank) | ⏳ landing 2026-06-12 |
| Full CPT (Gemma + Qwen) | Pending — measured 170 s/step → ~67 L40S-h/model at bs=1 (bs=2 probe planned) |
| Full SFT (Gemma + Qwen) | Pending — ~2-3 L40S-h each (950 steps, was misconfigured at 34 epochs) |
| 6-cell ablation grid | Pending |
| v2 inference + scoring | Pending — `make infer-v2-{gemma,qwen} ADAPTER=...`, `make score-v2` |
| RC/QA synthetic augmentation | Scaffolded (`training/data/generate_rc_qa.py`) — API-cost-gated, optional |
| v2 paper draft | After ablation lands |

**Compute path:** same V1 AWS EC2 g6e.xlarge (L40S 48 GB VRAM) at 18.233.5.108 (private 172.31.94.173). Key: `AWS/upsc-slm.pem` in repo root.

**Open items (none blocking):**
- Yojana / Kurukshetra magazines — `publicationsdivision.gov.in` timing out; deferred
- Synthetic Q/A augmentation — explicitly deferred; revisit only if v2 Task A misses the +10pp gate
- Hindi corpus — separate effort per [`v2-hindi-strategy.md`](v2-hindi-strategy.md)

---

## 6. Reference documents

| Doc | Purpose |
|---|---|
| [`experiment-report.md`](experiment-report.md) | Pre-registered scientific report — Aim/Setup/Procedure/Expected/Actual/Results/Inference (v1 §6-§8 complete) |
| [`eval-design.md`](eval-design.md) | ~45 Tier-1 metrics × 4 tasks × statistical protocol (paired bootstrap + dual-test + BH-FDR) |
| [`architecture.md`](architecture.md) | 4-plane system architecture (data / inference / scoring / dashboard) |
| [`project-brief.md`](project-brief.md) | Non-technical stakeholder one-pager |
| [`qa-status-cto.md`](qa-status-cto.md) | 6-section CTO-facing QA summary |
| [`v2-methodology.md`](v2-methodology.md) | CPT → SFT recipe, 6-cell ablation grid, acceptance gates |
| [`v2-hindi-strategy.md`](v2-hindi-strategy.md) | EN-only training decision + Hindi no-regression pulse + v3+ scope |
| [`v2-expert-input-plan.md`](v2-expert-input-plan.md) | UPSC + SLM expert hours needed, critical path |
| [`dashboard/DEPLOY.md`](dashboard/DEPLOY.md) | Streamlit Cloud deploy guide |
| [`test-strategy.md`](test-strategy.md) | 6-layer test strategy across data + training planes |

---

## 7. Locked decisions

| Decision | Rationale |
|---|---|
| Dual FT candidates: Gemma-4-E4B-it + Qwen3.5-4B | Isolates "Indic-via-FT vs Indic-via-pretraining" + architecture/family comparison |
| Comparator: Gemini-3.5-Flash (zero-shot + few-shot) | Production API benchmark prayas would otherwise route to |
| Tasks A/B/C/E (v1) + F/G (production prompts) | Skip Interview (T2 personalization) and Hindi-only tasks for v1 |
| Tier-1 deterministic metrics only (no LLM-judge) | Quantitative-first; reproducibility; no judge-family bias |
| Local M5 for dashboard + code; EC2 L40S for training | M5 has crashed twice on local GPU/OCR — heavy ML stays remote (see `feedback_m5_memory_pressure`) |
| Single source-of-truth: corpus build → leakage gate → tokenize | Leakage gate is non-optional (`--skip-leakage` removed); SHA-256 + 5-gram redundant checks |
| EN-only training corpus for v2 | Hindi separately addressed via pulse no-regression gate, full strategy in `v2-hindi-strategy.md` |
| `runs/ablation/`, no `v2_` cell prefixes | Production-clean naming; "v2" stays only where it disambiguates from v1 artifacts on disk |
| `predictions.parquet` + `eval_set.parquet` whitelisted for Cloud deploy | Per-row drill page needs them; user-approved expose |

---

## 8. Session log (compressed)

Detail logs for older work are in git history (commits since 2026-05-14). Recent material events:

- **2026-06-04** v1 published. 4-condition × 6-task eval, 12,800 predictions, 11 §6/§7 report tables filled, §8 Inference written. Cost ~$33.
- **2026-06-04** v2 methodology authored. CPT→SFT recipe, 6-cell ablation, length-penalty loss locked.
- **2026-06-04** Streamlit dashboard built (4 pages incl. live playground).
- **2026-06-05** v2 training code committed. 23 acquirer modules, full corpus pipeline (OCR/clean/leakage/tokenize), CPT + SFT trainers, smoke-cpt entrypoint, 80+ tests.
- **2026-06-08** Acquisition round-2: ORF + Wikipedia + NDMA + Newspapers + Constitution + MHA + curated framework. ~4.2 GB on disk.
- **2026-06-09** PMF Geography 2024 + NDMA OCR completed (128/128 docs); SFT corpus build runs (30K rows / 1.5K valid).
- **2026-06-12** Corpus build shipped + growth sweep:
  - **Corpus build COMPLETE on EC2** after 4 runs, each killed by a distinct real defect the gates caught: (1) ocr.py returned exit-1 on PDF-less sources → build died at `drdo`; (2) leakage gate caught 1,073 eval questions verbatim in local_db table dumps (the tables the eval set was drawn from) → new Stage 2c record-level purge (2,308 records excised); (3) 22 PYQ reprints in Mrunal pages/Laxmikanth appendix → purge extended to paragraph-level over all 65K .md files; (4) transformers-5.x `apply_chat_template` returns dict → `return_dict=False`. Final: gate CLEAN, 295/298M unique tokens, 0.37/0.38B weighted exposures, replay 31%, instruct 7%.
  - **FFFD-ligature rescue** added to ocr.py (pymupdf4llm emits U+FFFD for `ti`/`ft` ligatures in some fonts; plain `get_text` decodes them) — fired 305× across the corpus during the build.
  - **Gemma smoke PASSED on real base** (loss 2.4→2.05 over 100 steps, 26.8GB peak VRAM, adapter round-trip OK) after fixing the coverage assertion for Gemma-4-E4B's 18 KV-shared layers (258 eligible sites, not 294).
  - **Literature calibration**: published CPT wins cluster at 1.2–30B domain tokens; our 0.18B is below that floor → growth sweep launched (PIB 3-year back-extension ~45K new articles, DTE full archive to 1991 ~50K articles, AdaptLLM-style RC/QA generation over NCERT+refbooks ~$0.79 API). **r=128 analyzed and REJECTED** (memorization risk: adapter capacity would exceed corpus information; r64@scale-2.0 is the drift minimum — arXiv:2410.21228, 2507.21009).
  - **qa_bank source created** (repeat ×3): the DB question tables split out of local_db — mcqs + ai_generated + evaluation_questions + **recovered prelims_pyq_questions & pyqs** (the bilingual EN+HI blobs read as 56% Devanagari and the doc-level language filter had silently dropped 40M chars; fixed with char-level Devanagari stripping for record files) + fresh `article_generated_questions` pull from upscdev (129 rows, targeted SELECT). ~98M chars total.
- **2026-06-11** Full pipeline audit + fix sweep (`v2-audit-findings.md`):
  - 6 parallel literature-grounded review passes found 39 issues: Gemma LoRA regex matched zero modules (crash), sft.py missing imports (never ran), CPT LR 1e-5 a full-param rate on adapters (near-no-op), length-penalty triply dead (zero gradient), pulse gates never wired / Hindi gate parsed 0 items, replay .jsonl never entered the corpus (0% replay), SFT misconfigured at 34 epochs, no mix weighting existed, chat-template train/inference mismatch, Qwen EOS masked from loss, double BOS at inference, cell-6 checkpoint rotated away, + more.
  - All fixed in 4 phases (data plane / trainer plane / eval-orchestration / methodology doc). New: `data_mix_cpt.yaml` (enforced), `build_holdout.py`, `build_instruct_cpt.py`, `generate_rc_qa.py` (scaffold), `infer-v2-*`/`score-v2` Make targets, `--adapter-dir` on run_inference.
  - New leakage gate on the SFT corpus caught **v1-era contamination**: 583 rows where the English sibling of a Hindi eval question was in the training set (v1's ID check compared full ids incl. language suffix). v1's Hindi-stratum numbers should be read with this caveat.
  - EC2 env rebuilt: torch 2.11.0+cu130 restored, torchvision/audio removed (datasets VideoReader conflict), pymupdf4llm pin fixed (0.0.30 → 1.27.2.3). CUDA smoke (100 steps, GPT-2) passed for the first time ever — it had latent crashes (fp16 kwarg, tuple unpack) proving it never ran.
- **2026-06-10** Major additions:
  - **Dashboard deployed** to Streamlit Cloud (Yeeshan-prayasai/upsc-slm-eval). Added Inference column (one-line plain-English takeaway per task) + significance column (BH-FDR p + Cohen's d). Fixed pre-existing sign bug in hypothesis_tests parquet (delta now computed from paired means).
  - **PIB acquirer** built with 3 parallel workers covering 12-month window (~5-10 M tokens projected).
  - **Reference books added**: Spectrum + Laxmikanth + Satish Chandra + Bipan Chandra + Poonam Dahiya + Karthikeyan + PMF Geography (G.C. Leong dropped — Tesseract noise).
  - **Cleanup helpers added to `clean.py`**: `strip_dropcap_and_strikethrough()` (removes ~3,800 pymupdf4llm strikethrough artifacts from Laxmikanth+Spectrum), `fix_idrop_typos()` (broad `aton/iton/cton/tng` suffix fixes for the PDF font 'fi'-ligature drop bug, with `automaton/triton/briton` exclusions). 12 new tests pin behavior.
  - **Production-build cleanup** across 15 files: removed Phase-N framing (Phase 0/1/2/3 → descriptive language), removed v2-branding from internal docstrings/configs (keeping it only where it disambiguates output paths from v1 artifacts), renamed `runs/ablation_v2/` → `runs/ablation/`, `cell2_v2_sft_only` → `cell2_sft_only`, bumped `training.__version__` 0.1.0 → 1.0.0.
  - **Memory rules saved**: `feedback_m5_memory_pressure` (don't install heavy ML stacks on M5).
  - **Repo transferred** GitHub ownership: `YeeshanMalik` → `Yeeshan-prayasai`; local remote updated.

---

## 9. Critical command reference

```
# Local
make build-cpt-corpus           # OCR → clean → leakage → tokenize+pack
make build-sft-corpus           # Build SFT JSONL from v1 ft_corpus + pyqs word-count join
make export-corpus              # Parquet → JSONL for human inspection
make dashboard                  # Launch Streamlit locally
.venv/bin/python -m pytest training/tests/    # Run test suite

# EC2 (after `make verify-aws-env`)
make smoke-cpt-gemma            # 100-step smoke
make cpt-gemma / make cpt-qwen  # ~55 L40S-h each
make sft-gemma-v2 / make sft-qwen-v2   # ~20 L40S-h each
make ablation-gemma / make ablation-qwen   # 6-cell grid
make infer-v2 / make score-v2   # reuse v1 inference + scoring on new adapters
```
