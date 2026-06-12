# v2 Pipeline Audit — Findings vs Published Methodology

**Date:** 2026-06-11
**Method:** 6 parallel review passes (QLoRA/LoRA setup · SFT loss · scheduler/hyperparams · data pipeline · corpus mix · orchestration/eval), each checked against the primary literature the methodology cites. Every CRITICAL finding was independently re-verified against the code before inclusion.
**Headline:** the methodology document is largely sound; the implementation diverges from it in 6 places that would crash, and ~10 places that would silently train wrong or train nothing. All are fixable in code/config before any GPU hour is spent. The corpus data itself is fine.

---

## A. Crashes (would abort at startup)

| # | Finding | Where | Fix |
|---|---|---|---|
| A1 | Gemma target-module regex matches **zero modules** — `(?:language_model\.)?` sits before `model\.layers`, but Gemma-4-E4B names are `model.language_model.layers.N.*`. peft raises `ValueError` at `get_peft_model`. Verified empirically. Qwen unaffected. | `training/trainers/base.py:46-49` | `^(?:.*\.)?model\.(?:language_model\.)?layers\.\d+\.` + startup assert that adapter count == n_layers × 7 |
| A2 | `sft.py` calls `build_model_with_lora` / `config_summary` / `find_latest_checkpoint` without importing them → instant `NameError`. Proof the SFT entrypoint never ran end-to-end. Verified via AST. | `training/trainers/sft.py:56,179,193,219` | extend the `.base` import |
| A3 | Pulse callback would crash if wired: `on_step_end(..., tokenizer=None)` but transformers 5.x passes `processing_class=` | `training/eval/pulse.py:247-248` | read `kw.get("processing_class")` |

## B. Silent no-ops (runs "successfully", learns nothing / gates nothing)

| # | Finding | Where | Fix |
|---|---|---|---|
| B1 | **CPT LR 1e-5 is a full-parameter LR applied to fresh LoRA adapters.** Ibrahim/Gupta re-warming rationale doesn't transfer to frozen-base LoRA. Biderman 2024 + QLoRA paper: optimal LoRA LR ≈ 10×–20× higher (1e-4–2e-4). ΔW grows ∝ η² early (B=0 init) → ~400× smaller weight delta at 1e-5 vs 2e-4. CPT phase likely a near-no-op as configured. | `cpt_{gemma,qwen}.yaml`, `base.py:81` | peak LR 1e-4–2e-4; sweep {5e-5, 1e-4, 2e-4} on a 500-step pulse branch; amend methodology §4.3 |
| B2 | **Length-penalty loss is triply dead:** (a) trl 1.5 strips `target_word_count` before the collator (`_signature_columns` hardcoded) → penalty branch never executes; (b) even if it arrived, penalty derives from **label** token counts — constant w.r.t. parameters, zero gradient; (c) it would measure gold-answer length, a dataset property. §4.6's headline mechanism trains nothing. | `sft.py:144,151`, `length_penalty_math.py:52-56` | Either differentiable expected-EOS surrogate, or (simpler, literature-standard) inject "Answer in ~N words" into the prompt and let CE learn it. **Decision required** |
| B3 | **Gradient-accumulation normalization broken** by the `compute_loss` override: ignores `num_items_in_batch`, and transformers 5.x skips the GA division when the model accepts loss kwargs → backward loss ~64× too large → every step hard-clipped at max_grad_norm=1.0, LR schedule effectively destroyed. | `sft.py:141-162` | delegate to `super().compute_loss(..., num_items_in_batch=...)`, add penalty after |
| B4 | **Pulse eval never wired:** no caller constructs `PulseEvalCallback`; `extra_callbacks` is always None. The 500-step task pulse, MMLU pulse, and Hindi −5pp hard-stop do not run in any trainer. Verified by grep. | `run_cpt.py:113`, `run_sft.py:122` | construct + pass callback in both entrypoints |
| B5 | **Hindi hard-stop evaluates 0 items even if wired:** `_items_to_mcq` expects top-level `question/options/correct_option_letter`; eval_set keeps them inside `gold_payload` JSON → all rows filtered → `n=0` → gate skipped. | `pulse.py:132-176` | parse `gold_payload`; assert probe size ≥ 90% expected |
| B6 | **MMLU baseline never set** (`mmlu_baseline=None`, no caller) → MMLU regression check inert. | `pulse.py:62-65` | measure at step 0 in orchestrator |
| B7 | **Replay buffer never reaches the corpus:** `clean.py` globs only `.txt`/`.md`; slimpajama + wikipedia are `.jsonl` → never enter `cpt_clean_dedup` → 0% replay (not the planned 20%), and they'd bypass clean/dedup/leakage even if globbed at pack time. Verified. | `clean.py:398,408` | jsonl-aware clean pass; build-time assert replay share > 0 |
| B8 | **Task pulse can't run:** requires `data/eval_set_holdout.parquet` (doesn't exist) built by a `freeze_eval_set.py --holdout` flag (doesn't exist). Early-stop rule from §7 unimplemented (logs only). | `pulse.py:105-110` | build holdout + implement or descope |

## C. Trains wrong (measurable quality damage)

| # | Finding | Where | Fix |
|---|---|---|---|
| C1 | **Chat-template mismatch:** SFT trains on raw `instruction\n\ninput\n\noutput` concat; inference wraps the same text in `apply_chat_template(...)`. Model never sees `<start_of_turn>model` / `<\|im_start\|>assistant` framing it must answer under; `<end_of_turn>`/`<\|im_end\|>` never trained as terminator → weak stop supervision (also undermines length control). | `build_sft_corpus.py:94-103` vs `runners.py:389-506` | emit `{"prompt","completion"}` (or `messages`) rows; trl applies the template + enables completion-only loss in one change |
| C2 | **Full-sequence SFT loss** (no prompt masking): plain-`text` dataset → trl `completion_only_loss=False`. Code comments claim masking exists; they're wrong. | `build_sft_corpus.py:179`, `sft.py:150` | fixed by C1's format change |
| C3 | **SFT max_steps=16000 ≈ 34 epochs** over 29,892 rows (v1's 16000 was at effective batch 8; v2 runtime is 64). Methodology says 2 epochs. Deep memorization + only the last 3 checkpoints survive rotation. Verified arithmetic. | `sft_{gemma,qwen}.yaml`, `runtime_l40s.yaml` | max_steps ≈ 950–1400; add `load_best_model_at_end=True` |
| C4 | **CPT max_steps=15260 sized for a 4B corpus that doesn't exist** (actual ~1.15B) → silent ~3.5 epochs; comment says "one epoch". | `cpt_{gemma,qwen}.yaml:16` | set explicit 4-epoch budget on the post-fix corpus; rewrite comment |
| C5 | **Qwen EOS separators masked from loss:** pad==eos fallback + `DataCollatorForLanguageModeling` masks `labels==pad_id`. Packs are fixed-length (padding never occurs) so the only thing masked is every document boundary. Gemma safe (dedicated pad). | `base.py:111-112`, `cpt.py:171-173` | trivial collator: `labels=input_ids.clone()`, no pad dependence |
| C6 | **Gemma gets zero BOS in CPT packs** (`add_special_tokens=False`, EOS-only separators). Gemma is BOS-sensitive; pretraining format is BOS-prefixed docs. | `tokenize_pack.py:131-140,199` | per-doc `[bos]+ids+[eos]` for Gemma; EOS-only for Qwen |
| C7 | **No mix weighting exists** — `data_mix_cpt.yaml` never authored; batch share = disk-volume accident; ORF/MEA IR commentary dominates domain tokens while NCERT+refbooks (the Prelims gold) are <10%; §5.1's early oversampling also unimplemented. | `tokenize_pack.py:68-85`, missing config | implement per-source repetition/cap weights at pack time, carry `source` column. **Decision required** (see proposed mix below) |
| C8 | **CPT on `-it` models with zero instruction data in the mix** — raw-text CPT erodes instruction-following (AdaptLLM; arXiv 2401.03129); replay (raw web) doesn't guard chat ability; no IFEval-style pulse to detect it. | corpus composition | mix ~5-15% instruction-format data (v1 SFT pairs in chat template) into CPT |
| C9 | **Paragraph-level MinHash at 0.70 without Jaccard verification** — more aggressive than Lee 2022 (doc-level 0.8) / FineWeb (~0.75); LSH candidates dropped unverified (effective threshold even lower); first-seen-wins = alphabetical priority, not source quality. Risks deleting multi-source factual reinforcement Task A needs. | `clean.py:306-358` | doc-level near-dup at 0.8 with Jaccard verify; keep paragraph-level exact dedup; explicit source-priority order |
| C10 | **FineWeb floor mis-implemented:** `<3 non-blank lines` drops complete single-paragraph news articles (measured 5.5% of Hindu sample, 439–6,544-char real articles). Penedo floor is word-based. | `clean.py:46-47,263-273` | word floor (<50 words), drop the line floor |
| C11 | **`<<<END-RECORD>>>` delimiter survives into corpus**; DB extracts pack as one giant doc (no per-record EOS). | `clean.py` (absent), `tokenize_pack.py:109-113` | split on delimiter, drop it |
| C12 | **Hindi docs in the "English-only" corpus** (e.g. Devanagari Hindu articles); also invisible to the leakage gate (`[a-z0-9]+` tokenizer). | no language filter exists | Devanagari-ratio filter in clean |
| C13 | **fp32 upcast of Gemma's ~3B embedding/PLE params** by `prepare_model_for_kbit_training` — ~12 GB VRAM wasted (fits, but halves headroom). | `base.py:137-141` | re-cast embeddings to bf16 post-prep, keep norms fp32 |
| C14 | **SFT corpus never leakage-checked in v2** (`skip_ngram=True` in run_sft; build_sft_corpus has no gate) despite §5 claiming it. | `run_sft.py:115` | run 50-gram gate over sft_v2 text at build |
| C15 | **Ablation cell 6's midpoint checkpoint is rotated away** (`save_total_limit=5` keeps last 5; checkpoint-7630 never even created) → fallback silently substitutes a 72% checkpoint. | `run_ablation.py:233-242`, yaml | midpoint-save callback exempt from rotation; error if nearest >5% off |
| C16 | **Ablation driver never evaluates** (docstring steps 3-5 unimplemented) and `run_inference.py` hardcodes v1 adapter paths — no `--adapter` flag, no merge step, no `infer-v2`/`score-v2` Make targets. | `run_ablation.py:245-284`, `scripts/run_inference.py:53-58`, Makefile | plumb adapter override + merge + wire eval into cells |
| C17 | **Double BOS at inference (v1 bug, affects v2 evals):** chat-template text already starts with `<bos>`, then `tokenizer(...)` adds BOS again. Gemma documented to degrade. | `runners.py:517,587,641` | `add_special_tokens=False` on templated strings |
| C18 | **Hard-stop indistinguishable from success:** sets `should_training_stop` but not `should_save`; orchestrators still write `final/` and exit 0; ablation would continue on a regressed adapter. | `pulse.py:265,277`, run_*.py | HARD_STOP marker + non-zero exit |
| C19 | **Pulse measures with a different instrument than its baseline:** raw MMLU-style prompt vs v1 baselines measured via JSON-prompt+chat-template → −5pp gate compares apples to oranges. | `mcq_inference.py:49-56` vs hardcoded baselines | measure step-0 baseline with the pulse's own prompt |
| C20 | **Truncation from the end** (`keep_start`) at max_length 4096 cuts long completions' tails + EOS (Task E rows likely) → teaches non-stopping. | `sft.py:93` | measure per-task token lengths at build; drop/split over-length rows |

## D. Spec-vs-code divergences to reconcile in `v2-methodology.md`

1. §4.4 claims "attention reset at separators" — code does naive concatenation (standard, but the claim is false; Llama 3 does reset). Either implement block-diagonal masks or amend the spec.
2. §4.4 batch geometry: spec 8×8, code 1×64 (same tokens/step; spec's own citation argues against heavy accumulation).
3. §4.3 SFT "cosine to 0.1× peak" — HF `cosine` decays to 0. Use `cosine_with_min_lr`.
4. §4.5 mix table + §5.1 oversampling — unimplemented (C7).
5. §4.6 length-penalty "gradient signal" claim — false as implemented (B2).
6. §7 pulse sizes (spec 80 Task-A; code 50) + early-stop rule unimplemented.
7. Cell 3/5 cost notes vs driver behavior; `Tasks B + G` docstring claim (corpus has A/B/C/E only).
8. Leakage n-gram unit: words (~65-70 BPE) vs Carlini's BPE-token definition — document or lower to ~35 words.

## E. Verified correct (non-exhaustive)

- NF4 + double-quant + bf16 compute; paged_adamw_8bit (β2 wiring verified into optimizer kwargs); non-reentrant gradient checkpointing via kbit-prep; rsLoRA scale α/√r = 2.0 exactly as intended; frozen embeddings appropriate for English-domain CPT.
- WSD scheduler math exact (boundaries, 0.1× floor, no off-by-one); resume restores LR correctly for unchanged configs; clipping/decay present.
- Packing loop invariants (EOS never elided, empty docs skipped, trailing partial dropped) well-tested; numbers/tables kept end-to-end (per the explicit corpus rule); leakage gate direction + coverage + fail-loud non-skippable wiring all correct; deterministic seeds throughout.
- `_eval_no_grad` correctly saves/restores train state; adapter continuation CPT→SFT is single-adapter, no stacking; ablation cell branching (dirs, rsLoRA-off cell, resumability) sound apart from C15/C16.
- v1 instruction strings byte-identical between SFT corpus and inference; left-padding for batched generation correct.

## F. Priority fix order

**Phase 1 — before corpus build finishes (data plane):**
B7 (.jsonl replay) → C11 (END-RECORD) → C10 (line floor) → C9 (dedup granularity/threshold) → C12 (language filter) → C6 (Gemma BOS) → C7 (mix weighting mechanism + `data_mix_cpt.yaml`) → C14 (SFT leakage gate)

**Phase 2 — trainer plane (before smoke-cpt):**
A1 (regex) → A2 (imports) → C5 (collator) → B1 (CPT LR) → C4 (CPT steps) → C3 (SFT steps) → B3 (GA delegation) → B2 (length-penalty redesign) → C1/C2 (prompt-completion format) → C13 (bf16 re-cast) → C20 (truncation)

**Phase 3 — eval/orchestration plane (before full runs):**
B4 (wire pulse) → A3 (callback signature) → B5 (gold_payload) → B6 (MMLU baseline) → C19 (pulse baseline instrument) → C18 (hard-stop semantics) → C15 (cell-6 checkpoint) → C16 (eval plumbing + Make targets) → C17 (double BOS) → B8 (holdout or descope)

**Phase 4 — methodology doc reconciliation** (section D items) before pre-registration is cited in the paper.

## G. Decisions required (alter the pre-registered design)

| Decision | Options | Audit recommendation |
|---|---|---|
| CPT peak LR | keep 1e-5 / 5e-5 / 1e-4 / 2e-4 | 1e-4 with 500-step sweep confirmation |
| Length control | differentiable EOS surrogate / prompt-injected target / drop §4.6 | prompt-injected ("Answer in ~N words") — simplest mechanism with actual evidence |
| SFT data format | keep raw text / prompt-completion / messages | prompt-completion (fixes C1+C2 in one move) |
| CPT mix | as-is (volume accident) / weighted per audit table | weighted: domain-dominant ~80/20 with NCERT×4, refbooks×6, ORF↓0.5×, replay capped ~115M, ≤4 epochs |
| Synthetic RC/QA data (AdaptLLM-style) | add (~40M tokens, generated offline) / skip | add — highest-leverage lever for the Task-A +10pp gate per Ovadia/Cheng |
| Instruction data in CPT | add 5-15% / skip | add (v1 SFT pairs, chat-templated) |
| Attention reset at boundaries | implement / amend spec | amend spec (naive concat is standard; implementing needs flash-attn varlen) |
