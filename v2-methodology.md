# v2 Training Methodology — English path

**Owner:** Yeeshan — Data Scientist, prayas.ai
**Last updated:** 2026-06-04
**Status:** proposed; not started
**Scope:** English-only training. Hindi adaptation is **deferred to a separate strategy doc** (`v2-hindi-strategy.md`, to be authored). All references in this doc to language splits, replay sources, and acceptance gates are English-only.
**Companion to:** [`v2-expert-input-plan.md`](v2-expert-input-plan.md) (resourcing), [`experiment-report.md`](experiment-report.md) (v1 results)

---

## 0. TL;DR — the recipe in one paragraph

Two-stage training per base model on **English content only**: **rank-64 LoRA-CPT** on a mix-weighted ~1.4 B-token-exposure pass over the UPSC corpus (NCERT/reference books repeated ×4 per Muennighoff 2023, IR commentary capped, ~20 % English replay, ~5-10 % instruction-format slice), followed by **rank-64 LoRA-SFT** continuing the same adapter on ~30 K chat-templated English instruction pairs with prompt-side word-count conditioning. LoRA targets all 7 projections in **every** decoder layer (Gemma 42, Qwen 32) → 294 / 224 adapters per model, 144 M / 114 M trainable parameters. Uses **RSLoRA scaling** (α/√r, not α/r) to stabilize the higher rank, **WSD learning-rate schedule** to enable continued training, and a **6-cell ablation grid** to attribute gains. Total compute: ~16 H100-hours CPT + ~6 SFT per model. Hindi is deferred — Qwen-HI catastrophe (v1: 0.426) and Gemma-HI strength (v1: 0.636) are both addressed in the separate Hindi strategy.

---

## 1. What v1 told us — the diagnosis driving v2

[experiment-report.md §6-§8](experiment-report.md) gives the numbers; this section translates them into training decisions.

| Observation from v1 | What it means for v2 |
|---|---|
| Task A loss EN: Gemini 0.88-0.89 vs FT 0.61-0.65 (d up to −0.67) | Pure knowledge gap — FT corpus didn't contain enough factual material. **CPT is required**; SFT alone won't close it. |
| Hindi-stratum results (Qwen 0.426, Gemma 0.636) | **Deferred** to `v2-hindi-strategy.md`. This English-path plan does not address them. |
| FT wins B/E with d=0.21–0.92 (English) | The Q→A SFT regime works for *style* / *synthesis*. Don't break what worked — keep SFT phase, just put CPT in front of it. |
| Format validity 0.61-0.70 universal | Not a training problem. Fix at inference with constrained decoding (Outlines / XGrammar). Out of scope for this doc. |
| ECE 0.37-0.89 universal | Not a training problem either — verbal confidence is broken. Fix with logit confidence at inference. Out of scope. |
| Word-count adherence FT 0.08-0.09 vs Gemini 0.30-0.48 | Trainable — prompt-side word-count conditioning in SFT (§4.6). |
| Long-tail subjects (Art & Culture, Misc) clear LOSS on English stratum | CPT corpus must over-sample these subjects relative to natural distribution. |
| `silly_mistake_prone=1` items lose universally | Calibration gap; deferred to inference fixes. Not a training lever. |
| Qwen-FT position bias χ² p=1.5e-5 (English) | Balance the answer-letter distribution in the SFT corpus (mechanical fix, no methodology impact). |
| v1 used rank 16, **last-16 layers only** | Surgical FT was OK for SFT; CPT needs **all layers + higher rank** ([Biderman 2024](https://arxiv.org/abs/2405.09673)). |

---

## 2. v2 acceptance criteria (pre-registered before training)

Adapter does **not** merge to the production candidate unless:

All gates evaluated on the **English stratum only**. Hindi gates live in `v2-hindi-strategy.md`.

| Gate | v1 baseline (EN) | v2 target | Failure means |
|---|---:|---:|---|
| Task A accuracy EN | 0.652 (Gemma) | **≥ 0.75** | CPT didn't deliver enough fact injection |
| Task A neg-mark score EN | 1.06 (Gemma) | **≥ 1.40** | Wrong-answer confidence still costs marks |
| Task B BERTScore | 0.833 | **≥ 0.825** (no regression band ±0.01) | CPT damaged the SFT win |
| Task B word-count adherence | 0.086 | **≥ 0.40** | Length conditioning didn't work |
| Task C Score MAE | 1.901 (Qwen) | **≤ 2.20** (no regression band) | Grading quality unchanged or better |
| Task E mains BERTScore | 0.873 (Qwen) | **≥ 0.865** | Current-affairs synthesis preserved |
| Dev loss trajectory | — | monotonic over 80 % of 500-step pulses | Training divergence; investigate |
| General-capability holdout (MMLU sample) | base baseline | **within −2 pp** of base | Catastrophic forgetting on general knowledge |
| Hindi-stratum results | — | **must not regress vs v1** (soft gate) | Hindi shouldn't get *worse* even though it's not the focus; if it does, the English-only CPT bled into Hindi capability and the Hindi strategy needs to start from a different checkpoint |

If a gate fails, the v1 SFT-only adapter remains the production candidate and v2 is re-scoped.

---

## 3. Approach — why CPT → SFT, and why LoRA both times

### 3.1 The decision

**Stage 1: LoRA-CPT** (raw text, next-token-prediction loss). Injects English-side UPSC facts and prose distribution into the LoRA delta.
**Stage 2: LoRA-SFT** (instruction pairs, masked loss on response only). Continues the same adapter; teaches task shape.

This is the [Gururangan et al. — Don't Stop Pretraining (ACL 2020)](https://arxiv.org/abs/2004.10964) sequencing, reaffirmed for SLMs by [Domain-Adaptive CPT of SLMs (arXiv 2504.09687)](https://arxiv.org/abs/2504.09687) and [Reuse Don't Retrain (arXiv 2407.07263)](https://arxiv.org/abs/2407.07263).

### 3.2 Why LoRA-CPT and not full-parameter CPT

[Biderman et al. — "LoRA Learns Less and Forgets Less" (arXiv 2405.09673)](https://arxiv.org/abs/2405.09673) showed full FT outperforms LoRA on CPT by 1-3 pp absolute, *but* LoRA preserves source-domain capability strictly better. Three reasons we accept the trade:

1. UPSC is distribution-intensification, not new-distribution. Gemma-4 and Qwen-3.5 already saw Wikipedia, government docs, Indian English news, and Hindi corpora — we're amplifying signal they already have. This is the regime where LoRA's gap to full FT is smallest.
2. Adapter compatibility — LoRA adapters are diff-able, swappable, and inspectable. Full-FT checkpoints aren't.
3. The 4B model is at the size where full-FT CPT requires ~80 GB optimizer state (AdamW master+momentum in fp32). LoRA at rank 64 needs ~22 GB. The savings buy us more tokens in budget.

### 3.3 Why not DoRA / PiSSA instead

[DoRA (arXiv 2405.17357)](https://arxiv.org/abs/2405.17357) and [PiSSA (arXiv 2404.02948)](https://arxiv.org/abs/2404.02948) both report +1-2 pp accuracy over vanilla LoRA. [Spheron 2026 PEFT Decision Guide](https://www.spheron.network/blog/peft-methods-2026-dora-galore-pissa-vera-guide/) recommends DoRA "if you have downstream accuracy benchmarks and want the extra 1-2%".

But [Shi et al. — "Learning Rate Matters: Vanilla LoRA May Suffice" (arXiv 2602.04998)](https://arxiv.org/abs/2602.04998), published February 2026, shows the DoRA/PiSSA gap closes to noise when LR is properly tuned. We stick with vanilla LoRA + RSLoRA scaling so we can isolate the v1→v2 delta cleanly. Re-visiting DoRA is a v3 lever, not v2.

---

## 4. The full parameter sheet

### 4.1 Architecture targets — what gets trained

| | Gemma-4-E4B | Qwen-3.5-4B |
|---|---|---|
| Total text decoder layers in base | 42 | 32 |
| LoRA-targeted layers (v2) | **42 (all)** | **32 (all)** |
| Projections per layer | 7 (q, k, v, o, gate, up, down) | 7 |
| **Total LoRA adapters** | **294** | **224** |
| Hidden dim | 2560 | 2560 |
| FFN intermediate dim | 10240 | 9216 |
| Embedding layer | **FROZEN** | **FROZEN** |
| `lm_head` | **FROZEN** | **FROZEN** |
| LayerNorms (input, post-attn) | **FROZEN** | **FROZEN** |
| Vision encoder (Gemma only) | **FROZEN** | n/a |
| Audio encoder (Gemma only) | **FROZEN** | n/a |
| Base weights (everywhere) | **FROZEN** | **FROZEN** |

Embeddings stay frozen because:
- The English vocabulary is fully covered by both base tokenizers (Gemma 262K, Qwen 248K); no token additions needed.
- Tokenizer-fertility questions for Devanagari are deferred to the Hindi strategy.

### 4.2 LoRA configuration

| Hyperparameter | Value | Rationale |
|---|---:|---|
| **Rank `r`** | **64** | 4× v1's rank-16. [Biderman 2024](https://arxiv.org/abs/2405.09673) shows CPT-style learning gains scale with rank up to ~128. We pick 64 as the value past which Biderman's curve flattens for 4B-class models. |
| **Alpha `α`** | **16** | With RSLoRA scaling α/√r, the effective scale is α/√64 = α/8 = 2.0. Matches v1's α/r = 32/16 = 2.0, so the *step magnitude* is comparable to v1; only capacity changes. |
| **`use_rslora`** | **true** | Vanilla LoRA's α/r causes [gradient collapse at high rank (Kalajdzievski 2023, arXiv 2312.03732)](https://arxiv.org/abs/2312.03732). RSLoRA's α/√r is necessary at r=64; without it, rank 64 trains no better than rank 16. v1 had `use_rslora: false` because rank was low enough not to matter. |
| **LoRA dropout** | **0.05** | Same as v1; standard. |
| **`use_dora`** | **false** | See §3.3. |
| **Target modules** | regex matching `^.*\.layers\.\d+\..*\.(q\|k\|v\|o\|gate\|up\|down)_proj$` for both | All decoder layers, all 7 projections. v1 restricted to layers 26-41 / 16-31 — we drop that restriction for CPT. |
| **`bias`** | `"none"` | Standard; biases not trained. |
| **`init_lora_weights`** | `true` (Gaussian A, zero B) | Standard init. Adapter is no-op at step 0; gradients build up the perturbation. |

### 4.3 Optimizer and learning-rate schedule

| Phase | CPT | SFT |
|---|---|---|
| **Optimizer** | AdamW (paged for VRAM) | AdamW (paged) |
| **β₁** | 0.9 | 0.9 |
| **β₂** | **0.95** | 0.999 |
| **ε** | 1e-8 | 1e-8 |
| **Weight decay** | 0.1 | 0.01 |
| **Gradient clipping** | 1.0 | 1.0 |
| **Peak LR** | **1.0e-5** | **2.0e-4** |
| **LR schedule** | **WSD (Warmup-Stable-Decay)** | Cosine |
| **Warmup** | 1 % of steps, linear | 3 % of steps, linear |
| **Stable phase** | 80 % of steps at peak | n/a |
| **Decay phase** | 19 % of steps cosine to 0.1× peak | cosine to 0.1× peak |
| **Min LR** | 1.0e-6 | 2.0e-5 |

**Why β₂ = 0.95 for CPT, 0.999 for SFT:** lower β₂ tracks gradient variance over a shorter window, which suits a stationary distribution (raw text). Higher β₂ smooths over more steps, which suits the noisy + diverse SFT mix. This is standard from GPT-3 onward.

**Why WSD over plain cosine:** [Wen et al. — "Understanding Warmup-Stable-Decay" (arXiv 2410.05192)](https://arxiv.org/abs/2410.05192) and [Beyond Cosine Decay for Continual Pretraining (arXiv 2503.02844)](https://arxiv.org/abs/2503.02844) show WSD outperforms cosine for continued pretraining and crucially allows extending training by resuming from the *stable* phase. If we want to add tokens later (e.g. continual current-affairs updates) we don't have to restart — we resume from the stable checkpoint.

**Why peak LR 1e-4 for CPT and 2e-4 for SFT:** this is **LoRA-adapter** CPT, not full-parameter CPT. The Ibrahim/Gupta re-warming rates (~1e-5) apply when the base weights themselves move; here the base is frozen and the rank-64 adapters start from B=0, where [Biderman et al. 2024](https://arxiv.org/abs/2405.09673) and the QLoRA paper both place the optimal LR ~10× above the full-FT optimum (1e-4–2e-4). At 1e-5 the realized adapter delta ΔW = (α/√r)·BA — which grows quadratically in LR early in training — stays in the noise floor for the whole step budget. Forgetting control comes from the frozen base + 20% replay + WSD decay, not from starving the adapter. *(Amended 2026-06-11 after audit; the original 1e-5 was a full-parameter rate applied to adapters.)*

### 4.4 Batch / sequence configuration

| | CPT | SFT |
|---|---|---|
| **Sequence length** | 4096 | 4096 (shared runtime config) |
| **Sequence packing** | Yes — `[BOS]+doc+[EOS]` per doc for Gemma (BOS-sensitive), `doc+[EOS]` for Qwen (no BOS in vocab) | No — one chat-templated prompt+completion per example |
| **Per-device batch size** | 1 | 1 |
| **Gradient accumulation** | 64 | 64 |
| **Effective batch (tokens/step)** | 1 × 64 × 4096 = **262,144** | 64 examples/step |
| **Steps** | derived at runtime: one epoch over the mix-weighted packed corpus (per-source repetition lives in the mix, §4.5) | derived at runtime: 2 epochs over the train split (~950 steps at ~30K rows) |
| **Precision** | bf16 mixed | bf16 mixed |
| **Gradient checkpointing** | on | on |
| **Loss masking** | next-token over all packed tokens. **No attention reset at document separators** — naive concatenation, the GPT-3/Pythia recipe. (Llama 3 does reset; implementing it needs block-diagonal/varlen attention, and the expected delta at this scale doesn't justify it.) | **mask prompt tokens; train on completion only** (trl prompt-completion format) |

On L40S 48 GB, per-device bs=1 + accumulation 64 is the VRAM-safe configuration at seq 4096 with rank-64 all-layer adapters. [Marshall et al. (arXiv 2507.07101)](https://arxiv.org/abs/2507.07101) argues against heavy accumulation where avoidable; if VRAM headroom after the bf16-embedding fix allows bs≥2, accumulation drops proportionally (same 262K tokens/step).

### 4.5 Data mix — the CPT corpus (English only)

All sources filtered to English at the clean stage (documents with >30% Devanagari among alphabetic characters are dropped; scraped sources like The Hindu carry Hindi-language articles). Bilingual rows in prayas DB are kept on the English side only.

The acquired corpus is ~1.1-1.2 B unique tokens (28-30% of the original 4 B aspiration — comprehensive on named-text coverage, lighter on volume). Per [Muennighoff et al. 2023 (arXiv 2305.16264)](https://arxiv.org/abs/2305.16264), repeating unique data up to ~4 epochs costs almost nothing vs fresh tokens, so the mix repeats the high-yield core rather than padding with replay.

**The mix is enforced** — per-source `repeat` (epochs) and `cap_tokens` weights live in [`training/configs/data_mix_cpt.yaml`](training/configs/data_mix_cpt.yaml) and are applied by `tokenize_pack` at document granularity before packing. Summary:

| Source group | Weight | Rationale |
|---|---|---|
| NCERT, reference books, Constitution | repeat ×4 | Foundational Prelims fact base — highest yield per token |
| Exam-framed notes (PMF IAS, Mrunal, PRS) | repeat ×3 | Mid-density, exam-shaped |
| Government primary (Eco Survey, ARC, NITI, ministries, IPCC) | repeat ×2 | Factual but verbose |
| Current affairs (PIB, newspapers, ISRO, DRDO) + Budget | ×1 | Dated / boilerplate-heavy; one pass |
| ORF + MEA (IR commentary) | cap 30 M each | Low Prelims-fact density; must not dominate by raw volume |
| prayas DB extracts | ×1 | House style |
| **Instruction slice** (v2 SFT train pairs, chat-templated at pack time) | ×1 | CPT on `-it` checkpoints erodes chat formatting unless instruction-format data stays in the stream ([AdaptLLM, arXiv 2309.09530](https://arxiv.org/abs/2309.09530); [arXiv 2401.03129](https://arxiv.org/abs/2401.03129)) |
| **Replay** (FineWeb-Edu sample cap 80 M + Wikipedia-India cap 35 M) | ~20% of exposures | Anti-forgetting on general English capability |

**Replay ratio ≈ 20%.** [Continual Learning of LLMs Survey (ACM CSUR 2025)](https://dl.acm.org/doi/10.1145/3735633) finds the operating band is 5-30%, modal 15-20%. The build fails loud if the replay share lands at 0 (the failure mode where the `.jsonl` replay never enters the corpus).

**Optional RC/QA augmentation** (`training/data/generate_rc_qa.py`): AdaptLLM-style synthetic reading-comprehension Q&A over the NCERT/reference-book core — facts are learned from many phrasings ([Ovadia et al. 2024, arXiv 2312.05934](https://arxiv.org/abs/2312.05934)). Generation is API-cost-gated; enable by adding an `rc_qa` mix entry after generating.

**Interleaving:** the weighted document list is shuffled with a fixed seed before packing, and HF Trainer reshuffles sequences per epoch — every gradient step sees a cross-source mix. Block ordering triggers the forgetting pattern documented in the continual-learning literature.

### 4.6 Length control

**CPT phase:** standard next-token cross-entropy. No modifications.

**SFT phase:** length control is **in the data, not the loss**. Task-B rows with a known `pyqs.word_count` target carry an explicit "Answer in approximately N words." instruction in the prompt; plain cross-entropy learns the association between the stated target and the answer length — the mechanism [Plan-and-Write Length Control (arXiv 2511.01807)](https://arxiv.org/abs/2511.01807) actually uses at SFT time.

*(Amended 2026-06-11 after audit. The originally-specified auxiliary loss `λ·|len(gen)−target|/target` is not implementable as written: at training time there is no "generated length" — only teacher-forced labels, whose length is a constant w.r.t. parameters. The term carried zero gradient and could only distort the reported loss. A differentiable surrogate (expected-EOS-position) exists but adds complexity with no evidence advantage over prompt-side conditioning for a ±50-word target.)*

This addresses v1's Task B word-count adherence 0.086. Two supporting fixes land in the same change: the SFT data is chat-templated identically to inference (so the turn-end token is trained as the terminator — a model that can't stop can't hit a word count), and over-length rows are dropped rather than tail-truncated (truncation removed gold EOS, teaching non-stopping).

---

## 5. Curriculum + ordering

### 5.1 Within CPT

**No explicit curriculum.** Random shuffle across all sources (seeded, at document level before packing; HF Trainer reshuffles sequences per epoch). [Beyond Repetition (arXiv 2509.24356)](https://arxiv.org/abs/2509.24356) finds curriculum gains for *data-constrained* pretraining are small relative to data-mix variance.

*(The originally-planned early 1.3× long-tail-subject over-sampling schedule is descoped — subject-level weighting requires per-document topic tags the corpus doesn't carry; the per-source mix weights in §4.5 are the implemented prioritization.)*

### 5.2 Within SFT

Difficulty-mixed ordering — UPSC questions have difficulty tags (`silly_mistake_prone`, `tier_1_difficulty`). v1 had no curriculum; we add a soft easy-to-hard sort within each epoch's shuffle window. Easy first → model gets the format right before facing edge cases. Minor lift expected (3-5 % on `silly=1` cells).

---

## 6. Hindi — deferred

Hindi is handled in a **separate strategy document** (`v2-hindi-strategy.md`, to be authored). That doc will address:

- Whether to train a separate Hindi-only adapter or extend this English adapter with a second Hindi CPT pass
- Tokenizer-fertility check on Devanagari (both base tokenizers include the block; fertility numbers TBD)
- Whether to use Gemma (v1 HI 0.636) or Qwen (v1 HI 0.426) as the Hindi base — current evidence favors Gemma
- Hindi corpus sourcing (NCERT-HI editions, IndicLLMSuite sample, prayas internal HI content)
- Bilingual evaluation and code-mixing rate measurement
- Hindi acceptance gates and rollback policy

**What this English plan promises about Hindi:** nothing positive. The §2 soft gate requires Hindi accuracy not to *regress* from v1, but no improvement is targeted here. If English CPT inadvertently hurts Hindi (possible if the base's Hindi capability lives in the same MLP regions we're perturbing), the Hindi strategy starts from the pre-v2 v1 checkpoint instead of the v2-English checkpoint — adapter-stacking gives us that fallback for free.

---

## 7. Evaluation harness during training

| Pulse | Cadence | What runs | Action on failure |
|---|---|---|---|
| **Loss pulse** | every 100 steps | dev-set NLL on 200 held-out (not-eval) English UPSC passages | log; investigate if 3 consecutive divergences |
| **Task pulse** | every 500 steps | held-out probe (`data/eval_set_holdout.parquet`, built by `training/eval/build_holdout.py`): 50 A-EN MCQs run inline; B/C metrics are too expensive mid-train | logged to pulse.jsonl (trend monitoring) |
| **General-capability pulse** | every 1000 steps | 100-Q MMLU sample (no UPSC overlap) | if drops >2 pp from base, raise replay ratio next phase |
| **Hindi no-regression pulse** | every 1000 steps | 50 v1 Hindi MCQs | catch English-CPT-induced Hindi regression early; hard-stop if drops >5 pp from v1 baseline |

Dev probe questions are sampled from `prod.mcqs` rows NOT in either CPT corpus or v1 locked eval set. SHA-256 leakage check before training start (extends v1's check).

Hard checkpoint at 50 %, 75 %, 100 % of CPT — enables ablation 4 (CPT-only at three depths).

---

## 8. Ablation matrix

Pre-registered to attribute v2's gains correctly:

| # | Config | Trains | Compares | Answers |
|---|---|---|---|---|
| 1 | **v1 baseline** | already done | — | reference |
| 2 | **v2 SFT-only** (skip CPT) | r=64 SFT on v1's English pairs (no new pairs added) | vs #1 | "Did rank-64 + RSLoRA help, independent of CPT?" |
| 3 | **CPT-only** (no v2 SFT — use v1 adapter merged after CPT) | r=64 CPT only | vs #1 | "Did CPT alone inject knowledge?" |
| 4 | **v2 full pipeline (CPT→SFT)** | both phases | vs #2, #3 | "Did stacking the two help over either alone?" |
| 5 | **v2 full with vanilla LoRA** (no RSLoRA) | same as #4 but α/r scaling | vs #4 | "Did RSLoRA actually matter at r=64?" |
| 6 | **CPT-50% checkpoint→SFT** | use 50 % CPT ckpt then SFT | vs #4 | "Diminishing returns on CPT tokens?" |

All ablations evaluated on the same locked v1 Tier-1 eval set with the same BH-FDR test infrastructure. Compute cost: ~3× #4 (because #2, #3, #5 each cost ~half). Run on H100 cluster; full ablation grid ~80 H100-hours per base model.

---

## 9. Risk register + mitigations

| Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|
| Catastrophic forgetting on general English capability | Medium | High | 20 % replay ratio; MMLU pulse every 1000 steps; rollback to previous checkpoint if MMLU drops >2 pp |
| English-CPT bleeds into Hindi capability (Hindi regresses below v1) | Medium | Medium | No-regression Hindi pulse every 1000 steps (§7); hard-stop if Hindi drops >5 pp; fallback = Hindi strategy starts from v1 checkpoint, not v2-English |
| CPT corpus leaks into v1 locked eval set | Low | Critical | SHA-256 hash gate + 5-gram dedup ([Kandpal 2022](https://arxiv.org/abs/2202.06539)); fails-loud at training start |
| Rank 64 underperforms expectation (Biderman ceiling) | Low | Medium | Ablation #5 detects this; budget includes one re-run at rank 128 if so |
| Replay buffer too small / too large | Low | Medium | 20 % is mid-range; pulse #3 detects drift in either direction; can adjust mid-training |
| NCERT licensing dispute | Low | High | prayas is a free product — Indian Copyright Act §52 fair-use applies; legal review before publishing trained adapter weights publicly |
| Compute cost overrun (>$3K total) | Low | Medium | H100 spot pricing ~$2/hr; full ablation grid ~$320 per model; budget headroom 5× |
| Adapter incompatibility with merged MLX checkpoints | Medium | Low | Merge after each phase; re-test inference path with `scripts/runners.py` after CPT |

---

## 10. Compute budget + timeline

**Target hardware: NVIDIA L40S (48 GB VRAM)** on EC2 — same path v1 used via `scripts/run_ft_aws.py`. No H100 access available. L40S is ~3-4× slower than H100 for bf16 training; estimates below reflect L40S timings.

| Phase | Hours (per model, 1× L40S) | Wall clock (calendar days assuming 12 h/day) |
|---|---:|---|
| Data acquisition (NCERT downloads + scraping current affairs + SlimPajama sample) | 60 (mostly network + CPU) | ~7-10 |
| Data prep (OCR + dedup + tokenization) | 80 (CPU mostly) | ~7 |
| CPT pass | **55** | ~5 |
| SFT pass | **20** | ~2 |
| Eval (full Tier-1, reuse v1 pipeline) | 6 | ~0.5 |
| **Per-model subtotal (GPU only)** | **~80** | **~7-8 days GPU calendar** |
| **× 2 base models (serial)** | **~160** | **~14 days serial; ~8 days parallel on 2× L40S** |
| Ablation grid (cells 2, 3, 5, 6 share Phase 0/1 outputs; cells 4 is the headline) | ~550 | +10-14 days parallel |
| **Total GPU** | **~870 L40S-hours** | **~30-45 days end-to-end** |

Cost at L40S spot pricing (~$1.65/hr on AWS as of 2026): ~$1,440 GPU-only; ~$1,700 including storage + S3 egress + dev-loop iteration. The headline run (cell 4 of ablation, both models) is ~$280. Within the v2 budget envelope from [`v2-expert-input-plan.md`](v2-expert-input-plan.md).

**VRAM headroom** at rank 64 + all layers + seq 4096 on 48 GB L40S: predicted ~22-30 GB usage with QLoRA NF4 + gradient checkpointing + paged_adamw_8bit + bs=1, grad_accum=64. Smoke test at Phase 2 boundary will confirm; fallback drops seq to 2048 and raises grad_accum to 128 (preserves 262K-token effective batch).

---

## 11. What this plan does NOT solve (out of scope)

- **All Hindi improvement work.** Tracked in `v2-hindi-strategy.md` (to be authored). This plan only guards against Hindi *regression*, doesn't target Hindi *gains*.
- **Format validity ceiling (v1 was 0.61-0.70).** Constrained-decoding fix at inference. Tracked as P0 in [`v2-expert-input-plan.md`](v2-expert-input-plan.md).
- **ECE calibration (v1 was 0.37-0.89).** Confidence head needs separate retraining + temperature scaling. Tracked as P0 in `v2-expert-input-plan.md`.
- **Task A residual gap to Gemini (English).** Expected v2 lift on Task A EN is +10-15 pp. Gemini's 0.88 ceiling likely remains above us. Hybrid deployment (Gemini for MCQ, FT-SLM for B/C/E/F/G) stays the recommendation.
- **Tokenizer rebuilds.** Question lives in Hindi strategy, not here.
- **Multi-turn conversation FT.** v1 was single-turn; v2 stays single-turn. Multi-turn is a separate adapter pass tracked in `v2-expert-input-plan.md` P3.

---

## 12. Configuration files to produce

When this plan is approved, the following files will be authored from it:

```
configs/lora_v2_cpt_gemma.yaml      # §4.1-4.4 for Gemma CPT
configs/lora_v2_cpt_qwen.yaml       # §4.1-4.4 for Qwen CPT
configs/lora_v2_sft_gemma.yaml      # §4.2-4.6 for Gemma SFT
configs/lora_v2_sft_qwen.yaml       # §4.2-4.6 for Qwen SFT
configs/data_mix_cpt.yaml           # §4.5 corpus weights
configs/data_mix_sft.yaml           # SFT pair sources
scripts/build_cpt_corpus.py         # NCERT OCR + dedup + tokenization
scripts/train_cpt.py                # HF Trainer wrapper with WSD scheduler
scripts/train_sft.py                # HF Trainer wrapper with length-penalty loss
scripts/run_ablation.py             # orchestrates the 6-cell grid
```

The existing `scripts/runners.py` and `scripts/score_tier1.py` are reused unchanged.

---

## 13. References

Live links to every paper cited in this plan, ordered roughly by where they appear:

- Biderman et al., "LoRA Learns Less and Forgets Less" — [arXiv 2405.09673](https://arxiv.org/abs/2405.09673)
- Gururangan et al., "Don't Stop Pretraining" — [arXiv 2004.10964](https://arxiv.org/abs/2004.10964)
- Domain-Adaptive CPT of Small Language Models — [arXiv 2504.09687](https://arxiv.org/abs/2504.09687)
- Reuse, Don't Retrain (NVIDIA CPT recipe) — [arXiv 2407.07263](https://arxiv.org/abs/2407.07263)
- Liu et al., DoRA — [arXiv 2405.17357](https://arxiv.org/abs/2405.17357)
- Meng et al., PiSSA — [arXiv 2404.02948](https://arxiv.org/abs/2404.02948)
- Spheron 2026 PEFT Decision Guide — [Spheron blog](https://www.spheron.network/blog/peft-methods-2026-dora-galore-pissa-vera-guide/)
- Shi et al., "Learning Rate Matters: Vanilla LoRA May Suffice" — [arXiv 2602.04998](https://arxiv.org/abs/2602.04998)
- Kalajdzievski, RSLoRA (α/√r scaling) — [arXiv 2312.03732](https://arxiv.org/abs/2312.03732)
- Wen et al., "Understanding Warmup-Stable-Decay" — [arXiv 2410.05192](https://arxiv.org/abs/2410.05192)
- Beyond Cosine Decay for Continual Pretraining — [arXiv 2503.02844](https://arxiv.org/abs/2503.02844)
- Ibrahim et al., "Simple and Scalable Strategies to Continually Pre-train" — [arXiv 2403.08763](https://arxiv.org/abs/2403.08763)
- Continual Pretraining: How to Re-warm — [arXiv 2308.04014](https://arxiv.org/abs/2308.04014)
- Marshall et al., "Small Batch Size Training" — [arXiv 2507.07101](https://arxiv.org/abs/2507.07101)
- Continual Learning of LLMs Survey — [ACM CSUR 2025](https://dl.acm.org/doi/10.1145/3735633)
- Plan-and-Write Length Control — [arXiv 2511.01807](https://arxiv.org/abs/2511.01807)
- Just Enough Thinking (Length Penalty RL) — [arXiv 2506.05256](https://arxiv.org/abs/2506.05256)
- Beyond Repetition (Curriculum + Simplification) — [arXiv 2509.24356](https://arxiv.org/abs/2509.24356)
- Kandpal et al., n-gram dedup — [arXiv 2202.06539](https://arxiv.org/abs/2202.06539)
- LoRA original — Hu et al., [arXiv 2106.09685](https://arxiv.org/abs/2106.09685)
- QLoRA — Dettmers et al., [arXiv 2305.14314](https://arxiv.org/abs/2305.14314)
- Geva et al., "Transformer Feed-Forward Layers Are Key-Value Memories" — [arXiv 2012.14913](https://arxiv.org/abs/2012.14913)

**Deferred to `v2-hindi-strategy.md`** (Hindi-specific citations, kept for the future doc):

- Rethinking Multilingual CPT — [arXiv 2504.04152](https://arxiv.org/abs/2504.04152)
- Code-Switching Curriculum Learning — [arXiv 2411.02460](https://arxiv.org/abs/2411.02460)
- Multilingual Pretraining Improves Cross-Lingual Knowledge Alignment — [arXiv 2404.04659](https://arxiv.org/abs/2404.04659)
- IRIS: Interleaved Reinforcement with Incremental Staged Curriculum — [arXiv 2604.24114](https://arxiv.org/abs/2604.24114)
- Nemotron-Mini-4B Hindi Adaptation — [arXiv 2410.14815](https://arxiv.org/abs/2410.14815)
- IndicLLMSuite (AI4Bharat) — [arXiv 2403.06350](https://arxiv.org/abs/2403.06350)
- Parity-Aware BPE — [arXiv 2508.04796](https://arxiv.org/abs/2508.04796)
