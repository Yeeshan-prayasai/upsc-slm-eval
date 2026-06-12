# v2 Hindi strategy

**Owner:** Yeeshan
**Status:** Decision recorded; v2 trains EN-only.
**Companion:** [v2-methodology.md](v2-methodology.md), [experiment-report.md §6.2 / §6.3](experiment-report.md)

## v1 Hindi performance (the data)

Pre-FT base-model Hindi MCQ accuracy (50-Q probe from `data/hindi_probe.parquet`,
random = 0.25):

| Base model | Hindi acc | Binomial p (vs random) | Verdict |
|---|---:|---:|---|
| `gemma-4-E4B-it` | **0.520** | 0.0001 | Passes α=0.05 gate |
| `Qwen3.5-4B` | **0.280** | 0.363 | **Fails** — Hindi indistinguishable from chance |

Post-FT Task-A Hindi-stratum accuracy (347 Hindi items, [experiment-report.md §6.3](experiment-report.md)):

| Condition | Acc (en) | Acc (hi) | EN−HI gap |
|---|---:|---:|---:|
| Gemma-FT (v1) | 0.673 | **0.636** | −3.7 pp |
| Qwen-FT (v1) | 0.595 | **0.426** | **−16.9 pp** |
| Gemini-3-Flash | ~0.85 | ~0.85+ | ≈ 0 |

**Pre-registered prediction refuted:** v1 predicted Qwen (explicit-Indic
pretraining enumeration) would beat Gemma (pool-only Indic) on Hindi.
Empirically Gemma is the stronger Hindi base. Pretraining-pool
inclusion in the Gemma 140-language tier appears to deliver more
usable Hindi than Qwen's 201-language *list-membership* alone.

## v2 decision: train EN-only, gate Hindi as no-regression

### What v2 does

1. **CPT corpus filtered to English** (`build_cpt_corpus` consumes
   only the EN slices of the §4.5 source mix; no Hindi NCERT, no
   Hindi PIB, no Hindi reference texts).
2. **SFT corpus filtered to English** ([build_sft_corpus.py](training/data/build_sft_corpus.py)
   drops the 9,108 Hindi rows from v1's `ft_corpus.parquet`).
3. **Hindi no-regression pulse during training** ([pulse.py](training/eval/pulse.py)):
   - Every 1000 steps, score on 50 v1 Hindi Task-A items
   - **Hard-stop** if accuracy drops more than 5 pp from the v1 baseline
     (Gemma 0.636 / Qwen 0.426)
4. **Evaluation reports Hindi as a separate stratum**, not folded into
   the bilingual aggregate (matches v1 reporting convention).

### Why EN-only (the reasoning)

1. **Marginal-return calculus.** v1's English-only SFT bumped Gemma
   Hindi *up* from 0.520 → 0.636 (+11.6 pp) — i.e. EN training already
   improves Hindi via cross-lingual transfer. We don't *need* Hindi
   training to maintain Hindi quality at Gemma's level; we need it to
   reach Gemini-Flash's level (~0.85), which is a larger architectural
   ask than v2's CPT→SFT scope can deliver.
2. **Token budget pressure.** v2's L40S compute target is ~4 B CPT
   tokens. Adding a parallel Hindi track at proportional volume would
   either halve the EN factual signal or double the wall-clock — neither
   is acceptable for v2's primary goal (close the v1 word-count + Article
   citation gap on Mains).
3. **Tokenizer mismatch risk.** Gemma and Qwen tokenizers split Devanagari
   differently. A small Hindi corpus risks under-training the Hindi BPE
   subwords, which can *degrade* Hindi rather than improve it — exactly
   the scenario the no-regression pulse is designed to catch.
4. **v1 Qwen Hindi (0.426) is below the floor at which v2 SFT can help.**
   Pre-FT Hindi was at chance (0.280); SFT pushed it to 0.426 but Qwen
   never learned Hindi *factual recall* during pretraining. No amount
   of fine-tuning on a small Hindi corpus changes that.

## v3+ scope (explicit defer)

Items below are **not** in v2. They go on the v3 backlog only after v2
ships and we measure the headline EN deltas:

- **Continued pretraining on a Hindi corpus** (~0.3-0.5 B tokens —
  Indian-language Wikipedia + AI4Bharat datasets + Hindi-language
  current affairs from Yojana / DD News transcripts). Would push
  Qwen's Hindi from 0.426 toward Gemma's 0.636 baseline.
- **Hindi-specific tokenizer extension** if subword fragmentation
  remains a measurable issue on Mains-length Hindi outputs.
- **Bilingual SFT** with paired EN-HI Mains answers — only after the
  CPT Hindi gap is closed (otherwise the SFT signal goes into the
  wrong subwords).
- **Hindi-tier Task-B (Mains generation) eval expansion** — currently
  v2 only measures Hindi on Task A (MCQ). Task B/C Hindi evaluation
  needs Hindi UPSC-grade rubrics, which don't exist in v1 data and
  would need annotator-time investment.

## Success criteria for v2 (Hindi)

The v2 Hindi report card is **no-regression only**:

| Metric | Gemma-v2 target | Qwen-v2 target |
|---|---|---|
| Task-A Hindi accuracy | ≥ 0.586 (v1 − 5 pp) | ≥ 0.376 (v1 − 5 pp) |
| Pulse hard-stop trips | 0 across the run | 0 across the run |

If either model's Hindi accuracy drops below the no-regression floor
mid-training, the trainer hard-stops via [pulse.py:_run_hindi_pulse](training/eval/pulse.py),
the checkpoint at the last green pulse is kept, and we treat the run
as a partial success — EN improvements only.

A **positive Hindi delta** vs v1 (Gemma > 0.636 or Qwen > 0.426) would
be a bonus and gets reported, but it's not a v2 acceptance criterion.
