# Gemma CPT — Status-Check Log

Chronological record of each training-status query during the Gemma v2 CPT run
(`adapters/gemma4-e4b-upsc-v2-cpt`, tmux `gemma_pipe`, bs=1 × accum 64,
1853-step WSD epoch, ~170 s/step on the EC2 L40S). Pulse probe = n=50 holdout
MCQ, terse parser (absolute values noisy; trend trustworthy). Each row is what
that query actually returned.

**Run-clock note:** "elapsed" is HF Trainer's internal timer. It reset on the
EC2 shutdown/resume (resumed from checkpoint-1000), so post-resume elapsed
restarts from ~0 even though training continued.

| # | Step / 1853 | % | Run-clock elapsed | Loss (latest logged) | Pulse Task A / MMLU / Hindi | GPU |
|---|---|---|---|---|---|---|
| — | preflight | — | — | — | gate CLEAN (2120 eval rows, grams [25,50]) | — |
| 1 | 156 | 8% | — | 1.909 (step150) | not yet fired | 100% / 30.2 GB |
| 2 | 197 | 11% | ~9.3 h | 1.909 | **308: 0.78 / 0.61 / 0.40** | 100% |
| 3 | 243 | 13% | — | ~1.85 | 308 (unchanged) | 100% |
| 4 | 415 | 22% | 19.7 h | 1.815 | 308 | 100% |
| 5 | 465 | 25% | 22.1 h | 1.841 | 308 | 100% |
| 6 | 654 | 35% | 31.0 h | 1.781 | **616: 0.76 / 0.63 / 0.46** | 100% |
| 7 | 697 | 38% | 33.1 h | 1.781 | 616 | 100% |
| 8 | 762 | 41% | 36.1 h | 1.785 | 616 | 100% |
| 9 | 866 | 47% | 41.1 h | 1.816 | 616 (924 not yet) | 100% |
| — | *(full loss table pulled: 50→2.379, 200→1.855, 400→1.815, 600→1.807, 850→1.816)* | | | | | |
| — | **EC2 shutdown by external party → resumed from checkpoint-1000 (new IP)** | | | | step-924 pulse had fired: **0.76 / 0.60 / 0.50** | |
| 10 | 1002 | 54% | resume (~0) | — (dataloader fast-forward) | 924 | 100% / 30.1 GB |
| 11 | 1107 | 60% | 5.1 h | 1.76 | 924 | 100% |
| 12 | 1151 | 62% | — | 1.742 | 924 | 100% |
| 13 | 1195 | 64% | 9.3 h | 1.742 | 924 | 100% |
| 14 | 1221 | 66% | 10.5 h | 1.763 | 924 (1232 imminent) | 100% |
| 15 | 1234 | 67% | — | 1.763 | **1232: 0.74 / 0.61 / 0.50** | 100% |
| 16 | 1498 | 81% | 23.7 h | 1.734 | 1232 (entering decay tail) | 100% |
| 17 | 1539 | 83% | — | 1.743 | **1540: 0.76 / 0.60 / 0.40** (first decay pulse) | 100% |
| 18 | 1594 | 86% | 28.2 h | 1.723 | 1540 | 100% |
| 19 | 1669 | 90% | 31.8 h | 1.70 | 1540 | 100% |
| 20 | 1804 | 97% | 38.2 h | 1.70 | 1540 | 100% |
| 21 | **1853 (DONE)** | 100% | 40.5 h | ~1.70 | **1848: 0.76 / 0.60 / 0.40** (final) | — |

## Pulse summary 

| Pulse @ step | Task A EN | MMLU | Hindi | read |
|---|---|---|---|---|
| 308 | 0.78 | 0.61 | 0.40 | first gate — learning, no regression |
| 616 | 0.76 | 0.63 | 0.46 | Task A plateau; Hindi ↑ |
| 924 | 0.76 | 0.60 | 0.50 | stable; Hindi still ↑ |
| 1232 | 0.74 | 0.61 | 0.50 | Task A within-noise wiggle; MMLU/Hindi flat |
| 1540 | 0.76 | 0.60 | 0.40 | first decay-tail pulse — Task A back to 0.76; Hindi noise-dip |
| 1848 | 0.76 | 0.60 | 0.40 | FINAL pulse — Task A locked ~0.76, MMLU flat, Hindi in noise band |

**CPT COMPLETE @ step 1853 (~40.5 GPU-h post-resume; ~50+ h wall incl. the
shutdown).** Task A held 0.74–0.78 across all 6 pulses (above the 0.70 honest
target / at old 0.75 gate); MMLU flat ~0.61 (no forgetting); Hindi 0.40–0.50
noise band (no collapse). Final loss ~1.70 after the decay tail. → SFT next.
All pulse numbers are the lightweight n=50 probe — real magnitudes pending the
full post-SFT eval.

- **Task A EN**: plateaued ~0.76 (vs v1 0.645, honest target 0.70, old gate 0.75) — above expectation, pending real eval
- **MMLU**: flat ~0.61 — no catastrophic forgetting (ignore the 0.10 step-0 baseline = parser artifact)
- **Hindi**: 0.40→0.46→0.50, rising — no English-CPT bleed into Hindi

## Loss trajectory
Steep early drop 2.38→1.86 (steps 50–200, the 1e-4-LR adapter ramp), stable
plateau ~1.80 (steps 200–900), easing to ~1.74–1.76 by step ~1150. WSD decay
tail (steps ~1500→1853) expected to add a final sharp step-down.

**Caveat on every pulse number:** lightweight n=50 probe with a terse answer
parser; absolute values are noisy and tend to read high. Authoritative metrics
come from the full v1 inference+scoring pipeline post-SFT (2000 items, robust
extraction, BH-FDR). See [`v2-target-metrics.md`](v2-target-metrics.md) for the
honest recalibrated gates.
