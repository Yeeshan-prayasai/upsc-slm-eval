"""Trainer implementations.

- `base` — shared QLoRA + LoRA wrappers + dataclass configs.
- `schedulers` — Warmup-Stable-Decay LR schedule (Wen et al. 2024).
- `cpt` — Continued-pretraining trainer (HF Trainer + LM collator).
- `sft` — Supervised fine-tuning trainer with length-penalty `compute_loss`.
"""
