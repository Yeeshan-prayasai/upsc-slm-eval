"""Supervised fine-tuning (SFT) trainer.

Uses trl `SFTTrainer` directly on **conversational prompt-completion**
rows (`{"prompt": [{role,user}], "completion": [{role,assistant}]}`),
which gives three properties the previous raw-`text` setup lacked:

1. **Chat-template consistency** — trl renders rows through the
   tokenizer's own chat template, the same framing
   `scripts/runners.py` applies at inference. Training and eval see
   identical special-token contexts (`<start_of_turn>model` /
   `<|im_start|>assistant`), and the turn-end token is trained as the
   completion terminator.
2. **Completion-only loss** — prompt tokens are masked; capacity goes
   to the answer distribution, not to regenerating instruction
   boilerplate.
3. **Correct gradient-accumulation normalization** — no compute_loss
   override; trl's own loss path handles `num_items_in_batch` (the
   previous override bypassed it, inflating backward loss ~64× and
   hard-clipping every step).

Length control: handled in the DATA, not the loss. Rows with a known
`target_word_count` carry an "Answer in approximately N words."
instruction inside the prompt (see `build_sft_corpus`), and plain CE
learns the association — the soft-penalty term it replaces was
non-differentiable as formulated (computed from label lengths — a
constant w.r.t. parameters) and trained nothing.

SFT continues from the CPT-trained adapter via
`base.attach_lora_from_checkpoint`. Both phases use the same LoRA
config (rank 64, alpha 16, RSLoRA, all decoder layers × 7 projections)
so the SFT phase keeps training the same delta matrices.

`max_steps` defaults to **2 epochs** over the train split (methodology
§4.4), derived at build time from the dataset size and batch geometry.
The previous hardcoded 16,000 (inherited from v1's effective-batch-8
runs) was ~34 epochs at v2's effective batch 64.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .base import (
    LoRAConfig,
    OptimConfig,
    RuntimeConfig,
    build_model_with_lora,
    config_summary,
    find_latest_checkpoint,
)


@dataclass(frozen=True)
class SFTConfigV2:
    """SFT-training-loop configuration."""
    base_model: str
    train_jsonl: Path
    valid_jsonl: Path
    output_dir: Path
    resume_lora_from: Path | None        # path to CPT adapter; None = fresh LoRA
    # Optimizer steps. None → derived as `epochs` over the train split.
    max_steps: int | None = None
    epochs: float = 2.0                  # methodology §4.4
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    # SFT optimizer differs from CPT: beta2 back to default, lower
    # weight_decay (Ibrahim 2024). LR 2e-4 ≈ the LoRA-appropriate rate.
    optim: OptimConfig = field(default_factory=lambda: OptimConfig(
        learning_rate=2e-4, adam_beta2=0.999, weight_decay=0.01,
    ))
    warmup_ratio: float = 0.03           # 3% of max_steps (cosine SFT schedule)
    min_lr_rate: float = 0.1             # cosine floor = 0.1× peak (methodology §4.3)
    save_steps: int = 100
    save_total_limit: int = 3
    logging_steps: int = 50
    eval_steps: int = 100
    report_to: str = "none"


def resolve_max_steps(cfg: SFTConfigV2, n_train: int) -> int:
    """Explicit `max_steps` wins; otherwise `cfg.epochs` epochs over the
    train split at the configured batch geometry."""
    if cfg.max_steps is not None:
        return cfg.max_steps
    examples_per_step = (cfg.runtime.per_device_train_batch_size
                         * cfg.runtime.gradient_accumulation_steps)
    steps = -(-int(n_train * cfg.epochs) // examples_per_step)   # ceil
    print(f"[sft] max_steps derived: {n_train:,} rows × {cfg.epochs} epochs "
          f"/ {examples_per_step} per step = {steps:,} steps")
    return steps


def _build_sft_args(cfg: SFTConfigV2, max_steps: int):
    """Map our typed config into trl 1.5's SFTConfig."""
    from trl import SFTConfig    # lazy: trl is GPU-only dep
    return SFTConfig(
        output_dir=str(cfg.output_dir),
        max_steps=max_steps,
        per_device_train_batch_size=cfg.runtime.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.runtime.gradient_accumulation_steps,
        max_length=cfg.runtime.max_seq_length,
        learning_rate=cfg.optim.learning_rate,
        weight_decay=cfg.optim.weight_decay,
        adam_beta1=cfg.optim.adam_beta1,
        adam_beta2=cfg.optim.adam_beta2,
        adam_epsilon=cfg.optim.adam_epsilon,
        max_grad_norm=cfg.optim.max_grad_norm,
        # HF's plain `cosine` decays to 0; methodology §4.3 specifies a
        # 0.1× floor.
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": cfg.min_lr_rate},
        warmup_ratio=cfg.warmup_ratio,
        bf16=cfg.runtime.bf16,
        optim=cfg.runtime.optim,
        gradient_checkpointing=False,    # already enabled via peft prepare
        # Prompt-completion rows → trl masks prompt tokens automatically;
        # set explicitly so a data-format regression fails loud instead
        # of silently flipping to full-sequence loss.
        completion_only_loss=True,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        # Keep the best-eval-loss checkpoint; with rotation alone, only
        # the most-overfit tail checkpoints survived.
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        per_device_eval_batch_size=1,
        seed=cfg.runtime.seed,
        data_seed=cfg.runtime.seed,
        report_to=cfg.report_to,
        log_level="info",
        disable_tqdm=False,
        packing=False,
    )


def build_sft_trainer(cfg: SFTConfigV2, extra_callbacks: list | None = None):
    """Construct the SFT trainer + model + tokenizer ready to `.train()`."""
    from trl import SFTTrainer   # lazy: GPU-only dep

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[sft] base model: {cfg.base_model}  "
          f"resume_lora_from: {cfg.resume_lora_from}")
    model, tokenizer = build_model_with_lora(
        cfg.base_model,
        lora_cfg=cfg.lora,
        resume_adapter=cfg.resume_lora_from,
    )

    from datasets import load_dataset
    train_ds = load_dataset("json", data_files=str(cfg.train_jsonl), split="train")
    valid_ds = load_dataset("json", data_files=str(cfg.valid_jsonl), split="train")
    print(f"[sft] train={len(train_ds):,}  valid={len(valid_ds):,}")

    sample = train_ds[0]
    if "prompt" not in sample or "completion" not in sample:
        raise ValueError(
            "SFT rows must be conversational prompt-completion "
            "({'prompt': [...], 'completion': [...]}); got columns "
            f"{list(sample)}. Rebuild with `make build-sft-corpus`."
        )

    max_steps = resolve_max_steps(cfg, len(train_ds))
    sft_args = _build_sft_args(cfg, max_steps)

    summary = {
        **config_summary(cfg.lora, cfg.runtime, cfg.optim),
        "base_model": cfg.base_model,
        "resume_lora_from": str(cfg.resume_lora_from) if cfg.resume_lora_from else None,
        "max_steps": max_steps,
        "epochs": cfg.epochs,
        "warmup_ratio": cfg.warmup_ratio,
        "min_lr_rate": cfg.min_lr_rate,
        "output_dir": str(cfg.output_dir),
    }
    print(f"[sft] config summary:")
    for k, v in sorted(summary.items()):
        print(f"    {k} = {v}")

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        processing_class=tokenizer,
        callbacks=extra_callbacks or [],
    )
    return trainer, model, tokenizer


def maybe_resume(cfg: SFTConfigV2) -> str | None:
    cp = find_latest_checkpoint(cfg.output_dir)
    if cp is None:
        return None
    print(f"[sft] resuming from {cp}")
    return str(cp)
