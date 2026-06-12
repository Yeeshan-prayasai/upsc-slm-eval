"""Continued-pretraining (CPT) trainer.

Wraps HuggingFace `Trainer` with a trivial stack-collator (every row is
a packed `{input_ids: list[int]}` exactly `seq_len` tokens long — no
padding ever occurs, and `labels = input_ids` unmasked). NOT
`DataCollatorForLanguageModeling`: that collator masks
`labels == pad_token_id`, and on tokenizers where pad falls back to eos
(Qwen) it silently masked every EOS document separator out of the loss.

CPT is the standard recipe used by Llama 3, Gemma, and Qwen tech
reports: concatenate documents with (BOS+)EOS, pack to fixed-length,
train with causal-LM loss. No attention-mask reset at document
boundaries — naive concatenation (GPT-3/Pythia style); every position
contributes to the loss.

`max_steps` may be given explicitly, or derived from the corpus as
exactly one epoch over the packed parquet (the mix-weighting stage in
tokenize_pack already encodes per-source repetition, so one pass over
the parquet = the intended token exposures).

The trainer composes:
- LoRA via `training.trainers.base.build_model_with_lora`
- WSD scheduler via `training.trainers.schedulers.build_wsd_scheduler`
- HuggingFace `Trainer` + `TrainingArguments`
- Optional pulse-eval callback (passed by the orchestration script)

The output adapter is saved to `<output_dir>/final/`, with intermediate
`checkpoint-N/` directories for resume + ablation cell 6 (50 %-ckpt
branching).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

from .base import (
    LoRAConfig,
    OptimConfig,
    RuntimeConfig,
    build_model_with_lora,
    config_summary,
    find_latest_checkpoint,
)
from .schedulers import WSDConfig, build_wsd_scheduler


@dataclass(frozen=True)
class CPTConfig:
    """Training-loop configuration for one CPT run."""
    base_model: str
    corpus_parquet: Path                  # pre-tokenized output of tokenize_pack
    output_dir: Path
    # Optimizer steps. None → derived as one epoch over the packed
    # corpus (mix weighting already encodes per-source repetition).
    max_steps: int | None = None
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    wsd_warmup_frac: float = 0.01
    wsd_stable_frac: float = 0.80
    wsd_min_lr_ratio: float = 0.1
    save_steps: int = 1000
    save_total_limit: int = 5
    logging_steps: int = 50
    report_to: str = "none"


def resolve_max_steps(cfg: CPTConfig, n_sequences: int) -> int:
    """Explicit `max_steps` wins; otherwise exactly one epoch over the
    packed corpus: ceil(n_sequences / sequences-per-optimizer-step)."""
    if cfg.max_steps is not None:
        return cfg.max_steps
    seqs_per_step = (cfg.runtime.per_device_train_batch_size
                     * cfg.runtime.gradient_accumulation_steps)
    steps = -(-n_sequences // seqs_per_step)   # ceil
    print(f"[cpt] max_steps derived from corpus: {n_sequences:,} sequences "
          f"/ {seqs_per_step} per step = {steps:,} steps (one epoch)")
    return steps


def load_packed_corpus(parquet_path: Path):
    """Load the pre-tokenized Parquet as a HF Dataset.
    Schema: `{input_ids: list[int32]}` per row; torch's embedding
    lookup accepts int32 indices directly."""
    from datasets import load_dataset

    ds = load_dataset("parquet", data_files=str(parquet_path), split="train")
    return ds.with_format("torch", columns=["input_ids"])


def packed_collator(features: list[dict]) -> dict:
    """Collator for pre-packed fixed-length rows: stack input_ids,
    labels = input_ids (unmasked — every position trains, including
    the EOS document separators), full attention.

    Replaces DataCollatorForLanguageModeling, which sets
    `labels[labels == pad_token_id] = -100`: with Qwen's pad==eos
    fallback that silently masked every document boundary out of the
    loss. Packed rows are exactly seq_len long, so padding never
    occurs and no pad-token dependence is needed at all."""
    input_ids = torch.stack([f["input_ids"] for f in features])
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids.clone(),
    }


def _build_training_arguments(cfg: CPTConfig, max_steps: int) -> TrainingArguments:
    """Map our typed config into HF's monolithic TrainingArguments."""
    return TrainingArguments(
        output_dir=str(cfg.output_dir),
        max_steps=max_steps,
        per_device_train_batch_size=cfg.runtime.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.runtime.gradient_accumulation_steps,
        learning_rate=cfg.optim.learning_rate,
        weight_decay=cfg.optim.weight_decay,
        adam_beta1=cfg.optim.adam_beta1,
        adam_beta2=cfg.optim.adam_beta2,
        adam_epsilon=cfg.optim.adam_epsilon,
        max_grad_norm=cfg.optim.max_grad_norm,
        # We supply our own WSD scheduler via create_scheduler override
        # below; HF's built-in type is set to `constant` purely as a
        # placeholder that the override replaces.
        lr_scheduler_type="constant",
        warmup_ratio=0.0,
        bf16=cfg.runtime.bf16,
        gradient_checkpointing=False,    # already enabled inside prepare_model_for_kbit_training
        optim=cfg.runtime.optim,
        dataloader_num_workers=cfg.runtime.dataloader_num_workers,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        logging_steps=cfg.logging_steps,
        seed=cfg.runtime.seed,
        data_seed=cfg.runtime.seed,
        report_to=cfg.report_to,
        log_level="info",
        disable_tqdm=False,
    )


class MidpointCheckpointCallback(TrainerCallback):
    """Copy the exact-midpoint checkpoint aside as `checkpoint-midpoint/`
    (rotation-exempt). Ablation cell 6 branches SFT from the 50%-CPT
    point; HF's `save_total_limit` keeps only the most RECENT N
    checkpoints, so without this the midpoint is rotated away long
    before the run ends (and `save_steps` may never land on the exact
    midpoint step at all)."""

    def on_step_end(self, args, state, control, **kw):
        if state.max_steps and state.global_step == max(1, state.max_steps // 2):
            control.should_save = True
        return control

    def on_save(self, args, state, control, **kw):
        if not state.max_steps:
            return control
        mid = max(1, state.max_steps // 2)
        if state.global_step == mid:
            import shutil
            src = Path(args.output_dir) / f"checkpoint-{mid}"
            dst = Path(args.output_dir) / "checkpoint-midpoint"
            if src.exists() and not dst.exists():
                shutil.copytree(src, dst)
                print(f"[cpt] midpoint checkpoint copied aside → {dst} "
                      f"(rotation-exempt, for ablation cell 6)")
        return control


class CPTTrainer(Trainer):
    """`transformers.Trainer` subclass that:

    1. Replaces HF's default LR scheduler with our WSD scheduler
       (Wen 2024 / Ibrahim 2024).
    2. Logs the resolved config dict once at the start of `train()`
       so the run record contains every hyperparameter.
    """

    def __init__(self, *, _wsd_cfg: WSDConfig, _summary_dict: dict, **kw):
        super().__init__(**kw)
        self._wsd_cfg = _wsd_cfg
        self._summary_dict = _summary_dict

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        opt = optimizer if optimizer is not None else self.optimizer
        self.lr_scheduler = build_wsd_scheduler(opt, self._wsd_cfg)
        return self.lr_scheduler

    def train(self, *a, **kw):
        print(f"[cpt] config summary:")
        for k, v in sorted(self._summary_dict.items()):
            print(f"    {k} = {v}")
        return super().train(*a, **kw)


def build_cpt_trainer(
    cfg: CPTConfig,
    extra_callbacks: list[TrainerCallback] | None = None,
) -> tuple[CPTTrainer, PreTrainedModel, PreTrainedTokenizerBase]:
    """Construct the trainer + model + tokenizer ready to call `.train()`.

    `extra_callbacks` is for the orchestration script to inject the
    pulse-eval callback. The trainer itself doesn't know about pulses.
    """
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[cpt] base model: {cfg.base_model}")
    model, tokenizer = build_model_with_lora(cfg.base_model, cfg.lora)

    print(f"[cpt] corpus: {cfg.corpus_parquet}")
    train_ds = load_packed_corpus(cfg.corpus_parquet)
    # The parquet is packed to a fixed length by tokenize_pack --seq-len;
    # if runtime.max_seq_length disagrees (e.g. someone selects the
    # seq-2048 OOM-fallback runtime without re-tokenizing) the packed
    # collator would feed 4096-token rows under a 2048 assumption. Assert.
    row_len = len(train_ds[0]["input_ids"])
    if row_len != cfg.runtime.max_seq_length:
        raise RuntimeError(
            f"Packed parquet row length {row_len} != runtime.max_seq_length "
            f"{cfg.runtime.max_seq_length}. Re-tokenize with "
            f"`tokenize_pack --seq-len {cfg.runtime.max_seq_length}` or use "
            f"the matching runtime config."
        )
    max_steps = resolve_max_steps(cfg, len(train_ds))
    print(f"[cpt] sequences: {len(train_ds):,}  "
          f"(seq_len={cfg.runtime.max_seq_length}, effective batch="
          f"{cfg.runtime.per_device_train_batch_size * cfg.runtime.gradient_accumulation_steps} "
          f"× {cfg.runtime.max_seq_length} = "
          f"{cfg.runtime.per_device_train_batch_size * cfg.runtime.gradient_accumulation_steps * cfg.runtime.max_seq_length:,} tokens/step)")

    training_args = _build_training_arguments(cfg, max_steps)
    wsd_cfg = WSDConfig(
        total_steps=max_steps,
        warmup_frac=cfg.wsd_warmup_frac,
        stable_frac=cfg.wsd_stable_frac,
        min_lr_ratio=cfg.wsd_min_lr_ratio,
    )

    summary = {
        **config_summary(cfg.lora, cfg.runtime, cfg.optim),
        "base_model": cfg.base_model,
        "max_steps": max_steps,
        "corpus_parquet": str(cfg.corpus_parquet),
        "output_dir": str(cfg.output_dir),
        "wsd": {
            "warmup_steps": wsd_cfg.warmup_steps,
            "stable_end_step": wsd_cfg.stable_end_step,
            "min_lr_ratio": wsd_cfg.min_lr_ratio,
        },
    }

    trainer = CPTTrainer(
        _wsd_cfg=wsd_cfg,
        _summary_dict=summary,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=packed_collator,
        processing_class=tokenizer,
        callbacks=[MidpointCheckpointCallback()] + (extra_callbacks or []),
    )
    return trainer, model, tokenizer


def maybe_resume(cfg: CPTConfig) -> str | None:
    """Return the path of the latest checkpoint dir, or None for fresh run."""
    cp = find_latest_checkpoint(cfg.output_dir)
    if cp is None:
        return None
    print(f"[cpt] resuming from {cp}")
    return str(cp)
