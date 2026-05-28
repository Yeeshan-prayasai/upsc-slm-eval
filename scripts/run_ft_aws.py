"""Stage 3 (AWS path) — fine-tune via PyTorch + peft + bitsandbytes (QLoRA).

Production-grade equivalent of scripts/run_ft.py for NVIDIA hardware (e.g.
an EC2 g6e.xlarge with an L40S 48 GB). The recipe is symbolically synced
with configs/lora.yaml; only the framework differs.

Key design choices and why:
- 4-bit NF4 + double-quant (Dettmers et al., QLoRA, NeurIPS 2023) — matched
  to the MLX 4-bit pipeline.
- `prepare_model_for_kbit_training(model)` — canonical QLoRA prep; resolves
  several known gradient-checkpointing-vs-quantization interactions.
- target_modules are scoped via *regex on full module path* rather than bare
  module names, because both Qwen3.5 and Gemma-4 are multimodal models with
  identically-named projection layers in their vision/audio towers. Without
  scoping, peft would LoRA-tune both towers — wasted parameters + unstable
  gradients.
- num_hidden_layers is read via a multi-attempt walk: top-level config,
  then `text_config`, then `model.config` post-load — multimodal configs
  in transformers 5.x nest the text-tower attrs differently across models.
- We assert `model.num_parameters(only_trainable=True) > 0` after
  `get_peft_model` so a target_modules mismatch surfaces immediately
  rather than silently producing a no-op LoRA.
- Auto-resume passes the actual latest checkpoint path (trl 1.x), not a
  bool (trl 0.x).
- Mid-training memory peak fits in 24 GiB VRAM at rank=16, num_layers=16,
  seq=2048, batch=1 × grad_accum=8.

Inputs (must be present on disk BEFORE running):
    data/ft_split/train.jsonl    chat-format pairs from `make materialize-ft-split`
    data/ft_split/valid.jsonl    same
    HuggingFace login active     `hf auth login` first; needed for gated Gemma weights

Outputs:
    adapters/<adapter-out>/                  PEFT adapter (HF format, ~100 MB)
    adapters/<adapter-out>/checkpoint-N/     intermediate checkpoints (rotated, max 4)
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data" / "ft_split"
SEED = 20260514

# Mirrors configs/lora.yaml — the YAML is the source of truth on the MLX side;
# these constants are the symbolic translation for PyTorch + peft.
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
# Bare module names to LoRA-tune. We combine these with a per-base regex below
# so we hit the text decoder only, not the multimodal towers.
LORA_PROJ_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj")
NUM_LORA_LAYERS = 16            # last N transformer blocks of the *text* decoder
MAX_SEQ_LENGTH = 2048
PER_DEVICE_TRAIN_BATCH = 1
GRAD_ACCUM_STEPS = 8            # effective batch 8 (QLoRA standard)
LEARNING_RATE = 2.0e-4
MAX_STEPS = 16000               # ≈ 3 epochs at effective batch 8 over ~42 K pairs
EVAL_STEPS = 500
SAVE_STEPS = 1000
LOGGING_STEPS = 50
WARMUP_STEPS = 100


def _get_num_text_layers(base: str) -> int:
    """Robustly extract the number of text-decoder transformer layers.

    Multimodal configs in transformers 5.x nest text attrs under `text_config`
    for some models (Qwen3.5_5), at the top level for others (older). Try
    both. Fail loudly if neither works — never silently fall back to a wrong
    value, since that would produce a malformed LoRA.
    """
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(base, trust_remote_code=True)
    for path in ("num_hidden_layers", "n_layer", "num_layers"):
        v = getattr(cfg, path, None)
        if isinstance(v, int) and 0 < v < 1024:
            return v
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None:
        for path in ("num_hidden_layers", "n_layer", "num_layers"):
            v = getattr(text_cfg, path, None)
            if isinstance(v, int) and 0 < v < 1024:
                return v
    raise RuntimeError(
        f"Could not determine num_hidden_layers for {base}. "
        f"Top-level config type: {type(cfg).__name__}, "
        f"text_config type: {type(text_cfg).__name__ if text_cfg else None}. "
        f"Available attrs: {sorted(vars(cfg).keys())[:10]}..."
    )


def _build_target_modules_regex(base: str, total_layers: int,
                                lora_layers: int) -> str:
    """Regex matching exactly the projection modules we want LoRA on,
    scoped to the text decoder's last N layers only.

    The text decoder's qualified path varies by model:
      Qwen3.5-4B (Qwen3_5ForConditionalGeneration):
          `model.layers.N.{self_attn,mlp}.X_proj`
      Gemma-4-E4B-it (Gemma4ForConditionalGeneration):
          `model.language_model.layers.N.{self_attn,mlp}.X_proj`

    Vision and audio towers live at `model.vision_tower....layers.N.*` and
    `model.audio_tower.layers.N.*` respectively; we must NOT match those.

    We anchor on `^(?:.*\.)?model\.(?:language_model\.)?layers\.<N>\.`
    — the `language_model.` lives between `model.` and `layers.`,
    making it OPTIONAL so the same regex covers both bases.
    Vision/audio paths contain `vision_tower.encoder.` or `audio_tower.`
    between `model.` and `layers.`, so they're correctly excluded.
    """
    start = total_layers - lora_layers
    layer_idx_re = "|".join(str(i) for i in range(start, total_layers))
    proj_re = "|".join(LORA_PROJ_MODULES)
    return (
        r"^(?:.*\.)?model\.(?:language_model\.)?layers\."
        rf"(?:{layer_idx_re})\."
        rf".*\.(?:{proj_re})$"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True,
                    help="HF repo id (e.g. Qwen/Qwen3.5-4B or google/gemma-4-E4B-it)")
    ap.add_argument("--adapter-out", type=Path, required=True)
    ap.add_argument("--train-jsonl", type=Path, default=DATA_DIR / "train.jsonl")
    ap.add_argument("--valid-jsonl", type=Path, default=DATA_DIR / "valid.jsonl")
    args = ap.parse_args()

    if not args.train_jsonl.exists() or not args.valid_jsonl.exists():
        print(f"[FAIL] {args.train_jsonl} or {args.valid_jsonl} missing — "
              f"run `make materialize-ft-split` on the source M5 first, then "
              f"SCP the resulting data/ft_split/*.jsonl to this box.")
        return 1

    # Heavy imports deferred — keeps stdlib-only argparse / path-check fast and
    # the failure modes more legible.
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig

    # --- Resolve layer count / target_modules BEFORE loading model weights so
    #     a config-introspection error surfaces in seconds, not minutes.
    total_layers = _get_num_text_layers(args.base)
    if NUM_LORA_LAYERS > total_layers:
        raise RuntimeError(
            f"NUM_LORA_LAYERS={NUM_LORA_LAYERS} exceeds the model's "
            f"text-decoder depth ({total_layers}); reduce the constant."
        )
    target_re = _build_target_modules_regex(args.base, total_layers, NUM_LORA_LAYERS)
    print(f"[arch] {args.base}: {total_layers} text layers; "
          f"LoRA on last {NUM_LORA_LAYERS} (layers "
          f"{total_layers - NUM_LORA_LAYERS}..{total_layers - 1})")
    print(f"[arch] target_modules regex: {target_re}")

    # --- 4-bit NF4 + double-quant (QLoRA default, Dettmers et al. NeurIPS 2023)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"[load] {args.base} (4-bit NF4, compute=bf16)")
    # `dtype=` is the transformers 5.x replacement for the deprecated
    # `torch_dtype=`. `attn_implementation` is left unset so HF picks the
    # best available kernel (sdpa where available, eager fallback otherwise).
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        quantization_config=bnb,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # Canonical QLoRA prep. Among other things this:
    #   - casts the layer-norms to fp32 for numerical stability,
    #   - registers a forward hook to upcast embeddings for grad-flow,
    #   - calls model.gradient_checkpointing_enable() with use_reentrant=False.
    # We do NOT call model.gradient_checkpointing_enable() separately because
    # peft handles it inside prepare_model_for_kbit_training.
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # --- Tokenizer setup
    tokenizer = AutoTokenizer.from_pretrained(
        args.base, padding_side="right", trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Propagate to model config — required for some attention impls / loss masking.
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # --- Build LoRA config + apply
    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=target_re,             # regex; peft 0.7+ supports this
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # Production-grade assertion: a target_modules regex that matches zero
    # modules produces a silent no-op LoRA. Always verify trainable params > 0
    # AND that the count is in the sane range for our recipe.
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * n_trainable / max(1, n_total)
    print(f"[lora] trainable: {n_trainable:,} / {n_total:,} ({pct:.2f}%)")
    if n_trainable < 1_000_000:
        raise RuntimeError(
            f"LoRA wired up only {n_trainable:,} trainable params — likely a "
            f"target_modules regex mismatch. Expected ~10-50M for r=16 over "
            f"{NUM_LORA_LAYERS} layers × 7 projections."
        )
    if pct > 5.0:
        raise RuntimeError(
            f"LoRA trainable share is {pct:.1f}%, far above the QLoRA effective "
            f"band (0.5-2%). Regex may be matching unintended modules."
        )

    # --- Datasets
    train_ds = load_dataset("json", data_files=str(args.train_jsonl), split="train")
    eval_ds = load_dataset("json", data_files=str(args.valid_jsonl), split="train")
    print(f"[data] train={len(train_ds):,}  valid={len(eval_ds):,}")

    args.adapter_out.mkdir(parents=True, exist_ok=True)

    # --- Trainer config. trl 1.5 SFTConfig keys (all verified against
    #     dataclasses.fields(SFTConfig) at install time):
    #   - max_length          → trl 1.x renamed `max_seq_length` to this
    #   - eval_strategy       → kept (trl deprecated `evaluation_strategy` in 0.14)
    #   - dataset_text_field  → omit; trl 1.5 auto-detects 'messages' chat format
    #   - packing             → kept
    sft_cfg = SFTConfig(
        output_dir=str(args.adapter_out),
        max_steps=MAX_STEPS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        max_length=MAX_SEQ_LENGTH,        # was max_seq_length in trl 0.x
        warmup_steps=WARMUP_STEPS,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=4,
        bf16=True,
        optim="paged_adamw_8bit",         # paged for memory-spike safety
        gradient_checkpointing=False,     # we already enabled it via peft
        seed=SEED,
        report_to="none",
        packing=False,
        # transformers 5.x defaults log_level="passive" which silences INFO-level
        # training logs (loss / lr / grad_norm every `logging_steps`). Force INFO
        # so we can see the loss curve in stdout as it trains.
        log_level="info",
        disable_tqdm=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,       # trl 1.x renamed `tokenizer=` to `processing_class=`
    )

    # --- Auto-resume: find latest checkpoint dir; trl 1.x accepts either a bool
    #     (auto-detect) or a path. We pass the explicit path so resume failure
    #     surfaces as a clear FileNotFoundError instead of "started from step 0".
    checkpoints = sorted(args.adapter_out.glob("checkpoint-*"),
                         key=lambda p: int(p.name.split("-")[1]))
    resume_from = str(checkpoints[-1]) if checkpoints else None
    if resume_from:
        print(f"[resume] resuming from {Path(resume_from).name}")
    else:
        print(f"[start]  no existing checkpoints; starting fresh")

    print(f"[train] max_steps={MAX_STEPS}  effective_batch="
          f"{PER_DEVICE_TRAIN_BATCH * GRAD_ACCUM_STEPS}  "
          f"max_seq={MAX_SEQ_LENGTH}  lr={LEARNING_RATE}")
    trainer.train(resume_from_checkpoint=resume_from)

    trainer.save_model(str(args.adapter_out))
    tokenizer.save_pretrained(str(args.adapter_out))
    print(f"\n[OK] adapter → {args.adapter_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
