"""Stage 3 (AWS path) — fine-tune via PyTorch + peft + bitsandbytes (QLoRA).

Drop-in for scripts/run_ft.py when running on NVIDIA hardware (e.g. an EC2
g6.xlarge with an L4 24 GB). Recipe is held symbolically in sync with
configs/lora.yaml — same rank, alpha, num_layers, target_modules, batch,
grad-accum, max_seq, iters. Only the framework differs.

Inputs (must be present on the box BEFORE running):
    data/ft_split/train.jsonl    (chat-format pairs, produced locally by run_ft.py)
    data/ft_split/valid.jsonl    (same)
    HuggingFace login: `huggingface-cli login` if pulling Gemma-4-E4B-it.

Outputs:
    adapters/<adapter-out>/                  -- PEFT adapter (HF format)
    adapters/<adapter-out>/training.log     -- stdout teed
The PEFT adapter is HF-format; convert to MLX for local M5 inference via
`python -m mlx_lm convert --hf-path adapters/<name>` if/when needed.
"""
from __future__ import annotations
import argparse
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
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
NUM_LORA_LAYERS = 16            # last N transformer blocks
MAX_SEQ_LENGTH = 2048
PER_DEVICE_TRAIN_BATCH = 1
GRAD_ACCUM_STEPS = 8            # effective batch 8 (QLoRA standard)
LEARNING_RATE = 2.0e-4
MAX_STEPS = 16000               # ≈ 3 epochs at effective batch 8 over 42 701 pairs
EVAL_STEPS = 500
SAVE_STEPS = 1000
LOGGING_STEPS = 50
WARMUP_STEPS = 100


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
              f"materialize JSONL locally then aws s3 sync into data/ft_split/")
        return 1

    import torch
    from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    from peft import LoraConfig, get_peft_model
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig

    # 4-bit NF4 — QLoRA default (Dettmers et al., NeurIPS 2023).
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"[load] {args.base} (4-bit NF4, compute=bf16)")
    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(args.base, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    total_layers = AutoConfig.from_pretrained(args.base).num_hidden_layers
    layers_to_transform = list(range(total_layers - NUM_LORA_LAYERS, total_layers))
    print(f"[lora] r={LORA_R} alpha={LORA_ALPHA} dropout={LORA_DROPOUT} "
          f"layers={layers_to_transform[0]}..{layers_to_transform[-1]} "
          f"(of {total_layers}) target={TARGET_MODULES}")

    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        layers_to_transform=layers_to_transform,
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_ds = load_dataset("json", data_files=str(args.train_jsonl), split="train")
    eval_ds = load_dataset("json", data_files=str(args.valid_jsonl), split="train")
    print(f"[data] train={len(train_ds):,}  valid={len(eval_ds):,}")

    args.adapter_out.mkdir(parents=True, exist_ok=True)
    sft_cfg = SFTConfig(
        output_dir=str(args.adapter_out),
        max_steps=MAX_STEPS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        max_seq_length=MAX_SEQ_LENGTH,
        warmup_steps=WARMUP_STEPS,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=4,
        bf16=True,
        optim="paged_adamw_8bit",   # paged optimizer — survives memory spikes
        gradient_checkpointing=True,
        seed=SEED,
        report_to="none",
        dataset_text_field=None,    # let SFTTrainer auto-detect chat 'messages' key
        packing=False,
    )

    trainer = SFTTrainer(
        model=model, args=sft_cfg,
        train_dataset=train_ds, eval_dataset=eval_ds,
        processing_class=tokenizer,
    )
    print(f"[train] max_steps={MAX_STEPS}  effective_batch={PER_DEVICE_TRAIN_BATCH*GRAD_ACCUM_STEPS}  "
          f"max_seq={MAX_SEQ_LENGTH}  lr={LEARNING_RATE}")
    trainer.train()
    trainer.save_model(str(args.adapter_out))
    tokenizer.save_pretrained(str(args.adapter_out))
    print(f"\n[OK] adapter → {args.adapter_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
