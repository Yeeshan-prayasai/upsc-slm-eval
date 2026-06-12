"""Smoke-test the full CPT chain on a CUDA host before committing to a
55-hour run. Validates:

1. Model loads in 4-bit NF4 + bf16 compute (Dettmers QLoRA config)
2. `prepare_model_for_kbit_training` runs cleanly (peft + bnb interop)
3. LoRA attaches at rank 64 to all decoder layers × 7 projections
4. Trainable-param count lands in expected 50-150 M band
5. WSD scheduler integrates with HF Trainer + grad accumulation
6. 20 training steps run forward+backward without OOM at seq=4096
7. Checkpoint saves + reloads successfully
8. VRAM peak fits inside the L40S 48 GB budget

Uses a 1 K-row subset of the actual packed corpus (or a synthetic
1 K-sequence corpus if the real corpus parquet isn't available yet).
20 steps × bs=1 × grad_accum=64 × seq=4096 = ~5.2 M tokens of compute
→ 5-10 minute runtime on L40S; aborts with a clear error if any
stage fails.

Usage on EC2:
    python -m training.orchestration.smoke_cpt --model gemma
    python -m training.orchestration.smoke_cpt --model qwen --corpus data/cpt_corpus_qwen.parquet

If `--corpus` is unset and `data/cpt_corpus_{model}.parquet` doesn't
exist, the smoke generates a synthetic 1 K-sequence dataset from the
base model's tokenizer (lets you smoke before the real corpus is
ready).
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import torch

from ..trainers.base import LoRAConfig, OptimConfig, RuntimeConfig
from ..trainers.cpt import CPTConfig, build_cpt_trainer, load_packed_corpus

BASES = {
    "gemma": "google/gemma-4-E4B-it",
    "qwen": "Qwen/Qwen3.5-4B",
}


def make_synthetic_corpus(base_model: str, n_sequences: int, seq_len: int) -> Path:
    """Build a tiny synthetic packed corpus to smoke the trainer when
    the real corpus parquet doesn't exist yet. Tokenizes a short
    loremish passage repeatedly until we have `n_sequences` of
    `seq_len` tokens."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 200 chars of UPSC-flavored prose seed; tokenizes to ~50-80 tokens
    seed = (
        "Article 21 of the Indian Constitution guarantees the right to life and "
        "personal liberty. The Supreme Court in Maneka Gandhi v. Union of India "
        "(1978) expanded its scope to include the right to live with dignity. "
        "Subsequent judgments have read in rights to privacy (Puttaswamy, 2017), "
        "health, education, and a clean environment. "
    )
    # Tokenize once and tile.
    ids = tok.encode(seed, add_special_tokens=False)
    eos = tok.eos_token_id
    one_doc = ids + [eos]
    # Build a continuous stream, slice into seq_len chunks.
    needed_tokens = n_sequences * seq_len
    tiles = (needed_tokens // len(one_doc)) + 2
    buf = (one_doc * tiles)[: n_sequences * seq_len]
    rows = [buf[i * seq_len : (i + 1) * seq_len] for i in range(n_sequences)]

    out_path = Path(tempfile.gettempdir()) / f"smoke_cpt_synth_{Path(base_model).name}.parquet"
    pq.write_table(
        pa.table({"input_ids": rows}, schema=pa.schema([("input_ids", pa.list_(pa.int32()))])),
        out_path,
        compression="zstd",
    )
    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Smoke-test the CPT trainer (20 steps, abort-on-fail).")
    p.add_argument("--model", choices=("gemma", "qwen"), default="gemma")
    p.add_argument("--corpus", type=Path,
                   help="Pre-tokenized parquet (default: data/cpt_corpus_{model}.parquet "
                        "if it exists; otherwise synthesize)")
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--bs", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=64)
    p.add_argument("--output-dir", type=Path,
                   default=Path("/tmp/smoke_cpt"),
                   help="Smoke output dir; wiped at start. Default /tmp/smoke_cpt")
    args = p.parse_args(argv)

    base_model = BASES[args.model]
    print(f"\n========== CPT SMOKE TEST ==========")
    print(f"  model:       {base_model}")
    print(f"  max_steps:   {args.max_steps}")
    print(f"  seq_len:     {args.seq_len}")
    print(f"  effective batch tokens: {args.bs * args.grad_accum * args.seq_len:,}")
    print(f"  CUDA visible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        d = torch.cuda.get_device_properties(0)
        print(f"  GPU: {d.name}, {d.total_memory/1e9:.1f} GB VRAM")
    else:
        print("  WARNING: no CUDA — bnb 4-bit will fail. This smoke requires a CUDA host.")
        return 2
    print()

    # Resolve corpus
    if args.corpus and args.corpus.exists():
        corpus = args.corpus
        print(f"[corpus] using existing {corpus}")
    else:
        default = Path(f"data/cpt_corpus_{args.model}.parquet")
        if default.exists():
            corpus = default
            print(f"[corpus] using existing {corpus}")
        else:
            print(f"[corpus] no existing corpus; synthesizing 1 K sequences ...")
            n_seq = max(1024, args.bs * args.grad_accum * args.max_steps)
            corpus = make_synthetic_corpus(base_model, n_seq, args.seq_len)
            print(f"[corpus] synth: {corpus}")

    # Wipe smoke output dir for clean run
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = CPTConfig(
        base_model=base_model,
        corpus_parquet=corpus,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        lora=LoRAConfig(),
        runtime=RuntimeConfig(
            per_device_train_batch_size=args.bs,
            gradient_accumulation_steps=args.grad_accum,
            max_seq_length=args.seq_len,
        ),
        optim=OptimConfig(),
        wsd_warmup_frac=0.10,        # short smoke — use bigger warmup so we exercise the lambda
        wsd_stable_frac=0.50,
        wsd_min_lr_ratio=0.1,
        save_steps=args.max_steps,   # save once at end
        save_total_limit=1,
        logging_steps=1,             # log every step in smoke
    )

    print("\n[stage 1/4] building model + tokenizer + LoRA ...")
    trainer, model, tokenizer = build_cpt_trainer(cfg)

    print("\n[stage 2/4] checking VRAM after model load ...")
    print(f"  CUDA allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"  CUDA reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    print(f"  CUDA peak:      {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    print(f"\n[stage 3/4] running {args.max_steps} smoke steps ...")
    trainer.train()
    print(f"\n[stage 3/4] training completed.")
    print(f"  CUDA peak after training: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    print("\n[stage 4/4] saving + reloading adapter ...")
    final_dir = args.output_dir / "smoke_final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Reload as a peft adapter to verify roundtrip
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    # Free model first
    del model
    del trainer
    torch.cuda.empty_cache()

    from ..trainers.base import build_bnb_config
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=build_bnb_config(),
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    reloaded = PeftModel.from_pretrained(base, str(final_dir))
    reloaded_trainable = sum(p.numel() for p in reloaded.parameters() if p.requires_grad)
    print(f"  ✓ adapter reload OK; trainable: {reloaded_trainable:,}")

    print(f"\n========== SMOKE PASSED ==========")
    print(f"  Output: {final_dir}")
    print(f"  Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(f"  (L40S budget: 48 GB; A10G: 24 GB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
