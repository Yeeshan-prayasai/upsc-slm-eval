"""Stage 3.5 — merge a PEFT LoRA adapter into its base model.

Produces a single self-contained HuggingFace model directory that can be
loaded without `peft` and converted to MLX format on the M5 via
`python -m mlx_lm convert --hf-path <out>`.

Run on the AWS box (NVIDIA GPU available) once both adapters are trained.
Saves the merged model in bf16; ~8 GB per merged model.

Workflow:
    EC2:   python scripts/merge_adapter.py \\
              --base Qwen/Qwen3.5-4B \\
              --adapter adapters/qwen35-4b-upsc-v1 \\
              --merged-out adapters/qwen35-4b-upsc-v1-merged

    M5:    scp <merged-out> ~/SLM/adapters/
           python -m mlx_lm convert \\
              --hf-path adapters/qwen35-4b-upsc-v1-merged \\
              --mlx-path adapters/qwen35-4b-upsc-v1-mlx \\
              -q --q-bits 4 --q-group-size 64

    Then  mlx_lm.load("adapters/qwen35-4b-upsc-v1-mlx") works with no adapter_path.

Implementation notes:
- We load the base in bf16 (NOT 4-bit) because PEFT's merge_and_unload() does
  the merge in fp16/bf16 weight space. Merging into a 4-bit-quantized base
  would require dequantize → add LoRA → requantize, which loses precision.
- The 4B Qwen3.5 base in bf16 is ~8 GB, the 4B Gemma-4-E4B in bf16 is ~10 GB.
  L40S 48 GB VRAM comfortably fits both during merge.
- After merge, the resulting model is a plain HF AutoModelForCausalLM —
  no PEFT wrapping. AutoTokenizer + AutoConfig save alongside.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True,
                    help="HF repo id of the base model (e.g. Qwen/Qwen3.5-4B)")
    ap.add_argument("--adapter", type=Path, required=True,
                    help="Path to the PEFT/HuggingFace adapter directory")
    ap.add_argument("--merged-out", type=Path, required=True,
                    help="Output path for the merged HF model")
    args = ap.parse_args()

    if not args.adapter.exists():
        print(f"[FAIL] Adapter directory not found: {args.adapter}")
        return 1
    if not (args.adapter / "adapter_config.json").exists():
        print(f"[FAIL] Missing adapter_config.json in {args.adapter} — "
              f"not a valid PEFT adapter directory")
        return 1
    if args.merged_out.exists() and any(args.merged_out.iterdir()):
        print(f"[FAIL] Merged-out directory {args.merged_out} is non-empty. "
              f"Refusing to overwrite. Delete it first if intentional.")
        return 1

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from peft import PeftModel

    print(f"[load] Base model {args.base} in bf16 ...")
    # No quantization — we want clean fp16/bf16 weights for the merge.
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    print(f"       base loaded: {type(base_model).__name__}")

    print(f"[load] Attaching adapter from {args.adapter} ...")
    model = PeftModel.from_pretrained(base_model, str(args.adapter))

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"       adapter wired: {n_trainable:,} trainable params")

    print(f"[merge] Calling merge_and_unload() — folds LoRA into base weights ...")
    merged = model.merge_and_unload()
    print(f"        merged model class: {type(merged).__name__}")

    # Sanity: merged model should have NO trainable params (LoRA gone)
    remaining_trainable = sum(p.numel() for p in merged.parameters() if p.requires_grad)
    if remaining_trainable > 0:
        print(f"[WARN] {remaining_trainable:,} trainable params still in merged "
              f"model — merge may not have fully unwrapped PEFT")

    args.merged_out.mkdir(parents=True, exist_ok=True)
    print(f"[save] Writing merged model + tokenizer to {args.merged_out} ...")
    merged.save_pretrained(
        str(args.merged_out),
        safe_serialization=True,
        max_shard_size="5GB",
    )

    # Save tokenizer + config from the adapter directory (it has the
    # post-training chat template) — falling back to the base if missing.
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(args.adapter),
                                                  trust_remote_code=True)
        print(f"       loaded tokenizer from adapter dir (has trained chat template)")
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
        print(f"       loaded tokenizer from base (adapter dir had no tokenizer)")
    tokenizer.save_pretrained(str(args.merged_out))

    print(f"\n[OK] Merged model written to {args.merged_out}")
    print(f"     Files:")
    for f in sorted(args.merged_out.iterdir()):
        size_mb = f.stat().st_size / 1024 ** 2 if f.is_file() else 0
        print(f"       {f.name:<40} {size_mb:>8.1f} MB" if size_mb > 0
              else f"       {f.name}/ (dir)")
    print(f"\n     Next: convert to MLX 4-bit on M5:")
    print(f"       python -m mlx_lm convert \\")
    print(f"         --hf-path {args.merged_out.name} \\")
    print(f"         --mlx-path {args.merged_out.name.replace('-merged', '-mlx')} \\")
    print(f"         -q --q-bits 4 --q-group-size 64")
    return 0


if __name__ == "__main__":
    sys.exit(main())
