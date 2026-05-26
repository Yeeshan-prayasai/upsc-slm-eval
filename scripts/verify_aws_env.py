"""Pre-flight environment verifier for the AWS fine-tune path.

Hard-fails (exit code 1) the first thing that's wrong so we never burn hours
of GPU time on a misconfigured box. Invoked by the Makefile as the first
step of `ft-qwen-aws` and `ft-gemma-aws` — *before* anything that allocates
GPU memory or touches the HuggingFace network.

Checks (in order, fail-fast):
1.  CUDA-capable GPU with >= 24 GiB VRAM, BF16 supported.
2.  PyTorch sees the GPU and can run a BF16 matmul on it.
3.  bitsandbytes NF4 4-bit Linear layer loads and runs forward — confirms
    the bnb<->CUDA linkage is intact (the May-2026 CUDA-13 system has been
    a known source of bnb load failures with older wheels).
4.  HuggingFace token present and authenticates against the Hub API.
5.  At least 50 GiB free on the volume holding this repo.
6.  data/ft_split/{train,valid}.jsonl exist and have plausible line counts.
7.  configs/lora.yaml exists.

Designed to be a stdlib-only entry point at the top, so a syntax error
in this file shows up immediately rather than after a 30-second torch
import. Heavyweight imports are deferred to the functions that use them.
"""
from __future__ import annotations
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data" / "ft_split"
LORA_CONFIG = REPO / "configs" / "lora.yaml"

MIN_VRAM_GIB = 24
MIN_DISK_GIB = 50
MIN_TRAIN_LINES = 1000   # actual is ~41k; this catches catastrophic truncation
MIN_VALID_LINES = 100    # actual is ~2k


def _ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}", file=sys.stderr)
    sys.exit(1)


def check_gpu_and_torch() -> None:
    try:
        import torch
    except ImportError as e:
        _fail(f"torch not installed: {e}. Run `pip install -r requirements-aws.txt`")
    if not torch.cuda.is_available():
        _fail("CUDA not available to PyTorch. Check `nvidia-smi` and torch's CUDA build.")
    name = torch.cuda.get_device_name(0)
    vram_gib = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    if vram_gib < MIN_VRAM_GIB:
        _fail(f"GPU '{name}' has {vram_gib:.1f} GiB VRAM; need >= {MIN_VRAM_GIB} for "
              f"4B QLoRA at rank=16, num_layers=16, seq=2048")
    if not torch.cuda.is_bf16_supported():
        _fail(f"GPU '{name}' does not support BF16; recipe assumes bf16 compute_dtype")
    # End-to-end kernel-launch test
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    y = x @ x.T
    torch.cuda.synchronize()
    assert y.shape == (1024, 1024)
    _ok(f"GPU '{name}', {vram_gib:.1f} GiB VRAM, BF16 supported, matmul kernel OK")
    _ok(f"torch {torch.__version__} (built for CUDA {torch.version.cuda})")


def check_trl_sft_trainer() -> None:
    """Import SFTTrainer specifically — it lazy-loads rich/other deps."""
    try:
        from trl import SFTTrainer, SFTConfig
    except ImportError as e:
        _fail(f"trl.SFTTrainer import failed: {e}. "
              f"`pip install -r requirements-aws.txt` may be stale.")
    _ok(f"trl.SFTTrainer + SFTConfig importable")


def check_bitsandbytes() -> None:
    try:
        import bitsandbytes as bnb
        import torch
        from bitsandbytes.nn import Linear4bit
    except ImportError as e:
        _fail(f"bitsandbytes not installed: {e}")
    try:
        layer = Linear4bit(
            512, 512, bias=False, quant_type="nf4",
            compute_dtype=torch.bfloat16,
        ).cuda()
        inp = torch.randn(2, 512, device="cuda", dtype=torch.bfloat16)
        out = layer(inp)
        torch.cuda.synchronize()
        assert out.shape == (2, 512), f"unexpected output shape {out.shape}"
    except Exception as e:
        _fail(f"bitsandbytes NF4 forward pass failed: {type(e).__name__}: {e}. "
              f"May indicate CUDA library mismatch; check `nvidia-smi` driver vs "
              f"torch's bundled CUDA runtime.")
    _ok(f"bitsandbytes {bnb.__version__} NF4 4-bit kernels functional")


def check_hf_auth() -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        _fail(f"huggingface_hub not installed: {e}")
    try:
        info = HfApi().whoami()
    except Exception as e:
        _fail(f"HuggingFace not authenticated: {type(e).__name__}: {e}. "
              f"Run `hf auth login` and paste your hf_... token.")
    name = info.get("name") or info.get("email") or "unknown"
    _ok(f"HuggingFace authenticated as: {name}")


def check_disk() -> None:
    free_gib = shutil.disk_usage(REPO).free / 1024 ** 3
    if free_gib < MIN_DISK_GIB:
        _fail(f"only {free_gib:.1f} GiB free on the volume holding {REPO}; "
              f"need >= {MIN_DISK_GIB} for model weights + checkpoints + logs")
    _ok(f"{free_gib:.1f} GiB free on volume holding {REPO}")


def check_ft_data() -> None:
    train = DATA_DIR / "train.jsonl"
    valid = DATA_DIR / "valid.jsonl"
    if not train.exists():
        _fail(f"{train} not found. SCP it from the M5 first.")
    if not valid.exists():
        _fail(f"{valid} not found. SCP it from the M5 first.")
    train_lines = sum(1 for _ in train.open(encoding="utf-8"))
    valid_lines = sum(1 for _ in valid.open(encoding="utf-8"))
    if train_lines < MIN_TRAIN_LINES:
        _fail(f"{train} has only {train_lines:,} lines; expected >= {MIN_TRAIN_LINES:,}. "
              f"Truncated during SCP?")
    if valid_lines < MIN_VALID_LINES:
        _fail(f"{valid} has only {valid_lines:,} lines; expected >= {MIN_VALID_LINES:,}")
    _ok(f"data/ft_split: train.jsonl {train_lines:,} pairs, "
        f"valid.jsonl {valid_lines:,} pairs")


def check_lora_config() -> None:
    if not LORA_CONFIG.exists():
        _fail(f"{LORA_CONFIG} missing")
    _ok(f"configs/lora.yaml present")


def main() -> int:
    print(f"\nPre-flight verification for AWS fine-tune ({REPO.name})\n")
    check_gpu_and_torch()
    check_bitsandbytes()
    check_trl_sft_trainer()
    check_hf_auth()
    check_disk()
    check_ft_data()
    check_lora_config()
    print(f"\n[OK] all pre-flight checks passed; safe to start training\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
