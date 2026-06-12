"""Shared QLoRA + LoRA setup for the CPT and SFT trainers.

Centralizes the model-loading + adapter-attaching boilerplate so the
CPT and SFT trainers stay focused on their loss/data semantics. Two
dataclass configs (`LoRAConfig`, `RuntimeConfig`) make every tunable
explicit, and a single `build_model_with_lora()` function applies the
canonical QLoRA recipe from **Dettmers et al. 2023** ("QLoRA",
NeurIPS 2023, arXiv 2305.14314) + **Kalajdzievski 2023** ("Rank
Stabilization Scaling for LoRA", arXiv 2312.03732).

The recipe:
- 4-bit NF4 base via `bitsandbytes` BitsAndBytesConfig
- bf16 compute dtype
- Double quantization (Dettmers §3.2)
- `prepare_model_for_kbit_training` — fp32 layernorms, gradient flow
- LoRA via `peft.LoraConfig` with `use_rslora=True` (α/√r scaling at
  rank ≥ 32, per Kalajdzievski 2023)
- Target modules regex matching all q/k/v/o/gate/up/down projections
  across all decoder layers (broader scope than v1's last-16-layers)

Loadable on any CUDA host without modification; the Makefile's
`verify-aws-env` target hard-fails before training if bnb/CUDA/HF
aren't reachable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch

# Defer heavy GPU-only imports (peft, bnb-using bits of transformers) until
# `build_model_with_lora` / `find_latest_checkpoint` are actually called.
# This keeps the pure-Python dataclasses (LoRAConfig/OptimConfig/RuntimeConfig)
# importable on machines without those deps — e.g. for unit tests on the M5
# laptop. The GPU host (EC2 L40S) has all deps installed and pays no extra
# cost from the deferred imports.
if TYPE_CHECKING:
    from peft import PeftModel
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

# All q/k/v/o/gate/up/down projections across every decoder layer.
# Gemma-4-E4B (multimodal wrapper) names its text decoder
# `model.language_model.layers.N.…`; Qwen-3.5 uses `model.layers.N.…` —
# so the optional `language_model.` segment sits BETWEEN `model.` and
# `layers.`. (The previous regex put it before `model.` and matched
# ZERO Gemma modules; peft raises ValueError on zero matches, so the
# Gemma run crashed at get_peft_model. attach_lora now also asserts
# the exact per-layer adapter count to catch partial matches.)
TARGET_MODULES_ALL_LAYERS = (
    r"^(?:.*\.)?model\.(?:language_model\.)?layers\.\d+\."
    r".*\.(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
)


@dataclass(frozen=True)
class LoRAConfig:
    """LoRA hyperparameters. Defaults per methodology §4.2."""
    r: int = 64
    alpha: int = 16              # with use_rslora=True the effective scale is α/√r
    dropout: float = 0.05
    use_rslora: bool = True
    target_modules: str = TARGET_MODULES_ALL_LAYERS
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass(frozen=True)
class RuntimeConfig:
    """Hardware / precision runtime settings. Defaults target L40S 48 GB."""
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 64
    max_seq_length: int = 4096
    bf16: bool = True
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4
    optim: str = "paged_adamw_8bit"
    seed: int = 20260514


@dataclass(frozen=True)
class OptimConfig:
    """Optimizer hyperparameters. CPT defaults; SFT subclass overrides
    `learning_rate`, `weight_decay`, and `adam_beta2`.

    CPT LR 1e-4: this is LoRA-adapter CPT, not full-parameter CPT.
    Adapters start from B=0 and need ~10× the full-FT learning rate
    (Biderman et al. 2024 "LoRA Learns Less and Forgets Less"; QLoRA
    used 1e-4–2e-4). The previous 1e-5 was a full-parameter re-warming
    rate (Ibrahim 2024) that doesn't transfer to frozen-base LoRA —
    at 1e-5 the realized weight delta after the full step budget stays
    in the noise floor ("trains but learns nothing")."""
    learning_rate: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95           # 0.95 for CPT, 0.999 for SFT (Ibrahim 2024)
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.1          # 0.1 CPT, 0.01 SFT
    max_grad_norm: float = 1.0


def build_bnb_config(compute_dtype: torch.dtype = torch.bfloat16):
    """The canonical QLoRA quantization config (Dettmers 2023 §3.2)."""
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )


def load_tokenizer(base_model: str) -> "PreTrainedTokenizerBase":
    """Load the base-model tokenizer with the right pad-token wiring.

    `padding_side='right'` is correct for training (loss masks pad on
    the right); inference flips to left in scripts/runners.py for
    causal-LM batched generation.
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        base_model, padding_side="right", trust_remote_code=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base_model(
    base_model: str,
    bnb_cfg=None,
) -> "PreTrainedModel":
    """Load the base model in 4-bit NF4 + bf16 compute. Returns a
    `prepare_model_for_kbit_training`-prepared model ready to attach
    a LoRA adapter to."""
    from transformers import AutoModelForCausalLM
    from peft import prepare_model_for_kbit_training
    if bnb_cfg is None:
        bnb_cfg = build_bnb_config()
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_cfg,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    # use_cache=False is required when gradient_checkpointing is on
    # (the cache is invalid mid-recomputation; HF logs a warning otherwise).
    model.config.use_cache = False
    # Gemma-4's wrapper nests the decoder config; the top-level flag
    # doesn't always reach it.
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    # prepare_model_for_kbit_training upcasts EVERY non-quantized bf16
    # param to fp32. QLoRA (Dettmers 2023 §3.2) only needs the *norms*
    # in fp32 for stability; on Gemma-4-E4B the blanket upcast also hits
    # the ~3B-param embedding / per-layer-embedding tables (~12 GB of
    # frozen weights held at 4 bytes instead of 2). Re-cast embeddings
    # back to bf16; keep norms (and any 1-D params) fp32.
    import torch.nn as nn
    n_recast = 0
    for module in model.modules():
        if isinstance(module, nn.Embedding) and module.weight.dtype == torch.float32:
            module.weight.data = module.weight.data.to(torch.bfloat16)
            n_recast += module.weight.numel()
    if n_recast:
        print(f"[kbit] re-cast {n_recast/1e9:.2f}B embedding params fp32 → bf16 "
              f"(norms stay fp32 per QLoRA)")
    return model


def attach_lora(model: "PreTrainedModel", lora_cfg: LoRAConfig) -> "PreTrainedModel":
    """Wrap `model` with a PEFT LoRA adapter from the given config.
    Validates trainable-param count to catch silent target-module
    mismatches early."""
    from peft import LoraConfig, get_peft_model
    peft_cfg = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
        use_rslora=lora_cfg.use_rslora,
        target_modules=lora_cfg.target_modules,
        bias=lora_cfg.bias,
        task_type=lora_cfg.task_type,
    )
    # Expected adapter sites, computed from the ACTUAL model with a scan
    # independent of the target regex (so a regex anchoring bug can't
    # fool the comparison): every decoder-layer module ending in one of
    # the 7 projection names, excluding the frozen multimodal towers.
    # NOTE: not n_layers × 7 — Gemma-4-E4B's top `num_kv_shared_layers`
    # (18) share KV from earlier layers and have no k/v_proj of their
    # own (42×7 − 18×2 = 258 sites, not 294).
    _PROJ_NAMES = {"q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"}
    expected_sites = {
        n for n, _ in model.named_modules()
        if n.rsplit(".", 1)[-1] in _PROJ_NAMES
        and ".layers." in n
        and not any(t in n for t in ("vision_tower", "audio_tower", "multi_modal"))
    }

    model = get_peft_model(model, peft_cfg)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * n_trainable / max(1, n_total)
    print(f"[lora] trainable: {n_trainable:,} / {n_total:,} ({pct:.2f}%)")

    if n_trainable == 0:
        raise RuntimeError("LoRA attached zero trainable params.")

    # Production-scale guards apply only to the all-layers regex config;
    # custom target_modules (e.g. the GPT-2 CUDA smoke test) skip them.
    if lora_cfg.target_modules == TARGET_MODULES_ALL_LAYERS:
        # Exact-coverage assertion: the regex must have matched EVERY
        # eligible projection site. A trainable-param floor alone can't
        # catch a PARTIAL match (attention-only at r=64 ≈ 55M, above any
        # reasonable floor while silently missing the MLP projections).
        targeted = getattr(model.base_model, "targeted_module_names", None)
        if targeted is not None and expected_sites:
            missed = expected_sites - set(targeted)
            extra = set(targeted) - expected_sites
            if missed or extra:
                raise RuntimeError(
                    f"LoRA coverage mismatch: {len(missed)} eligible sites "
                    f"unmatched (e.g. {sorted(missed)[:3]}), {len(extra)} "
                    f"unexpected matches (e.g. {sorted(extra)[:3]}) — "
                    f"target_modules regex is wrong for this architecture."
                )
            print(f"[lora] adapter coverage verified: {len(targeted)} / "
                  f"{len(expected_sites)} eligible projection sites")

        # Sanity bounds for r=64 + all-layer scope. Methodology §4.1 expects
        # ~108-144 M trainable for Gemma/Qwen at this config.
        if n_trainable < 50_000_000:
            raise RuntimeError(
                f"LoRA wired up only {n_trainable:,} trainable params — likely a "
                f"target_modules regex mismatch. Expected ~100-150M at "
                f"r={lora_cfg.r} over all decoder layers × 7 projections."
            )
        if pct > 8.0:
            raise RuntimeError(
                f"LoRA trainable share is {pct:.2f}%, far above the QLoRA "
                f"effective band (~2-5% at r=64). Regex may be matching "
                f"unintended modules."
            )
    return model


def attach_lora_from_checkpoint(
    model: "PreTrainedModel", adapter_path: Path
) -> "PreTrainedModel":
    """Resume from an existing LoRA adapter directory (used by SFT to
    continue training from the CPT checkpoint)."""
    from peft import PeftModel
    if not (adapter_path / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"No adapter_config.json under {adapter_path} — "
            f"can't resume LoRA from this path."
        )
    return PeftModel.from_pretrained(model, str(adapter_path), is_trainable=True)


def build_model_with_lora(
    base_model: str,
    lora_cfg: LoRAConfig | None = None,
    resume_adapter: Path | None = None,
) -> "tuple[PreTrainedModel, PreTrainedTokenizerBase]":
    """End-to-end model+tokenizer builder: load base → kbit-prep → attach LoRA
    (or resume from an existing adapter). Returns (model, tokenizer)."""
    lora_cfg = lora_cfg or LoRAConfig()
    tok = load_tokenizer(base_model)
    model = load_base_model(base_model)
    # Make sure the model.config.pad_token_id matches the tokenizer's pad.
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tok.pad_token_id
    if resume_adapter is not None:
        model = attach_lora_from_checkpoint(model, resume_adapter)
        print(f"[lora] resumed from {resume_adapter}")
    else:
        model = attach_lora(model, lora_cfg)
    return model, tok


def find_latest_checkpoint(adapter_out: Path) -> Path | None:
    """Find the most-recent `checkpoint-N` dir under `adapter_out` for
    auto-resume. Returns None if no checkpoints exist."""
    cps = sorted(
        adapter_out.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1]) if p.name.split("-")[-1].isdigit() else 0,
    )
    return cps[-1] if cps else None


def config_summary(*configs) -> dict:
    """Flatten one or more dataclass configs into a single dict for logging."""
    out: dict = {}
    for c in configs:
        out.update({k: v for k, v in asdict(c).items()})
    return out
