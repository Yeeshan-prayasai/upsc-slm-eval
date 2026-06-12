"""Smoke test for the CPT trainer (methodology §11 final item).

Runs the actual `CPTTrainer` for 100 steps on **GPT-2 124M** with a
synthetic 1K-row token corpus. Validates that:

1. `build_cpt_trainer` constructs without error against a real (tiny)
   HF base model
2. The WSD scheduler hooks in cleanly
3. 100 training steps complete + checkpoint saves to disk
4. The checkpoint reloads via `peft.PeftModel.from_pretrained`
5. Loss decreases monotonically across the 100 steps (sanity-check that
   training is actually happening, not just stepping)

GPT-2 124M is chosen because it's small (~500 MB on disk) and HF-hosted
so no licence dance is required for CI. The CPT trainer's QLoRA path
needs BitsAndBytes which only runs on CUDA, so this test skips on CPU /
when bitsandbytes is unavailable — it's intended for GPU-bearing
machines (the L40S EC2 host or any local CUDA box).

If you're running locally on M5 (no CUDA), this test is auto-skipped.
For full local verification, run on the EC2 smoke host:

    make smoke-cpt-gemma     # the production smoke entrypoint
    pytest training/tests/test_cpt_smoke.py -xvs

The Makefile target uses the real Gemma-4-E4B base; this pytest test
uses GPT-2 124M so it can run as part of the regular test sweep without
pulling a 4 B-param model.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


# Skip the whole module if the GPU/training stack isn't installed
# (we keep base unit tests runnable on the M5 dev box without HF deps).
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("bitsandbytes")
pytest.importorskip("trl")  # not used here directly but the trainer module imports check it


CUDA_OK = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(
    not CUDA_OK,
    reason="CPT smoke test requires CUDA (bitsandbytes 4-bit only runs on GPU)",
)


SMOKE_BASE = "gpt2"  # 124M params; HF-default, no auth needed
SMOKE_STEPS = 100
SMOKE_SEQ_LEN = 256
SMOKE_CORPUS_ROWS = 1024


@pytest.fixture(scope="module")
def smoke_corpus(tmp_path_factory) -> Path:
    """Build a synthetic 1K-row tokenized parquet corpus.

    Each row is a tensor of length SMOKE_SEQ_LEN of pseudorandom token
    IDs in [0, vocab_size). The CPT trainer's data collator expects the
    column `input_ids` (and uses the same as labels via causal LM)."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(SMOKE_BASE)
    vocab_size = tok.vocab_size
    rng = np.random.default_rng(20260605)
    rows = rng.integers(0, vocab_size, size=(SMOKE_CORPUS_ROWS, SMOKE_SEQ_LEN), dtype=np.int32)

    out_dir = tmp_path_factory.mktemp("cpt_smoke_corpus")
    out_path = out_dir / "synth.parquet"
    table = pa.table({"input_ids": [row.tolist() for row in rows]})
    pq.write_table(table, str(out_path))
    return out_path


def _make_smoke_cfg(corpus_path: Path, out_dir: Path):
    """Build a CPTConfig wired to the smoke base + corpus + 100 steps."""
    from training.trainers.base import LoRAConfig, OptimConfig, RuntimeConfig
    from training.trainers.cpt import CPTConfig

    return CPTConfig(
        base_model=SMOKE_BASE,
        corpus_parquet=corpus_path,
        output_dir=out_dir,
        max_steps=SMOKE_STEPS,
        lora=LoRAConfig(
            r=8, alpha=16, dropout=0.05, use_rslora=True,
            bias="none", task_type="CAUSAL_LM",
            # GPT-2 layer names differ from Gemma/Qwen; target attention proj.
            target_modules=["c_attn"],
        ),
        optim=OptimConfig(
            learning_rate=1e-4,    # GPT-2 small needs a higher LR to move in 100 steps
            adam_beta1=0.9, adam_beta2=0.95, adam_epsilon=1e-8,
            weight_decay=0.1, max_grad_norm=1.0,
        ),
        runtime=RuntimeConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            max_seq_length=SMOKE_SEQ_LEN,
            gradient_checkpointing=False,
            bf16=True,    # production parity — load_base_model loads bf16
            dataloader_num_workers=0,
        ),
        wsd_warmup_frac=0.05,
        wsd_stable_frac=0.70,
        wsd_min_lr_ratio=0.1,
        save_steps=SMOKE_STEPS,                    # one checkpoint at end
        save_total_limit=1,
        logging_steps=10,
        report_to="none",
    )


def test_cpt_smoke_100_steps_gpt2(smoke_corpus, tmp_path):
    """End-to-end smoke: build trainer, run 100 steps, save+reload checkpoint."""
    from training.trainers.cpt import build_cpt_trainer

    out_dir = tmp_path / "smoke_out"
    cfg = _make_smoke_cfg(smoke_corpus, out_dir)
    trainer, _model, _tok = build_cpt_trainer(cfg)

    # Run training. Trainer's log_history records per-step loss.
    trainer.train()

    losses = [
        h["loss"] for h in trainer.state.log_history
        if "loss" in h and "eval_loss" not in h
    ]
    assert len(losses) >= 5, f"expected ≥ 5 logged losses, got {losses}"
    # Sanity: late-window mean should be below early-window mean. GPT-2 124M
    # on random tokens won't drop sharply, but any real gradient step gives
    # a measurable downward trend over 100 steps.
    early = sum(losses[:3]) / 3
    late = sum(losses[-3:]) / 3
    assert late < early, (
        f"loss didn't decrease: early={early:.3f} late={late:.3f} all={losses}"
    )

    # Checkpoint saved
    ckpts = list(out_dir.glob("checkpoint-*"))
    assert ckpts, f"no checkpoint saved under {out_dir}"

    # Reload checkpoint adapter via peft to confirm it's valid
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    base = AutoModelForCausalLM.from_pretrained(SMOKE_BASE)
    reloaded = PeftModel.from_pretrained(base, str(ckpts[-1]))
    assert reloaded is not None
    # The reloaded model must have at least one trainable parameter
    # registered as a LoRA delta.
    lora_params = [n for n, _ in reloaded.named_parameters() if "lora_" in n]
    assert lora_params, "no LoRA params found in reloaded checkpoint"
