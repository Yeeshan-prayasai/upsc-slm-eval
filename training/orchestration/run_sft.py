"""CLI entrypoint for one SFT run.

Loads a YAML config, builds the trl SFT trainer (chat-templated
prompt-completion data, completion-only loss), optionally resumes the
LoRA adapter from a CPT-phase checkpoint, runs `trainer.train()` with
auto-resume on the SFT checkpoints themselves.

Usage:
    python -m training.orchestration.run_sft \\
        --config training/configs/sft_gemma.yaml \\
        --runtime training/configs/runtime_l40s.yaml \\
        --train data/ft_split/train.jsonl \\
        --valid data/ft_split/valid.jsonl \\
        --resume-lora-from adapters/gemma4-e4b-upsc-v2-cpt/final \\
        --max-steps 5000 \\
        --output-dir adapters/gemma4-e4b-upsc-v2-sft
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from ..eval import preflight
from ..eval.pulse import PulseConfig, PulseEvalCallback
from ..trainers.base import LoRAConfig, OptimConfig, RuntimeConfig
from ..trainers.sft import SFTConfigV2, build_sft_trainer, maybe_resume


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


def _build_cfg(args: argparse.Namespace) -> SFTConfigV2:
    main = _load_yaml(args.config) if args.config else {}
    runtime_yaml = _load_yaml(args.runtime) if args.runtime else {}

    lora_dict = {**runtime_yaml.get("lora", {}), **main.get("lora", {})}
    rt_dict = {**runtime_yaml.get("runtime", {}), **main.get("runtime", {})}
    opt_dict = {**runtime_yaml.get("optim", {}), **main.get("optim", {})}

    lora_cfg = LoRAConfig(**lora_dict) if lora_dict else LoRAConfig()
    rt_cfg = RuntimeConfig(**rt_dict) if rt_dict else RuntimeConfig()
    # SFT optim defaults differ from CPT — but if the YAML or CLI doesn't
    # override, fall through to the SFTConfigV2 default (lr=2e-4, β2=0.999, wd=0.01)
    # which is set in the dataclass via its default_factory.
    if opt_dict:
        opt_cfg = OptimConfig(**opt_dict)
    else:
        opt_cfg = OptimConfig(learning_rate=2e-4, adam_beta2=0.999, weight_decay=0.01)

    base_model = args.base_model or main.get("base_model")
    if not base_model:
        raise ValueError("base_model required (via --base-model or YAML)")
    train = args.train or main.get("train_jsonl")
    valid = args.valid or main.get("valid_jsonl")
    if not (train and valid):
        raise ValueError("train/valid JSONL required (via --train/--valid or YAML)")
    output_dir = args.output_dir or main.get("output_dir")
    if not output_dir:
        raise ValueError("output_dir required (via --output-dir or YAML)")
    # None → derived as `epochs` epochs over the train split.
    max_steps = args.max_steps or main.get("max_steps")

    resume_lora = args.resume_lora_from or main.get("resume_lora_from")
    resume_lora_path = Path(resume_lora) if resume_lora else None

    return SFTConfigV2(
        base_model=base_model,
        train_jsonl=Path(train),
        valid_jsonl=Path(valid),
        output_dir=Path(output_dir),
        resume_lora_from=resume_lora_path,
        max_steps=int(max_steps) if max_steps else None,
        epochs=float(main.get("epochs", 2.0)),
        lora=lora_cfg,
        runtime=rt_cfg,
        optim=opt_cfg,
        warmup_ratio=float(main.get("warmup_ratio", 0.03)),
        min_lr_rate=float(main.get("min_lr_rate", 0.1)),
        save_steps=int(main.get("save_steps", 100)),
        save_total_limit=int(main.get("save_total_limit", 3)),
        logging_steps=int(main.get("logging_steps", 50)),
        eval_steps=int(main.get("eval_steps", 100)),
        report_to=str(main.get("report_to", "none")),
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run SFT (with length-penalty loss).")
    p.add_argument("--config", type=Path,
                   help="Per-model SFT YAML (sft_gemma.yaml or sft_qwen.yaml)")
    p.add_argument("--runtime", type=Path, help="Runtime YAML (shared with CPT)")
    p.add_argument("--base-model", help="Override config base_model")
    p.add_argument("--train", type=Path, help="Override train_jsonl path")
    p.add_argument("--valid", type=Path, help="Override valid_jsonl path")
    p.add_argument("--resume-lora-from", type=Path,
                   help="LoRA adapter dir to continue training from (typically a CPT 'final' dir)")
    p.add_argument("--max-steps", type=int, help="Override config max_steps")
    p.add_argument("--output-dir", type=Path, help="Override config output_dir")
    p.add_argument("--skip-preflight", action="store_true",
                   help="Skip the eval-set leakage gate (DANGEROUS — debug only)")
    args = p.parse_args(argv)

    cfg = _build_cfg(args)

    # Pre-flight gate — SFT also reads from data/ft_corpus.parquet (v1
    # locked) but the eval-set + manifest check is still cheap and catches
    # accidental contamination introduced after corpus build.
    if not args.skip_preflight:
        # SFT doesn't read the tokenized CPT corpus directly, so we skip
        # the tokenized-existence check (--no-tokenized).
        rep = preflight.run_preflight(["gemma", "qwen"], skip_ngram=True,
                                       require_tokenized=False)
        print(preflight.render(rep))
        if not rep.is_clean():
            print("\nPRE-FLIGHT FAILED — aborting SFT.", file=sys.stderr)
            return 2

    model_family = "qwen" if "qwen" in cfg.base_model.lower() else "gemma"
    pulse_cb = PulseEvalCallback(
        PulseConfig(model_family=model_family), cfg.output_dir,
    )

    trainer, model, tokenizer = build_sft_trainer(
        cfg, extra_callbacks=[pulse_cb],
    )

    resume_from = maybe_resume(cfg)
    trainer.train(resume_from_checkpoint=resume_from)

    hard_stop_marker = cfg.output_dir / "HARD_STOP"
    if hard_stop_marker.exists():
        print(f"\nHARD-STOP: {hard_stop_marker.read_text().strip()} — "
              f"final adapter NOT saved.", file=sys.stderr)
        return 3

    final_dir = cfg.output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"[sft] saved final adapter to {final_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
