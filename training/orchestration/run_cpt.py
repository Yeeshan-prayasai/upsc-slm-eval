"""CLI entrypoint for one CPT run.

Loads a YAML config, builds the CPT trainer + model, invokes
`trainer.train()` with auto-resume. The full run record (config dict +
WSD curve metadata) is logged to stdout at start; HF Trainer's native
logging handles per-step loss/lr/grad-norm.

Usage:
    python -m training.orchestration.run_cpt \\
        --config training/configs/cpt_gemma.yaml \\
        --runtime training/configs/runtime_l40s.yaml \\
        --corpus data/cpt_corpus_gemma.parquet \\
        --max-steps 16000 \\
        --output-dir adapters/gemma4-e4b-upsc-v2-cpt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from ..eval import preflight
from ..eval.pulse import PulseConfig, PulseEvalCallback
from ..trainers.base import LoRAConfig, OptimConfig, RuntimeConfig
from ..trainers.cpt import CPTConfig, build_cpt_trainer, maybe_resume
from ._guards import clear_stale_hard_stop, hard_stopped_this_run, schedule_resume_guard


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


def _build_cfg(args: argparse.Namespace) -> CPTConfig:
    """Merge YAML configs + CLI overrides into a typed CPTConfig.

    Precedence (high → low): CLI flag → runtime-yaml → main-yaml → defaults.
    """
    main = _load_yaml(args.config) if args.config else {}
    runtime_yaml = _load_yaml(args.runtime) if args.runtime else {}

    # Pull subsections; main config takes precedence over the runtime YAML
    # for keys that overlap (config-level explicit > runtime-default).
    lora_dict = {**runtime_yaml.get("lora", {}), **main.get("lora", {})}
    rt_dict = {**runtime_yaml.get("runtime", {}), **main.get("runtime", {})}
    opt_dict = {**runtime_yaml.get("optim", {}), **main.get("optim", {})}

    lora_cfg = LoRAConfig(**lora_dict) if lora_dict else LoRAConfig()
    rt_cfg = RuntimeConfig(**rt_dict) if rt_dict else RuntimeConfig()
    opt_cfg = OptimConfig(**opt_dict) if opt_dict else OptimConfig()

    base_model = args.base_model or main.get("base_model")
    if not base_model:
        raise ValueError("base_model required (via --base-model or YAML)")
    corpus = args.corpus or main.get("corpus_parquet")
    if not corpus:
        raise ValueError("corpus required (via --corpus or YAML)")
    # None → derived as one epoch over the mix-weighted packed corpus.
    max_steps = args.max_steps or main.get("max_steps")
    output_dir = args.output_dir or main.get("output_dir")
    if not output_dir:
        raise ValueError("output_dir required (via --output-dir or YAML)")

    return CPTConfig(
        base_model=base_model,
        corpus_parquet=Path(corpus),
        output_dir=Path(output_dir),
        max_steps=int(max_steps) if max_steps else None,
        lora=lora_cfg,
        runtime=rt_cfg,
        optim=opt_cfg,
        wsd_warmup_frac=float(main.get("wsd_warmup_frac", 0.01)),
        wsd_stable_frac=float(main.get("wsd_stable_frac", 0.80)),
        wsd_min_lr_ratio=float(main.get("wsd_min_lr_ratio", 0.1)),
        save_steps=int(main.get("save_steps", 1000)),
        save_total_limit=int(main.get("save_total_limit", 5)),
        logging_steps=int(main.get("logging_steps", 50)),
        report_to=str(main.get("report_to", "none")),
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run continued pretraining (CPT).")
    p.add_argument("--config", type=Path,
                   help="Per-model CPT YAML (cpt_gemma.yaml or cpt_qwen.yaml)")
    p.add_argument("--runtime", type=Path,
                   help="Runtime YAML (runtime_l40s.yaml) for VRAM-tuned batch/seq/optim")
    p.add_argument("--base-model", help="Override config base_model")
    p.add_argument("--corpus", type=Path, help="Override config corpus_parquet")
    p.add_argument("--max-steps", type=int, help="Override config max_steps")
    p.add_argument("--output-dir", type=Path, help="Override config output_dir")
    p.add_argument("--skip-preflight", action="store_true",
                   help="Skip the leakage gate (DANGEROUS — only for debugging)")
    args = p.parse_args(argv)

    cfg = _build_cfg(args)

    # Pre-flight leakage gate — refuse to start if eval contamination
    # made it into the tokenized corpus.
    if not args.skip_preflight:
        # Detect tokenizer key from corpus filename (cpt_corpus_<key>.parquet)
        tok_key = cfg.corpus_parquet.stem.replace("cpt_corpus_", "")
        if tok_key not in ("gemma", "qwen"):
            tok_key = "gemma"
        rep = preflight.run_preflight([tok_key], skip_ngram=False,
                                       require_tokenized=True)
        print(preflight.render(rep))
        if not rep.is_clean():
            print("\nPRE-FLIGHT FAILED — aborting CPT.", file=sys.stderr)
            return 2

    # In-training pulse: task probe + MMLU + Hindi no-regression gates
    # (methodology §7). Baselines are measured at step 0 by the callback
    # itself, with the pulse's own prompt format.
    model_family = "qwen" if "qwen" in cfg.base_model.lower() else "gemma"
    pulse_cb = PulseEvalCallback(
        PulseConfig(model_family=model_family), cfg.output_dir,
    )

    trainer, model, tokenizer = build_cpt_trainer(
        cfg, extra_callbacks=[pulse_cb],
    )

    resume_from = maybe_resume(cfg)
    # Clear a stale HARD_STOP BEFORE deciding to resume: a resume into a
    # stale marker would otherwise burn the whole remaining budget then
    # discard the adapter.
    launch_ts = clear_stale_hard_stop(cfg.output_dir)

    # WSD-resume guard: LambdaLR only persists the step counter; the curve
    # is rebuilt from config at resume, so a changed schedule silently
    # reshapes it around the restored step.
    if not schedule_resume_guard(cfg.output_dir, {
        "max_steps": int(trainer.args.max_steps),
        "warmup_frac": cfg.wsd_warmup_frac,
        "stable_frac": cfg.wsd_stable_frac,
        "min_lr_ratio": cfg.wsd_min_lr_ratio,
    }, kind="wsd"):
        return 2

    trainer.train(resume_from_checkpoint=resume_from)

    # A pulse hard-stop checkpoints + stops the trainer; it must NOT
    # look like a successful run (downstream SFT/ablation would train
    # on the regressed adapter).
    reason = hard_stopped_this_run(cfg.output_dir, launch_ts)
    if reason:
        print(f"\nHARD-STOP: {reason} — final adapter NOT saved.", file=sys.stderr)
        return 3

    final_dir = cfg.output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"[cpt] saved final adapter to {final_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
