"""Stage the instruction-format slice of the CPT corpus.

CPT on instruction-tuned checkpoints erodes chat formatting when the
stream is 100% raw text (Cheng et al. 2023 AdaptLLM; arXiv 2401.03129).
The standard mitigation is to mix instruction-format data into the CPT
stream. This script stages the v2 SFT train split (already
eval-leakage-gated at its own build time) under
`data/cpt_clean_dedup/instruct/` as `{"prompt": str, "completion": str}`
rows; `tokenize_pack` renders those rows through each model's own chat
template at pack time, so Gemma sees Gemma formatting and Qwen sees
Qwen formatting from the same staged file.

Run AFTER `build_sft_corpus` and AFTER the clean+dedup stage (the
build_cpt_corpus orchestrator sequences this); the leakage gate then
re-checks everything under cpt_clean_dedup including this dir.

Only the TRAIN split is staged — the valid split stays unseen for SFT
eval-loss / best-checkpoint selection.

CLI:
    python -m training.data.build_instruct_cpt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .acquire._base import RepoPaths

REPO = RepoPaths.root()
SFT_TRAIN = REPO / "data" / "sft_v2" / "train.jsonl"
OUT_PATH = REPO / "data" / "cpt_clean_dedup" / "instruct" / "sft_pairs.jsonl"


def _content(turns) -> str:
    """Extract the content string from a conversational turn list."""
    if isinstance(turns, list) and turns and isinstance(turns[0], dict):
        return str(turns[0].get("content") or "").strip()
    if isinstance(turns, str):
        return turns.strip()
    return ""


def build(sft_train: Path = SFT_TRAIN, out_path: Path = OUT_PATH) -> int:
    if not sft_train.exists():
        print(f"ERROR: {sft_train} not found — run `make build-sft-corpus` first.",
              file=sys.stderr)
        return 1
    n_in = n_out = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with sft_train.open(encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = _content(d.get("prompt"))
            completion = _content(d.get("completion"))
            if not (prompt and completion):
                continue
            fout.write(json.dumps(
                {"prompt": prompt, "completion": completion},
                ensure_ascii=False) + "\n")
            n_out += 1
    print(f"Instruct CPT slice: {n_out:,}/{n_in:,} rows → "
          f"{out_path.relative_to(REPO)}")
    return 0 if n_out else 1


def main(argv: "list[str] | None" = None) -> int:
    p = argparse.ArgumentParser(description="Stage instruction data for the CPT mix.")
    p.add_argument("--sft-train", type=Path, default=SFT_TRAIN)
    p.add_argument("--out", type=Path, default=OUT_PATH)
    args = p.parse_args(argv)
    return build(sft_train=args.sft_train, out_path=args.out)


if __name__ == "__main__":
    sys.exit(main())
