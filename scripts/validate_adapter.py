"""Stage 3.4 — adapter sanity check.

Runs the FT'd model on 50 held-out items per task (from data/ft_split/valid.jsonl,
which scripts/run_ft.py wrote when the adapter was trained). Verifies each output
is non-empty and parseable per task convention. Halts the pipeline if the
unparseable rate exceeds 5%, catching catastrophic regressions before downstream
inference burns time and tokens.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VALID_JSONL = REPO / "data" / "ft_split" / "valid.jsonl"
TASK_TAG_RE = re.compile(r"\[TASK=([ABCE])\]")
LETTER_RE = re.compile(r"\b[ABCD]\b")
DEFAULT_PER_TASK = 50
FAIL_RATE_THRESHOLD = 0.05


def _task_of(prompt: str) -> str:
    m = TASK_TAG_RE.search(prompt)
    return m.group(1) if m else "?"


def _is_parseable(task: str, output: str) -> bool:
    if not output or not output.strip():
        return False
    if task == "A":
        return bool(LETTER_RE.search(output.upper()))
    if task == "B":
        return True
    if task in ("C", "E"):
        try:
            json.loads(output)
            return True
        except (json.JSONDecodeError, ValueError):
            return False
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="HF id of MLX base model")
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--per-task", type=int, default=DEFAULT_PER_TASK)
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    if not VALID_JSONL.exists():
        print(f"[FAIL] {VALID_JSONL} not found; run `make ft-gemma` (or `ft-qwen`) first")
        return 1

    by_task: dict[str, list[dict]] = {"A": [], "B": [], "C": [], "E": []}
    for line in VALID_JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        user = item["messages"][0]["content"]
        task = _task_of(user)
        if task in by_task and len(by_task[task]) < args.per_task:
            by_task[task].append(item)

    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    print(f"Loading {args.base} + adapter {args.adapter} …")
    model, tokenizer = load(args.base, adapter_path=str(args.adapter))
    sampler = make_sampler(temp=0.0)

    fail_counts: Counter[str] = Counter()
    n_total = 0
    for task in ("A", "B", "C", "E"):
        for item in by_task[task]:
            user = item["messages"][0]["content"]
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user}],
                add_generation_prompt=True, tokenize=False,
            )
            out = generate(model, tokenizer, prompt=prompt,
                           max_tokens=args.max_tokens, sampler=sampler, verbose=False)
            if not _is_parseable(task, out):
                fail_counts[task] += 1
            n_total += 1

    print()
    for task in ("A", "B", "C", "E"):
        n = len(by_task[task])
        f = fail_counts[task]
        rate = f / max(1, n)
        status = "OK" if rate <= FAIL_RATE_THRESHOLD else "FAIL"
        print(f"  [{status}] task {task}: {n - f}/{n} parseable ({(n - f) / max(1, n):.1%})")

    overall_fails = sum(fail_counts.values())
    overall_rate = overall_fails / max(1, n_total)
    if overall_rate > FAIL_RATE_THRESHOLD:
        print(f"\n[FAIL] overall unparseable rate {overall_rate:.1%} > {FAIL_RATE_THRESHOLD:.0%}")
        return 1
    print(f"\n[OK] adapter passes sanity check ({n_total - overall_fails}/{n_total} parseable)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
