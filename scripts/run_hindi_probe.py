"""Stage 2.2 — A2 Hindi-capability probe.

Runs a base SLM (no LoRA) on 200 deterministically-sampled Hindi MCQs from
`upsc_prelims_ai_generated_que.question_hindi`. Pass-1 forced-choice protocol:
the model returns a single letter A/B/C/D; we compare to gold.

Appends per-model rows to results/pre_ft_hindi_probe.parquet. If the same
--model is run again, its prior rows are replaced (idempotent).

The probe reads from the local SQLite snapshot — no remote DB access.
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from local_db import read_table

PASS1_TEMPLATE = """You are taking the UPSC Prelims (Indian Civil Services examination).

Question: {question}

Options:
A) {a}
B) {b}
C) {c}
D) {d}

Answer with ONLY the letter (A, B, C, or D). Do not explain.
Answer:"""

LETTER_RE = re.compile(r"\b([ABCD])\b")


def _normalize_options(opts: dict) -> dict[str, str]:
    return {k.upper(): v for k, v in opts.items() if k.upper() in ("A", "B", "C", "D")}


def _parse_letter(text: str) -> str | None:
    m = LETTER_RE.search(text.upper())
    return m.group(1) if m else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id of an MLX-compatible base model")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260514)
    ap.add_argument("--out", type=Path, default=Path("results/pre_ft_hindi_probe.parquet"))
    args = ap.parse_args()

    df = read_table("upsc_prelims_ai_generated_que")
    df = df[df["question_hindi"].notna()
            & df["options_hindi"].apply(lambda x: isinstance(x, dict))
            & df["answer"].notna()
            & df["quality_pass_flag"].fillna(False).astype(bool)]

    if len(df) < args.n:
        print(f"[FAIL] only {len(df)} eligible Hindi MCQs available; need {args.n}")
        return 1

    sample = df.sample(n=args.n, random_state=args.seed).to_dict("records")

    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    print(f"Loading {args.model} …")
    model, tokenizer = load(args.model)
    sampler = make_sampler(temp=0.0)

    def _format(prompt_text: str) -> str:
        """Wrap the prompt in the model's chat template; disable hybrid-reasoning
        when supported (Qwen3.x), since the probe only wants a forced-choice letter."""
        messages = [{"role": "user", "content": prompt_text}]
        try:
            return tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
            )

    rows: list[dict] = []
    for i, r in enumerate(sample, 1):
        opts = _normalize_options(r["options_hindi"])
        if set(opts.keys()) != {"A", "B", "C", "D"}:
            continue
        prompt = PASS1_TEMPLATE.format(
            question=r["question_hindi"],
            a=opts["A"], b=opts["B"], c=opts["C"], d=opts["D"],
        )
        text = generate(model, tokenizer, prompt=_format(prompt),
                        max_tokens=24, sampler=sampler, verbose=False)
        predicted = _parse_letter(text)
        gold = r["answer"].upper().strip()
        rows.append(dict(
            model=args.model, question_id=str(r["id"]),
            gold=gold, predicted=predicted,
            is_correct=(predicted == gold),
            raw_output=text,
        ))
        if i % 20 == 0:
            running = sum(x["is_correct"] for x in rows) / len(rows)
            print(f"  [{i:>3d}/{args.n}] running acc: {running:.3f}")

    new_df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        existing = pd.read_parquet(args.out)
        existing = existing[existing["model"] != args.model]
        out_df = pd.concat([existing, new_df], ignore_index=True)
    else:
        out_df = new_df
    out_df.to_parquet(args.out, index=False, compression="snappy")

    acc = new_df["is_correct"].mean()
    correct = int(new_df["is_correct"].sum())
    print(f"\n[OK] {args.model}: {correct}/{len(new_df)} = {acc:.3f} → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
