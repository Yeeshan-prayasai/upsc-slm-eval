"""Minimal MCQ inference helper for in-training pulse evals.

NOT the production inference path — `scripts/runners.py` owns that,
with full JSON parsing, confidence calibration, and rate-limit handling.
This is the small, fast, deterministic variant used inside the training
loop's pulse callback: given a 4-choice question, greedy-decode a few
tokens, extract the first A/B/C/D character, compare to gold.

Why a separate helper:
- The pulse model is already in train() mode + memory; can't reuse
  the full runner stack which constructs its own model/tokenizer.
- VRAM is tight (4-bit + grad-ckpt + activations); we need
  short generations only and inference-mode dropout off.
- The pulse only needs accuracy, not the production-runner JSON
  schema or confidence numbers.
"""
from __future__ import annotations

import re
from contextlib import contextmanager

import torch


_LETTER_RE = re.compile(r"\b([A-D])\b")


@contextmanager
def _eval_no_grad(model):
    """Flip the model to eval() + disable grad/cache for the duration of
    the block, then restore train mode."""
    was_training = model.training
    model.eval()
    prev_cache = getattr(model.config, "use_cache", None)
    # gradient checkpointing + use_cache=True can be coexistent in eval,
    # but `use_cache=True` materially speeds inference.
    try:
        if prev_cache is False:
            model.config.use_cache = True
        with torch.inference_mode():
            yield
    finally:
        if prev_cache is not None:
            model.config.use_cache = prev_cache
        if was_training:
            model.train()


def format_mcq_prompt(question: str, options: "dict[str, str]") -> str:
    """Format a 4-choice MCQ into a short, model-agnostic prompt.

    Matches the in-context style used by MMLU / Big-bench: question,
    options labeled A-D, then 'Answer:' to elicit a single letter.
    """
    opts_block = "\n".join(f"{k}. {v}" for k, v in options.items())
    return f"{question}\n{opts_block}\nAnswer:"


def extract_letter(text: str) -> "str | None":
    """First A/B/C/D character in `text`. Returns None on no match."""
    m = _LETTER_RE.search(text.upper())
    return m.group(1) if m else None


def mcq_accuracy(
    model,
    tokenizer,
    items: "list[dict]",
    max_new_tokens: int = 4,
) -> "tuple[float, int]":
    """Greedy-decode a few tokens for each item and compare against gold.

    Each item must have:
      `question`, `options` (dict of A/B/C/D → text), `gold_letter`.

    Returns `(accuracy, n_evaluated)`. Returns `(0.0, 0)` on empty input.

    Per-item cost: 1 forward + max_new_tokens generation steps. With
    max_new_tokens=4 and 100 items, this is ~200ms on L40S — cheap
    enough to run mid-training as a pulse.
    """
    if not items:
        return 0.0, 0
    device = next(model.parameters()).device
    correct = 0
    n = 0
    with _eval_no_grad(model):
        for it in items:
            opts = it["options"]
            if not isinstance(opts, dict) or not all(k in opts for k in "ABCD"):
                continue
            prompt = format_mcq_prompt(it["question"], opts)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
            )
            # Slice off the prompt, decode just the generated tail.
            tail = gen[0][inputs["input_ids"].shape[1]:]
            decoded = tokenizer.decode(tail, skip_special_tokens=True)
            pred = extract_letter(decoded)
            n += 1
            if pred is not None and pred == it["gold_letter"]:
                correct += 1
    return (correct / n if n else 0.0), n
