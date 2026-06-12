"""Inference runners for the four eval conditions.

C1a/C1b — local MLX-LM (base + LoRA adapter), in-process.
C2/C3   — Gemini API (zero-shot / few-shot) via google-genai SDK.

All runners share a `predict(item) -> Prediction` shape so the orchestrator
in run_inference.py dispatches on --condition with one code path.
"""
from __future__ import annotations
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

LETTER_RE = re.compile(r"\b[ABCD]\b")
# Confidence parsing — see _parse_confidence below for why we explicitly
# search the LAST 1-3 digit run in the model's response.
INT_RE = re.compile(r"\b(\d{1,3})\b")


def _parse_confidence(raw: str) -> int | None:
    """Extract a 0-100 confidence integer from the model's Pass-2 response.

    The confidence prompt instructs the model to "Respond with ONLY the
    integer between 0 and 100", but Gemini sometimes echoes the prompt
    (e.g. "On a scale of 0 to 100, I'm 85% sure"). To avoid latching onto
    the echo's "0" or "100", we take the LAST matching integer in the
    response — typically the actual answer.

    Returns the parsed confidence clamped to [0, 100], or None if no usable
    integer is found. (Earlier behavior of defaulting to 50 silently biased
    Brier loss toward 0.25 for all parse failures.)
    """
    if not raw:
        return None
    matches = INT_RE.findall(raw)
    if not matches:
        return None
    return max(0, min(100, int(matches[-1])))  # LAST match — see docstring
GEMINI_FLASH_IN_USD_PER_M = 0.50
GEMINI_FLASH_OUT_USD_PER_M = 3.00
# Task A/B/C/E are FT-corpus tasks; F/G are production-prompt capability tests
# reusing Task A and Task B eval items respectively (see eval-design.md §4.6/§4.7).
# Task A bumped 1024 → 1536 — Qwen3.5's Hindi explanations averaged ~1100 tokens
# in our FT corpus, and the original 1024 cap was truncating ~30% of Hindi rows
# mid-JSON-string. _extract_json now recovers from truncation but more headroom
# is cheap and removes a known systematic bias against the Hindi stratum.
MAX_OUT_TOKENS = {"A": 1536, "B": 1500, "C": 1024, "E": 1500, "F": 1500, "G": 1500}

# Tasks F and G derive their eval items from A and B respectively.
EVAL_TASK_TO_INFERENCE_TASKS = {
    "A": ["A", "F"],   # each Task-A eval item produces both A and F predictions
    "B": ["B", "G"],   # each Task-B eval item produces both B and G predictions
    "C": ["C"],
    "E": ["E"],
}

# Path A2: the SAME instruction strings used at FT-corpus build time
# (scripts/build_ft_corpus.py TASK_INSTRUCTIONS). Train-test alignment is the
# point — the FT'd model sees identical user-message structure at both stages.
TASK_INSTRUCTIONS = {
    "A": (
        '[TASK=A] You are taking the UPSC Prelims (Indian Civil Services examination). '
        'Read the question and the four options. Return ONLY a JSON object: '
        '{"answer": "<A|B|C|D>", "explanation": "<step-by-step reasoning citing specific '
        'Article numbers / dates / scheme names; explain why the correct option is right '
        'and why each wrong option is wrong>"}'
    ),
    "B": (
        '[TASK=B] You are answering a UPSC Mains question. Write a complete answer at '
        'approximately the given word count, following UPSC structure (introduction, '
        'body with multi-dimensional analysis, conclusion). Cite specific Article numbers, '
        'dates, schemes, court cases where applicable. Return ONLY a JSON object: '
        '{"answer": "<full Mains answer text>"}'
    ),
    "C": (
        '[TASK=C] You are a UPSC Mains evaluator. Grade the student answer against the '
        'maximum marks. Return ONLY a JSON object: {"score": <float 0..max>, '
        '"strengths": [<2-4 specific strength bullets>], '
        '"improvements": {"intro": [<...>], "body": [<...>], "conclusion": [<...>]}}'
    ),
    "E": (
        '[TASK=E] You are creating UPSC study material from a news article. Produce a '
        'synthesis suitable for an aspirant. Return ONLY a JSON object: '
        '{"prelims_info": "<2-4 paragraphs of Prelims-relevant facts: scheme names, dates, '
        'key figures, definitions>", "mains_info": "<3-6 paragraphs of Mains-relevant '
        'analysis: causes, impacts, multi-dimensional framing, way forward>"}'
    ),
}


def _load_production_prompt(name: str) -> str:
    """Load a prayas production prompt from configs/prompts/ at first call.

    Cached after first load. The on-disk markdown files include explanatory
    front-matter (provenance, usage notes) which we strip — only the prompt
    body below the first horizontal rule '---' is returned.
    """
    repo_root = Path(__file__).resolve().parent.parent
    path = repo_root / "configs" / "prompts" / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(
            f"Production prompt {name!r} not found at {path}. "
            f"Configure configs/prompts/ first."
        )
    text = path.read_text(encoding="utf-8")
    # Front-matter is everything up to the first top-level horizontal rule
    # (a standalone `\n---\n` line). Don't use bare `---` — markdown table
    # separators contain `---` substrings and would split incorrectly.
    sep = "\n---\n"
    parts = text.split(sep, maxsplit=1)
    return parts[1].strip() if len(parts) == 2 else text.strip()


def get_production_prompt(task: str) -> str:
    """Resolve Task F / Task G to its prayas-canonical production prompt."""
    return {
        "F": _load_production_prompt("prelims_explanation"),
        "G": _load_production_prompt("mains_model_answer"),
    }[task]


@dataclass
class EvalItem:
    question_id: str
    task: str
    paper: str
    subject: str
    language: str
    stratum_key: str
    gold: dict

    @classmethod
    def from_row(cls, r: dict) -> "EvalItem":
        return cls(
            question_id=r["question_id"], task=r["task"],
            paper=r["paper"], subject=r["subject"],
            language=r["language"], stratum_key=r["stratum_key"],
            gold=json.loads(r["gold_payload"]),
        )


@dataclass
class Prediction:
    raw: str
    parsed: dict
    latency_ms: int
    ttft_ms: int
    input_tokens: int
    output_tokens: int
    extras: dict = field(default_factory=dict)


# ------------- Prompts -------------

def _options_lookup(opts) -> dict[str, str]:
    if isinstance(opts, dict):
        return {k.upper(): v for k, v in opts.items()}
    if isinstance(opts, list):
        return {o["id"].upper(): o["text"] for o in opts if isinstance(o, dict)}
    return {}


def _input_for(item: EvalItem, inference_task: str | None = None) -> dict:
    """Build the JSON-input payload for `inference_task` against `item`.

    `inference_task` defaults to `item.task` (i.e. eval-task == inference-task);
    pass explicitly for Tasks F + G which derive from A + B eval items but
    use different input schemas (F includes the gold answer; G is a Mains-only
    prompt without max_score).
    """
    g = item.gold
    task = inference_task or item.task
    if task == "A":
        return {"question": g["question"],
                "options": _options_lookup(g["options"]),
                "paper": item.paper}
    if task == "B":
        return {"question": g["question"], "paper": item.paper, "subject": item.subject,
                "word_count": int(g.get("word_count") or 250),
                "max_score": float(g.get("max_score") or 15)}
    if task == "C":
        return {"question_text": g["question_text"], "answer_text": g["answer_text"],
                "max_score": float(g["max_score"])}
    if task == "E":
        return {"date": g["date"], "title": g["title"], "article": g["source_text"]}
    if task == "F":
        # Task F (Prelims Explanation Generation, production prompt): given the
        # gold correct option, generate the bilingual explanation. Reuses Task A
        # eval items but the model is NOT asked to pick — it's asked to explain
        # the gold answer.
        return {
            "question": g["question"],
            "options": _options_lookup(g["options"]),
            "correct_answer": (g.get("correct_option") or "").upper(),
            "subject": item.subject,
            "paper": item.paper,
            "language": item.language,
        }
    if task == "G":
        # Task G (Mains Model-Answer Generation, production prompt): same eval
        # items as Task B, but the prayas DSL prompt expects a different
        # input shape (no `max_score`, no `paper`/`subject` JSON; the DSL
        # itself handles all of that internally from the question text).
        return {
            "question": g["question"],
            "word_count": int(g.get("word_count") or 250),
            "additional_context": "",   # prayas pipeline injects this from
                                        # a separate search step (see
                                        # configs/prompts/current_affairs_queries.md);
                                        # we leave empty for the capability test.
        }
    raise ValueError(f"unknown task: {task}")


def build_prompt(item: EvalItem, inference_task: str | None = None) -> str:
    """Build the prompt for `inference_task` against `item`.

    A/B/C/E use Path-A2 FT-corpus instructions (TASK_INSTRUCTIONS).
    F/G use prayas's production prompts loaded from configs/prompts/.
    """
    task = inference_task or item.task
    if task in TASK_INSTRUCTIONS:
        instruction = TASK_INSTRUCTIONS[task]
    elif task in ("F", "G"):
        instruction = get_production_prompt(task)
    else:
        raise ValueError(f"no prompt available for task: {task}")
    return f"{instruction}\n\n{json.dumps(_input_for(item, task), ensure_ascii=False)}"


def build_confidence_prompt(item: EvalItem, letter: str) -> str:
    """Separate Pass-2 call for verbal confidence elicitation (Task A only)."""
    return (
        f"The Prelims question was: {item.gold['question']}\n\n"
        f"You answered option {letter}.\n\n"
        f"On a scale of 0 to 100, how confident are you that {letter} is correct?\n"
        "Respond with ONLY the integer between 0 and 100.\nConfidence:"
    )


# ------------- Output parsing -------------

def _extract_json(text: str) -> dict | None:
    """Best-effort JSON recovery from possibly-truncated model output.

    Strategies, in order:
    1. Parse the full text directly.
    2. Extract `{...}` from a markdown fence (```json ... ```).
    3. Extract `{...}` from anywhere in the text (greedy `\\{.*\\}`).
    4. Recover from truncated JSON (no closing `}` because generation hit
       max_tokens mid-string). We re-quote-balance by appending `"}` for
       common shapes and retrying.

    The truncation recovery is critical for long-output cases (e.g. Qwen3.5's
    verbose Hindi Task-A explanations that can run past 1024 tokens). Without
    it we'd lose both the answer letter AND the full explanation; with it we
    keep both, just slightly truncated at the explanation tail.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    # Truncation recovery: find the LAST `{` start and try to balance.
    start = text.rfind("{")
    if start < 0:
        return None
    truncated = text[start:]
    # Append progressively-aggressive closers and retry.
    # Order: `"}` (mid-string), `}` (closed missing brace),
    # `]}` (mid-array), `"]}` (mid-array-string), etc.
    for closer in ('"}', '}', ']}', '"]}', '"}]}', '"}}', ']}}'):
        try:
            return json.loads(truncated + closer)
        except json.JSONDecodeError:
            continue
    return None


def parse_output(task: str, raw: str) -> dict:
    """Parse the raw model output for `task`.

    A/B/C/E use Path-A2 JSON instructions. Task F uses the prayas production
    prompt which also emits JSON ({"english", "hindi"}). Task G emits raw
    markdown (per the Mains DSL prompt — no JSON enforced).
    """
    if task == "G":
        # Mains DSL prompt: raw markdown body, no JSON wrapper expected.
        text = (raw or "").strip()
        return {"answer": text} if text else {"_parse_error": True}

    parsed = _extract_json(raw)
    if parsed is None:
        text = (raw or "").strip()
        if task == "A":
            # Task A fallback: model may emit a bare letter despite the JSON instruction
            m = LETTER_RE.search(raw.upper())
            if m:
                return {"answer": m.group(0), "explanation": "", "_format_warn": True}
        if task == "F":
            # Task F fallback: keep raw text under english if the model bypassed JSON.
            if text:
                return {"english": text, "hindi": "", "_format_warn": True}
        # Lenient text-key fallbacks for B/C/E — v1 saw 11-34 % parse-error rate on
        # these tasks (especially Gemma-FT on Mains-B at 34 %) where the model wrote
        # prose without the JSON wrapper. The raw text is still scoreable on
        # BERTScore / ROUGE-L / chrF++ as long as we expose it under the expected
        # field. Tagged `_format_warn` so format-validity metrics still count this
        # as a wrapper violation.
        if task == "B" and text:
            return {"answer": text, "_format_warn": True}
        if task == "E" and text:
            # Best-effort split: assume the first half is prelims, second is mains.
            # If we can't split, mirror the full text into both — better than
            # dropping the row entirely.
            return {"prelims_info": text, "mains_info": text, "_format_warn": True}
        if task == "C" and text:
            # C needs a numeric score to be useful; without one, we can only
            # preserve the rubric prose under strengths-as-text. Downstream score
            # MAE will be missing for this row.
            return {
                "score": None,
                "strengths": [text],
                "improvements": {"intro": [], "body": [], "conclusion": []},
                "_format_warn": True,
            }
        return {"_parse_error": True}
    if task == "A":
        ans = parsed.get("answer")
        # Normalize whatever the model emitted into a single letter A/B/C/D
        # (or None). Models occasionally return a list (`["C"]`) or null;
        # this guards against downstream `.strip().upper()` crashes in
        # score_task_A which expects a string.
        candidate: str | None = None
        if isinstance(ans, str):
            candidate = ans
        elif isinstance(ans, list) and ans:
            head = ans[0]
            if isinstance(head, str):
                candidate = head
        elif isinstance(ans, (int, float)) and 0 <= int(ans) <= 3:
            # Some models index options 0..3 instead of A..D.
            candidate = "ABCD"[int(ans)]
        if candidate is not None:
            m = LETTER_RE.search(candidate.upper())
            parsed["answer"] = m.group(0) if m else None
        else:
            parsed["answer"] = None
    return parsed


# ------------- Runners -------------

class ConditionRunner(Protocol):
    def predict(self, item: EvalItem) -> Prediction: ...
    def confidence(self, item: EvalItem, letter: str) -> int: ...
    def explanation(self, item: EvalItem, letter: str) -> str: ...


class MLXLoRARunner:
    """C1a/C1b — local MLX with optional LoRA adapter. Model loads once and stays warm."""

    def __init__(self, base: str, adapter: str | None = None):
        from mlx_lm import load, stream_generate
        from mlx_lm.sample_utils import make_sampler
        self.model, self.tokenizer = load(base, adapter_path=adapter)
        self.sampler = make_sampler(temp=0.0)
        self._stream = stream_generate

    def _render(self, prompt: str) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True, tokenize=False,
            )
        except (AttributeError, ValueError):
            return prompt

    def _generate(self, prompt: str, max_tokens: int) -> Prediction:
        # Start the wall-clock timer BEFORE the chat-template render so
        # latency_ms is a fair comparison against the Gemini runner (which
        # ships the prompt as a single string to the API and includes any
        # server-side templating in its measured time). Excluding the render
        # systematically under-reported MLX latency by 1-5 ms per call.
        t0 = time.perf_counter()
        full = self._render(prompt)
        ttft_ms = 0
        out_parts: list[str] = []
        in_tokens = 0
        out_tokens = 0
        for resp in self._stream(self.model, self.tokenizer, prompt=full,
                                 max_tokens=max_tokens, sampler=self.sampler):
            if ttft_ms == 0:
                ttft_ms = int((time.perf_counter() - t0) * 1000)
            out_parts.append(resp.text)
            in_tokens = getattr(resp, "prompt_tokens", in_tokens)
            out_tokens = getattr(resp, "generation_tokens", out_tokens + 1)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return Prediction(raw="".join(out_parts), parsed={},
                          latency_ms=latency_ms, ttft_ms=ttft_ms,
                          input_tokens=in_tokens, output_tokens=out_tokens)

    def predict(self, item: EvalItem, inference_task: str | None = None) -> Prediction:
        task = inference_task or item.task
        pred = self._generate(build_prompt(item, task), MAX_OUT_TOKENS[task])
        pred.parsed = parse_output(task, pred.raw)
        return pred

    def confidence(self, item: EvalItem, letter: str) -> int | None:
        """Returns a 0-100 integer confidence, or None on parse failure.
        Caller records None so score_task_A's `conf = float(raw)/100` branch
        treats it as missing rather than averaging in a silent 0.5 fake."""
        p = self._generate(build_confidence_prompt(item, letter), max_tokens=16)
        return _parse_confidence(p.raw)


class HFTransformersRunner:
    """C1a/C1b on EC2 (NVIDIA GPU) — loads the merged HF model in bf16 via
    `transformers.AutoModelForCausalLM` and generates with greedy decoding.

    This is the EC2-side counterpart to MLXLoRARunner (which targets M5 with
    MLX 4-bit). The merged HF dir was produced by scripts/merge_adapter.py
    (PEFT adapter folded into base weights, saved as safetensors). No PEFT
    wrapper at inference time — just plain transformers.

    Streaming generation captures TTFT (time-to-first-token) by yielding
    each token as it's produced; we measure `time.perf_counter()` deltas
    against the initial `generate` call start.
    """

    def __init__(self, hf_path: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._torch = torch
        print(f"[hf-runner] loading {hf_path} in bf16 ...")
        self.tokenizer = AutoTokenizer.from_pretrained(hf_path)
        # Prefer flash-attention 2 if available; falls back to sdpa, then
        # eager. flash-attention-2 gives ~3-5× generation throughput on L40S
        # for 4B-class models — worth probing.
        load_kwargs = dict(dtype=torch.bfloat16, device_map="auto")
        last_err = None
        for attn in ("flash_attention_2", "sdpa", "eager"):
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    hf_path, attn_implementation=attn, **load_kwargs,
                )
                print(f"[hf-runner] attention impl = {attn}")
                break
            except (TypeError, ValueError, ImportError) as e:
                last_err = e
                # Some transformers versions used `torch_dtype` instead of `dtype`.
                if "dtype" in str(e) or isinstance(e, TypeError):
                    load_kwargs = {k: v for k, v in load_kwargs.items() if k != "dtype"}
                    load_kwargs["torch_dtype"] = torch.bfloat16
                continue
        else:
            raise RuntimeError(f"failed to load model with any attention impl: {last_err}")
        self.model.eval()
        # Use the tokenizer's EOS as pad if pad_token_id is missing.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        print(f"[hf-runner] loaded: {type(self.model).__name__}, "
              f"{sum(p.numel() for p in self.model.parameters()) / 1e9:.2f}B params")

    def _render(self, prompt: str) -> str:
        # Pass `enable_thinking=False` to suppress Qwen3.5's `<think>...</think>`
        # reasoning preamble — without this it consumes hundreds of tokens of
        # internal reasoning before reaching the actual answer, blowing through
        # MAX_OUT_TOKENS. The flag is a no-op on tokenizers that don't support
        # it (Gemma 4), so this is safe across both architectures.
        try:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True, tokenize=False,
                enable_thinking=False,
            )
        except TypeError:
            # Older tokenizers don't accept enable_thinking — fall back.
            try:
                return self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True, tokenize=False,
                )
            except (AttributeError, ValueError):
                return prompt
        except (AttributeError, ValueError):
            return prompt

    def _generate(self, prompt: str, max_tokens: int) -> Prediction:
        from transformers import TextIteratorStreamer
        from threading import Thread

        # Include chat-template render time in latency_ms (parity with the
        # other runners; symmetric to how the Gemini runner counts every
        # millisecond from the moment we hand the prompt over).
        t0 = time.perf_counter()
        full = self._render(prompt)
        # add_special_tokens=False: `full` came from apply_chat_template,
        # which already embeds BOS where the model expects it. The default
        # (True) prepends a SECOND BOS — Gemma is documented to degrade on
        # duplicated BOS.
        enc = self.tokenizer(
            full, return_tensors="pt", add_special_tokens=False,
        ).to(self.model.device)
        in_tokens = int(enc["input_ids"].shape[-1])

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True,
        )
        gen_kwargs = dict(
            **enc,
            max_new_tokens=max_tokens,
            do_sample=False,           # greedy / temperature=0 ⇒ deterministic
            num_beams=1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
            use_cache=True,
        )
        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        out_parts: list[str] = []
        ttft_ms = 0
        for token_text in streamer:
            if ttft_ms == 0 and token_text:
                ttft_ms = int((time.perf_counter() - t0) * 1000)
            out_parts.append(token_text)
        thread.join()

        latency_ms = int((time.perf_counter() - t0) * 1000)
        raw = "".join(out_parts)
        # Output-token count via re-tokenize (avoids hooking into generate's
        # internal token list which streamer doesn't expose).
        out_tokens = int(len(self.tokenizer(raw, return_tensors="pt")["input_ids"][0])) if raw else 0
        return Prediction(raw=raw, parsed={},
                          latency_ms=latency_ms, ttft_ms=ttft_ms,
                          input_tokens=in_tokens, output_tokens=out_tokens)

    def predict(self, item: EvalItem, inference_task: str | None = None) -> Prediction:
        task = inference_task or item.task
        pred = self._generate(build_prompt(item, task), MAX_OUT_TOKENS[task])
        pred.parsed = parse_output(task, pred.raw)
        return pred

    def predict_batch(
        self,
        items: list[EvalItem],
        inference_tasks: list[str],
        max_tokens_override: int | None = None,
    ) -> list[Prediction]:
        """Batched generation — feeds N prompts into a single `model.generate`
        call to amortize GPU matmul cost. Trades per-row TTFT precision (we
        only capture batch-level first-token time, attributed to each row) for
        ~5-10× higher throughput on L40S-class hardware.

        All items in a batch are padded to the longest prompt; max_new_tokens
        is the largest needed across the batch (or max_tokens_override). The
        result list is aligned positionally with the input.
        """
        if not items:
            return []
        assert len(items) == len(inference_tasks), \
            "items and inference_tasks must have the same length"
        torch = self._torch
        prompts = [build_prompt(it, t) for it, t in zip(items, inference_tasks)]
        rendered = [self._render(p) for p in prompts]
        max_tokens = max_tokens_override or max(MAX_OUT_TOKENS[t] for t in inference_tasks)

        t0 = time.perf_counter()
        # Left-padding matters for causal LMs — without it, generated tokens
        # are emitted at positions that respect the padding-shifted logits.
        self.tokenizer.padding_side = "left"
        # add_special_tokens=False — rendered text already carries the
        # template's BOS (see _generate).
        enc = self.tokenizer(
            rendered, return_tensors="pt", padding=True, truncation=False,
            add_special_tokens=False,
        ).to(self.model.device)
        in_token_counts = [int(m.sum()) for m in enc["attention_mask"]]
        prompt_len = int(enc["input_ids"].shape[-1])

        with torch.inference_mode():
            output_ids = self.model.generate(
                **enc,
                max_new_tokens=max_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        batch_latency_ms = int((time.perf_counter() - t0) * 1000)
        # Slice off the prompt portion; decode each row's new tokens.
        new_ids = output_ids[:, prompt_len:]
        results: list[Prediction] = []
        for i, (item, task) in enumerate(zip(items, inference_tasks)):
            row_new = new_ids[i]
            # Trim trailing pads from this row's output.
            row_new = row_new[row_new != self.tokenizer.pad_token_id]
            raw = self.tokenizer.decode(row_new, skip_special_tokens=True)
            out_tok = int(len(row_new))
            pred = Prediction(
                raw=raw, parsed=parse_output(task, raw),
                latency_ms=batch_latency_ms,
                # We don't have true per-row TTFT in a batch — attribute the
                # batch's total wall-clock divided by batch size as a coarse
                # estimate. Tagged so downstream readers know it's not exact.
                ttft_ms=batch_latency_ms // max(1, len(items)),
                input_tokens=in_token_counts[i],
                output_tokens=out_tok,
                extras={"batch_size": len(items)},
            )
            results.append(pred)
        return results

    def confidence(self, item: EvalItem, letter: str) -> int | None:
        """Pass-2 verbal confidence elicitation — see MLXLoRARunner.confidence
        docstring. Returns 0-100 int or None on parse failure."""
        p = self._generate(build_confidence_prompt(item, letter), max_tokens=16)
        return _parse_confidence(p.raw)

    def confidence_batch(self, items: list[EvalItem], letters: list[str]) -> list[int | None]:
        """Batched confidence elicitation — 16-token cap per row, batched."""
        if not items:
            return []
        prompts = [build_confidence_prompt(it, l) for it, l in zip(items, letters)]
        rendered = [self._render(p) for p in prompts]
        torch = self._torch
        self.tokenizer.padding_side = "left"
        enc = self.tokenizer(
            rendered, return_tensors="pt", padding=True, truncation=False,
            add_special_tokens=False,
        ).to(self.model.device)
        prompt_len = int(enc["input_ids"].shape[-1])
        with torch.inference_mode():
            out = self.model.generate(
                **enc, max_new_tokens=16, do_sample=False, num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id, use_cache=True,
            )
        new_ids = out[:, prompt_len:]
        result: list[int | None] = []
        for i in range(len(items)):
            row = new_ids[i]
            row = row[row != self.tokenizer.pad_token_id]
            text = self.tokenizer.decode(row, skip_special_tokens=True)
            result.append(_parse_confidence(text))
        return result


# Tasks where the prompt asks the model to emit a JSON object.
# Task G is markdown (Mains DSL) — explicitly excluded so we don't fight the prompt.
JSON_RESPONSE_TASKS = frozenset({"A", "B", "C", "E", "F"})


def _gemini_safety_settings():
    """Disable Gemini safety blocks for academic / UPSC eval content.

    v1 hit a 65 % empty-output rate on Task F Gemini (521/800 rows) because
    `gemini-3.5-flash` was silently blocking the prayas production prompts —
    chunks streamed back with `chunk.text=None` and consumed output tokens
    without emitting visible content. UPSC syllabus material (Article 21,
    treaty articles, current-affairs commentary on policy) frequently
    intersects with HARM_CATEGORY_CIVIC_INTEGRITY (elections / political
    process), which defaults to BLOCK_MEDIUM_AND_ABOVE. We set every category
    to BLOCK_NONE so the eval reflects model competence, not safety-policy
    sensitivity — appropriate for benign educational content used inside a
    controlled eval pipeline.

    Documented in experiment-report.md §8.4 alongside the 65 % rate.
    """
    from google.genai.types import HarmBlockThreshold, HarmCategory, SafetySetting
    cats = [
        HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        HarmCategory.HARM_CATEGORY_HARASSMENT,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
    ]
    return [SafetySetting(category=c, threshold=HarmBlockThreshold.BLOCK_NONE) for c in cats]


class GeminiRunner:
    """C2/C3 base. Subclasses inject few-shot exemplars (or none)."""

    def __init__(self, model: str = "gemini-3-flash"):
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) not set")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        # Cached safety settings — built once per runner instance (constructors
        # for SafetySetting / HarmCategory enums are cheap but we'd otherwise
        # rebuild ~5K times per condition run).
        self._safety = _gemini_safety_settings()
        # Exemplar prefix per inference-task. F/G are kept empty by design:
        # they are capability tests of the prayas production prompts themselves,
        # not generalization tests. Adding FT-corpus exemplars (which use the
        # different Path-A2 prompts) would contaminate the prompt.
        self.exemplar_block: dict[str, str] = {
            "A": "", "B": "", "C": "", "E": "", "F": "", "G": "",
        }

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=16),
        retry=retry_if_exception_type((TimeoutError, ConnectionError)),
        reraise=True,
    )
    def _generate(
        self,
        prompt: str,
        max_tokens: int,
        response_mime_type: str | None = None,
    ) -> Prediction:
        """Stream a Gemini response.

        `response_mime_type="application/json"` activates Gemini's structured-
        output mode for JSON tasks (A/B/C/E/F) — the model guarantees valid
        JSON in `response.text`, eliminating the parse-error tail v1 saw on
        Tasks B/C (11-18 % invalid-JSON rate on Gemini).

        Also captures `finish_reason` + `prompt_feedback.block_reason` from
        the streamed candidates so the empty-output case (raw="") records
        WHY in `Prediction.extras` instead of leaving downstream metrics to
        guess.
        """
        from google.genai.types import GenerateContentConfig

        cfg_kwargs = dict(
            temperature=0.0,
            max_output_tokens=max_tokens,
            safety_settings=self._safety,
        )
        if response_mime_type:
            cfg_kwargs["response_mime_type"] = response_mime_type
        cfg = GenerateContentConfig(**cfg_kwargs)

        t0 = time.perf_counter()
        ttft_ms = 0
        out_parts: list[str] = []
        in_tokens = 0
        out_tokens = 0
        finish_reason: str | None = None
        block_reason: str | None = None
        safety_blocked: list[str] = []

        for chunk in self.client.models.generate_content_stream(
            model=self.model, contents=prompt, config=cfg,
        ):
            if ttft_ms == 0:
                ttft_ms = int((time.perf_counter() - t0) * 1000)
            if getattr(chunk, "text", None):
                out_parts.append(chunk.text)
            usage = getattr(chunk, "usage_metadata", None)
            if usage:
                in_tokens = getattr(usage, "prompt_token_count", in_tokens) or in_tokens
                out_tokens = getattr(usage, "candidates_token_count", out_tokens) or out_tokens
            # Capture finish_reason from any candidate that has one this chunk.
            # The streaming API yields finish_reason on the last chunk per
            # candidate; we just keep the most recent non-None value.
            for cand in (getattr(chunk, "candidates", None) or []):
                fr = getattr(cand, "finish_reason", None)
                if fr is not None:
                    finish_reason = getattr(fr, "name", str(fr))
                # SAFETY blocks surface here even when finish_reason != "SAFETY"
                for sr in (getattr(cand, "safety_ratings", None) or []):
                    if getattr(sr, "blocked", False):
                        safety_blocked.append(getattr(sr.category, "name", str(sr.category)))
            pf = getattr(chunk, "prompt_feedback", None)
            if pf is not None:
                br = getattr(pf, "block_reason", None)
                if br is not None:
                    block_reason = getattr(br, "name", str(br))

        latency_ms = int((time.perf_counter() - t0) * 1000)
        raw = "".join(out_parts)
        extras: dict = {}
        if finish_reason:
            extras["gemini_finish_reason"] = finish_reason
        if block_reason:
            extras["gemini_block_reason"] = block_reason
        if safety_blocked:
            extras["gemini_safety_blocked"] = sorted(set(safety_blocked))
        if not raw and (finish_reason or block_reason or safety_blocked):
            extras["gemini_empty_response"] = True

        return Prediction(
            raw=raw, parsed={},
            latency_ms=latency_ms, ttft_ms=ttft_ms,
            input_tokens=in_tokens, output_tokens=out_tokens,
            extras=extras,
        )

    def predict(self, item: EvalItem, inference_task: str | None = None) -> Prediction:
        task = inference_task or item.task
        prompt = self.exemplar_block.get(task, "") + build_prompt(item, task)
        mime = "application/json" if task in JSON_RESPONSE_TASKS else None
        pred = self._generate(prompt, MAX_OUT_TOKENS[task], response_mime_type=mime)
        pred.parsed = parse_output(task, pred.raw)
        return pred

    def confidence(self, item: EvalItem, letter: str) -> int | None:
        """Returns a 0-100 integer confidence, or None on parse failure.
        Caller records None so score_task_A's `conf = float(raw)/100` branch
        treats it as missing rather than averaging in a silent 0.5 fake."""
        # No JSON mime type — confidence prompt expects a plain integer.
        p = self._generate(build_confidence_prompt(item, letter), max_tokens=16)
        return _parse_confidence(p.raw)


class GeminiZeroShotRunner(GeminiRunner):
    pass


class GeminiFewShotRunner(GeminiRunner):
    """Prepends 3 task-matched exemplars from the FT corpus. Deterministic pick."""

    def __init__(self, ft_corpus_path: Path, model: str = "gemini-3-flash"):
        super().__init__(model=model)
        df = pd.read_parquet(ft_corpus_path)
        for task in ("A", "B", "C", "E"):
            picked = df[df["task"] == task].sort_values("pair_id").head(3)
            blocks: list[str] = []
            for _, r in picked.iterrows():
                blocks.append(
                    "=== EXAMPLE ===\n"
                    f"{r['instruction']}\n\n{r['input']}\n\n"
                    f"ANSWER:\n{r['output']}"
                )
            print(f"[C3 few-shot] task {task} exemplars: {picked['pair_id'].tolist()}")
            self.exemplar_block[task] = "\n\n".join(blocks) + "\n\n=== YOUR TURN ===\n\n"


def estimate_gemini_cost(eval_set_path: Path, few_shot: bool) -> float:
    """Pre-run cost estimate (USD). Conservative — uses upper-bound token estimates.

    Accounts for the inference-task fan-out via EVAL_TASK_TO_INFERENCE_TASKS:
    each Task-A eval row drives one A and one F call; each Task-B drives one
    B and one G call. F + G use the prayas production prompts which are
    substantially longer than the FT-corpus instructions.
    """
    df = pd.read_parquet(eval_set_path)
    # Rough upper bounds (input includes the instruction text).
    # F input ~ 2600 tokens (large prayas prompt + question + options + gold).
    # G input ~ 3500 tokens (giant Mains DSL prompt + question).
    in_per_task = {"A": 250, "B": 200, "C": 700, "E": 1800,
                   "F": 2600, "G": 3500}
    out_per_task = {"A": 250, "B": 800, "C": 600, "E": 1100,
                    "F": 700, "G": 900}
    # Few-shot exemplars are only prepended for A/B/C/E (see GeminiFewShotRunner).
    few_shot_extra = {"A": 1800, "B": 1800, "C": 1800, "E": 1800,
                      "F": 0, "G": 0} if few_shot else {k: 0 for k in in_per_task}
    counts = df.groupby("task").size().to_dict()  # eval-task counts
    in_tok = out_tok = 0
    for eval_task, n in counts.items():
        for inf_task in EVAL_TASK_TO_INFERENCE_TASKS.get(eval_task, [eval_task]):
            in_tok += n * (in_per_task[inf_task] + few_shot_extra[inf_task])
            out_tok += n * out_per_task[inf_task]
    return ((in_tok / 1_000_000) * GEMINI_FLASH_IN_USD_PER_M
            + (out_tok / 1_000_000) * GEMINI_FLASH_OUT_USD_PER_M)
