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
INT_RE = re.compile(r"\b(\d{1,3})\b")
GEMINI_FLASH_IN_USD_PER_M = 0.50
GEMINI_FLASH_OUT_USD_PER_M = 3.00
MAX_OUT_TOKENS = {"A": 1024, "B": 1500, "C": 1024, "E": 1500}

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


# Production prompts for the capability-test tasks F and G.
# These are loaded lazily from configs/prompts/ so changes don't require
# code edits. NOT used at FT-corpus build time — Path A2 design.
PRODUCTION_TASK_INSTRUCTIONS = {
    # Task F — Prelims Explanation Generation (bilingual). Reuses Task A's
    # 800 eval items but feeds the GOLD letter as input + production prompt.
    "F": _load_production_prompt,   # loaded on demand via _load_production_prompt("prelims_explanation")
    # Task G — Mains Model-Answer Generation. Reuses Task B's 400 eval items
    # with the prayas DSL prompt (L1-L4, banned words, mandatory diagram, …).
    "G": _load_production_prompt,
}


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


def _input_for(item: EvalItem) -> dict:
    """Build the JSON-input payload that mirrors FT-corpus `input` for this task."""
    g = item.gold
    if item.task == "A":
        return {"question": g["question"],
                "options": _options_lookup(g["options"]),
                "paper": item.paper}
    if item.task == "B":
        return {"question": g["question"], "paper": item.paper, "subject": item.subject,
                "word_count": int(g.get("word_count") or 250),
                "max_score": float(g.get("max_score") or 15)}
    if item.task == "C":
        return {"question_text": g["question_text"], "answer_text": g["answer_text"],
                "max_score": float(g["max_score"])}
    if item.task == "E":
        return {"date": g["date"], "title": g["title"], "article": g["source_text"]}
    raise ValueError(item.task)


def build_prompt(item: EvalItem) -> str:
    """Same instruction + JSON input as the FT corpus saw at train time."""
    return f"{TASK_INSTRUCTIONS[item.task]}\n\n{json.dumps(_input_for(item), ensure_ascii=False)}"


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
    return None


def parse_output(task: str, raw: str) -> dict:
    """All tasks output JSON under Path A2. Fallback for Task A: regex-extract letter."""
    parsed = _extract_json(raw)
    if parsed is None:
        # Task A fallback: model may emit a bare letter despite the JSON instruction
        if task == "A":
            m = LETTER_RE.search(raw.upper())
            if m:
                return {"answer": m.group(0), "explanation": "", "_format_warn": True}
        return {"_parse_error": True}
    if task == "A":
        ans = parsed.get("answer")
        if isinstance(ans, str):
            m = LETTER_RE.search(ans.upper())
            parsed["answer"] = m.group(0) if m else None
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
        full = self._render(prompt)
        t0 = time.perf_counter()
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

    def predict(self, item: EvalItem) -> Prediction:
        pred = self._generate(build_prompt(item), MAX_OUT_TOKENS[item.task])
        pred.parsed = parse_output(item.task, pred.raw)
        return pred

    def confidence(self, item: EvalItem, letter: str) -> int:
        p = self._generate(build_confidence_prompt(item, letter), max_tokens=8)
        m = INT_RE.search(p.raw)
        return max(0, min(100, int(m.group(1)))) if m else 50


class GeminiRunner:
    """C2/C3 base. Subclasses inject few-shot exemplars (or none)."""

    def __init__(self, model: str = "gemini-3-flash"):
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) not set")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.exemplar_block: dict[str, str] = {"A": "", "B": "", "C": "", "E": ""}

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=16),
        retry=retry_if_exception_type((TimeoutError, ConnectionError)),
        reraise=True,
    )
    def _generate(self, prompt: str, max_tokens: int) -> Prediction:
        from google.genai.types import GenerateContentConfig
        cfg = GenerateContentConfig(temperature=0.0, max_output_tokens=max_tokens)
        t0 = time.perf_counter()
        ttft_ms = 0
        out_parts: list[str] = []
        in_tokens = 0
        out_tokens = 0
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
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return Prediction(raw="".join(out_parts), parsed={},
                          latency_ms=latency_ms, ttft_ms=ttft_ms,
                          input_tokens=in_tokens, output_tokens=out_tokens)

    def predict(self, item: EvalItem) -> Prediction:
        prompt = self.exemplar_block[item.task] + build_prompt(item)
        pred = self._generate(prompt, MAX_OUT_TOKENS[item.task])
        pred.parsed = parse_output(item.task, pred.raw)
        return pred

    def confidence(self, item: EvalItem, letter: str) -> int:
        p = self._generate(build_confidence_prompt(item, letter), max_tokens=8)
        m = INT_RE.search(p.raw)
        return max(0, min(100, int(m.group(1)))) if m else 50


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
    """Pre-run cost estimate (USD). Conservative — uses upper-bound token estimates."""
    df = pd.read_parquet(eval_set_path)
    in_per_task = {"A": 250, "B": 200, "C": 700, "E": 1800}
    out_per_task = {"A": 250, "B": 800, "C": 600, "E": 1100}
    few_shot_in = 1800 if few_shot else 0
    counts = df.groupby("task").size().to_dict()
    in_tok = sum(counts.get(t, 0) * (in_per_task[t] + few_shot_in) for t in ("A", "B", "C", "E"))
    out_tok = sum(counts.get(t, 0) * out_per_task[t] for t in ("A", "B", "C", "E"))
    return (in_tok / 1_000_000) * GEMINI_FLASH_IN_USD_PER_M + (out_tok / 1_000_000) * GEMINI_FLASH_OUT_USD_PER_M
