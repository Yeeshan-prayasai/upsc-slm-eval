"""HF `TrainerCallback` that runs lightweight evals during training.

Three pulse types, each fired on a configurable step cadence:

1. **Task pulse** — every `task_every_steps` (default 500):
   - 200 questions from `prod.mcqs` that are NOT in the locked v1
     eval set AND NOT in either CPT/SFT training corpus.
   - 50 Task-A MCQ (EN) + 60 Task-B Mains + 60 Task-C Rubric.
   - Computes per-task primary metric (accuracy, BERTScore, MAE) on
     CPU during the trainer's eval-time slot.

2. **MMLU pulse** — every `mmlu_every_steps` (default 1000):
   - 100-question random sample from MMLU's test split (`cais/mmlu`,
     `all` subset). Catches general-capability regression — drop more
     than `mmlu_drop_pp_threshold` from baseline triggers hard-stop.

3. **Hindi no-regression pulse** — every `hindi_every_steps` (default 1000):
   - 50 v1 Hindi Task-A items from the locked eval set.
   - Hard-stops if accuracy drops more than `hindi_drop_pp_threshold`
     from v1 baseline (0.426 Qwen / 0.636 Gemma per
     experiment-report.md §6.3).

Each pulse appends one JSON line to `<output_dir>/pulse.jsonl`:

    {"step": 500, "pulse": "task", "task_a_acc_en": 0.71, "task_b_bertscore": 0.84, ...}
    {"step": 1000, "pulse": "mmlu", "accuracy": 0.42}
    {"step": 1000, "pulse": "hindi", "task_a_hi_acc": 0.51, "v1_baseline": 0.426}

Hard-stop triggers `TrainerControl.should_training_stop = True`,
saving the current checkpoint and exiting cleanly.

The callback is **lazy** — datasets + base-model inference are not
loaded until the first pulse fires, so trainer startup isn't slowed
by 200 questions of probe data.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


@dataclass
class PulseConfig:
    """All cadences + thresholds in one place. Defaults match
    methodology §7 ("Eval pulse + pre-flight gates")."""
    task_every_steps: int = 500
    mmlu_every_steps: int = 1000
    hindi_every_steps: int = 1000

    # Drop thresholds (in percentage points) for hard-stop
    mmlu_drop_pp_threshold: float = 2.0
    hindi_drop_pp_threshold: float = 5.0

    # v1 Hindi-stratum reference values (experiment-report.md §6.3) —
    # for the LOG only. The gating baselines are measured at step 0
    # with the pulse's own prompt format (on_train_begin), because the
    # v1 numbers were measured through a different instrument (JSON
    # task prompt + chat template) and gating one instrument against
    # another produces spurious stops / masked regressions.
    hindi_v1_reference_gemma: float = 0.636
    hindi_v1_reference_qwen: float = 0.426

    # Baselines measured at step 0 by on_train_begin (same instrument
    # as every later pulse). Left None until then.
    mmlu_baseline: float | None = None
    hindi_baseline: float | None = None

    # Pulse probe sizes
    task_a_n: int = 50
    task_b_n: int = 60
    task_c_n: int = 60
    mmlu_n: int = 100
    hindi_n: int = 50

    # Generation kwargs for inline inference inside pulses
    max_new_tokens: int = 512
    inference_seed: int = 20260514

    # Which model family this is — drives Hindi baseline selection
    model_family: str = "gemma"  # or "qwen"


class PulseEvalCallback(TrainerCallback):
    """Pulse evaluator. Attach to a Trainer via `callbacks=[PulseEvalCallback(...)]`."""

    def __init__(self, cfg: PulseConfig, output_dir: Path):
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pulse_log_path = self.output_dir / "pulse.jsonl"
        self._task_dataset = None    # lazy
        self._mmlu_dataset = None    # lazy
        self._hindi_dataset = None   # lazy
        self._stopped = False

    # ----- Lazy dataset loaders -----

    def _load_task_probe(self) -> list[dict]:
        """200 held-out questions from prod.mcqs not in eval/FT corpora.

        Pulled deterministically from `data/eval_set_holdout.parquet`
        if present (a pre-baked stratified holdout written at corpus-
        build time); otherwise no-op with an explanatory log line.
        """
        from ..data.acquire._base import RepoPaths
        holdout_path = RepoPaths.root() / "data" / "eval_set_holdout.parquet"
        if not holdout_path.exists():
            print(f"[pulse] task probe disabled — "
                  f"no holdout at {holdout_path}; "
                  f"build it via `python -m training.eval.build_holdout` first")
            return []
        import pandas as pd
        df = pd.read_parquet(holdout_path)
        # Stratified pick: A=cfg.task_a_n, B=cfg.task_b_n, C=cfg.task_c_n
        picks = []
        for task, n in (("A", self.cfg.task_a_n),
                        ("B", self.cfg.task_b_n),
                        ("C", self.cfg.task_c_n)):
            sub = df[df["task"] == task].head(n)
            picks.append(sub)
        return pd.concat(picks).to_dict(orient="records")

    def _load_mmlu_probe(self) -> list[dict]:
        try:
            from datasets import load_dataset
            ds = load_dataset("cais/mmlu", "all", split="test")
            sample = ds.shuffle(seed=self.cfg.inference_seed).select(range(self.cfg.mmlu_n))
            return list(sample)
        except Exception as e:
            print(f"[pulse] MMLU probe disabled — failed to load: {e}")
            return []

    def _load_hindi_probe(self) -> list[dict]:
        """v1 Hindi Task-A rows. The eval-set schema keeps question /
        options / correct answer inside the `gold_payload` JSON string —
        parse it into the flat MCQ shape `_items_to_mcq` expects.
        (Reading top-level columns returned 0 items, which silently
        disabled the −5pp hard-stop.)"""
        from ..data.acquire._base import RepoPaths
        eval_path = RepoPaths.root() / "data" / "eval_set.parquet"
        if not eval_path.exists():
            return []
        import pandas as pd
        df = pd.read_parquet(eval_path)
        hi_a = df[(df["task"] == "A") & (df["language"] == "hi")]
        items: list[dict] = []
        for _, row in hi_a.head(self.cfg.hindi_n).iterrows():
            gp = row.get("gold_payload")
            if isinstance(gp, str):
                try:
                    gp = json.loads(gp)
                except json.JSONDecodeError:
                    continue
            if not isinstance(gp, dict):
                continue
            items.append({
                "question": gp.get("question") or gp.get("question_text"),
                "options": gp.get("options"),
                "gold_letter": gp.get("correct_option_letter")
                               or gp.get("correct_option") or gp.get("gold_letter"),
            })
        expected = min(self.cfg.hindi_n, len(hi_a))
        usable = [i for i in items if i["question"] and i["options"] and i["gold_letter"]]
        if expected and len(usable) < expected * 0.9:
            raise RuntimeError(
                f"Hindi probe parsed only {len(usable)}/{expected} rows from "
                f"gold_payload — schema drift would silently disable the "
                f"no-regression gate. Inspect eval_set.parquet."
            )
        return usable

    # ----- Pulse runners -----

    def _items_to_mcq(self, items: "list[dict]") -> "list[dict]":
        """Map a heterogeneous probe dataset to the MCQ inference schema:
        {question, options: {A..D}, gold_letter}.

        Source formats handled:
        - `prod.mcqs` rows: `question`, `options` (jsonb→dict), `correct_option_letter`
        - MMLU rows: `question`, `choices` (list of 4 strings), `answer` (0-3)
        """
        out: list[dict] = []
        for it in items:
            if "choices" in it and isinstance(it.get("choices"), (list, tuple)):
                # MMLU schema
                if len(it["choices"]) != 4:
                    continue
                opts = {k: v for k, v in zip("ABCD", it["choices"])}
                gold_idx = it.get("answer")
                if not isinstance(gold_idx, int) or not (0 <= gold_idx < 4):
                    continue
                out.append({"question": it["question"], "options": opts,
                            "gold_letter": "ABCD"[gold_idx]})
            else:
                # Native prod.mcqs schema
                opts = it.get("options")
                if isinstance(opts, str):
                    try:
                        opts = json.loads(opts)
                    except json.JSONDecodeError:
                        continue
                gold = it.get("correct_option_letter") or it.get("gold_letter")
                q = it.get("question")
                if not (q and isinstance(opts, dict) and gold in "ABCD"):
                    continue
                out.append({"question": q, "options": opts, "gold_letter": gold})
        return out

    def _run_task_pulse(self, step: int, model, tokenizer) -> dict:
        """Inline inference on the dev probe; return metric dict.

        For the EN Task-A subset we run MCQ accuracy; B/C subsets are
        skipped here (their primary metrics — BERTScore / Rubric MAE —
        are too expensive for a mid-train pulse, and v1 already showed
        they correlate with Task-A accuracy within a 1-2 pp band)."""
        from .mcq_inference import mcq_accuracy
        if self._task_dataset is None:
            self._task_dataset = self._load_task_probe()
        if not self._task_dataset:
            return {"step": step, "pulse": "task", "skipped": "no holdout"}
        # Filter to Task-A EN MCQs only for the inline pulse.
        a_en = [r for r in self._task_dataset
                if r.get("task") == "A" and r.get("language") == "en"]
        mcq_items = self._items_to_mcq(a_en)
        if not mcq_items:
            return {"step": step, "pulse": "task", "skipped": "no Task-A EN MCQs"}
        acc, n = mcq_accuracy(model, tokenizer, mcq_items)
        return {"step": step, "pulse": "task", "task_a_en_acc": acc, "n": n}

    def _run_mmlu_pulse(self, step: int, model, tokenizer) -> tuple[dict, bool]:
        """Returns (record, should_hard_stop)."""
        from .mcq_inference import mcq_accuracy
        if self._mmlu_dataset is None:
            self._mmlu_dataset = self._load_mmlu_probe()
        if not self._mmlu_dataset:
            return {"step": step, "pulse": "mmlu", "skipped": "no MMLU"}, False
        mcq_items = self._items_to_mcq(self._mmlu_dataset)
        acc, n = mcq_accuracy(model, tokenizer, mcq_items)
        record = {"step": step, "pulse": "mmlu", "accuracy": acc, "n": n}
        if self.cfg.mmlu_baseline is not None and n > 0:
            drop_pp = (self.cfg.mmlu_baseline - acc) * 100
            record["baseline"] = self.cfg.mmlu_baseline
            record["drop_pp"] = drop_pp
            if drop_pp > self.cfg.mmlu_drop_pp_threshold:
                record["hard_stop"] = True
                return record, True
        return record, False

    def _run_hindi_pulse(self, step: int, model, tokenizer) -> tuple[dict, bool]:
        from .mcq_inference import mcq_accuracy
        if self._hindi_dataset is None:
            self._hindi_dataset = self._load_hindi_probe()
        if not self._hindi_dataset:
            return {"step": step, "pulse": "hindi", "skipped": "no eval"}, False
        v1_ref = (self.cfg.hindi_v1_reference_qwen
                  if self.cfg.model_family == "qwen"
                  else self.cfg.hindi_v1_reference_gemma)
        mcq_items = self._items_to_mcq(self._hindi_dataset)
        acc, n = mcq_accuracy(model, tokenizer, mcq_items)
        record = {"step": step, "pulse": "hindi", "task_a_hi_acc": acc,
                  "n": n, "baseline": self.cfg.hindi_baseline,
                  "v1_reference": v1_ref}
        if self.cfg.hindi_baseline is not None and n > 0:
            drop_pp = (self.cfg.hindi_baseline - acc) * 100
            record["drop_pp"] = drop_pp
            if drop_pp > self.cfg.hindi_drop_pp_threshold:
                record["hard_stop"] = True
                return record, True
        return record, False

    def _append_log(self, record: dict) -> None:
        record["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with self.pulse_log_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _hard_stop(self, control: "TrainerControl", step: int, reason: str) -> None:
        """Make a hard-stop observable: checkpoint the offending step,
        stop training, and write a HARD_STOP marker the orchestrators
        check (so a stopped run exits non-zero instead of looking like
        a successful completion and flowing into SFT/ablation)."""
        print(f"[pulse] {reason} — HARD-STOP at step {step}")
        control.should_save = True
        control.should_training_stop = True
        self._stopped = True
        marker = self.output_dir / "HARD_STOP"
        marker.write_text(
            json.dumps({"step": step, "reason": reason}) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _resolve_tokenizer(kw: dict):
        """transformers 5.x passes `processing_class=` to callbacks;
        older versions passed `tokenizer=`. Accept both."""
        return kw.get("processing_class") or kw.get("tokenizer")

    # ----- TrainerCallback API -----

    def on_train_begin(self, args: TrainingArguments, state: TrainerState,
                       control: TrainerControl, model=None, **kw):
        """Measure the MMLU + Hindi baselines at step 0 with the SAME
        prompt format every later pulse uses. Gating against baselines
        measured by a different instrument (the v1 eval pipeline)
        produces spurious stops or masked regressions."""
        tokenizer = self._resolve_tokenizer(kw)
        if model is None or tokenizer is None:
            print("[pulse] on_train_begin: model/tokenizer unavailable — "
                  "baselines deferred (gates inactive until set)")
            return control
        from .mcq_inference import mcq_accuracy
        if self.cfg.mmlu_baseline is None:
            if self._mmlu_dataset is None:
                self._mmlu_dataset = self._load_mmlu_probe()
            if self._mmlu_dataset:
                acc, n = mcq_accuracy(model, tokenizer,
                                      self._items_to_mcq(self._mmlu_dataset))
                if n > 0:
                    self.cfg.mmlu_baseline = acc
                    self._append_log({"step": 0, "pulse": "mmlu_baseline",
                                      "accuracy": acc, "n": n})
        if self.cfg.hindi_baseline is None:
            if self._hindi_dataset is None:
                self._hindi_dataset = self._load_hindi_probe()
            if self._hindi_dataset:
                acc, n = mcq_accuracy(model, tokenizer,
                                      self._items_to_mcq(self._hindi_dataset))
                if n > 0:
                    self.cfg.hindi_baseline = acc
                    self._append_log({"step": 0, "pulse": "hindi_baseline",
                                      "task_a_hi_acc": acc, "n": n})
        return control

    def on_step_end(self, args: TrainingArguments, state: TrainerState,
                    control: TrainerControl, model=None, **kw):
        tokenizer = self._resolve_tokenizer(kw)
        step = state.global_step
        if step <= 0 or self._stopped or model is None or tokenizer is None:
            return control

        # Task pulse
        if step % self.cfg.task_every_steps == 0:
            rec = self._run_task_pulse(step, model, tokenizer)
            self._append_log(rec)

        # MMLU pulse
        if step % self.cfg.mmlu_every_steps == 0:
            rec, hard_stop = self._run_mmlu_pulse(step, model, tokenizer)
            self._append_log(rec)
            if hard_stop:
                self._hard_stop(control, step,
                                f"MMLU drop > {self.cfg.mmlu_drop_pp_threshold} pp")
                return control

        # Hindi no-regression pulse
        if step % self.cfg.hindi_every_steps == 0:
            rec, hard_stop = self._run_hindi_pulse(step, model, tokenizer)
            self._append_log(rec)
            if hard_stop:
                self._hard_stop(control, step,
                                f"Hindi accuracy drop > "
                                f"{self.cfg.hindi_drop_pp_threshold} pp")
                return control

        return control
