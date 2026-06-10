"""Cached parquet loaders for the dashboard.

All loaders are decorated with `@st.cache_data` so Streamlit memoizes the
DataFrame across re-runs of the same session (TTL = 1 hour). Files are read
from `results/` at the repo root, resolved via `REPO`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parent.parent.parent
RESULTS = REPO / "results"
DATA = REPO / "data"

CONDITION_LABELS = {
    "C1a": "Gemma-FT",
    "C1b": "Qwen-FT",
    "C2": "Gemini ZS",
    "C3": "Gemini FS",
}

TASK_LABELS = {
    "A": "Prelims MCQ",
    "B": "Mains generation",
    "C": "Rubric grading",
    "E": "Current affairs",
    "F": "Prelims expl. (prod)",
    "G": "Mains DSL (prod)",
}

# Headline metric per task — matches experiment-report.md §8.1
HEADLINE_METRIC = {
    "A": "is_correct",
    "B": "answer_bertscore_f1",
    "C": "score_abs_err",
    "E": "mains_bertscore_f1",
    "F": "explanation_bertscore_f1",
    "G": "answer_bertscore_f1",
}

# Direction: True = higher is better, False = lower is better
HIGHER_IS_BETTER = {
    "is_correct": True,
    "answer_bertscore_f1": True,
    "score_abs_err": False,
    "mains_bertscore_f1": True,
    "explanation_bertscore_f1": True,
    "format_fail": False,
    "brier_loss": False,
    "hallucination_rate": False,
    "word_count_adherence": True,
    "upsc_neg_marking_score": True,
}


@st.cache_data(ttl=3600)
def load_aggregate() -> pd.DataFrame:
    """Per-(condition, task, language, metric) means + 95 % bootstrap CIs."""
    return pd.read_parquet(RESULTS / "aggregate.parquet")


@st.cache_data(ttl=3600)
def load_hypothesis_tests() -> pd.DataFrame:
    """Paired t / Wilcoxon results with BH-FDR correction across ~40 cells."""
    return pd.read_parquet(RESULTS / "hypothesis_tests.parquet")


@st.cache_data(ttl=3600)
def load_scores_tier1() -> pd.DataFrame:
    """Per-(condition, question_id) scored row — 85 metric columns."""
    return pd.read_parquet(RESULTS / "scores_tier1.parquet")


@st.cache_data(ttl=3600)
def load_predictions() -> pd.DataFrame:
    """Per-(condition, question_id, inference_task) raw outputs + gold."""
    return pd.read_parquet(RESULTS / "predictions.parquet")


@st.cache_data(ttl=3600)
def load_stratum_heatmap() -> pd.DataFrame:
    """Per-stratum champion-vs-C3 deltas with WIN/TIE/LOSS verdicts."""
    return pd.read_parquet(RESULTS / "stratum_heatmap.parquet")


@st.cache_data(ttl=3600)
def load_hindi_probe() -> pd.DataFrame:
    """Pre-FT Hindi binomial-gate probe results (50 items × 2 base models)."""
    return pd.read_parquet(RESULTS / "pre_ft_hindi_probe.parquet")


@st.cache_data(ttl=3600)
def load_eval_set() -> pd.DataFrame:
    """Frozen 2K eval items — gold + question_id + task + paper + subject + language."""
    return pd.read_parquet(DATA / "eval_set.parquet")


@st.cache_data(ttl=3600)
def list_question_ids(task: str | None = None) -> list[str]:
    """Available question_ids in the eval set, optionally filtered by task."""
    eval_set = load_eval_set()
    if task:
        eval_set = eval_set[eval_set["task"] == task]
    return sorted(eval_set["question_id"].unique().tolist())


def headline_summary() -> pd.DataFrame:
    """One row per (task, condition) at the headline metric — for the home page."""
    agg = load_aggregate()
    rows = []
    for task, metric in HEADLINE_METRIC.items():
        sub = agg[(agg["task"] == task) & (agg["metric"] == metric) & (agg["language"] == "all")]
        for _, r in sub.iterrows():
            rows.append({
                "task": task,
                "task_label": TASK_LABELS[task],
                "condition": r["condition"],
                "condition_label": CONDITION_LABELS[r["condition"]],
                "metric": metric,
                "mean": r["mean"],
                "ci_lo": r["ci_lo"],
                "ci_hi": r["ci_hi"],
                "n": r["n"],
            })
    return pd.DataFrame(rows)


def _champion_for_task(task: str, metric: str, agg: "pd.DataFrame") -> str:
    """Pick the better of C1a (Gemma-FT) vs C1b (Qwen-FT) on the task's
    primary metric. Direction is set by HIGHER_IS_BETTER."""
    sub = agg[
        (agg["task"] == task)
        & (agg["metric"] == metric)
        & (agg["language"] == "all")
        & (agg["condition"].isin(["C1a", "C1b"]))
    ]
    if sub.empty:
        return "C1a"
    higher = HIGHER_IS_BETTER.get(metric, True)
    return (sub.sort_values("mean", ascending=not higher)
              .iloc[0]["condition"])


def _fmt_p(p: float) -> str:
    if pd.isna(p):
        return "—"
    if p < 1e-5:
        return f"{p:.1e}"
    return f"{p:.4f}"


def _fmt_delta(d: float, metric: str) -> str:
    """Format the champion−C3 delta with a sign that reflects 'higher is
    better' semantics. For score_abs_err (lower is better), we flip so a
    positive figure always means 'champion wins by this much'."""
    if pd.isna(d):
        return "—"
    if not HIGHER_IS_BETTER.get(metric, True):
        d = -d
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.3f}"


def headline_with_significance() -> pd.DataFrame:
    """Per-task: pick the FT-SLM champion, compare to Gemini few-shot (C3),
    surface BH-FDR-corrected p + significance verdict.

    Returns one row per task with columns:
        Task, Metric, Champion, Gemma-FT, Qwen-FT, Gemini ZS, Gemini FS,
        Δ (Champion vs FS), p (BH-FDR), Significant?, Effect

    The Δ is sign-normalized so positive always means 'champion beats C3',
    even for lower-is-better metrics like Task C's score_abs_err."""
    agg = load_aggregate()
    tests = load_hypothesis_tests()
    # Only the no-stratum, language=all primary cells
    primary = tests[
        (tests["stratum_dim"].astype(str).str.strip() == "")
        | tests["stratum_dim"].isna()
    ]
    rows = []
    for task, metric in HEADLINE_METRIC.items():
        # Per-condition means
        sub = agg[(agg["task"] == task) & (agg["metric"] == metric)
                  & (agg["language"] == "all")]
        means = {r["condition"]: r["mean"] for _, r in sub.iterrows()}
        champion = _champion_for_task(task, metric, agg)
        # Pairwise champion vs C3 (or C3 vs champion — we just need the row)
        pair = primary[
            (primary["task"] == task)
            & (primary["metric"] == metric)
            & (
                ((primary["condition_a"] == champion) & (primary["condition_b"] == "C3"))
                | ((primary["condition_a"] == "C3") & (primary["condition_b"] == champion))
            )
        ]
        if not pair.empty:
            r = pair.iloc[0]
            # Compute delta directly from the paired means rather than
            # trusting the parquet's `mean_diff_a_minus_b` column — that
            # column is misnamed in the producer (`scripts/test_hypotheses.py`)
            # and actually carries `mean_b − mean_a`. Computing from means
            # ourselves sidesteps the sign-convention bug.
            champ_mean = r["mean_a"] if r["condition_a"] == champion else r["mean_b"]
            c3_mean = r["mean_b"] if r["condition_a"] == champion else r["mean_a"]
            delta = champ_mean - c3_mean
            p_fdr = r["paired_t_p_fdr"]
            sig = bool(r["significant_fdr"])
            effect = f"{r['effect_size']:+.2f} ({r['effect_interpretation']})"
        else:
            delta = float("nan")
            p_fdr = float("nan")
            sig = False
            effect = "—"

        rows.append({
            "Task": f"{task} — {TASK_LABELS[task]}",
            "Metric": metric,
            "Champion": CONDITION_LABELS[champion],
            "Gemma-FT": means.get("C1a", float("nan")),
            "Qwen-FT": means.get("C1b", float("nan")),
            "Gemini ZS": means.get("C2", float("nan")),
            "Gemini FS": means.get("C3", float("nan")),
            "Δ (Champ − FS)": _fmt_delta(delta, metric),
            "p (BH-FDR)": _fmt_p(p_fdr),
            "Sig?": "✓" if sig else "·",
            "Effect": effect,
        })
    return pd.DataFrame(rows)
