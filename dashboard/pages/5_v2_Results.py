"""v2 Results — Gemma-4-E4B CPT→SFT vs Gemini zero-shot, per-row drill.

Shows the v2 Gemma run (scores_v2_gemma.parquet) alongside Gemini ZS (C2
from scores_tier1.parquet) for a direct Gemma-v2 vs Gemini comparison.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import data as data_utils

REPO = Path(__file__).resolve().parent.parent.parent
RESULTS = REPO / "results"

st.set_page_config(page_title="v2 Results — UPSC SLM", page_icon="🧪", layout="wide")

st.title("🧪 v2 Results — Gemma-4-E4B CPT→SFT")
st.caption(
    "Run `gemma-v2-20260617-102048` · adapter `gemma4-e4b-upsc-v2-sft/final` · "
    "evaluated over the locked 2,000-item eval set. "
    "Comparator: Gemini-3-Flash zero-shot (C2) from the v1 eval run."
)


# ── data loaders ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_v2() -> pd.DataFrame:
    return pd.read_parquet(RESULTS / "scores_v2_gemma.parquet")


@st.cache_data(ttl=3600)
def load_gemini_zs() -> pd.DataFrame:
    v1 = pd.read_parquet(RESULTS / "scores_tier1.parquet")
    return v1[v1["condition"] == "C2"].copy()


@st.cache_data(ttl=3600)
def load_predictions_v2() -> pd.DataFrame:
    return pd.read_parquet(
        RESULTS / "predictions_gemma-v2-20260617-102048_C1a.parquet"
    )


@st.cache_data(ttl=3600)
def load_predictions_gemini() -> pd.DataFrame:
    p = pd.read_parquet(RESULTS / "predictions.parquet")
    return p[p["condition"] == "C2"].copy()


v2 = load_v2()
gemini = load_gemini_zs()


# ── headline summary table ────────────────────────────────────────────────────

st.header("Headline results")

HEADLINE_ROWS = [
    # (display label,           column,                   task, lang,  higher, gate)
    ("Task A — Accuracy EN",    "is_correct",             "A",  "en",  True,  "≥ 0.69"),
    ("Task A — Accuracy HI",    "is_correct",             "A",  "hi",  True,  "no-regress"),
    ("Task A — Neg-mark EN",    "upsc_neg_marking_score", "A",  "en",  True,  "≥ 1.10"),
    ("Task B — BERTScore",      "answer_bertscore_f1",    "B",  "all", True,  "≥ 0.825"),
    ("Task B — Word-count adh.","word_count_adherence",   "B",  "all", True,  "≥ 0.40"),
    ("Task C — Score MAE (↓)",  "score_abs_err",          "C",  "all", False, "≤ 2.20"),
    ("Task E — BERTScore",      "mains_bertscore_f1",     "E",  "all", True,  "≥ 0.865"),
    ("Task F — BERTScore",      "explanation_bertscore_f1","F", "all", True,  "≥ 0.814"),
    ("Task G — BERTScore",      "answer_bertscore_f1",    "G",  "all", True,  "≥ 0.735"),
]


def _mean(df: pd.DataFrame, task: str, col: str, lang: str) -> float:
    sub = df[df["task"] == task]
    if lang != "all":
        sub = sub[sub["language"] == lang]
    vals = sub[col].dropna()
    return float(vals.mean()) if len(vals) else float("nan")


summary_rows = []
for label, col, task, lang, higher, gate in HEADLINE_ROWS:
    v2_val  = _mean(v2,     task, col, lang)
    gem_val = _mean(gemini, task, col, lang)
    if not (pd.isna(v2_val) or pd.isna(gem_val)):
        delta = v2_val - gem_val
        wins  = delta > 0 if higher else delta < 0
        verdict = "✅" if wins else ("⚠️" if abs(delta) < 0.05 else "❌")
        delta_str = f"{delta:+.3f}"
    else:
        delta_str = "—"
        verdict   = "—"
    summary_rows.append({
        "Metric": label,
        "Gemma v2": round(v2_val, 3) if not pd.isna(v2_val) else None,
        "Gemini ZS": round(gem_val, 3) if not pd.isna(gem_val) else None,
        "Δ (v2 − Gem)": delta_str,
        "Gate": gate,
        "": verdict,
    })

st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)
st.caption(
    "Δ = Gemma v2 − Gemini ZS. For MAE (↓) a negative Δ means Gemma v2 has "
    "lower error (better). Gate from `v2-target-metrics.md`."
)


# ── per-row drill ─────────────────────────────────────────────────────────────

st.divider()
st.header("🔍 Per-row drill — Gemma v2 vs Gemini ZS")
st.caption("Pick a task and question to compare what each model produced, side-by-side.")

task_opts = sorted(v2["task"].unique().tolist())
task = st.sidebar.selectbox(
    "Task",
    options=task_opts,
    format_func=lambda t: f"{t} — {data_utils.TASK_LABELS.get(t, t)}",
    key="v2_task",
)
lang = st.sidebar.radio(
    "Language",
    options=["all", "en", "hi"],
    index=0,
    horizontal=True,
    key="v2_lang",
)

task_rows = v2[v2["task"] == task].copy()
if lang != "all":
    task_rows = task_rows[task_rows["language"] == lang]

if task_rows.empty:
    st.warning(f"No v2 rows for task={task}, language={lang}.")
    st.stop()

task_rows = task_rows.sort_values("question_id")
qid_opts   = task_rows["question_id"].tolist()
qid_labels = {
    row["question_id"]: (
        f"{row['question_id']} — "
        f"{row.get('paper','?')} / {row.get('subject','?')} / {row.get('language','?')}"
    )
    for _, row in task_rows.iterrows()
}

question_id = st.selectbox(
    "Eval question",
    options=qid_opts,
    format_func=lambda q: qid_labels[q],
    key="v2_qid",
)

v2_score_row  = v2[v2["question_id"] == question_id]
gem_score_row = gemini[gemini["question_id"] == question_id]

# Metadata strip
if not v2_score_row.empty:
    r = v2_score_row.iloc[0]
    mc = st.columns(4)
    mc[0].metric("Task", task)
    mc[1].metric("Paper", r.get("paper", "—"))
    mc[2].metric("Subject", r.get("subject", "—"))
    mc[3].metric("Language", r.get("language", "—"))

# Load prediction text
try:
    preds_v2  = load_predictions_v2()
    preds_gem = load_predictions_gemini()
    has_preds = True
except Exception:
    has_preds = False

HEADLINE_COL = data_utils.HEADLINE_METRIC.get(task, "")


def _score_badges(s: pd.Series, task: str) -> str:
    badges = []
    if task == "A" and pd.notna(s.get("is_correct")):
        badges.append("✅ correct" if s["is_correct"] else "❌ wrong")
    if pd.notna(s.get("format_valid")):
        badges.append("📐 valid" if s["format_valid"] else "📐 invalid")
    if HEADLINE_COL and pd.notna(s.get(HEADLINE_COL)):
        badges.append(f"{HEADLINE_COL}={s[HEADLINE_COL]:.3f}")
    return " · ".join(badges)


def _render_condition(col, title: str, score_df: pd.DataFrame,
                      pred_df: pd.DataFrame | None, qid: str, task: str):
    col.subheader(title)
    if score_df.empty:
        col.caption("_no score row_")
        return

    badge = _score_badges(score_df.iloc[0], task)
    if badge:
        col.caption(badge)

    if pred_df is not None:
        prow = pred_df[pred_df["question_id"] == qid]
        if not prow.empty:
            pr  = prow.iloc[0]
            raw = pr.get("raw_output") or ""
            try:
                parsed = (
                    json.loads(pr["prediction"])
                    if isinstance(pr.get("prediction"), str)
                    else pr.get("prediction")
                )
            except Exception:
                parsed = None

            if parsed and not (isinstance(parsed, dict) and parsed.get("_parse_error")):
                col.json(parsed)
            elif raw.strip():
                col.text(raw[:2000])
            else:
                col.caption("_empty response_")

            with col.expander("Raw output"):
                col.text(raw if raw.strip() else "(empty)")

            col.caption(
                f"⏱ {int(pr.get('latency_ms') or 0)} ms  ·  "
                f"📥 {int(pr.get('input_tokens') or 0)} in  ·  "
                f"📤 {int(pr.get('output_tokens') or 0)} out"
            )
        else:
            col.caption("_no prediction row for this question_")
    else:
        col.caption("_predictions file unavailable_")


st.subheader("Outputs")
col_v2, col_gem = st.columns(2)
_render_condition(col_v2,  "Gemma v2 (CPT→SFT)",
                  v2_score_row,
                  preds_v2  if has_preds else None,
                  question_id, task)
_render_condition(col_gem, "Gemini ZS (C2)",
                  gem_score_row,
                  preds_gem if has_preds else None,
                  question_id, task)

# ── per-metric delta for this question ───────────────────────────────────────

st.subheader("Per-metric comparison for this question")

SKIP = {
    "run_id","condition","question_id","task","language","paper","subject",
    "stratum_key","silly_mistake_prone","predicted_letter","correct_letter",
    "directive_class",
}

if not v2_score_row.empty and not gem_score_row.empty:
    v2s  = v2_score_row.iloc[0]
    gems = gem_score_row.iloc[0]
    shared = [
        c for c in v2_score_row.columns
        if c in gem_score_row.columns
        and c not in SKIP
        and pd.notna(v2s.get(c)) and pd.notna(gems.get(c))
        and isinstance(v2s[c], (int, float))
    ]
    if shared:
        cmp = pd.DataFrame({
            "Metric":     shared,
            "Gemma v2":  [round(float(v2s[c]), 4)  for c in shared],
            "Gemini ZS": [round(float(gems[c]), 4) for c in shared],
            "Δ":         [round(float(v2s[c]) - float(gems[c]), 4) for c in shared],
        })
        st.dataframe(cmp, hide_index=True, use_container_width=True)
    else:
        st.caption("No shared numeric metrics with data for this question.")
else:
    st.caption("Score rows missing for one or both models on this question.")

st.sidebar.divider()
st.sidebar.caption(
    "Sources: `results/scores_v2_gemma.parquet` · "
    "`results/scores_tier1.parquet` (C2) · "
    "`results/predictions_gemma-v2-20260617-102048_C1a.parquet`."
)
