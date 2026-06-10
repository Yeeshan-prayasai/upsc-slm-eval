"""Per-question drill-down — see what each condition produced for one question_id."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import data as data_utils

st.set_page_config(page_title="Per-Row Drill — UPSC SLM v1", page_icon="🔍", layout="wide")

st.title("🔍 Per-Row Drill")
st.caption(
    "Inspect one eval question across all four conditions side-by-side: "
    "the input given to the model, the model's output, and the gold answer. "
    "Useful for sanity-checking aggregate-level findings."
)

# Per-Row Drill reads two PII-bearing files that are intentionally
# `.gitignore`'d: `results/predictions.parquet` (model outputs on real
# UPSC eval items) and `data/eval_set.parquet` (the frozen question
# set). On a public-cloud deploy these files are absent — surface a
# friendly notice and stop the page early instead of throwing
# FileNotFoundError.
_pred_path = data_utils.RESULTS / "predictions.parquet"
_eval_path = data_utils.DATA / "eval_set.parquet"
if not _pred_path.exists() or not _eval_path.exists():
    st.warning(
        "**Per-Row Drill unavailable in this deployment.**\n\n"
        "This page reads `results/predictions.parquet` and `data/eval_set.parquet`, "
        "both of which contain real UPSC question items + per-student-derived "
        "predictions. They're gitignored for privacy and aren't shipped to the "
        "public Streamlit Cloud deploy.\n\n"
        "To run this page:\n"
        "1. Clone the repo locally.\n"
        "2. Restore the eval-set + predictions parquets from the secure store "
        "(see `scripts/freeze_eval_set.py` + `scripts/run_inference.py`).\n"
        "3. `streamlit run dashboard/app.py`"
    )
    st.stop()

predictions = data_utils.load_predictions()
scores = data_utils.load_scores_tier1()
eval_set = data_utils.load_eval_set()

# ---------- Filter eval set by task ----------
task = st.sidebar.selectbox(
    "Task",
    options=sorted(eval_set["task"].unique().tolist()),
    format_func=lambda t: f"{t} — {data_utils.TASK_LABELS.get(t, t)}",
    key="drill_task",
)
language = st.sidebar.radio(
    "Language",
    options=["en", "hi", "all"],
    index=2,
    horizontal=True,
    key="drill_lang",
)

filtered = eval_set[eval_set["task"] == task]
if language != "all":
    filtered = filtered[filtered["language"] == language]

if filtered.empty:
    st.warning(f"No eval rows for task={task}, language={language}.")
    st.stop()

# Question id picker — show subject + paper for readability
filtered = filtered.sort_values("question_id")
options = filtered["question_id"].tolist()
labels = {
    qid: f"{qid} — {row['paper']} / {row['subject']} / {row['language']}"
    for qid, (_, row) in zip(options, filtered.iterrows())
}

question_id = st.selectbox(
    "Eval question",
    options=options,
    format_func=lambda qid: labels[qid],
    key="drill_qid",
)

# ---------- Show gold ----------
row = filtered[filtered["question_id"] == question_id].iloc[0]
gold = json.loads(row["gold_payload"])

st.header(f"Question {question_id}")
meta_cols = st.columns(4)
meta_cols[0].metric("Task", task)
meta_cols[1].metric("Paper", row["paper"])
meta_cols[2].metric("Subject", row["subject"])
meta_cols[3].metric("Language", row["language"])

with st.expander("Gold payload (what the model was asked + the correct answer)", expanded=True):
    st.json(gold)

# ---------- Predictions per condition ----------
qid_preds = predictions[predictions["question_id"] == question_id]
qid_scores = scores[scores["question_id"] == question_id]

if qid_preds.empty:
    st.warning(f"No predictions found for question_id={question_id}.")
    st.stop()

inference_tasks = sorted(qid_preds["inference_task"].unique().tolist())
inference_task = st.radio(
    "Inference task variant",
    options=inference_tasks,
    horizontal=True,
    key="drill_inf_task",
    help=(
        "Tasks F and G reuse Task A / B eval items but with production prompts. "
        "Pick which prompt variant you want to drill into."
    ),
)
qid_preds = qid_preds[qid_preds["inference_task"] == inference_task]

st.header("Outputs by condition")
condition_order = ["C1a", "C1b", "C2", "C3"]
cols = st.columns(len(condition_order))
for col, cond in zip(cols, condition_order):
    sub = qid_preds[qid_preds["condition"] == cond]
    score_sub = qid_scores[qid_scores["condition"] == cond]
    col.subheader(f"{cond} — {data_utils.CONDITION_LABELS[cond]}")
    if sub.empty:
        col.caption("_no prediction_")
        continue
    pred_row = sub.iloc[0]
    # Score badges
    if not score_sub.empty:
        s = score_sub.iloc[0]
        badges = []
        if task == "A" and pd.notna(s.get("is_correct")):
            badges.append("✅ correct" if s["is_correct"] else "❌ wrong")
        if pd.notna(s.get("format_valid")):
            badges.append("📐 valid" if s["format_valid"] else "📐 invalid")
        if task != "A" and pd.notna(s.get(data_utils.HEADLINE_METRIC[task])):
            v = s[data_utils.HEADLINE_METRIC[task]]
            badges.append(f"{data_utils.HEADLINE_METRIC[task]}={v:.3f}")
        if badges:
            col.caption(" · ".join(badges))
    # Parsed output (clean view)
    try:
        parsed = json.loads(pred_row["prediction"]) if isinstance(pred_row["prediction"], str) else pred_row["prediction"]
    except (json.JSONDecodeError, TypeError):
        parsed = None
    raw_text = pred_row["raw_output"] or ""
    out_tok = int(pred_row.get("output_tokens") or 0)
    if parsed and parsed.get("_parse_error"):
        # Distinguish the two failure modes — they look the same in the JSON
        # column but mean very different things.
        if not raw_text.strip():
            col.error(
                f"**Empty response from model.** API reported {out_tok} output "
                "tokens but no text content. Common Gemini failure mode on the "
                "production prompts (Task F/G) — model refuses or emits "
                "non-text safety chunks. Parser had nothing to parse."
            )
        else:
            col.warning(
                "**Parser failed on non-empty output.** Model produced text but "
                "no JSON could be recovered. See raw output below."
            )
    elif parsed:
        col.json(parsed)
    else:
        col.caption("_no parsed output_")
    # Raw output (collapsed)
    with col.expander("Raw model output"):
        col.text(raw_text if raw_text.strip() else "(empty — model returned no text)")
    # Latency + tokens
    col.caption(
        f"⏱ {int(pred_row['latency_ms'])} ms  ·  "
        f"📥 {int(pred_row['input_tokens'])} in  ·  "
        f"📤 {int(pred_row['output_tokens'])} out"
    )

st.sidebar.divider()
st.sidebar.caption(
    "Source: `results/predictions.parquet` + `results/scores_tier1.parquet`. "
    "Predictions are the raw model outputs from `scripts/run_inference.py`."
)
