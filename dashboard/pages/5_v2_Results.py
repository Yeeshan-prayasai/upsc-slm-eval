"""v2 Results — Gemma-4-E4B CPT→SFT."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import data as data_utils

st.set_page_config(page_title="v2 Results — UPSC SLM", page_icon="🚀", layout="wide")

st.title("🚀 v2 Results — Gemma-4-E4B (CPT→SFT)")
st.caption(
    "Run `gemma-v2-20260617-102048` · adapter `gemma4-e4b-upsc-v2-sft/final` merged → bf16. "
    "Evaluated on the locked 2,000-item eval set (isolated v2 shard, not the pooled aggregate). "
    "Comparator gates from `v2-target-metrics.md`."
)

# ---------- Headline summary ----------
st.header("Headline — primary objective met ✅")
st.success(
    "**CPT+SFT closed the Task-A factual-recall gap to Gemini-3-Flash zero-shot parity.** "
    "Task A accuracy jumped 0.645→0.884 EN (+0.239) and 0.636→0.932 HI (+0.296). "
    "Generation quality (Tasks B, E, F, G) held above gate or improved. "
    "Every task clears its pre-registered gate — this is a clean positive result."
)

headline_rows = [
    {"Metric": "Task A acc EN",          "v1": 0.645, "v2": 0.884, "Δ": "+0.239", "Gate": "≥0.69",  "Verdict": "✅ MET"},
    {"Metric": "Task A acc HI",          "v1": 0.636, "v2": 0.932, "Δ": "+0.296", "Gate": "no-regress", "Verdict": "✅"},
    {"Metric": "Task A neg-mark EN",     "v1": 1.06,  "v2": 1.764, "Δ": "+0.704", "Gate": "≥1.10",  "Verdict": "✅"},
    {"Metric": "Task B BERTScore",       "v1": 0.833, "v2": 0.872, "Δ": "+0.039", "Gate": "≥0.825", "Verdict": "✅ improved"},
    {"Metric": "Task B word-count adh.", "v1": 0.086, "v2": 0.484, "Δ": "+0.398", "Gate": "≥0.40",  "Verdict": "✅ improved"},
    {"Metric": "Task C MAE (↓)",         "v1": 1.90,  "v2": 2.158, "Δ": "+0.258", "Gate": "≤2.20",  "Verdict": "⚠️ within gate by 0.042"},
    {"Metric": "Task E BERTScore",       "v1": 0.873, "v2": 0.866, "Δ": "−0.007", "Gate": "≥0.865", "Verdict": "✅ clears"},
    {"Metric": "Task F BERTScore",       "v1": 0.824, "v2": 0.847, "Δ": "+0.023", "Gate": "≥0.814", "Verdict": "✅ improved"},
    {"Metric": "Task G BERTScore",       "v1": 0.745, "v2": 0.849, "Δ": "+0.104", "Gate": "≥0.735", "Verdict": "✅ improved"},
]
headline_df = pd.DataFrame(headline_rows)
st.dataframe(
    headline_df,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Metric":   st.column_config.TextColumn("Metric",   width="medium"),
        "v1":       st.column_config.NumberColumn("v1",     format="%.3f", width="small"),
        "v2":       st.column_config.NumberColumn("v2",     format="%.3f", width="small"),
        "Δ":        st.column_config.TextColumn("Δ",        width="small"),
        "Gate":     st.column_config.TextColumn("Gate",     width="small"),
        "Verdict":  st.column_config.TextColumn("Verdict",  width="medium"),
    },
)

# ---------- Key findings ----------
st.header("Key findings")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Wins")
    st.markdown(
        "- **Task A (Prelims MCQ)**: 0.645→0.884 EN; 0.636→0.932 HI — reaches Gemini-3-Flash zero-shot parity.\n"
        "- **Task A negative-marking**: 1.06→1.764 — wide margin above ≥1.10 gate.\n"
        "- **Task B word-count adherence**: 0.086→0.484 — v1's standout regression reversed cleanly.\n"
        "- **Task B BERTScore**: 0.833→0.872 (+0.039) — generation quality improved.\n"
        "- **Task F BERTScore**: 0.824→0.847 — Prelims-explanation prompt clears gate with room.\n"
        "- **Task G BERTScore**: 0.745→0.849 (+0.104) — largest single-task BERTScore gain."
    )
with col2:
    st.subheader("Marginal regressions (still within gate)")
    st.markdown(
        "- **Task C MAE**: 1.90→2.158 — within ≤2.20 gate by 0.042. Grading-error magnitude drifted; worth watching in v3.\n"
        "- **Task E BERTScore**: 0.873→0.866 (−0.007) — essentially flat, still clears ≥0.865 gate.\n\n"
        "No task failed its gate. Significance testing (paired bootstrap + BH-FDR) was **not re-run** "
        "on the isolated v2 shard — the deltas above are point estimates from n=200–500 per task."
    )

st.header("Production recommendation")
st.info(
    "**v2 (CPT→SFT) is the Prelims candidate.** It closes the factual-recall gap v1 left open, "
    "retains all generation leads, and reverses the word-count adherence regression. "
    "Hybrid routing (v2 for all tasks; Gemini only as fallback for Hindi-heavy edge cases) "
    "is viable. v3 focus: push Task C MAE and Task E BERTScore further clear of their gates."
)

# ---------- Per-row drill ----------
st.header("Per-row drill (v2 Gemma shard)")
st.caption(
    "Inspect any eval question: v2 Gemma-FT output alongside scores. "
    "Only C1a (v2 run) is available in this shard — compare against v1 via the v1 Per-Row Drill page."
)

try:
    scores_v2 = data_utils.load_scores_v2_gemma()
    preds_v2 = data_utils.load_predictions_v2_gemma()
    eval_set = data_utils.load_eval_set()
except Exception as e:
    st.error(f"Could not load v2 data: {e}")
    st.stop()

# Sidebar filters
task = st.sidebar.selectbox(
    "Task",
    options=sorted(eval_set["task"].unique().tolist()),
    format_func=lambda t: f"{t} — {data_utils.TASK_LABELS.get(t, t)}",
    key="v2_task",
)
language = st.sidebar.radio(
    "Language",
    options=["en", "hi", "all"],
    index=2,
    horizontal=True,
    key="v2_lang",
)

filtered = eval_set[eval_set["task"] == task]
if language != "all":
    filtered = filtered[filtered["language"] == language]

if filtered.empty:
    st.warning(f"No eval rows for task={task}, language={language}.")
    st.stop()

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
    key="v2_qid",
)

row = filtered[filtered["question_id"] == question_id].iloc[0]
gold = json.loads(row["gold_payload"])

st.subheader(f"Question {question_id}")
meta_cols = st.columns(4)
meta_cols[0].metric("Task", task)
meta_cols[1].metric("Paper", row["paper"])
meta_cols[2].metric("Subject", row["subject"])
meta_cols[3].metric("Language", row["language"])

with st.expander("Gold payload", expanded=True):
    st.json(gold)

# Predictions for this question
qid_preds = preds_v2[preds_v2["question_id"] == question_id]
qid_scores = scores_v2[scores_v2["question_id"] == question_id]

if qid_preds.empty:
    st.warning(f"No v2 predictions found for question_id={question_id}.")
    st.stop()

inference_tasks = sorted(qid_preds["inference_task"].unique().tolist())
inference_task = st.radio(
    "Inference task variant",
    options=inference_tasks,
    horizontal=True,
    key="v2_inf_task",
)
qid_preds = qid_preds[qid_preds["inference_task"] == inference_task]

st.subheader("v2 Gemma-FT output (C1a)")
sub = qid_preds[qid_preds["condition"] == "C1a"] if "condition" in qid_preds.columns else qid_preds
score_sub = qid_scores[(qid_scores["condition"] == "C1a") if "condition" in qid_scores.columns else qid_scores.index >= 0]

if sub.empty:
    st.warning("No prediction row for this question/inference_task.")
    st.stop()

pred_row = sub.iloc[0]

# Score badges
if not score_sub.empty:
    s = score_sub.iloc[0]
    badges = []
    if task == "A" and pd.notna(s.get("is_correct")):
        badges.append("✅ correct" if s["is_correct"] else "❌ wrong")
    if pd.notna(s.get("format_valid")):
        badges.append("📐 valid" if s["format_valid"] else "📐 invalid")
    headline_metric = data_utils.HEADLINE_METRIC.get(task)
    if task != "A" and headline_metric and pd.notna(s.get(headline_metric)):
        v = s[headline_metric]
        badges.append(f"{headline_metric}={v:.3f}")
    if badges:
        st.caption(" · ".join(badges))

# Parsed output
try:
    parsed = json.loads(pred_row["prediction"]) if isinstance(pred_row["prediction"], str) else pred_row["prediction"]
except (json.JSONDecodeError, TypeError):
    parsed = None

raw_text = pred_row.get("raw_output") or ""
out_tok = int(pred_row.get("output_tokens") or 0)

if parsed and isinstance(parsed, dict) and parsed.get("_parse_error"):
    if not raw_text.strip():
        st.error(f"**Empty response from model.** API reported {out_tok} output tokens but no text content.")
    else:
        st.warning("**Parser failed on non-empty output.** See raw output below.")
elif parsed:
    st.json(parsed)
else:
    st.caption("_no parsed output_")

with st.expander("Raw model output"):
    st.text(raw_text if raw_text.strip() else "(empty — model returned no text)")

st.caption(
    f"⏱ {int(pred_row['latency_ms'])} ms  ·  "
    f"📥 {int(pred_row['input_tokens'])} in  ·  "
    f"📤 {int(pred_row['output_tokens'])} out"
)

st.sidebar.divider()
st.sidebar.markdown(
    "**v2 run details**\n"
    "- Run: `gemma-v2-20260617-102048`\n"
    "- Adapter: `gemma4-e4b-upsc-v2-sft/final`\n"
    "- Shard: `results/scores_v2_gemma.parquet`\n"
    "- n=3,200 scored rows (C1a only)\n\n"
    "Source: `v2-results-gemma.md`, `v2-target-metrics.md`."
)
