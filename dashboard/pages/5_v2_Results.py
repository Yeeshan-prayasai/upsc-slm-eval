"""v2 Results — Gemma-4-E4B CPT→SFT headline summary."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import data as data_utils  # noqa: F401

st.set_page_config(page_title="v2 Results — UPSC SLM", page_icon="🧪", layout="wide")

st.title("🧪 v2 Results — Gemma-4-E4B CPT→SFT")
st.caption(
    "Run `gemma-v2-20260617-102048` · adapter `gemma4-e4b-upsc-v2-sft/final` · "
    "evaluated over the locked 2,000-item eval set. "
    "Source: `v2-results-gemma.md`."
)

# ── headline summary ──────────────────────────────────────────────────────────

st.header("Headline results — primary objective met ✅")
st.success(
    "**CPT+SFT closed the Task-A factual-recall gap to Gemini-3-Flash zero-shot parity.** "
    "Task A accuracy: 0.645→0.884 EN (+0.239), 0.636→0.932 HI (+0.296). "
    "Generation quality held or improved across B, F, G. Every task clears its pre-registered gate."
)

# Numbers from v2-results-gemma.md (isolated v2 shard)
HEADLINE_ROWS = [
    # (label,                    v1,    v2,    gate,          verdict)
    ("Task A — Accuracy EN",     0.645, 0.884, "≥ 0.69",     "✅ MET"),
    ("Task A — Accuracy HI",     0.636, 0.932, "no-regress", "✅"),
    ("Task A — Neg-mark EN",     1.06,  1.764, "≥ 1.10",     "✅"),
    ("Task B — BERTScore",       0.833, 0.872, "≥ 0.825",    "✅ improved"),
    ("Task B — Word-count adh.", 0.086, 0.484, "≥ 0.40",     "✅ improved"),
    ("Task C — Score MAE (↓)",   1.90,  2.158, "≤ 2.20",     "⚠️ within gate by 0.042"),
    ("Task E — BERTScore",       0.873, 0.866, "≥ 0.865",    "✅ clears"),
    ("Task F — BERTScore",       0.824, 0.847, "≥ 0.814",    "✅ improved"),
    ("Task G — BERTScore",       0.745, 0.849, "≥ 0.735",    "✅ improved"),
]

rows = []
for label, v1_val, v2_val, gate, verdict in HEADLINE_ROWS:
    rows.append({
        "Metric":   label,
        "v1":       v1_val,
        "Gemma v2": v2_val,
        "Δ vs v1":  f"{v2_val - v1_val:+.3f}",
        "Gate":     gate,
        "":         verdict,
    })

st.dataframe(
    pd.DataFrame(rows),
    hide_index=True,
    use_container_width=True,
    column_config={
        "v1":       st.column_config.NumberColumn("v1",       format="%.3f", width="small"),
        "Gemma v2": st.column_config.NumberColumn("Gemma v2", format="%.3f", width="small"),
        "Δ vs v1":  st.column_config.TextColumn("Δ vs v1",   width="small"),
        "Gate":     st.column_config.TextColumn("Gate",       width="small"),
        "":         st.column_config.TextColumn("",           width="medium"),
    },
)
st.caption("Gate targets from `v2-target-metrics.md`.")

# ── key findings ──────────────────────────────────────────────────────────────

st.header("Key findings")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Wins")
    st.markdown(
        "- **Task A EN**: 0.645→0.884 — reaches Gemini-3-Flash zero-shot parity.\n"
        "- **Task A HI**: 0.636→0.932.\n"
        "- **Task A neg-mark EN**: 1.06→1.764.\n"
        "- **Task B word-count adherence**: 0.086→0.484 — v1's standout regression reversed.\n"
        "- **Task B BERTScore**: +0.039.\n"
        "- **Task F BERTScore**: 0.824→0.847.\n"
        "- **Task G BERTScore**: 0.745→0.849 (+0.104)."
    )
with col2:
    st.subheader("Marginal (still within gate)")
    st.markdown(
        "- **Task C MAE**: 1.90→2.158 — within ≤2.20 gate by 0.042. Watch in v3.\n"
        "- **Task E BERTScore**: −0.007, flat but clears ≥0.865.\n\n"
        "No task failed its gate."
    )

st.info(
    "**Production recommendation:** v2 (CPT→SFT) is the Prelims candidate — "
    "it closes the factual-recall gap v1 left open. "
    "v3 focus: push Task C MAE and Task E BERTScore further clear of their gates."
)

# ── per-row drill ─────────────────────────────────────────────────────────────

st.divider()
st.header("🔍 Per-row drill — Gemma v1 vs Gemini ZS")
st.caption(
    "Row-level outputs use the v1 eval data (the v2 prediction shard is not committed to the repo). "
    "Useful for inspecting output quality on specific questions."
)

try:
    scores   = data_utils.load_scores_tier1()
    preds    = data_utils.load_predictions()
    eval_set = data_utils.load_eval_set()
    gemma_scores  = scores[scores["condition"] == "C1a"]
    gemini_scores = scores[scores["condition"] == "C2"]
    gemma_preds   = preds[preds["condition"] == "C1a"]
    gemini_preds  = preds[preds["condition"] == "C2"]
except Exception as e:
    st.error(f"Could not load eval data: {e}")
    st.stop()

import json

task = st.sidebar.selectbox(
    "Task",
    options=sorted(eval_set["task"].unique().tolist()),
    format_func=lambda t: f"{t} — {data_utils.TASK_LABELS.get(t, t)}",
    key="v2_task",
)
lang = st.sidebar.radio("Language", ["all", "en", "hi"], index=0, horizontal=True, key="v2_lang")

filtered = eval_set[eval_set["task"] == task]
if lang != "all":
    filtered = filtered[filtered["language"] == lang]
filtered = filtered.sort_values("question_id")

if filtered.empty:
    st.warning(f"No rows for task={task}, language={lang}.")
    st.stop()

qid_opts = filtered["question_id"].tolist()
qid_labels = {
    row["question_id"]: f"{row['question_id']} — {row.get('paper','?')} / {row.get('subject','?')} / {row.get('language','?')}"
    for _, row in filtered.iterrows()
}
question_id = st.selectbox("Eval question", options=qid_opts, format_func=lambda q: qid_labels[q], key="v2_qid")

row = filtered[filtered["question_id"] == question_id].iloc[0]
mc = st.columns(4)
mc[0].metric("Task", task)
mc[1].metric("Paper", row.get("paper", "—"))
mc[2].metric("Subject", row.get("subject", "—"))
mc[3].metric("Language", row.get("language", "—"))

with st.expander("Gold payload", expanded=True):
    st.json(json.loads(row["gold_payload"]))

inf_tasks = sorted(preds[preds["question_id"] == question_id]["inference_task"].unique().tolist())
if inf_tasks:
    inf_task = st.radio("Inference task variant", inf_tasks, horizontal=True, key="v2_inf")
else:
    st.warning("No predictions for this question.")
    st.stop()

HEADLINE_COL = data_utils.HEADLINE_METRIC.get(task, "")


def _render(col, title, score_df, pred_df, qid, task):
    col.subheader(title)
    s_row = score_df[score_df["question_id"] == qid]
    p_row = pred_df[(pred_df["question_id"] == qid) & (pred_df["inference_task"] == inf_task)]
    if s_row.empty:
        col.caption("_no score row_")
        return
    s = s_row.iloc[0]
    badges = []
    if task == "A" and pd.notna(s.get("is_correct")):
        badges.append("✅ correct" if s["is_correct"] else "❌ wrong")
    if pd.notna(s.get("format_valid")):
        badges.append("📐 valid" if s["format_valid"] else "📐 invalid")
    if HEADLINE_COL and pd.notna(s.get(HEADLINE_COL)):
        badges.append(f"{HEADLINE_COL}={s[HEADLINE_COL]:.3f}")
    if badges:
        col.caption(" · ".join(badges))
    if p_row.empty:
        col.caption("_no prediction_")
        return
    pr = p_row.iloc[0]
    raw = pr.get("raw_output") or ""
    try:
        parsed = json.loads(pr["prediction"]) if isinstance(pr.get("prediction"), str) else pr.get("prediction")
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
    col.caption(f"⏱ {int(pr.get('latency_ms') or 0)} ms · 📥 {int(pr.get('input_tokens') or 0)} in · 📤 {int(pr.get('output_tokens') or 0)} out")


col_g, col_gem = st.columns(2)
_render(col_g,   "Gemma v1 (C1a)", gemma_scores,  gemma_preds,  question_id, task)
_render(col_gem, "Gemini ZS (C2)", gemini_scores, gemini_preds, question_id, task)

st.sidebar.divider()
st.sidebar.caption("Source: `v2-results-gemma.md` · per-row data from v1 eval parquets.")
