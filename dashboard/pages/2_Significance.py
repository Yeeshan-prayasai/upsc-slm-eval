"""BH-FDR-corrected hypothesis tests + 230-cell stratum heatmap."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import data as data_utils
from utils import render

st.set_page_config(page_title="Significance — UPSC SLM v1", page_icon="🎯", layout="wide")

st.title("🎯 Statistical Significance — Paired Tests + Stratum Heatmap")
st.caption(
    "Pairwise paired-bootstrap tests with BH-FDR correction at q=0.05. "
    "Dual-test agreement (paired-t AND Wilcoxon) for significance. "
    "Effect sizes labeled per Cohen's conventions."
)

# ---------- Pairwise tests ----------
st.header("Pairwise hypothesis tests")

tests = data_utils.load_hypothesis_tests()

# Filter to the primary cells (no stratum sub-slicing — those are in the heatmap)
primary = tests[tests["stratum_dim"].astype(str).str.strip() == ""].copy()
if "stratum_dim" in primary.columns:
    primary = primary.fillna({"stratum_dim": "", "stratum_val": ""})
primary = primary.sort_values(["task", "metric", "condition_a", "condition_b"]).reset_index(drop=True)

task_filter = st.sidebar.multiselect(
    "Tasks",
    options=sorted(primary["task"].unique().tolist()),
    default=sorted(primary["task"].unique().tolist()),
    key="sig_tasks",
)
metric_filter = st.sidebar.multiselect(
    "Metrics",
    options=sorted(primary["metric"].unique().tolist()),
    default=sorted(primary["metric"].unique().tolist()),
    key="sig_metrics",
)
sig_only = st.sidebar.checkbox("Show only BH-FDR-significant rows", value=False, key="sig_only")

filtered = primary[primary["task"].isin(task_filter) & primary["metric"].isin(metric_filter)]
if sig_only:
    filtered = filtered[filtered["significant_fdr"]]

def _fmt_p(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 1e-5:
        return f"{p:.1e}"
    return f"{p:.4f}"

def _condition_label(cond: str) -> str:
    return data_utils.CONDITION_LABELS.get(cond, cond)

display = filtered.copy()
display["A − B"] = display.apply(
    lambda r: f"{_condition_label(r['condition_a'])} − {_condition_label(r['condition_b'])}", axis=1
)
display["Δ [95 % CI]"] = display.apply(
    lambda r: render.format_delta(r["mean_diff_a_minus_b"], r["diff_ci_lo"], r["diff_ci_hi"]), axis=1
)
display["p (raw)"] = display["paired_t_p"].apply(_fmt_p)
display["p (BH-FDR)"] = display["paired_t_p_fdr"].apply(_fmt_p)
display["Effect"] = display.apply(
    lambda r: f"{r['effect_size']:+.3f} ({r['effect_interpretation']})", axis=1
)
display["Sig?"] = display["significant_fdr"].apply(lambda b: "✓" if b else "")

view_cols = ["task", "metric", "A − B", "n_paired", "Δ [95 % CI]", "p (raw)", "p (BH-FDR)", "Effect", "Sig?"]
st.dataframe(
    display[view_cols].rename(columns={"task": "Task", "metric": "Metric", "n_paired": "N"}),
    hide_index=True,
    use_container_width=True,
    height=500,
)
st.caption(
    f"{len(filtered)} of {len(primary)} primary tests displayed. "
    f"BH-FDR controls the expected proportion of false discoveries at q=0.05 across all ~40 primary cells."
)

# ---------- Stratum heatmap ----------
st.header("Per-stratum heatmap — champion vs Gemini few-shot")
st.caption(
    "230 strata at (task, subject, silly_mistake_prone, language). "
    "Δ = champion mean − C3 mean on the task's primary metric. "
    "Verdicts: WIN (CI excludes 0 in champion's favor), LOSS (CI excludes 0 against), TIE (CI crosses 0)."
)

heatmap = data_utils.load_stratum_heatmap()

heatmap_task = st.selectbox(
    "Heatmap task",
    options=sorted(heatmap["task"].unique().tolist()),
    format_func=lambda t: f"{t} — {data_utils.TASK_LABELS.get(t, t)}",
    key="heatmap_task",
)
hm_sub = heatmap[heatmap["task"] == heatmap_task].copy()

if hm_sub.empty:
    st.warning(f"No heatmap data for task {heatmap_task}.")
else:
    hm_sub["Δ [95 % CI]"] = hm_sub.apply(
        lambda r: render.format_delta(r["delta_champion_minus_c3"], r["ci_lo"], r["ci_hi"]), axis=1
    )
    hm_sub["Verdict"] = hm_sub["verdict"].apply(render.verdict_chip)
    hm_sub["Champion"] = hm_sub["champion"].apply(_condition_label)
    hm_display = hm_sub[[
        "stratum_key", "primary_metric", "Champion", "n_paired",
        "champion_mean", "c3_mean", "Δ [95 % CI]", "Verdict",
    ]].rename(columns={
        "stratum_key": "Stratum",
        "primary_metric": "Metric",
        "n_paired": "N",
        "champion_mean": "Champion mean",
        "c3_mean": "C3 mean",
    })
    st.dataframe(hm_display, hide_index=True, use_container_width=True, height=500)

    counts = hm_sub["verdict"].value_counts().to_dict()
    cols = st.columns(3)
    cols[0].metric("🟢 WIN cells", counts.get("WIN", 0))
    cols[1].metric("⚪ TIE cells", counts.get("TIE", 0))
    cols[2].metric("🔴 LOSS cells", counts.get("LOSS", 0))

st.sidebar.divider()
st.sidebar.caption(
    "Source: `results/hypothesis_tests.parquet` + `results/stratum_heatmap.parquet`. "
    "Generated by `scripts/test_hypotheses.py`."
)
