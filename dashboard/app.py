"""SLM v1 Dashboard — Home.

Launch:
    streamlit run dashboard/app.py

Or via Make:
    make dashboard
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure `utils` is importable regardless of how Streamlit invokes us
# (CLI `streamlit run` adds the script's dir; AppTest does not).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import data as data_utils  # noqa: E402

st.set_page_config(
    page_title="UPSC SLM v1 — Results Dashboard",
    page_icon="🎓",
    layout="wide",
)

st.title("UPSC SLM v1 — Evaluation Dashboard")
st.caption("UPSC-fine-tuned 4B open-source SLMs vs Google's gemini-3.5-flash. "
           "Six tasks × four conditions × ~12,800 scored rows.")

# ---------- Headline numbers ----------
st.header("v1 outcome — 3 of 4 core tasks WIN")
st.markdown(
    "The pre-registered headline ('FT-SLM beats or matches few-shot Gemini "
    "on ≥3 of 4 core tasks') is **partially confirmed**. Per-task champion "
    "vs few-shot Gemini-3.5-flash:"
)

headline = data_utils.headline_with_significance()
if not headline.empty:
    # Format the float-mean columns to 3dp for display (keep numeric for sort)
    fmt_cols = ["Gemma-FT", "Qwen-FT", "Gemini ZS", "Gemini FS"]
    styled = headline.style.format({c: "{:.3f}" for c in fmt_cols})
    st.dataframe(styled, hide_index=True, use_container_width=True)
    st.caption(
        "**Δ (Champ − FS)** is sign-normalized: positive = champion beats Gemini few-shot, "
        "for all metrics including Task C's `score_abs_err` (lower-is-better). "
        "**p (BH-FDR)** is the Benjamini-Hochberg-corrected paired-bootstrap p across all ~40 "
        "primary cells; **Sig? ✓** means the dual-test (paired-t AND Wilcoxon) both clear q=0.05. "
        "**Effect** is Cohen's d/h with the conventional small/medium/large label. "
        "All numbers are the language=all stratum — see **Results** for per-language breakdown."
    )

# ---------- Findings ----------
st.header("Key findings")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Where the FT-SLM wins")
    st.markdown(
        "- **Task B** Mains generation: Gemma BERTScore 0.833 vs Gemini few-shot 0.795 (+0.038, Cohen's d=0.21).\n"
        "- **Task C** Rubric grading: Qwen Score MAE **1.90 vs Gemini 2.52** — halves the error. QWK 0.778.\n"
        "- **Task E** Current affairs: Qwen Mains BERTScore 0.873 vs Gemini 0.851 (d=0.92, large).\n"
        "- **Task F** Prelims explanation (prod prompt): 3.6× higher distractor coverage than Gemini.\n"
        "- **Task G** Mains DSL (prod prompt): 2.8× more PESEE-dimension coverage than Gemini."
    )
with col2:
    st.subheader("Where the FT-SLM loses")
    st.markdown(
        "- **Task A** Prelims MCQ: Gemini wins by **23 pp EN** and **30-50 pp HI** (d up to −1.19).\n"
        "- **Qwen Hindi catastrophe**: 0.426 accuracy — below the binomial-gate threshold from §6.2.\n"
        "- **Format validity 0.61-0.70 universal** — below the 0.90 production threshold.\n"
        "- **ECE 0.37-0.89** — verbal confidence elicitation is broken across all four conditions."
    )

st.header("What this means for production")
st.markdown(
    "**Hybrid deployment** captures most of the win at a fraction of the API cost:\n"
    "- Route **Task A (Prelims MCQ) + Hindi-heavy queries** through Gemini API.\n"
    "- Route **Tasks B/C/E/F/G** through the FT-SLM (zero per-query cost).\n\n"
    "v2 is in scope: continued pretraining on NCERT + reference books + current affairs "
    "(see [`v2-methodology.md`](v2-methodology.md))."
)

st.header("Navigate")
st.markdown(
    "- **📊 Results** — full Tier-1 metric tables per task, with per-language breakdown\n"
    "- **🎯 Significance** — BH-FDR-corrected pairwise tests + 230-cell stratum heatmap\n"
    "- **🔍 Per-Row Drill** — pick a question_id and see the four conditions side-by-side\n"
    "- **🎮 Playground** — type a new question and run it through all four conditions live"
)

st.sidebar.title("About")
st.sidebar.markdown(
    "**Models compared**\n"
    "- C1a: `google/gemma-4-E4B-it` + LoRA (4B, MLX 4-bit)\n"
    "- C1b: `Qwen/Qwen3.5-4B` + LoRA (4B, MLX 4-bit)\n"
    "- C2: `gemini-3.5-flash` zero-shot\n"
    "- C3: `gemini-3.5-flash` few-shot (3 exemplars)\n\n"
    "**Eval set**: 2,000 stratified UPSC items (English + Hindi), frozen and SHA-pinned.\n\n"
    "**Statistical protocol**: paired bootstrap CI, paired-t + Wilcoxon dual-test, "
    "BH-FDR at q=0.05.\n\n"
    "**Source docs**: `experiment-report.md`, `eval-design.md`, `project-brief.md`."
)
