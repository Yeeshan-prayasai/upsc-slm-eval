"""Playground — type a new UPSC question and run all four conditions live.

Resource model on a 16 GB Mac:
- Default: only Gemini API conditions checked (network only, no local RAM).
- Toggling C1a (Gemma) or C1b (Qwen) lazy-loads the MLX model on demand.
- `max_entries=1` in utils/models.py guarantees only ONE MLX model resident
  at a time. Switching from Gemma to Qwen evicts Gemma automatically.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import streamlit as st

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The Playground runs models live — needs `mlx_lm` (MLX-only, Apple Silicon)
# + `google-generativeai` + checkpoints. On environments without these
# (e.g. Streamlit Cloud), surface a friendly message instead of a stack
# trace and stop the page early. The other three dashboard pages are
# read-only on parquet files and work on any Python env.
try:
    from runners import EvalItem  # noqa: E402
    from utils import data as data_utils  # noqa: E402
    from utils import models as model_utils  # noqa: E402
except ImportError as e:
    import streamlit as _st
    _st.set_page_config(page_title="Playground — UPSC SLM v1", page_icon="🎮")
    _st.title("🎮 Playground")
    _st.warning(
        f"**Playground unavailable in this deployment.**\n\n"
        f"This page runs live inference against local Gemma/Qwen MLX adapters "
        f"and the Gemini API. The deployed dashboard doesn't ship those "
        f"dependencies — it's a read-only viewer over the v1 result parquets.\n\n"
        f"To run the Playground:\n"
        f"1. Clone the repo locally (Apple Silicon Mac required for MLX).\n"
        f"2. `pip install -r requirements.txt`\n"
        f"3. Set `GEMINI_API_KEY` in your environment.\n"
        f"4. `streamlit run dashboard/app.py`\n\n"
        f"_Missing module: `{e.name}`_"
    )
    _st.stop()

st.set_page_config(
    page_title="Playground — UPSC SLM v1",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎮 Playground — Live 4-Condition Runner")
st.caption(
    "Type a UPSC question and run it through Gemma-FT, Qwen-FT, "
    "Gemini zero-shot, and Gemini few-shot. Compare outputs side-by-side."
)

# ---------- Main pane: task picker ----------
TASK_HELP = {
    "A": "Prelims MCQ — pick A/B/C/D + explanation.",
    "B": "Mains generation — write the model answer at the target word count.",
    "C": "Rubric grading — score a student answer against a max-score rubric.",
    "E": "Current affairs — produce prelims_info + mains_info from an article.",
    "F": "Prelims explanation (production prompt). Same eval items as A.",
    "G": "Mains model answer (production prompt). Same eval items as B.",
}

task_col, env_col = st.columns([2, 3])
with task_col:
    task = st.selectbox(
        "Task",
        options=list(TASK_HELP.keys()),
        format_func=lambda t: f"{t} — {data_utils.TASK_LABELS[t]}",
        help="What kind of UPSC task should the models attempt?",
        key="pg_task_main",
    )
    st.caption(TASK_HELP[task])

with env_col:
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        st.warning(
            "⚠️ `GEMINI_API_KEY` not set — C2 and C3 will fail. "
            "Add to `.env` at repo root: `GEMINI_API_KEY=AIza...`, then re-launch via `make dashboard`."
        )

# ---------- Main pane: condition checkboxes (4 columns, prominent) ----------
st.subheader("Conditions to run")
cond_cols = st.columns(4)
run_c1a = cond_cols[0].checkbox("**C1a** — Gemma-FT (MLX, ~5 GB)", value=True, key="pg_c1a_main")
run_c1b = cond_cols[1].checkbox("**C1b** — Qwen-FT (MLX, ~3 GB)", value=True, key="pg_c1b_main")
run_c2 = cond_cols[2].checkbox("**C2** — Gemini zero-shot (API)", value=True, key="pg_c2_main")
run_c3 = cond_cols[3].checkbox("**C3** — Gemini few-shot (API)", value=True, key="pg_c3_main")
st.caption(
    "MLX models lazy-load on first use (~10-15 s each). Only one MLX model is "
    "resident at a time; Streamlit auto-evicts the previous one when you switch."
)

st.sidebar.divider()
if st.sidebar.button("Free MLX memory", help="Evict any loaded MLX model from cache."):
    model_utils.evict_mlx()
    st.sidebar.success("MLX cache cleared.")

# ---------- Task-specific input form ----------
st.header("Input")

PAPERS = ["GS-I", "GS-II", "GS-III", "GS-IV", "CSAT"]
SUBJECTS_HINT = "e.g. Polity, Economy, History, Geography, Environment, Science & Tech"

with st.form("playground_input", clear_on_submit=False):
    if task in ("A", "F"):
        question_text = st.text_area("Question", height=120, key="pg_q_A",
                                     placeholder="With reference to Indian Constitution, …")
        opt_cols = st.columns(4)
        opts = {}
        for letter, col in zip("ABCD", opt_cols):
            opts[letter] = col.text_input(f"Option {letter}", key=f"pg_opt_{letter}")
        col_l, col_p, col_s, col_c = st.columns(4)
        language = col_l.selectbox("Language", options=["en", "hi"], key="pg_lang_A")
        paper = col_p.selectbox("Paper", options=PAPERS, key="pg_paper_A")
        subject = col_s.text_input("Subject", placeholder=SUBJECTS_HINT, key="pg_subj_A")
        correct = col_c.selectbox(
            "Correct option",
            options=["", "A", "B", "C", "D"],
            help="Required for Task F (explain the gold). Leave blank for Task A.",
            key="pg_correct_A",
        ) if task == "F" else ""
    elif task in ("B", "G"):
        question_text = st.text_area("Question", height=120, key="pg_q_B",
                                     placeholder="Discuss the implications of …")
        col_wc, col_p, col_s, col_l = st.columns(4)
        word_count = col_wc.number_input("Word count target", min_value=50, max_value=2000,
                                         value=250, step=50, key="pg_wc_B")
        paper = col_p.selectbox("Paper", options=PAPERS, key="pg_paper_B")
        subject = col_s.text_input("Subject", placeholder=SUBJECTS_HINT, key="pg_subj_B")
        language = col_l.selectbox("Language", options=["en", "hi"], key="pg_lang_B")
        max_score = st.number_input("Max score", min_value=1, max_value=50, value=15, key="pg_ms_B")
    elif task == "C":
        question_text = st.text_area("Question text", height=120, key="pg_q_C")
        answer_text = st.text_area("Student answer text", height=200, key="pg_a_C")
        col_ms, col_l = st.columns(2)
        max_score = col_ms.number_input("Max score", min_value=1, max_value=50, value=15, key="pg_ms_C")
        language = col_l.selectbox("Language", options=["en", "hi"], key="pg_lang_C")
        paper = "GS-II"  # Task C doesn't use paper but EvalItem requires it
        subject = ""
    elif task == "E":
        col_d, col_t = st.columns([1, 3])
        date = col_d.date_input("Article date", key="pg_date_E")
        title = col_t.text_input("Article title", key="pg_title_E",
                                 placeholder="India unveils new National Education Policy")
        article = st.text_area("Article text", height=300, key="pg_article_E",
                               placeholder="Paste the full news article body here …")
        language = st.selectbox("Language", options=["en", "hi"], key="pg_lang_E")
        paper = "GS-II"
        subject = ""

    submit = st.form_submit_button("▶ Run", type="primary")

# ---------- Build EvalItem from form ----------
def _build_item() -> EvalItem:
    """Construct a synthetic EvalItem from the form inputs.

    The gold payload here is the *input* — there is no ground-truth answer in
    Playground mode. Runners only consume the input fields (via _input_for in
    scripts/runners.py); they don't need actual gold answers to generate.
    """
    if task in ("A", "F"):
        gold = {
            "question": question_text,
            "options": {l: opts[l] for l in "ABCD" if opts[l].strip()},
        }
        if task == "F" and correct:
            gold["correct_option"] = correct
        return EvalItem(
            question_id="playground",
            task=task,
            paper=paper,
            subject=subject,
            language=language,
            stratum_key=f"{paper}|{subject}|silly=0|{language}",
            gold=gold,
        )
    if task in ("B", "G"):
        gold = {"question": question_text, "word_count": int(word_count), "max_score": float(max_score) if task == "B" else 15.0}
        return EvalItem(
            question_id="playground", task=task, paper=paper, subject=subject,
            language=language, stratum_key=f"{paper}|{subject}|silly=0|{language}",
            gold=gold,
        )
    if task == "C":
        gold = {
            "question_text": question_text,
            "answer_text": answer_text,
            "max_score": float(max_score),
        }
        return EvalItem(
            question_id="playground", task="C", paper="GS-II", subject="",
            language=language, stratum_key=f"GS-II||silly=0|{language}",
            gold=gold,
        )
    if task == "E":
        gold = {"date": str(date), "title": title, "source_text": article}
        return EvalItem(
            question_id="playground", task="E", paper="GS-II", subject="",
            language=language, stratum_key=f"GS-II||silly=0|{language}",
            gold=gold,
        )
    raise ValueError(f"unsupported task {task!r}")


def _validate() -> str | None:
    """Returns an error message if inputs are incomplete; None if OK."""
    if task in ("A", "F"):
        if not question_text.strip():
            return "Question text is required."
        if sum(1 for l in "ABCD" if opts[l].strip()) < 2:
            return "Provide at least 2 options."
        if task == "F" and not correct:
            return "Task F requires the correct option (so the model knows what to explain)."
    if task in ("B", "G"):
        if not question_text.strip():
            return "Question text is required."
    if task == "C":
        if not question_text.strip() or not answer_text.strip():
            return "Both question text and student answer text are required."
    if task == "E":
        if not article.strip():
            return "Article text is required."
    if not (run_c1a or run_c1b or run_c2 or run_c3):
        return "Pick at least one condition to run."
    return None


# ---------- Run inference ----------
if submit:
    err = _validate()
    if err:
        st.error(err)
        st.stop()

    item = _build_item()
    inference_task = task  # 1:1 for the playground

    st.divider()
    st.header("Outputs")

    # Capture conditions in display order
    plan: list[tuple[str, bool]] = [
        ("C1a", run_c1a), ("C1b", run_c1b), ("C2", run_c2), ("C3", run_c3),
    ]
    cols = st.columns(sum(1 for _, on in plan if on))
    col_iter = iter(cols)

    gemini_model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

    for cond, on in plan:
        if not on:
            continue
        col = next(col_iter)
        col.subheader(f"{cond} — {data_utils.CONDITION_LABELS[cond]}")
        try:
            t0 = time.perf_counter()
            with col.status(f"Running {cond} …", expanded=False) as status:
                if cond in ("C1a", "C1b"):
                    runner = model_utils.load_mlx_runner(cond)
                elif cond == "C2":
                    runner = model_utils.load_gemini_zs(gemini_model)
                elif cond == "C3":
                    runner = model_utils.load_gemini_fs(gemini_model)
                else:
                    raise ValueError(cond)
                pred = runner.predict(item, inference_task=inference_task)
                status.update(label=f"{cond} — done", state="complete", expanded=False)
            wall_ms = int((time.perf_counter() - t0) * 1000)
        except FileNotFoundError as e:
            col.error(f"Adapter missing: {e}")
            continue
        except Exception as e:  # noqa: BLE001 — surface arbitrary runner errors to UI
            col.error(f"{type(e).__name__}: {e}")
            continue

        # Parsed JSON output (clean view)
        if pred.parsed and not pred.parsed.get("_parse_error"):
            col.json(pred.parsed)
        else:
            col.warning("Output did not parse as expected JSON — see raw below.")

        # Raw output (collapsed)
        with col.expander("Raw model output"):
            col.text(pred.raw or "")

        # Stats
        col.caption(
            f"⏱ {pred.latency_ms} ms  ·  "
            f"📥 {pred.input_tokens} in  ·  "
            f"📤 {pred.output_tokens} out  ·  "
            f"⌛ TTFT {pred.ttft_ms} ms"
        )

st.sidebar.divider()
st.sidebar.caption(
    "Playground uses the same Runner classes as `scripts/run_inference.py`. "
    "MLX inference runs on Apple Metal; one model loaded at a time. "
    "Gemini calls require `GEMINI_API_KEY` env var."
)
