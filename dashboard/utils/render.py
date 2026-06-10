"""Shared render helpers — formatting, tables, CI bars, verdict chips."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from . import data as data_utils


def format_mean_ci(mean: float, ci_lo: float, ci_hi: float, decimals: int = 3) -> str:
    """Render a mean with 95 % CI as 'mean [lo, hi]'."""
    return f"{mean:.{decimals}f} [{ci_lo:.{decimals}f}, {ci_hi:.{decimals}f}]"


def format_delta(delta: float, ci_lo: float, ci_hi: float, decimals: int = 3) -> str:
    """Render a delta with sign + CI as '+0.038 [+0.020, +0.056]'."""
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.{decimals}f} [{ci_lo:+.{decimals}f}, {ci_hi:+.{decimals}f}]"


def winner_emoji(condition_means: dict[str, float], higher_is_better: bool) -> dict[str, str]:
    """Return a 🏆/blank label per condition based on the metric direction."""
    if not condition_means:
        return {}
    best = max(condition_means.values()) if higher_is_better else min(condition_means.values())
    return {c: "🏆" if v == best else "" for c, v in condition_means.items()}


def verdict_chip(verdict: str) -> str:
    """Map WIN/TIE/LOSS strings to colored emoji chips."""
    return {"WIN": "🟢 WIN", "TIE": "⚪ TIE", "LOSS": "🔴 LOSS"}.get(verdict, verdict)


def metric_table(
    agg: pd.DataFrame,
    task: str,
    metric: str,
    language: str = "all",
) -> pd.DataFrame:
    """Build a per-condition wide-format table for one (task, metric, language)."""
    sub = agg[
        (agg["task"] == task)
        & (agg["metric"] == metric)
        & (agg["language"] == language)
    ].copy()
    if sub.empty:
        return sub
    sub["condition_label"] = sub["condition"].map(data_utils.CONDITION_LABELS)
    sub["mean ± 95% CI"] = sub.apply(
        lambda r: format_mean_ci(r["mean"], r["ci_lo"], r["ci_hi"]), axis=1
    )
    sub = sub.sort_values("condition")
    return sub[["condition", "condition_label", "n", "mean", "ci_lo", "ci_hi", "mean ± 95% CI"]]


def render_metric_table(
    agg: pd.DataFrame, task: str, metric: str, language: str = "all", caption: str | None = None
) -> None:
    """Render one (task, metric, language) cell as a Streamlit table with winner highlight."""
    t = metric_table(agg, task, metric, language)
    if t.empty:
        st.caption(f"No data for task={task}, metric={metric}, language={language}.")
        return
    direction_known = metric in data_utils.HIGHER_IS_BETTER
    higher_better = data_utils.HIGHER_IS_BETTER.get(metric, True)
    means = dict(zip(t["condition"], t["mean"]))
    if direction_known:
        wins = winner_emoji(means, higher_better)
        t.insert(0, "🏆", t["condition"].map(wins))
    display_cols = ["🏆", "condition_label", "n", "mean ± 95% CI"] if direction_known else ["condition_label", "n", "mean ± 95% CI"]
    display = t[display_cols].rename(columns={"condition_label": "Condition", "n": "N"})
    if caption:
        st.caption(caption)
    st.dataframe(display, hide_index=True, use_container_width=True)


def language_picker(key: str, default: str = "all") -> str:
    """Sidebar radio for selecting language stratum."""
    return st.sidebar.radio(
        "Language stratum",
        options=["all", "en", "hi"],
        index=["all", "en", "hi"].index(default),
        key=key,
        horizontal=True,
    )
