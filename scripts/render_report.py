"""Stage 6.4 — render the §6 + §7 tables of experiment-report.md from
results/aggregate.parquet, results/aggregate.extras.json,
results/hypothesis_tests.parquet, and results/stratum_heatmap.parquet.

Idempotent in-place table fill: parses experiment-report.md, locates each
table header by exact match, replaces the rows beneath it with values
pulled from the result store. Headers and surrounding prose are preserved.

Run with --check to verify all expected metric values are present in the
aggregate store *without* writing the report; surfaces gaps before any
table is touched.

CLI:
    python scripts/render_report.py                # write tables in-place
    python scripts/render_report.py --check        # gap audit only
    python scripts/render_report.py --report PATH  # alternate report path

Conventions:
- BLEURT-20, SummaC-ZS, Glossary recall are *deferred per Revisions item 4*
  and rendered as `N/A` with no implementation expected.
- N/A is also used for any metric × condition cell with zero scored rows.
- Numeric format: 3 decimals for [0,1] metrics; thousands-sep for counts;
  scientific for p-values < 1e-3.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
REPORT = REPO / "experiment-report.md"
AGGREGATE = REPO / "results" / "aggregate.parquet"
EXTRAS = REPO / "results" / "aggregate.extras.json"
HYPOTHESES = REPO / "results" / "hypothesis_tests.parquet"
HEATMAP = REPO / "results" / "stratum_heatmap.parquet"

CONDITIONS = ("C1a", "C1b", "C2", "C3")
COND_LABELS = {
    "C1a": "C1a (Gemma-4-E4B-it + LoRA)",
    "C1b": "C1b (Qwen3.5-4B + LoRA)",
    "C2":  "C2 (zero-shot Gemini-3-Flash)",
    "C3":  "C3 (few-shot Gemini-3-Flash)",
}
DEFERRED = {"bleurt_20", "summac_zs", "glossary_recall"}


# ---------- helpers ----------

def _fmt(val: Any, kind: str = "ratio") -> str:
    """Format a metric cell. NaN → '—' (table-empty)."""
    if val is None:
        return "—"
    if isinstance(val, float) and (np.isnan(val) or val is None):
        return "—"
    try:
        f = float(val)
    except (TypeError, ValueError):
        return str(val)
    if np.isnan(f):
        return "—"
    if kind == "pct":          return f"{100 * f:.1f}"
    if kind == "ms":           return f"{f:.0f}"
    if kind == "tps":          return f"{f:.1f}"
    if kind == "cost":         return f"${f:.4f}"
    if kind == "count":        return f"{int(round(f)):,}"
    if kind == "pvalue":
        return f"{f:.2e}" if f < 1e-3 else f"{f:.3f}"
    return f"{f:.3f}"


def _lookup(agg: pd.DataFrame, condition: str, task: str, metric: str,
            language: str = "all") -> float:
    """Pull `metric` mean for (condition, task, language). Returns NaN if missing."""
    if metric in DEFERRED:
        return np.nan
    row = agg[(agg["condition"] == condition) & (agg["task"] == task)
              & (agg["language"] == language) & (agg["metric"] == metric)]
    if row.empty:
        return np.nan
    return float(row["mean"].iloc[0])


def _build_row(label: str, values: list[tuple[Any, str]]) -> str:
    cells = " | ".join(_fmt(v, k) for v, k in values)
    return f"| {label} | {cells} |"


# ---------- §6.3 tables ----------

def _t6a_correctness(agg: pd.DataFrame) -> list[str]:
    lines = ["| Condition | Accuracy (en) | Accuracy (hi) | UPSC neg-mark score | ECE | Brier | Refusal rate |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for c in CONDITIONS:
        lines.append(_build_row(COND_LABELS[c], [
            (_lookup(agg, c, "A", "is_correct", "en"), "ratio"),
            (_lookup(agg, c, "A", "is_correct", "hi"), "ratio"),
            (_lookup(agg, c, "A", "upsc_neg_marking_score"), "ratio"),
            (_lookup(agg, c, "A", "ece_15bin"), "ratio"),
            (_lookup(agg, c, "A", "brier_loss"), "ratio"),
            (_lookup(agg, c, "A", "refusal_rate"), "ratio"),
        ]))
    return lines


def _t6a_explanation(agg: pd.DataFrame) -> list[str]:
    lines = ["| Condition | Expl. BERTScore-F1 | Expl. ROUGE-L | Expl. Entity-F1 | Distractor coverage | Reasoning-step density | Article/scheme citation acc. | Position-bias χ² p | Sentence-len variance |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in CONDITIONS:
        lines.append(_build_row(c, [
            (_lookup(agg, c, "A", "explanation_bertscore_f1"), "ratio"),
            (_lookup(agg, c, "A", "explanation_rouge_l_f1"), "ratio"),
            (_lookup(agg, c, "A", "explanation_entity_f1"), "ratio"),
            (_lookup(agg, c, "A", "distractor_coverage"), "ratio"),
            (_lookup(agg, c, "A", "reasoning_step_density_per100w"), "ratio"),
            (_lookup(agg, c, "A", "citation_accuracy"), "ratio"),
            (_lookup(agg, c, "A", "position_bias_p_value"), "pvalue"),
            (_lookup(agg, c, "A", "sentence_length_variance"), "ratio"),
        ]))
    return lines


def _t6b(agg: pd.DataFrame) -> list[str]:
    lines = ["| Condition | BERTScore-F1 | BLEURT-20 | ROUGE-L F1 | chrF++ | Word-count adh. | Entity-F1 | Hindi code-mix | MATTR | F-K grade | Paragraph adh. | 4-gram rep. rate | UPSC fact prec. |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in CONDITIONS:
        lines.append(_build_row(c, [
            (_lookup(agg, c, "B", "answer_bertscore_f1"), "ratio"),
            (np.nan, "ratio"),  # BLEURT-20 deferred per Revisions item 4
            (_lookup(agg, c, "B", "answer_rouge_l_f1"), "ratio"),
            (_lookup(agg, c, "B", "answer_chrf"), "ratio"),
            (_lookup(agg, c, "B", "word_count_adherence"), "ratio"),
            (_lookup(agg, c, "B", "entity_f1"), "ratio"),
            (_lookup(agg, c, "B", "hindi_code_mixing_rate"), "ratio"),
            (_lookup(agg, c, "B", "mattr_100"), "ratio"),
            (_lookup(agg, c, "B", "flesch_kincaid_grade"), "ratio"),
            (_lookup(agg, c, "B", "paragraph_count_adherence"), "ratio"),
            (_lookup(agg, c, "B", "ngram4_repetition_rate"), "ratio"),
            (_lookup(agg, c, "B", "fact_lookup_precision"), "ratio"),
        ]))
    return lines


def _t6c(agg: pd.DataFrame) -> list[str]:
    # Per-criterion κ here = intro/body/conclusion improvements F1 trio
    # (clarified via AskUserQuestion 2026-05-29). Render as 'I/B/C' triplet.
    lines = ["| Condition | QWK vs gold | Score MAE | Spearman ρ | Per-criterion F1 (I/B/C) | Strengths F1 | Improvements F1 | Score var. ratio | JSON schema valid | Item-count adh. |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in CONDITIONS:
        intro = _lookup(agg, c, "C", "improvements_intro_token_f1")
        body = _lookup(agg, c, "C", "improvements_body_token_f1")
        conc = _lookup(agg, c, "C", "improvements_conclusion_token_f1")
        per_crit = (f"{_fmt(intro)} / {_fmt(body)} / {_fmt(conc)}"
                    if not (np.isnan(intro) and np.isnan(body) and np.isnan(conc))
                    else "—")
        # strengths+improvements count adherence mean
        s_ca = _lookup(agg, c, "C", "strengths_count_adherence")
        i_ca = _lookup(agg, c, "C", "improvements_count_adherence")
        item_ca = np.nanmean([s_ca, i_ca])
        lines.append(
            f"| {c} | {_fmt(_lookup(agg, c, 'C', 'qwk'))} | "
            f"{_fmt(_lookup(agg, c, 'C', 'score_mae'))} | "
            f"{_fmt(_lookup(agg, c, 'C', 'spearman_rho'))} | {per_crit} | "
            f"{_fmt(_lookup(agg, c, 'C', 'strengths_token_f1'))} | "
            f"{_fmt(_lookup(agg, c, 'C', 'improvements_token_f1'))} | "
            f"{_fmt(_lookup(agg, c, 'C', 'score_variance_ratio'))} | "
            f"{_fmt(_lookup(agg, c, 'C', 'schema_valid'))} | "
            f"{_fmt(item_ca)} |"
        )
    return lines


def _t6e(agg: pd.DataFrame) -> list[str]:
    lines = ["| Condition | BERTScore-F1 | Entity-F1 | Halluc. rate | Date F1 | SummaC-ZS | Subject-tag acc | Compression adh. | Glossary recall | Citation density | Lead-100 entity recall | UPSC fact prec. |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in CONDITIONS:
        lines.append(_build_row(c, [
            (_lookup(agg, c, "E", "mains_bertscore_f1"), "ratio"),
            (_lookup(agg, c, "E", "entity_f1_vs_gold"), "ratio"),
            (_lookup(agg, c, "E", "hallucination_rate"), "ratio"),
            (_lookup(agg, c, "E", "date_f1_vs_source"), "ratio"),
            (np.nan, "ratio"),  # SummaC-ZS deferred
            (_lookup(agg, c, "E", "subject_tag_acc"), "ratio"),
            (_lookup(agg, c, "E", "compression_ratio_score"), "ratio"),
            (np.nan, "ratio"),  # Glossary recall deferred
            (_lookup(agg, c, "E", "citation_density_per100w"), "ratio"),
            (_lookup(agg, c, "E", "lead100_entity_recall"), "ratio"),
            (_lookup(agg, c, "E", "fact_lookup_precision"), "ratio"),
        ]))
    return lines


def _t6f(agg: pd.DataFrame) -> list[str]:
    lines = ["| Condition | BERTScore-F1 (en) | BERTScore-F1 (hi) | ROUGE-L F1 | chrF++ | Entity-F1 | Distractor coverage | Reasoning-step density | Article citation acc. | Word-count adh. | Hindi code-mix | Δ BERTScore-F1 vs Task A |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in CONDITIONS:
        bs_en = _lookup(agg, c, "F", "explanation_bertscore_f1", "en")
        bs_hi = _lookup(agg, c, "F", "explanation_bertscore_f1", "hi")
        # Δ vs Task A's all-language explanation BERTScore-F1
        bs_a = _lookup(agg, c, "A", "explanation_bertscore_f1")
        bs_f_all = _lookup(agg, c, "F", "explanation_bertscore_f1")
        delta = (bs_f_all - bs_a) if not (np.isnan(bs_f_all) or np.isnan(bs_a)) else np.nan
        lines.append(_build_row(c, [
            (bs_en, "ratio"), (bs_hi, "ratio"),
            (_lookup(agg, c, "F", "explanation_rouge_l_f1"), "ratio"),
            (_lookup(agg, c, "F", "explanation_chrf"), "ratio"),
            (_lookup(agg, c, "F", "explanation_entity_f1"), "ratio"),
            (_lookup(agg, c, "F", "distractor_coverage"), "ratio"),
            (_lookup(agg, c, "F", "reasoning_step_density_per100w"), "ratio"),
            (_lookup(agg, c, "F", "citation_accuracy"), "ratio"),
            (_lookup(agg, c, "F", "word_count_adherence"), "ratio"),
            (_lookup(agg, c, "F", "hindi_code_mixing_rate"), "ratio"),
            (delta, "ratio"),
        ]))
    return lines


def _t6g(agg: pd.DataFrame) -> list[str]:
    lines = ["| Condition | BERTScore-F1 | ROUGE-L F1 | chrF++ | Word-count adh. | Paragraph adh. | Entity-F1 | Date/Num F1 | MATTR | F-K grade | 4-gram rep. | UPSC fact prec. | Dim-keyword cov. | Δ BERTScore-F1 vs Task B |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in CONDITIONS:
        date_f1 = _lookup(agg, c, "G", "date_exact_f1")
        num_f1 = _lookup(agg, c, "G", "numeric_exact_f1")
        dn = np.nanmean([date_f1, num_f1])
        bs_b = _lookup(agg, c, "B", "answer_bertscore_f1")
        bs_g = _lookup(agg, c, "G", "answer_bertscore_f1")
        delta = (bs_g - bs_b) if not (np.isnan(bs_g) or np.isnan(bs_b)) else np.nan
        lines.append(_build_row(c, [
            (bs_g, "ratio"),
            (_lookup(agg, c, "G", "answer_rouge_l_f1"), "ratio"),
            (_lookup(agg, c, "G", "answer_chrf"), "ratio"),
            (_lookup(agg, c, "G", "word_count_adherence"), "ratio"),
            (_lookup(agg, c, "G", "paragraph_count_adherence"), "ratio"),
            (_lookup(agg, c, "G", "entity_f1"), "ratio"),
            (dn, "ratio"),
            (_lookup(agg, c, "G", "mattr_100"), "ratio"),
            (_lookup(agg, c, "G", "flesch_kincaid_grade"), "ratio"),
            (_lookup(agg, c, "G", "ngram4_repetition_rate"), "ratio"),
            (_lookup(agg, c, "G", "fact_lookup_precision"), "ratio"),
            (_lookup(agg, c, "G", "dimension_keyword_coverage"), "ratio"),
            (delta, "ratio"),
        ]))
    return lines


# ---------- §6.4 universal ----------

def _t6_universal(agg: pd.DataFrame) -> list[str]:
    lines = ["| Condition | Latency p50 (ms) | TTFT (ms) | Tokens/sec | Cost/query (USD) | Format-validity rate |",
             "|---|---:|---:|---:|---:|---:|"]
    for c in CONDITIONS:
        lines.append(_build_row(c, [
            (_lookup(agg, c, "universal", "latency_p50_ms"), "ms"),
            (_lookup(agg, c, "universal", "ttft_p50_ms"), "ms"),
            (_lookup(agg, c, "universal", "tokens_per_sec_mean"), "tps"),
            (_lookup(agg, c, "universal", "cost_per_query_usd"), "cost"),
            (_lookup(agg, c, "universal", "format_validity_rate"), "ratio"),
        ]))
    return lines


# ---------- §7.1 / §7.2 / §7.3 ----------

def _t7_pairwise(tests: pd.DataFrame) -> list[str]:
    """§7.1 — primary-metric pairwise comparisons across the six condition
    pairs, for the headline metrics per task."""
    HEADLINES = {
        "A": ("is_correct", "accuracy"),
        "B": ("answer_bertscore_f1", "BERTScore-F1"),
        "C": ("score_abs_err", "score_MAE"),
        "E": ("mains_bertscore_f1", "BERTScore-F1"),
        "F": ("explanation_bertscore_f1", "BERTScore-F1"),
        "G": ("answer_bertscore_f1", "BERTScore-F1"),
    }
    lines = ["| Task | Metric | Comparison | Δ (mean) | 95% CI | p (raw) | p (BH-FDR) | Effect size | Significant? |",
             "|---|---|---|---:|---|---:|---:|---:|---|"]
    sub = tests[(tests["stratum_dim"].fillna("") == "")
                & (tests["metric"].isin(HEADLINES[t][0] for t in HEADLINES))]
    for task, (metric, _) in HEADLINES.items():
        for _, r in sub[(sub["task"] == task) & (sub["metric"] == metric)].iterrows():
            comp = f"{r['condition_a']} − {r['condition_b']}"
            ci = f"({_fmt(r.get('diff_ci_lo'))}, {_fmt(r.get('diff_ci_hi'))})"
            es = f"{_fmt(r.get('effect_size'))} ({r.get('effect_interpretation','')})"
            sig = "✓" if bool(r.get("significant_fdr")) else "—"
            lines.append(
                f"| {task} | {metric} | {comp} | {_fmt(r['mean_diff_a_minus_b'])} | {ci} | "
                f"{_fmt(r['paired_t_p'], 'pvalue')} | {_fmt(r.get('paired_t_p_fdr'), 'pvalue')} | "
                f"{es} | {sig} |"
            )
    # Append language-stratified accuracy tests for Task A (report §7.1 lists
    # accuracy_en pairwise explicitly).
    sub_lang = tests[(tests["stratum_dim"] == "language")
                     & (tests["task"] == "A")
                     & (tests["metric"] == "is_correct")]
    for _, r in sub_lang.iterrows():
        comp = f"{r['condition_a']} − {r['condition_b']}"
        ci = f"({_fmt(r.get('diff_ci_lo'))}, {_fmt(r.get('diff_ci_hi'))})"
        es = f"{_fmt(r.get('effect_size'))} ({r.get('effect_interpretation','')})"
        sig = "✓" if bool(r.get("significant_fdr")) else "—"
        lines.append(
            f"| A | accuracy_{r['stratum_val']} | {comp} | {_fmt(r['mean_diff_a_minus_b'])} | {ci} | "
            f"{_fmt(r['paired_t_p'], 'pvalue')} | {_fmt(r.get('paired_t_p_fdr'), 'pvalue')} | "
            f"{es} | {sig} |"
        )
    return lines


def _t7_heatmap(heatmap: pd.DataFrame) -> list[str]:
    lines = ["| Task | Stratum | Δ (champion − C3) | 95% CI | Verdict |",
             "|---|---|---:|---|---|"]
    for _, r in heatmap.iterrows():
        ci = f"({_fmt(r.get('ci_lo'))}, {_fmt(r.get('ci_hi'))})"
        lines.append(
            f"| {r['task']} | {r['stratum_key']} | {_fmt(r['delta_champion_minus_c3'])} | "
            f"{ci} | {r['verdict']} |"
        )
    if len(lines) == 2:
        lines.append("| (no stratum cells with paired data ≥ 5) | | | | |")
    return lines


def _t7_effects(tests: pd.DataFrame) -> list[str]:
    """§7.3 — effect sizes for significant pairwise comparisons."""
    lines = ["| Task | Metric | Comparison | Effect size | Interpretation |",
             "|---|---|---|---:|---|"]
    sub = tests[(tests["significant_fdr"] == True)  # noqa: E712
                & (tests["stratum_dim"].fillna("") == "")]
    for _, r in sub.iterrows():
        comp = f"{r['condition_a']} − {r['condition_b']}"
        lines.append(
            f"| {r['task']} | {r['metric']} | {comp} | {_fmt(r['effect_size'])} | "
            f"{r['effect_interpretation']} |"
        )
    if len(lines) == 2:
        lines.append("| (no significant comparisons after BH-FDR) | | | | |")
    return lines


# ---------- in-place table replacement ----------

# Each entry: (anchor_substring_in_header, builder_function).
# `anchor_substring` must uniquely identify the table header in the report;
# we replace the table (header + separator + body) immediately following
# the next matching `|...|` line after the anchor's position.
TABLE_SPECS: list[tuple[str, str, Any]] = [
    ("#### Task A — Prelims MCQ (correctness & calibration)", "6.3-A-correctness", _t6a_correctness),
    ("#### Task A — Explanation quality (Tier 1)", "6.3-A-explanation", _t6a_explanation),
    ("#### Task B — Mains generation", "6.3-B", _t6b),
    ("#### Task C — Mains rubric grading", "6.3-C", _t6c),
    ("#### Task E — Current Affairs synthesis", "6.3-E", _t6e),
    ("#### Task F — Prelims Explanation Generation (prayas production prompt)", "6.3-F", _t6f),
    ("#### Task G — Mains Model-Answer Generation (prayas production prompt)", "6.3-G", _t6g),
    ("### 6.4 Universal metrics", "6.4", _t6_universal),
    ("### 7.1 Pairwise hypothesis tests (BH-FDR-corrected)", "7.1", _t7_pairwise),
    ("### 7.2 Per-stratum heatmap data", "7.2", _t7_heatmap),
    ("### 7.3 Effect sizes", "7.3", _t7_effects),
]


def _replace_table_after(text: str, anchor: str, new_table_lines: list[str]) -> str:
    """Locate the anchor in `text`, find the next markdown table after it,
    and splice in `new_table_lines` in place of the existing table block."""
    idx = text.find(anchor)
    if idx < 0:
        raise ValueError(f"anchor not found in report: {anchor!r}")
    # Walk forward to find the first table-header line (begins with `| ` and
    # has at least one column delimiter `|`).
    tail = text[idx:]
    # `[ \t]*\n` instead of `\s*\n?` so we consume EXACTLY one newline per
    # row — `\s` matches `\n` and the greedy quantifier would eat the
    # following blank line that separates the table from the next section.
    m = re.search(
        r"^\|[^\n]*\|[ \t]*\n\|[-: |]+\|[ \t]*\n(?:\|[^\n]*\|[ \t]*\n)+",
        tail, re.MULTILINE,
    )
    if not m:
        raise ValueError(f"no markdown table found after anchor {anchor!r}")
    start = idx + m.start()
    end = idx + m.end()
    replacement = "\n".join(new_table_lines) + "\n"
    return text[:start] + replacement + text[end:]


# ---------- main ----------

def _load_inputs() -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    if not AGGREGATE.exists():
        raise FileNotFoundError(f"{AGGREGATE} not found; run `make aggregate` first")
    agg = pd.read_parquet(AGGREGATE)
    extras = json.loads(EXTRAS.read_text()) if EXTRAS.exists() else {}
    tests = pd.read_parquet(HYPOTHESES) if HYPOTHESES.exists() else pd.DataFrame()
    heatmap = pd.read_parquet(HEATMAP) if HEATMAP.exists() else pd.DataFrame()
    return agg, extras, tests, heatmap


def _gap_audit(agg: pd.DataFrame, tests: pd.DataFrame, heatmap: pd.DataFrame) -> int:
    """Report which (condition, task, metric) cells would be filled vs '—'."""
    n_filled = n_empty = 0
    for c in CONDITIONS:
        for task, metrics in {
            "A": ["is_correct", "upsc_neg_marking_score", "ece_15bin",
                  "brier_loss", "refusal_rate", "explanation_bertscore_f1",
                  "distractor_coverage"],
            "B": ["answer_bertscore_f1", "answer_chrf", "word_count_adherence",
                  "fact_lookup_precision"],
            "C": ["qwk", "score_mae", "spearman_rho", "strengths_token_f1",
                  "improvements_intro_token_f1"],
            "E": ["mains_bertscore_f1", "hallucination_rate",
                  "subject_tag_acc", "compression_ratio_score"],
            "F": ["explanation_bertscore_f1", "explanation_chrf",
                  "distractor_coverage"],
            "G": ["answer_bertscore_f1", "dimension_keyword_coverage"],
            "universal": ["latency_p50_ms", "tokens_per_sec_mean",
                          "cost_per_query_usd", "format_validity_rate"],
        }.items():
            for m in metrics:
                v = _lookup(agg, c, task, m)
                if np.isnan(v):
                    n_empty += 1
                else:
                    n_filled += 1
    print(f"[gap-audit] table cells: {n_filled} filled / {n_empty} empty")
    print(f"[gap-audit] pairwise tests: {len(tests)} rows")
    print(f"[gap-audit] heatmap strata: {len(heatmap)} rows")
    return 0 if n_empty == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, default=REPORT)
    ap.add_argument("--check", action="store_true",
                    help="run gap audit only; do not modify the report")
    args = ap.parse_args()

    agg, extras, tests, heatmap = _load_inputs()

    if args.check:
        return _gap_audit(agg, tests, heatmap)

    text = args.report.read_text(encoding="utf-8")
    for anchor, label, builder in TABLE_SPECS:
        if builder is _t7_heatmap:
            new_lines = builder(heatmap)
        elif builder in (_t7_pairwise, _t7_effects):
            new_lines = builder(tests)
        else:
            new_lines = builder(agg)
        try:
            text = _replace_table_after(text, anchor, new_lines)
            print(f"[ok] {label} ({anchor[:50]}…)")
        except ValueError as e:
            print(f"[skip] {label}: {e}")

    args.report.write_text(text, encoding="utf-8")
    print(f"\n[OK] rendered tables → {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
