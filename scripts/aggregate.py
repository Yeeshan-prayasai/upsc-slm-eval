"""Stage 6.2 — aggregate per-row scores into per-condition metric tables.

Reads `results/scores_tier1.parquet`, emits `results/aggregate.parquet` with:
- mean + bootstrap-95% CI for every per-row scalar metric, grouped by
  (condition, task) and (condition, task, language)
- aggregation-level metrics that need the full row set:
    Task A: ECE, Brier Skill Score, Position-bias χ², Silly-mistake delta,
            Bilingual accuracy delta
    Task C: QWK, Spearman ρ, Pearson r, Score-variance ratio
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
SCORES = REPO / "results" / "scores_tier1.parquet"
OUT = REPO / "results" / "aggregate.parquet"
SEED = 20260514
BOOTSTRAP_N = 1000

# Columns that are identifiers, not metrics
NON_METRIC = {
    "run_id", "condition", "question_id", "task", "language", "paper",
    "subject", "stratum_key", "predicted_letter", "correct_letter",
    "pred_score", "gold_score", "max_score",
    # Universal columns aggregated separately by _aggregate_universal()
    "latency_ms", "ttft_ms", "input_tokens", "output_tokens",
    "tokens_per_sec", "cost_usd", "format_valid",
}


def _bootstrap_ci(values: np.ndarray, n: int = BOOTSTRAP_N,
                  alpha: float = 0.05) -> tuple[float, float, float]:
    """Percentile bootstrap mean + (lo, hi) CI. NaN-safe."""
    v = values[~np.isnan(values)]
    if len(v) == 0:
        return (np.nan, np.nan, np.nan)
    mean = float(v.mean())
    if len(v) == 1:
        return (mean, mean, mean)
    rng = np.random.default_rng(SEED)
    means = rng.choice(v, size=(n, len(v)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (mean, float(lo), float(hi))


def _ece(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error (L1). NaN entries are excluded."""
    mask = ~(np.isnan(confidences) | np.isnan(correct))
    c, y = confidences[mask], correct[mask]
    if len(c) == 0:
        return np.nan
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (c > edges[i]) & (c <= edges[i + 1]) if i > 0 else (c >= edges[0]) & (c <= edges[1])
        if not in_bin.any():
            continue
        acc = y[in_bin].mean()
        conf = c[in_bin].mean()
        ece += (in_bin.sum() / len(c)) * abs(acc - conf)
    return float(ece)


def _brier_skill(brier: np.ndarray, correct: np.ndarray) -> float:
    """1 - Brier / Brier_of_baserate_predictor. Higher = better."""
    b = brier[~np.isnan(brier)]
    y = correct[~np.isnan(correct)]
    if len(b) == 0 or len(y) == 0:
        return np.nan
    base_rate = float(y.mean())
    baseline_brier = float(((base_rate - y) ** 2).mean())
    if baseline_brier <= 0:
        return np.nan
    return 1.0 - float(b.mean()) / baseline_brier


def _position_bias(letters: pd.Series) -> tuple[float, float]:
    """χ² test of predicted-letter distribution vs uniform (1/4 each).

    Returns (chi2_statistic, p_value). Skips empty/format-fail entries.
    """
    valid = [x for x in letters if x in {"A", "B", "C", "D"}]
    if len(valid) < 8:
        return (np.nan, np.nan)
    obs = np.array([valid.count(x) for x in ("A", "B", "C", "D")])
    expected = np.full(4, len(valid) / 4.0)
    chi2, p = stats.chisquare(obs, expected)
    return (float(chi2), float(p))


def _bilingual_delta(df: pd.DataFrame) -> dict:
    """Task A: per-question-stem accuracy delta between en and hi."""
    a = df[df["task"] == "A"].copy()
    if a.empty:
        return {}
    # pair en/hi rows by stripping the language suffix on question_id
    a["stem"] = a["question_id"].str.replace(r":(en|hi)$", "", regex=True)
    pairs = (
        a.groupby(["condition", "stem", "language"])["is_correct"].first()
         .unstack("language")
    )
    if "en" not in pairs.columns or "hi" not in pairs.columns:
        return {}
    out = {}
    for cond, sub in pairs.dropna(subset=["en", "hi"]).groupby(level="condition"):
        en = sub["en"].astype(float).values
        hi = sub["hi"].astype(float).values
        if len(en) < 5:
            continue
        diff = en - hi
        t, p = stats.ttest_rel(en, hi)
        out[cond] = {
            "n_paired": int(len(en)),
            "acc_en": float(en.mean()),
            "acc_hi": float(hi.mean()),
            "delta_en_minus_hi": float(diff.mean()),
            "paired_t": float(t),
            "p_value": float(p),
        }
    return out


def _silly_breakdown(df: pd.DataFrame) -> dict:
    a = df[df["task"] == "A"]
    if a.empty:
        return {}
    out = {}
    for cond, sub in a.groupby("condition"):
        silly = sub[sub["silly_mistake_prone"] == 1]["is_correct"]
        normal = sub[sub["silly_mistake_prone"] == 0]["is_correct"]
        if len(silly) < 5 or len(normal) < 5:
            continue
        out[cond] = {
            "n_silly": int(len(silly)),
            "n_normal": int(len(normal)),
            "acc_silly": float(silly.mean()),
            "acc_normal": float(normal.mean()),
            "delta_silly_minus_normal": float(silly.mean() - normal.mean()),
        }
    return out


def _task_c_rank(df: pd.DataFrame) -> dict:
    c = df[df["task"] == "C"].copy()
    if c.empty:
        return {}
    c = c.dropna(subset=["pred_score", "gold_score"])
    out = {}
    for cond, sub in c.groupby("condition"):
        if len(sub) < 5:
            continue
        max_s = float(sub["max_score"].iloc[0]) or 1.0

        def _band(s):
            r = s / max_s
            return "low" if r <= 0.30 else ("mid" if r <= 0.60 else "high")

        gold_bins = sub["gold_score"].round().astype(int)
        pred_bins = sub["pred_score"].round().astype(int)
        try:
            from sklearn.metrics import cohen_kappa_score, confusion_matrix
            qwk = float(cohen_kappa_score(gold_bins, pred_bins, weights="quadratic"))
            gold_bands = sub["gold_score"].map(_band)
            pred_bands = sub["pred_score"].map(_band)
            labels = ["low", "mid", "high"]
            cm = confusion_matrix(gold_bands, pred_bands, labels=labels).tolist()
        except Exception:
            qwk, cm = np.nan, []
        s_rho, _ = stats.spearmanr(sub["gold_score"], sub["pred_score"])
        p_r, _ = stats.pearsonr(sub["gold_score"], sub["pred_score"])
        var_pred = float(sub["pred_score"].var())
        var_gold = float(sub["gold_score"].var())
        var_ratio = (var_pred / var_gold) if var_gold > 0 else np.nan
        out[cond] = {
            "n": int(len(sub)),
            "qwk": qwk,
            "spearman_rho": float(s_rho),
            "pearson_r": float(p_r),
            "score_mae": float((sub["pred_score"] - sub["gold_score"]).abs().mean()),
            "score_variance_ratio": var_ratio,
            "confusion_matrix_low_mid_high": cm,
        }
    return out


def _aggregate_scalars(df: pd.DataFrame) -> pd.DataFrame:
    """Per-(condition, task, language) mean + bootstrap CI for every numeric col."""
    rows = []
    metric_cols = [c for c in df.columns
                   if c not in NON_METRIC and pd.api.types.is_numeric_dtype(df[c])]
    for (cond, task, lang), sub in df.groupby(["condition", "task", "language"], dropna=False):
        for col in metric_cols:
            vals = sub[col].astype(float).values
            if np.all(np.isnan(vals)):
                continue
            mean, lo, hi = _bootstrap_ci(vals)
            rows.append({
                "condition": cond, "task": task, "language": lang,
                "metric": col, "n": int((~np.isnan(vals)).sum()),
                "mean": mean, "ci_lo": lo, "ci_hi": hi,
            })
    # Also produce a (condition, task, language='all') view
    for (cond, task), sub in df.groupby(["condition", "task"]):
        for col in metric_cols:
            vals = sub[col].astype(float).values
            if np.all(np.isnan(vals)):
                continue
            mean, lo, hi = _bootstrap_ci(vals)
            rows.append({
                "condition": cond, "task": task, "language": "all",
                "metric": col, "n": int((~np.isnan(vals)).sum()),
                "mean": mean, "ci_lo": lo, "ci_hi": hi,
            })
    return pd.DataFrame(rows)


def _aggregate_special(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregation-level metrics that aren't per-row scalars."""
    rows = []
    # Task A: ECE + Brier Skill Score + Position bias + Refusal rate per condition.
    a = df[df["task"] == "A"]
    for cond, sub in a.groupby("condition"):
        conf = sub["confidence_prob"].astype(float).values
        cor = sub["is_correct"].astype(float).values
        ece = _ece(conf, cor)
        bss = _brier_skill(sub["brier_loss"].astype(float).values, cor)
        chi2, p_pos = _position_bias(sub["predicted_letter"])
        refusal_rate = (float(sub["refusal"].mean()) if "refusal" in sub.columns
                        else np.nan)
        for m_name, m_val in (
            ("ece_15bin", ece), ("brier_skill_score", bss),
            ("position_bias_chi2", chi2), ("position_bias_p_value", p_pos),
            ("refusal_rate", refusal_rate),
        ):
            rows.append({
                "condition": cond, "task": "A", "language": "all",
                "metric": m_name, "n": int(len(sub)),
                "mean": m_val, "ci_lo": np.nan, "ci_hi": np.nan,
            })

    # Task C: QWK / Spearman ρ / Pearson r / MAE / variance ratio per condition.
    # These need the full row set (not per-row scalars) — promote from extras
    # so the renderer can read everything from aggregate.parquet.
    c_rank = _task_c_rank(df)
    for cond, m in c_rank.items():
        for key in ("qwk", "spearman_rho", "pearson_r", "score_mae",
                    "score_variance_ratio"):
            val = m.get(key)
            if val is None:
                continue
            rows.append({
                "condition": cond, "task": "C", "language": "all",
                "metric": key, "n": int(m.get("n") or 0),
                "mean": float(val) if val == val else np.nan,
                "ci_lo": np.nan, "ci_hi": np.nan,
            })

    return pd.DataFrame(rows)


def _aggregate_universal(df: pd.DataFrame) -> pd.DataFrame:
    """§6.4 universal metrics — per-condition latency percentiles, throughput,
    cost, and format-validity rate. Latency is reported across all tasks/items
    for that condition (single-row table per condition, language='all').
    """
    if "latency_ms" not in df.columns:
        return pd.DataFrame()
    rows = []
    for cond, sub in df.groupby("condition"):
        lat = sub["latency_ms"].astype(float).dropna().values
        ttft = sub["ttft_ms"].astype(float).dropna().values
        tps = sub["tokens_per_sec"].astype(float).dropna().values
        cost = sub["cost_usd"].astype(float).dropna().values
        fv = sub["format_valid"].astype(float).dropna().values
        for m_name, vals, agg in (
            ("latency_p50_ms", lat, lambda v: float(np.percentile(v, 50)) if len(v) else np.nan),
            ("latency_p95_ms", lat, lambda v: float(np.percentile(v, 95)) if len(v) else np.nan),
            ("latency_p99_ms", lat, lambda v: float(np.percentile(v, 99)) if len(v) else np.nan),
            ("ttft_p50_ms",    ttft, lambda v: float(np.percentile(v, 50)) if len(v) else np.nan),
            ("tokens_per_sec_mean", tps, lambda v: float(np.mean(v)) if len(v) else np.nan),
            ("cost_per_query_usd", cost, lambda v: float(np.mean(v)) if len(v) else np.nan),
            ("format_validity_rate", fv, lambda v: float(np.mean(v)) if len(v) else np.nan),
        ):
            rows.append({
                "condition": cond, "task": "universal", "language": "all",
                "metric": m_name, "n": int(len(vals)),
                "mean": agg(vals), "ci_lo": np.nan, "ci_hi": np.nan,
            })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, default=SCORES)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    if not args.scores.exists():
        print(f"[FAIL] {args.scores} not found; run `make score-tier1` first")
        return 1
    df = pd.read_parquet(args.scores)
    print(f"[load] {len(df):,} scored rows from {args.scores}")

    scalar = _aggregate_scalars(df)
    special = _aggregate_special(df)
    universal = _aggregate_universal(df)
    agg = pd.concat([scalar, special, universal], ignore_index=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(args.out, index=False, compression="snappy")

    # Side files for the non-tabular Task-A and Task-C breakdowns
    extras = {
        "bilingual_delta": _bilingual_delta(df),
        "silly_breakdown": _silly_breakdown(df),
        "task_c_rank": _task_c_rank(df),
    }
    extras_path = args.out.with_suffix(".extras.json")
    import json
    extras_path.write_text(json.dumps(extras, indent=2, default=str))

    print(f"\n[OK] wrote {len(agg):,} aggregate rows → {args.out}")
    print(f"     + extras → {extras_path}")
    print(f"     by task: {agg.groupby('task').size().to_dict()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
