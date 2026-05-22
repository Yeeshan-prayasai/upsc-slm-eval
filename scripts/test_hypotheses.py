"""Stage 6.3 — pairwise hypothesis tests with BH-FDR correction.

For every (task, metric, condition_pair) where both conditions have scored the
same question_id, run a paired test on the difference and a percentile-
bootstrap 95% CI on the mean difference. Apply BH-FDR correction across the
full test family (per [eval-design.md §6](eval-design.md)).

Output: `results/hypothesis_tests.parquet` — one row per pairwise test.
"""
from __future__ import annotations
import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

REPO = Path(__file__).resolve().parent.parent
SCORES = REPO / "results" / "scores_tier1.parquet"
OUT = REPO / "results" / "hypothesis_tests.parquet"

SEED = 20260514
BOOTSTRAP_N = 1000

# Direction we *expect* a higher value to be better. Metrics whose smaller-
# value direction is better (e.g. brier_loss) are flipped before testing.
LOWER_IS_BETTER = {
    "brier_loss", "format_fail", "score_abs_err", "hallucination_rate",
    "ngram4_repetition_rate",
}

NON_METRIC = {
    "run_id", "condition", "question_id", "task", "language", "paper",
    "subject", "stratum_key", "predicted_letter", "correct_letter",
    "pred_score", "gold_score", "max_score",
}


def _paired_bootstrap_ci(diff: np.ndarray, n: int = BOOTSTRAP_N,
                         alpha: float = 0.05) -> tuple[float, float, float]:
    if len(diff) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(SEED)
    if len(diff) == 1:
        return (float(diff[0]), float(diff[0]), float(diff[0]))
    means = rng.choice(diff, size=(n, len(diff)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(diff.mean()), float(lo), float(hi))


def _pairwise(df: pd.DataFrame) -> pd.DataFrame:
    conditions = sorted(df["condition"].unique())
    metric_cols = [c for c in df.columns
                   if c not in NON_METRIC and pd.api.types.is_numeric_dtype(df[c])]
    rows = []
    for task, t_sub in df.groupby("task"):
        for col in metric_cols:
            if t_sub[col].isna().all():
                continue
            pivot = t_sub.pivot_table(index="question_id", columns="condition",
                                      values=col, aggfunc="first")
            for ca, cb in combinations(conditions, 2):
                if ca not in pivot.columns or cb not in pivot.columns:
                    continue
                paired = pivot[[ca, cb]].dropna()
                if len(paired) < 5:
                    continue
                a = paired[ca].astype(float).values
                b = paired[cb].astype(float).values
                # Orient so that positive diff = ca beats cb.
                diff = (b - a) if col in LOWER_IS_BETTER else (a - b)
                try:
                    t_stat, p_t = stats.ttest_rel(a, b)
                    if col in LOWER_IS_BETTER:
                        t_stat = -t_stat
                    w_stat, p_w = stats.wilcoxon(a, b)
                except (ValueError, FloatingPointError):
                    t_stat, p_t, w_stat, p_w = np.nan, np.nan, np.nan, np.nan
                mean_d, lo, hi = _paired_bootstrap_ci(diff)
                rows.append({
                    "task": task, "metric": col,
                    "condition_a": ca, "condition_b": cb,
                    "n_paired": int(len(paired)),
                    "mean_a": float(a.mean()),
                    "mean_b": float(b.mean()),
                    "mean_diff_a_minus_b": mean_d,
                    "diff_ci_lo": lo, "diff_ci_hi": hi,
                    "paired_t": float(t_stat),
                    "paired_t_p": float(p_t),
                    "wilcoxon_stat": float(w_stat),
                    "wilcoxon_p": float(p_w),
                })
    return pd.DataFrame(rows)


def _apply_fdr(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Add BH-FDR-adjusted p-values across the full test family."""
    df = df.copy()
    if df.empty or "paired_t_p" not in df.columns:
        df["paired_t_p_fdr"] = pd.Series(dtype=float)
        df["wilcoxon_p_fdr"] = pd.Series(dtype=float)
        df["significant_fdr"] = pd.Series(dtype=bool)
        return df
    ok = df["paired_t_p"].notna()
    if not ok.any():
        df["paired_t_p_fdr"] = np.nan
        df["wilcoxon_p_fdr"] = np.nan
        df["significant_fdr"] = False
        return df
    _, p_t_adj, _, _ = multipletests(df.loc[ok, "paired_t_p"], alpha=alpha, method="fdr_bh")
    _, p_w_adj, _, _ = multipletests(df.loc[ok, "wilcoxon_p"].fillna(1.0), alpha=alpha,
                                     method="fdr_bh")
    df.loc[ok, "paired_t_p_fdr"] = p_t_adj
    df.loc[ok, "wilcoxon_p_fdr"] = p_w_adj
    df["significant_fdr"] = (df["paired_t_p_fdr"] < alpha).fillna(False)
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, default=SCORES)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    if not args.scores.exists():
        print(f"[FAIL] {args.scores} not found; run `make score-tier1` first")
        return 1

    df = pd.read_parquet(args.scores)
    print(f"[load] {len(df):,} scored rows; "
          f"{df['condition'].nunique()} conditions × {df['task'].nunique()} tasks")

    tests = _pairwise(df)
    tests = _apply_fdr(tests, alpha=args.alpha)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tests.to_parquet(args.out, index=False, compression="snappy")

    n_sig = int(tests["significant_fdr"].sum())
    print(f"\n[OK] wrote {len(tests):,} pairwise tests → {args.out}")
    print(f"     {n_sig}/{len(tests)} significant after BH-FDR (α={args.alpha})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
