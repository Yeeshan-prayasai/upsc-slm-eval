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
    "latency_ms", "ttft_ms", "input_tokens", "output_tokens",
    "tokens_per_sec", "cost_usd", "format_valid",
}

# Metrics treated as proportions (0/1 per row) for Cohen's h effect size.
PROPORTION_METRICS = {
    "is_correct", "format_fail", "refusal", "schema_valid", "format_valid",
    "silly_mistake_prone", "subject_tag_acc",
}


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d (paired-sample variant): mean(a-b) / sd(a-b)."""
    diff = a - b
    sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    if sd == 0.0:
        return 0.0
    return float(np.mean(diff) / sd)


def _cohen_h(p1: float, p2: float) -> float:
    """Cohen's h for two proportions. Arcsin-transformed difference."""
    p1c = min(max(p1, 0.0), 1.0)
    p2c = min(max(p2, 0.0), 1.0)
    return float(2 * np.arcsin(np.sqrt(p1c)) - 2 * np.arcsin(np.sqrt(p2c)))


def _effect_size(col: str, a: np.ndarray, b: np.ndarray) -> tuple[float, str]:
    """Return (effect_size, kind). kind ∈ {'cohen_d', 'cohen_h'}."""
    if col in PROPORTION_METRICS:
        return (_cohen_h(float(a.mean()), float(b.mean())), "cohen_h")
    return (_cohen_d(a, b), "cohen_d")


def _interpret_effect(es: float, kind: str) -> str:
    """Cohen's conventional thresholds (small / medium / large)."""
    mag = abs(es)
    if kind == "cohen_d":
        if mag < 0.2: return "negligible"
        if mag < 0.5: return "small"
        if mag < 0.8: return "medium"
        return "large"
    # cohen_h
    if mag < 0.2: return "negligible"
    if mag < 0.5: return "small"
    if mag < 0.8: return "medium"
    return "large"


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


def _pairwise(df: pd.DataFrame, stratify_by: str | None = None) -> pd.DataFrame:
    """Pairwise paired tests across conditions for each (task, metric) cell.

    When `stratify_by` is set (e.g. 'language'), tests are run within each
    stratum so per-language and per-paper splits are produced separately.

    Surfaces a one-time warning if the (task, question_id, condition) tuple
    is non-unique — that signals a duplicate prediction (e.g. accidental
    re-run that wasn't deduped) and would cause the pivot's `aggfunc="first"`
    to silently drop later occurrences.
    """
    conditions = sorted(df["condition"].unique())
    metric_cols = [c for c in df.columns
                   if c not in NON_METRIC and pd.api.types.is_numeric_dtype(df[c])]
    dup_mask = df.duplicated(subset=["task", "question_id", "condition"], keep=False)
    if dup_mask.any():
        n_dup = int(dup_mask.sum())
        print(f"[WARN] {n_dup} duplicate (task, question_id, condition) rows "
              f"detected — pivot will keep first only. Dedupe the scores parquet "
              f"to avoid silent data loss.")
    rows = []
    group_cols = ["task"] + ([stratify_by] if stratify_by else [])
    for keys, t_sub in df.groupby(group_cols):
        # pandas can return scalar OR tuple keys depending on len(group_cols);
        # normalize to a list so unpacking is consistent.
        key_list = list(keys) if isinstance(keys, tuple) else [keys]
        task = str(key_list[0])
        stratum = "all" if not stratify_by else str(key_list[1])
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
                es, es_kind = _effect_size(col, a, b)
                rows.append({
                    "task": task,
                    "stratum_dim": stratify_by or "",
                    "stratum_val": stratum,
                    "metric": col,
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
                    "effect_size": es,
                    "effect_size_kind": es_kind,
                    "effect_interpretation": _interpret_effect(es, es_kind),
                })
    return pd.DataFrame(rows)


def _stratum_heatmap(df: pd.DataFrame, c3_id: str = "C3") -> pd.DataFrame:
    """§7.2 — per-(task, stratum_key) champion-vs-C3 delta on the primary
    headline metric. Champion = argmax over (C1a, C1b) on that stratum mean.

    Primary metric per task (mirrors eval-design §4):
        A → is_correct (accuracy); B → answer_bertscore_f1;
        C → score_abs_err (lower is better — orientation flipped);
        E → mains_bertscore_f1; F → explanation_bertscore_f1;
        G → answer_bertscore_f1.
    """
    PRIMARY = {
        "A": ("is_correct", False), "B": ("answer_bertscore_f1", False),
        "C": ("score_abs_err", True), "E": ("mains_bertscore_f1", False),
        "F": ("explanation_bertscore_f1", False), "G": ("answer_bertscore_f1", False),
    }
    rows = []
    for task, sub in df.groupby("task"):
        if task not in PRIMARY:
            continue
        metric, lower_better = PRIMARY[task]
        if metric not in sub.columns:
            continue
        for stratum, ssub in sub.groupby("stratum_key"):
            if not stratum:
                continue
            # Champion = whichever of C1a / C1b is better on this stratum mean.
            means_by_cond = ssub.groupby("condition")[metric].mean()
            cand = [c for c in ("C1a", "C1b") if c in means_by_cond.index]
            if not cand or c3_id not in means_by_cond.index:
                continue
            if lower_better:
                champion = min(cand, key=lambda c: means_by_cond[c])
            else:
                champion = max(cand, key=lambda c: means_by_cond[c])
            pivot = ssub.pivot_table(index="question_id", columns="condition",
                                     values=metric, aggfunc="first")
            if champion not in pivot.columns or c3_id not in pivot.columns:
                continue
            paired = pivot[[champion, c3_id]].dropna()
            if len(paired) < 5:
                continue
            a = paired[champion].astype(float).values
            b = paired[c3_id].astype(float).values
            diff = (b - a) if lower_better else (a - b)
            mean_d, lo, hi = _paired_bootstrap_ci(diff)
            verdict = ("WIN" if (lo > 0)
                       else ("LOSS" if (hi < 0) else "TIE"))
            rows.append({
                "task": task, "stratum_key": stratum,
                "primary_metric": metric, "champion": champion,
                "n_paired": int(len(paired)),
                "champion_mean": float(a.mean()), "c3_mean": float(b.mean()),
                "delta_champion_minus_c3": mean_d,
                "ci_lo": lo, "ci_hi": hi,
                "verdict": verdict,
            })
    return pd.DataFrame(rows)


def _apply_fdr(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Add BH-FDR-adjusted p-values across the full test family.

    A comparison is flagged `significant_fdr=True` only when BOTH the
    paired t-test AND the Wilcoxon signed-rank test agree at q=alpha.
    Requiring agreement guards against:
     - false positives from t-test on heavy-tailed metrics (e.g. brier_loss
       where outliers inflate t-stat magnitude)
     - false negatives from Wilcoxon on near-Gaussian metrics with ties
    """
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
    sig_t = (df["paired_t_p_fdr"] < alpha).fillna(False)
    sig_w = (df["wilcoxon_p_fdr"] < alpha).fillna(False)
    df["significant_fdr"] = sig_t & sig_w
    # Surface the single-test results too so downstream readers can see
    # whether a comparison failed agreement (one test fires, the other doesn't).
    df["significant_t_only"] = sig_t & ~sig_w
    df["significant_w_only"] = sig_w & ~sig_t
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

    # All-strata + per-language tests in one frame. BH-FDR is then applied
    # across the FULL family so report §7.1 corrects for the language splits.
    tests_all = _pairwise(df)
    tests_lang = _pairwise(df, stratify_by="language")
    tests = pd.concat([tests_all, tests_lang], ignore_index=True)
    tests = _apply_fdr(tests, alpha=args.alpha)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tests.to_parquet(args.out, index=False, compression="snappy")

    # §7.2 per-stratum heatmap (champion vs C3 on the headline primary metric).
    heatmap = _stratum_heatmap(df)
    heatmap_path = args.out.with_name("stratum_heatmap.parquet")
    heatmap.to_parquet(heatmap_path, index=False, compression="snappy")

    n_sig = int(tests["significant_fdr"].sum())
    print(f"\n[OK] wrote {len(tests):,} pairwise tests → {args.out}")
    print(f"     {n_sig}/{len(tests)} significant after BH-FDR (α={args.alpha})")
    print(f"     stratum-heatmap rows: {len(heatmap):,} → {heatmap_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
