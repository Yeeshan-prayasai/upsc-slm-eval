"""Stage 2.3 — A2 pass-criterion gate.

One-sided binomial test against the random-chance baseline (p=0.25 for 4-option
MCQs). H0: model accuracy = p_null. H1: model accuracy > p_null. A model passes
when the p-value of P(X ≥ correct | n, p_null) is below alpha.

At n=50, alpha=0.05 ⇒ critical value k=18 (36% accuracy).
At n=200 (legacy), alpha=0.05 ⇒ critical value k=64 (32% accuracy).

Exits 0 if all models pass; exits 1 otherwise. Failing models route their
Hindi-stratum results to separate post-FT reporting (per eval-design.md §10 A2).
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=Path, default=Path("results/pre_ft_hindi_probe.parquet"))
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="one-sided significance level")
    ap.add_argument("--p-null", type=float, default=0.25,
                    help="random-chance baseline (0.25 for 4-option MCQs)")
    args = ap.parse_args()

    if not args.probe.exists():
        print(f"[FAIL] probe file not found: {args.probe}")
        print(f"       run scripts/run_hindi_probe.py --model <hf_id> first")
        return 1

    df = pd.read_parquet(args.probe)
    if df.empty:
        print(f"[FAIL] {args.probe} is empty")
        return 1

    print(f"A2 Hindi-capability gate")
    print(f"  H0: accuracy = {args.p_null}  vs  H1: accuracy > {args.p_null}  (one-sided binomial)")
    print(f"  alpha = {args.alpha}\n")

    any_fail = False
    for model, g in df.groupby("model"):
        n = len(g)
        correct = int(g["is_correct"].sum())
        acc = correct / n
        p_value = binomtest(correct, n, p=args.p_null, alternative="greater").pvalue
        status = "PASS" if p_value < args.alpha else "FAIL"
        if status == "FAIL":
            any_fail = True
        print(f"  {status}  {model:<50s}  {correct:>3d}/{n:<3d} = {acc:.3f}   p = {p_value:.4f}")

    if any_fail:
        print(f"\n[A2 gate FAIL]")
        print(f"  Failing models do not reject H0 at alpha={args.alpha}; treat their pre-FT")
        print(f"  Hindi accuracy as indistinguishable from chance. Hindi-stratum results")
        print(f"  for those models will be reported separately post-FT — not folded into")
        print(f"  the bilingual aggregate.")
        return 1
    print(f"\n[A2 gate PASS] all models reject H0 at alpha={args.alpha}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
