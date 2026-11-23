#!/usr/bin/env python3
"""Say one result three ways: p-value, confidence interval, and effect size.

The point is the lesson from ASA 2019 ("Moving to a World Beyond p < 0.05"):
the bare p-value hides both magnitude and uncertainty. Lead with effect size + CI.

Usage:
    python three_ways.py --mean-a 0.410 --mean-b 0.382 --sd 0.021 --n 5

This is a teaching tool for a simple two-group comparison with a shared SD and n
per group. For multi-seed deep-learning comparisons, prefer the Almost Stochastic
Order test in `deep-significance` (current version; arXiv:2204.06815;
`pip install deepsig`) over a t-test.
"""
import argparse
import math
from scipy import stats  # SciPy 1.18.0 (2026-06-19) as of mid-2026


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mean-a", type=float, required=True, help="your method's mean")
    p.add_argument("--mean-b", type=float, required=True, help="baseline mean")
    p.add_argument("--sd", type=float, required=True, help="pooled std dev")
    p.add_argument("--n", type=int, required=True, help="runs/seeds per group")
    args = p.parse_args()

    diff = args.mean_a - args.mean_b
    se = args.sd * math.sqrt(2.0 / args.n)          # SE of the difference
    df = 2 * args.n - 2
    t = diff / se
    pval = 2 * stats.t.sf(abs(t), df)               # two-sided
    tcrit = stats.t.ppf(0.975, df)
    lo, hi = diff - tcrit * se, diff + tcrit * se   # 95% CI of the difference
    d = diff / args.sd                               # Cohen's d
    d_se = math.sqrt((2.0 / args.n) + (d * d) / (2 * df))
    d_lo, d_hi = d - 1.96 * d_se, d + 1.96 * d_se

    alpha = 0.05
    verdict = ("significantly different" if pval < alpha
               else f"not significantly different at alpha = {alpha}")
    print("\n--- one result, three ways ---\n")
    print(f"p-value form:     {verdict} (p = {pval:.3f}).  "
          "<- hides magnitude AND uncertainty; weakest sentence.")
    print(f"CI form:          improved by {diff:+.3f} (95% CI [{lo:+.3f}, {hi:+.3f}]).")
    print(f"effect-size form: Cohen's d = {d:.2f} (95% CI [{d_lo:.2f}, {d_hi:.2f}]); "
          f"{_band(d)} effect.")
    if lo <= 0 <= hi:
        print("\nNOTE: the difference CI includes 0 -> report as an uncertain trend, "
              "not a win. Test everything before you hold it fast.")


def _band(d: float) -> str:
    a = abs(d)
    # Cohen's 0.2/0.5/0.8 landmarks; Cohen himself called them arbitrary/field-dependent.
    return "small" if a < 0.5 else "moderate" if a < 0.8 else "large"


if __name__ == "__main__":
    main()
