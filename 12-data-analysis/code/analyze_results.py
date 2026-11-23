#!/usr/bin/env python3
"""analyze_results.py — honest analysis of a benchmark comparison.

Effect size + bootstrap CI + permutation p-value travel together, and every
comparison you run gets multiple-comparison correction. Point-in-time stack
(mid-2026): SciPy 1.18.0, statsmodels 0.15.0-dev. Pin your versions.

  pip install numpy scipy statsmodels
Optional multi-seed (DL) test:
  pip install deepsig   # the deep-significance package: deepsig.aso, ASO test, tau=0.2 threshold

Usage idea: load your real per-run scores from the Chapter 11 raw dataset,
not toy numbers. Lead with the EFFECT, carry the INTERVAL, let p bring up
the rear. Count every comparison honestly — including the ones you don't report.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import bootstrap, permutation_test
from statsmodels.stats.multitest import multipletests


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Standardized mean difference (pooled SD). Cohen's 0.2/0.5/0.8 are
    arbitrary & field-dependent — report the number, argue importance in
    your domain's terms (cf. Ortloff et al., CHI 2025)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1))
                     / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled)


def diff_ci(a: np.ndarray, b: np.ndarray, paired: bool = False, seed: int = 0):
    """95% bootstrap CI on the difference of means. Use for weird metrics
    (F1, BLEU) where analytic CIs are awkward."""
    if paired:
        data = (np.asarray(a, float) - np.asarray(b, float),)
        stat = np.mean
    else:
        data = (np.asarray(a, float), np.asarray(b, float))
        stat = lambda x, y: x.mean() - y.mean()
    res = bootstrap(data, stat, confidence_level=0.95,
                    n_resamples=10_000, random_state=seed)
    return res.confidence_interval.low, res.confidence_interval.high


def perm_p(a: np.ndarray, b: np.ndarray, seed: int = 0) -> float:
    """Permutation test for difference of means — no normality assumption.
    Prefer this over a blind Student's t when assumptions are shaky."""
    res = permutation_test((np.asarray(a, float), np.asarray(b, float)),
                           lambda x, y: x.mean() - y.mean(),
                           n_resamples=10_000, random_state=seed)
    return float(res.pvalue)


def report_one(name: str, mine, base, paired: bool = False) -> dict:
    d = cohens_d(mine, base)
    lo, hi = diff_ci(mine, base, paired=paired)
    p = perm_p(mine, base)
    # Lead with effect + interval; p last.
    print(f"{name}: delta={np.mean(mine)-np.mean(base):+.3f} "
          f"(95% CI [{lo:+.3f}, {hi:+.3f}]; d={d:+.2f}; perm p={p:.4f})")
    return {"name": name, "d": d, "ci": (lo, hi), "p": p}


def correct(pvals, method: str = "holm"):
    """Correct for the comparisons you ACTUALLY ran. holm=FWER (strict),
    fdr_bh=FDR (higher power for many tests). State the count in the paper."""
    reject, p_adj, _, _ = multipletests(pvals, alpha=0.05, method=method)
    return reject, p_adj


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # Replace with your real per-seed scores.
    mine = rng.normal(0.84, 0.02, size=7)
    base = rng.normal(0.81, 0.02, size=7)
    r = report_one("primary", mine, base)
    rej, adj = correct([r["p"], 0.013, 0.30, 0.044], method="holm")
    print("Holm-corrected:", np.round(adj, 4), "survives:", rej)
