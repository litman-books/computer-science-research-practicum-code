#!/usr/bin/env python3
"""plot_with_ci.py — honest-defaults figure for a results comparison.

A figure is an argument. Show uncertainty, keep the axis defensible, and write
a caption that stands alone. The honest plot is the EASY one here; the
misleading one (truncated axis, no error bars) takes extra effort.

  pip install pandas seaborn matplotlib

When AI writes your plotting code, the most common silent error is dropping the
error bars. Check for them every time.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def make_long(scores: dict[str, np.ndarray]) -> pd.DataFrame:
    """scores: {method_name: per-seed score array} -> long DataFrame."""
    rows = [{"method": m, "score": float(s)}
            for m, arr in scores.items() for s in np.asarray(arr, float)]
    return pd.DataFrame(rows)


def honest_bar(df: pd.DataFrame, out: str = "results.png") -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    # 95% CI bars over the per-seed mean...
    sns.barplot(data=df, x="method", y="score", errorbar=("ci", 95),
                ax=ax, alpha=0.85)
    # ...and the raw per-seed points so the spread is visible.
    sns.stripplot(data=df, x="method", y="score", color="k",
                  alpha=0.5, size=5, ax=ax)
    # Defensible axis: do NOT truncate to inflate a small gap.
    ax.set_ylim(0, max(1.0, df["score"].max() * 1.1))
    ax.set_ylabel("Test accuracy")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")
    # Caption that stands alone (paste into the paper, edit the n):
    print("Figure 1. Test accuracy by method (n=%d seeds each); bars are means "
          "with 95%% bootstrap CIs, points are individual seeds."
          % df.groupby("method").size().min())


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    df = make_long({
        "baseline": rng.normal(0.81, 0.02, 7),
        "ours":     rng.normal(0.84, 0.02, 7),
    })
    honest_bar(df)
