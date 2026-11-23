# pilot_sanity_check.py — five cheap checks to run on a pilot BEFORE you spend compute.
# Domain-agnostic: works for any supervised setup where you have y_true/y_pred arrays
# and named train/test index sets. Adapt the metric to your task.
#
# Usage:  python pilot_sanity_check.py
# Requires: numpy, scikit-learn  (pip install numpy scikit-learn)

import numpy as np
from sklearn.metrics import accuracy_score


def check_metric_in_range(score, lo=0.0, hi=1.0, name="metric"):
    """A metric outside its theoretical range means a buggy pipeline, not a result."""
    ok = lo <= score <= hi
    print(f"[{'PASS' if ok else 'RED FLAG'}] {name}={score:.4f} in [{lo},{hi}]")
    return ok


def check_beats_trivial_baseline(y_true, y_pred):
    """If your method can't beat 'always predict the majority class', stop and debug."""
    vals, counts = np.unique(y_true, return_counts=True)
    majority = vals[np.argmax(counts)]
    trivial = accuracy_score(y_true, np.full_like(y_true, majority))
    method = accuracy_score(y_true, y_pred)
    ok = method > trivial
    print(f"[{'PASS' if ok else 'RED FLAG'}] method={method:.4f} vs majority={trivial:.4f}")
    return ok


def check_no_split_leakage(train_idx, test_idx):
    """The single most common silent bug: the same sample in train AND test."""
    overlap = set(train_idx) & set(test_idx)
    ok = len(overlap) == 0
    print(f"[{'PASS' if ok else 'RED FLAG'}] {len(overlap)} samples leak across the split")
    return ok


def check_seed_variance(scores, max_spread=0.10):
    """One run is an anecdote. Report the spread across seeds (Bouthillier et al. 2021)."""
    spread = float(np.max(scores) - np.min(scores))
    ok = spread <= max_spread
    print(f"[{'PASS' if ok else 'WATCH'}] seed spread={spread:.4f} "
          f"(mean={np.mean(scores):.4f}, n={len(scores)})")
    return ok


if __name__ == "__main__":
    rng = np.random.default_rng(3407)  # seed everything, even the demo
    y_true = rng.integers(0, 2, size=200)
    y_pred = y_true.copy()
    y_pred[:40] = 1 - y_pred[:40]      # ~80% accurate toy "method"

    train_idx = list(range(0, 160))
    test_idx = list(range(160, 200))    # clean split
    seed_scores = np.array([0.81, 0.79, 0.83, 0.80, 0.82])

    print("=== Pilot sanity checks ===")
    results = [
        check_metric_in_range(accuracy_score(y_true, y_pred), name="accuracy"),
        check_beats_trivial_baseline(y_true, y_pred),
        check_no_split_leakage(train_idx, test_idx),
        check_seed_variance(seed_scores),
    ]
    print(f"\n{sum(results)}/{len(results)} checks passed. "
          f"{'Scale it.' if all(results) else 'Fix the red flags first.'}")
