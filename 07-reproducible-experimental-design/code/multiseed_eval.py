"""
multiseed_eval.py — a minimal, honest multi-seed evaluation harness.

The point: a single run that "beats the baseline" is statistically empty
(Bouthillier et al. 2021, "Accounting for Variance in ML Benchmarks", MLSys;
arXiv:2103.03098). Run several seeds, report mean +/- std, and test the
difference with a method that respects the multi-seed structure.

This is a TEMPLATE. Swap in your own train/eval; keep the seed discipline,
the leakage guard, and the significance test. Verify library versions live:
scikit-learn 1.9.0 and SciPy 1.18.0 as of mid-2026.

    pip install numpy scikit-learn scipy
    # optional, for the ASO test recommended for multi-seed DL comparisons:
    # pip install deepsig     # the deep-significance package, current version (arXiv:2204.06815), tau=0.2
"""
from __future__ import annotations
import random
import numpy as np
from scipy import stats


def set_all_seeds(seed: int) -> None:
    """Seed every RNG you touch. If you use torch, also seed it and set
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False."""
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def run_one_seed(seed: int, condition: str) -> float:
    """Train + evaluate ONE configuration under ONE seed; return the primary metric.

    Leakage guard: fit preprocessing on TRAIN only, then transform val/test.
    Replace this stub with your real pipeline; the structure is what matters.
    """
    set_all_seeds(seed)
    rng = np.random.default_rng(seed)
    base = 0.71 if condition == "baseline" else 0.745  # stand-in metric
    return float(base + rng.normal(0, 0.012))           # seed-to-seed variance


def evaluate(condition: str, seeds: list[int]) -> np.ndarray:
    return np.array([run_one_seed(s, condition) for s in seeds])


def report(name: str, scores: np.ndarray) -> None:
    print(f"{name:>12}: mean={scores.mean():.4f}  std={scores.std(ddof=1):.4f}  "
          f"n={len(scores)}  runs={np.round(scores, 4).tolist()}")


def main() -> None:
    seeds = [0, 1, 2, 3, 4]                    # >= 5; pin them in your design doc
    baseline = evaluate("baseline", seeds)
    method = evaluate("method", seeds)
    report("baseline", baseline)
    report("method", method)

    # A starting test. For multi-seed DL score vectors, prefer deep-significance
    # ASO (weaker distributional assumptions than a t-test). Report effect size,
    # not just p — a tiny, "significant" gap can be practically meaningless.
    t, p = stats.ttest_ind(method, baseline, equal_var=False)
    diff = method.mean() - baseline.mean()
    pooled = np.sqrt((baseline.var(ddof=1) + method.var(ddof=1)) / 2)
    cohens_d = diff / pooled if pooled else float("nan")
    print(f"\nabsolute diff = {diff:+.4f}   Welch t = {t:.3f}   p = {p:.4f}   "
          f"Cohen's d = {cohens_d:.2f}")
    print("Pre-register this test in your design doc. Do NOT pick it after seeing p.")


if __name__ == "__main__":
    main()
