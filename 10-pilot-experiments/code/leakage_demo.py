# leakage_demo.py — the classic data-leakage bug, and the fix.
# Kapoor & Narayanan 2023 (Patterns 4(9):100804) catalogued leakage across
# 294 papers in 17 fields. The most common form: you scale or feature-select
# using the WHOLE dataset, then split. The test set has already "seen" itself.
#
# This demo uses feature selection, because it shows the bug at full strength:
# 5,000 columns of pure noise, labels that are pure coin flips. Pick the 20
# columns that best "predict" the labels using ALL rows, then cross-validate,
# and the model looks skilled. Do the same selection inside each training fold
# (a Pipeline) and the skill collapses to chance, which is the truth.
# (Scaling on all rows leaks too, but usually by a hair; the habit that fixes
# both is the same: every fitted step goes inside the Pipeline.)
#
# Run it and watch the inflated number collapse when selection moves inside CV.
# Requires: numpy, scikit-learn

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(42)
X = rng.normal(size=(200, 5000))          # 200 rows, 5,000 noise features
y = rng.integers(0, 2, size=200)          # labels are PURE NOISE — truth is ~0.50
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)

# --- WRONG: select features on ALL rows, then cross-validate (leakage) -----
X_leaky = SelectKBest(f_classif, k=20).fit_transform(X, y)   # fit sees test rows
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
leaky = cross_val_score(model, X_leaky, y, cv=cv).mean()
print(f"[LEAKY ] accuracy = {leaky:.3f}   (selection fit on all 200 rows)")

# --- RIGHT: selection lives inside the Pipeline, fit on each TRAIN fold ----
pipe = make_pipeline(StandardScaler(), SelectKBest(f_classif, k=20),
                     LogisticRegression(max_iter=1000))
clean = cross_val_score(pipe, X, y, cv=cv).mean()
print(f"[CLEAN ] accuracy = {clean:.3f}   (selection fit inside each training fold)")
print(f"inflation from leakage = {leaky - clean:+.3f}")

# On pure-noise labels the CLEAN number should hover near 0.50 (a few points
# either side is ordinary cross-validation noise on 200 rows). If your real
# pipeline shows a result that looks "too good," run this pattern of thinking
# against it before you celebrate.
