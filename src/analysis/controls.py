"""Negative controls N1 and N2 — run before any experiment is interpreted.

N1 permutes the training labels. A leaking fold split or a feature cache that
carries label information would show accuracy well above chance here.
N2 predicts the training prior and ignores the features entirely; nothing in the
round is allowed to score below it.
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import accuracy_score

import config as C
from features.pipeline import Cache
from models import cv as cvmod


def run(seed: int = C.SEED) -> dict:
    import lightgbm as lgb

    cache = Cache.get()
    X = cache.base_train.to_numpy()
    y = cache.y
    folds = cvmod.load_folds()["group"][: C.N_SPLITS]
    rng = np.random.RandomState(seed)

    y_shuffled = rng.permutation(y)
    pred = np.zeros(len(y), dtype=int)
    for tr_idx, va_idx in folds:
        clf = lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=31,
            random_state=C.SEED, n_jobs=2, verbose=-1,
        )
        clf.fit(X[tr_idx], y_shuffled[tr_idx])
        pred[va_idx] = clf.predict(X[va_idx])
    n1 = float(accuracy_score(y_shuffled, pred))

    counts = np.bincount(y, minlength=len(C.LABELS))
    n2 = float(counts.max() / counts.sum())

    result = {
        "N1_shuffled_label_accuracy": n1,
        "N1_expected_range": [0.08, 0.20],
        "N1_passes": bool(n1 < 0.25),
        "N2_prior_only_accuracy": n2,
        "round_valid": bool(n1 < 0.25),
    }
    (C.REPORTS_DIR / "ablation2_controls.json").write_text(json.dumps(result, indent=2))
    return result
