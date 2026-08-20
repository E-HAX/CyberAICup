"""Model factory.

Every model is addressed by a short name plus a plain-dict parameter set, so a
grid point is a JSON-serialisable object that can be shipped to a container and
logged verbatim next to its score.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import config as C

# Models that accept per-sample weights rather than a class_weight argument.
SAMPLE_WEIGHT_MODELS = {"xgb", "lgbm", "cat", "hgb"}


def build_model(name: str, params: dict[str, Any], n_classes: int = len(C.LABELS)):
    """`n_classes` matters only for the boosters that take it explicitly; the
    hierarchical heads fit five- and two-class targets with the same factory."""
    p = dict(params)
    p.pop("class_weight_mode", None)

    if name == "xgb":
        from xgboost import XGBClassifier

        return XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            tree_method="hist",
            eval_metric="mlogloss",
            random_state=C.SEED,
            n_jobs=1,
            verbosity=0,
            **p,
        )
    if name == "lgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            objective="multiclass",
            num_class=n_classes,
            random_state=C.SEED,
            n_jobs=1,
            verbose=-1,
            **p,
        )
    if name == "cat":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            loss_function="MultiClass",
            random_seed=C.SEED,
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
            **p,
        )
    if name == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(random_state=C.SEED, **p)
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(random_state=C.SEED, n_jobs=1, **p)
    if name == "et":
        from sklearn.ensemble import ExtraTreesClassifier

        return ExtraTreesClassifier(random_state=C.SEED, n_jobs=1, **p)
    if name == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, random_state=C.SEED, **p),
        )
    if name == "svm":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC

        return make_pipeline(
            StandardScaler(), SVC(probability=True, random_state=C.SEED, **p)
        )
    if name == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_jobs=1, **p))
    if name == "tabpfn":
        # Prior-fitted transformer for small tabular problems: no task-specific
        # training, so there is nothing to search - the defaults are the method.
        # The published checkpoint is passed explicitly; the package's own
        # download path expects an interactive licence acceptance that a
        # container cannot perform.
        import os

        from tabpfn import TabPFNClassifier

        ckpt = p.pop("model_path", os.environ.get("TABPFN_CKPT"))
        if ckpt:
            p["model_path"] = ckpt
        return TabPFNClassifier(**p)
    if name == "tabicl":
        # In-context tabular classifier with openly downloadable weights - the
        # stand-in for TabPFN, whose weights now sit behind a licence token.
        from tabicl import TabICLClassifier

        return TabICLClassifier(**p)
    if name == "mlp":
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            StandardScaler(),
            MLPClassifier(max_iter=1500, random_state=C.SEED, **p),
        )
    raise ValueError(f"unknown model name: {name}")


def supports_class_weight(name: str) -> bool:
    return name in {"rf", "et", "logreg", "svm"}


def apply_class_weight(name: str, params: dict[str, Any]) -> dict[str, Any]:
    """Translate the searched class-weight mode into the model's own idiom."""
    mode = params.get("class_weight_mode", "none")
    p = {k: v for k, v in params.items() if k != "class_weight_mode"}
    if mode in ("none", None):
        return p
    if supports_class_weight(name):
        p["class_weight"] = mode
    return p


def sample_weights(name: str, mode: str, y: np.ndarray) -> np.ndarray | None:
    """Balanced sample weights for the boosted-tree models."""
    if mode not in ("balanced", "balanced_subsample") or name not in SAMPLE_WEIGHT_MODELS:
        return None
    counts = np.bincount(y, minlength=len(C.LABELS)).astype(np.float64)
    counts[counts == 0] = 1.0
    w = len(y) / (len(C.LABELS) * counts)
    return w[y]
