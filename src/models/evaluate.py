"""The fan-out unit: score one (model, hyperparameter, feature-set) point.

One call evaluates a single point under both cross-validation schemes and
writes its out-of-fold probability matrices to the work Volume under a path
keyed by the hash of the specification, so hundreds of containers can run
concurrently without ever touching the same file.

A specification is a plain dict:

    {"model": "xgb",
     "params": {"max_depth": 4, "learning_rate": 0.05, "class_weight_mode": "balanced"},
     "features": "full" | "top60" | "top120",
     "target_feats": true,
     "repeats": {"group": 2, "plain": 1}}
"""

from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss

import config as C
from features.likelihood import TargetAwareFeatures
from features.pipeline import Cache
from models import registry

N_CLASSES = len(C.LABELS)


def spec_hash(spec: dict) -> str:
    payload = json.dumps(spec, sort_keys=True, default=str).encode()
    return hashlib.sha1(payload).hexdigest()[:16]


def _feature_frame(cache: Cache, scheme: str, fold: int, spec: dict, tr_idx, va_idx):
    """Assemble fold-train and fold-validation matrices without leakage."""
    base_tr = cache.base_train.iloc[tr_idx]
    base_va = cache.base_train.iloc[va_idx]

    fset = spec.get("features", "full")
    if fset.startswith("top"):
        k = int(fset[3:])
        cols = cache.top_columns(scheme, fold, k)
        base_tr, base_va = base_tr[cols], base_va[cols]

    if spec.get("probes", False) and cache.has_probes:
        base_tr = pd.concat(
            [base_tr, cache.probes_train.iloc[tr_idx].set_index(base_tr.index)], axis=1
        )
        base_va = pd.concat(
            [base_va, cache.probes_train.iloc[va_idx].set_index(base_va.index)], axis=1
        )

    if not spec.get("target_feats", True):
        return base_tr, base_va

    extra_tr, extra_va = _target_aware(cache, spec, tr_idx, va_idx)
    return (
        pd.concat([base_tr, extra_tr], axis=1),
        pd.concat([base_va, extra_va], axis=1),
    )


def _target_aware(cache: Cache, spec: dict, tr_idx, va_idx):
    """Label-aware features with an inner out-of-fold loop on the training part.

    Fitting the class-conditional estimators on the fold's training rows and
    then scoring those same rows in-sample makes the likelihood and
    nearest-neighbour features far sharper on training data than on held-out
    data. A model trained on that mismatch over-trusts them and generalises
    worse - which is exactly what the first screening run showed. Generating
    the training-side values through an inner K-fold removes the mismatch:
    every row is scored by an estimator that never saw it.
    """
    from sklearn.model_selection import StratifiedKFold

    taf_params = spec.get("taf_params", {})
    n_inner = int(spec.get("taf_inner_folds", 5))
    y = cache.y
    raw_tr, raw_va = cache.train.iloc[tr_idx], cache.train.iloc[va_idx]
    base_tr_full = cache.base_train.iloc[tr_idx]
    base_va_full = cache.base_train.iloc[va_idx]

    inner = StratifiedKFold(n_inner, shuffle=True, random_state=C.SEED)
    parts = []
    for in_tr, in_va in inner.split(np.zeros(len(tr_idx)), y[tr_idx]):
        taf = TargetAwareFeatures(**taf_params)
        taf.fit(raw_tr.iloc[in_tr], base_tr_full.iloc[in_tr], y[tr_idx][in_tr])
        parts.append(taf.transform(raw_tr.iloc[in_va], base_tr_full.iloc[in_va]))
    extra_tr = pd.concat(parts).loc[raw_tr.index]

    taf_full = TargetAwareFeatures(**taf_params)
    taf_full.fit(raw_tr, base_tr_full, y[tr_idx])
    extra_va = taf_full.transform(raw_va, base_va_full)
    return extra_tr, extra_va


def _fit_predict(spec: dict, X_tr, y_tr, X_va) -> np.ndarray:
    """Fit and predict, averaging over seeds when the spec asks for bagging.

    In-context models are noticeably seed-sensitive at this sample size, and
    averaging several seeds is the cheapest variance reduction available - it
    needs no extra data and no tuning.
    """
    bag = int(spec.get("bag", 0))
    if bag > 1:
        probs = []
        for s in range(bag):
            sub = dict(spec)
            sub["bag"] = 0
            sub["params"] = {**spec.get("params", {}), "random_state": C.SEED + s}
            probs.append(_fit_predict(sub, X_tr, y_tr, X_va))
        return np.mean(probs, axis=0)
    name = spec["model"]
    raw_params = spec.get("params", {})
    mode = raw_params.get("class_weight_mode", "none")
    params = registry.apply_class_weight(name, raw_params)
    model = registry.build_model(name, params)
    sw = registry.sample_weights(name, mode, y_tr)
    if sw is not None:
        model.fit(X_tr.to_numpy(), y_tr, sample_weight=sw)
    else:
        model.fit(X_tr.to_numpy(), y_tr)
    proba = model.predict_proba(X_va.to_numpy())
    if proba.shape[1] != N_CLASSES:  # a fold can miss a rare class
        full = np.zeros((len(proba), N_CLASSES))
        full[:, model.classes_.astype(int)] = proba
        proba = full
    return proba


def evaluate_point(spec: dict, save: bool = True) -> dict:
    cache = Cache.get()
    y = cache.y
    result: dict = {
        "spec": spec,
        "hash": spec_hash(spec),
        "model": spec["model"],
        "features": spec.get("features", "full"),
        "target_feats": bool(spec.get("target_feats", True)),
    }
    oof_store: dict[str, np.ndarray] = {}
    t0 = time.time()

    for scheme, n_rep in spec.get("repeats", {"group": 2, "plain": 1}).items():
        splits = cache.folds[scheme][: n_rep * C.N_SPLITS]
        probs = np.zeros((n_rep, len(y), N_CLASSES), dtype=np.float32)
        for i, (tr_idx, va_idx) in enumerate(splits):
            rep = i // C.N_SPLITS
            X_tr, X_va = _feature_frame(cache, scheme, i, spec, tr_idx, va_idx)
            probs[rep, va_idx] = _fit_predict(spec, X_tr, y[tr_idx], X_va)

        accs = [accuracy_score(y, p.argmax(axis=1)) for p in probs]
        f1s = [f1_score(y, p.argmax(axis=1), average="macro") for p in probs]
        mean_p = probs.mean(axis=0)
        result[f"{scheme}_acc"] = float(np.mean(accs))
        result[f"{scheme}_acc_std"] = float(np.std(accs))
        result[f"{scheme}_macro_f1"] = float(np.mean(f1s))
        result[f"{scheme}_logloss"] = float(
            log_loss(y, np.clip(mean_p, 1e-9, 1), labels=list(range(N_CLASSES)))
        )
        oof_store[scheme] = mean_p

    extra_arrays: dict[str, np.ndarray] = {}
    if spec.get("with_test", False):
        # Finalists also refit on the full training set and store their test
        # probabilities, so the ensemble and the submission never need to refit.
        test_proba, _ = refit_full(spec)
        extra_arrays["test"] = test_proba.astype(np.float32)

    result["fit_seconds"] = round(time.time() - t0, 2)

    if save:
        C.OOF_DIR.mkdir(parents=True, exist_ok=True)
        path = C.OOF_DIR / f"{spec['model']}_{result['hash']}.npz"
        np.savez_compressed(
            path,
            meta=json.dumps(result),
            **{f"oof_{k}": v for k, v in oof_store.items()},
            **extra_arrays,
        )
        result["oof_path"] = str(path)
    return result


def refit_full(spec: dict) -> tuple[np.ndarray, np.ndarray]:
    """Refit on all training rows and predict the test set.

    Returns (test probabilities, fitted feature-column order).
    """
    cache = Cache.get()
    base_tr, base_te = cache.base_train, cache.base_test

    fset = spec.get("features", "full")
    if fset.startswith("top"):
        # Ranking on the full training set mirrors what each fold did locally.
        from sklearn.feature_selection import mutual_info_classif

        k = int(fset[3:])
        mi = mutual_info_classif(base_tr.to_numpy(), cache.y, random_state=C.SEED)
        cols = list(base_tr.columns.to_numpy()[np.argsort(mi)[::-1][:k]])
        base_tr, base_te = base_tr[cols], base_te[cols]

    if spec.get("probes", False) and cache.has_probes:
        base_tr = pd.concat([base_tr, cache.probes_train.set_index(base_tr.index)], axis=1)
        base_te = pd.concat([base_te, cache.probes_test.set_index(base_te.index)], axis=1)

    X_tr, X_te = base_tr, base_te
    if spec.get("target_feats", True):
        # Same inner out-of-fold construction as during cross-validation, so the
        # final model sees training-side label-aware features with the same
        # sharpness as the ones it will see at prediction time.
        from sklearn.model_selection import StratifiedKFold

        taf_params = spec.get("taf_params", {})
        n_inner = int(spec.get("taf_inner_folds", 5))
        inner = StratifiedKFold(n_inner, shuffle=True, random_state=C.SEED)
        parts = []
        for in_tr, in_va in inner.split(np.zeros(len(cache.train)), cache.y):
            taf = TargetAwareFeatures(**taf_params)
            taf.fit(cache.train.iloc[in_tr], cache.base_train.iloc[in_tr], cache.y[in_tr])
            parts.append(
                taf.transform(cache.train.iloc[in_va], cache.base_train.iloc[in_va])
            )
        extra_tr = pd.concat(parts).loc[cache.train.index]

        taf_full = TargetAwareFeatures(**taf_params)
        taf_full.fit(cache.train, cache.base_train, cache.y)
        extra_te = taf_full.transform(cache.test, cache.base_test)

        X_tr = pd.concat([base_tr, extra_tr], axis=1)
        X_te = pd.concat([base_te, extra_te], axis=1)

    proba = _fit_predict(spec, X_tr, cache.y, X_te)
    return proba, np.asarray(X_tr.columns)
