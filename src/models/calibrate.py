"""Experiment C1 — probability calibration on stored out-of-fold matrices.

The error study showed the model assigning 0.61–0.86 probability to the class it
gets wrong, so its probabilities are not trustworthy as they stand. Three
calibrators are compared, each fitted out-of-fold so the reported numbers are not
fitted to their own evaluation rows:

    temperature   one scalar on the logits — the standard minimal fix
    vector        per-class scale and offset
    dirichlet     a full linear map on log-probabilities

Accuracy is reported alongside expected calibration error and negative
log-likelihood, because a calibrator that leaves accuracy untouched while fixing
confidence is still a success: the decision layer in C2 and the ensemble in F1
both consume probabilities, not argmaxes.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold

import config as C

N_CLASSES = len(C.LABELS)
EPS = 1e-9


def expected_calibration_error(prob: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    conf = prob.max(axis=1)
    pred = prob.argmax(axis=1)
    acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def _logits(prob: np.ndarray) -> np.ndarray:
    return np.log(np.clip(prob, EPS, 1.0))


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(prob: np.ndarray, y: np.ndarray) -> float:
    """Single scalar chosen by grid search on negative log-likelihood."""
    z = _logits(prob)
    grid = np.linspace(0.4, 4.0, 73)
    losses = [
        log_loss(y, _softmax(z / t), labels=list(range(N_CLASSES))) for t in grid
    ]
    return float(grid[int(np.argmin(losses))])


def apply_temperature(prob: np.ndarray, t: float) -> np.ndarray:
    return _softmax(_logits(prob) / t)


def fit_vector(prob: np.ndarray, y: np.ndarray):
    """Per-class scale and offset, fitted by multinomial logistic regression."""
    z = _logits(prob)
    model = LogisticRegression(max_iter=2000, C=1e4, random_state=C.SEED)
    model.fit(z, y)
    return model


def fit_dirichlet(prob: np.ndarray, y: np.ndarray):
    """Full linear map on log-probabilities, mildly regularised."""
    z = _logits(prob)
    model = LogisticRegression(max_iter=3000, C=1.0, random_state=C.SEED)
    model.fit(z, y)
    return model


def evaluate(prob: np.ndarray, y: np.ndarray, n_splits: int = 5) -> dict:
    """Compare calibrators under an inner split so nothing scores itself."""
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=C.SEED)
    out = {name: np.zeros_like(prob) for name in ("temperature", "vector", "dirichlet")}

    for tr, va in cv.split(prob, y):
        t = fit_temperature(prob[tr], y[tr])
        out["temperature"][va] = apply_temperature(prob[va], t)
        for name, fitter in (("vector", fit_vector), ("dirichlet", fit_dirichlet)):
            model = fitter(prob[tr], y[tr])
            p = model.predict_proba(_logits(prob[va]))
            if p.shape[1] != N_CLASSES:
                full = np.zeros((len(p), N_CLASSES))
                full[:, model.classes_.astype(int)] = p
                p = full
            out[name][va] = p

    def summary(p: np.ndarray) -> dict:
        return {
            "accuracy": float(accuracy_score(y, p.argmax(axis=1))),
            "nll": float(log_loss(y, np.clip(p, EPS, 1), labels=list(range(N_CLASSES)))),
            "ece": expected_calibration_error(p, y),
            "mean_confidence_when_wrong": float(
                p.max(axis=1)[p.argmax(axis=1) != y].mean()
            ),
        }

    result = {"raw": summary(prob)}
    for name, p in out.items():
        result[name] = summary(p)
    names = list(result)
    result["best_by_nll"] = min(names, key=lambda k: result[k]["nll"])
    result["best_by_ece"] = min(names, key=lambda k: result[k]["ece"])
    # A calibrator that buys confidence at the cost of accuracy is not a
    # calibrator worth shipping here, so the selected one is the best-calibrated
    # option among those that stay within a point of the raw accuracy.
    raw_acc = result["raw"]["accuracy"]
    eligible = [k for k in names if result[k]["accuracy"] >= raw_acc - 0.01]
    result["selected"] = min(eligible, key=lambda k: result[k]["nll"])
    return result, out
