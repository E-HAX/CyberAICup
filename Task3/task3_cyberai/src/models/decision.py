"""Experiment C2/C3 — a decision layer on top of calibrated probabilities.

Inside the Zoom pair the model scores 0.6548 where a constant predictor scores
0.6825: it is losing to a rule that ignores the input. C2 asks whether a small
per-class bias on the log-probabilities recovers that loss without damaging the
classes that already work. C3 asks the narrower question — defer to the
application-level prior only when the top two candidates are the same
application in different modes and the margin is thin.

Both are evaluated **nested**: the bias is refitted inside each split and scored
on rows it never saw, because a bias fitted on all the data and scored on all the
data would improve by construction.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

import config as C

N_CLASSES = len(C.LABELS)
EPS = 1e-9
ZOOM = [C.LABEL_TO_ID["Zoom_voice"], C.LABEL_TO_ID["Zoom_video"]]


def _logits(prob: np.ndarray) -> np.ndarray:
    return np.log(np.clip(prob, EPS, 1.0))


def fit_bias(
    prob: np.ndarray, y: np.ndarray, l2: float = 0.5, rounds: int = 6, seed: int = C.SEED
) -> np.ndarray:
    """Coordinate ascent on accuracy with an L2 cap on the bias vector.

    Accuracy is not differentiable, so the search is a bounded coordinate sweep
    rather than a gradient method. The L2 penalty is what keeps the layer a
    correction rather than a licence to rewrite the prior wholesale.
    """
    z = _logits(prob)
    bias = np.zeros(N_CLASSES)
    grid = np.linspace(-1.2, 1.2, 25)
    rng = np.random.RandomState(seed)

    def score(b: np.ndarray) -> float:
        pred = (z + b).argmax(axis=1)
        return float((pred == y).mean()) - l2 * float(np.dot(b, b)) / N_CLASSES

    best = score(bias)
    for _ in range(rounds):
        improved = False
        for k in rng.permutation(N_CLASSES):
            base = bias[k]
            for v in grid:
                bias[k] = v
                s = score(bias)
                if s > best + 1e-12:
                    best, base, improved = s, v, True
            bias[k] = base
        if not improved:
            break
    return bias


def apply_bias(prob: np.ndarray, bias: np.ndarray) -> np.ndarray:
    z = _logits(prob) + bias
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def defer_within_pair(prob: np.ndarray, margin: float, prior: np.ndarray) -> np.ndarray:
    """C3: when the top two are one application's two modes and the call is
    close, answer with whichever mode that application shows more often."""
    order = np.argsort(prob, axis=1)[:, ::-1]
    pred = order[:, 0].copy()
    top1, top2 = order[:, 0], order[:, 1]
    gap = prob[np.arange(len(prob)), top1] - prob[np.arange(len(prob)), top2]
    same_app = np.array(
        [C.LABELS[a].rsplit("_", 1)[0] == C.LABELS[b].rsplit("_", 1)[0] for a, b in zip(top1, top2)]
    )
    ambiguous = same_app & (gap < margin)
    for i in np.where(ambiguous)[0]:
        pair = [top1[i], top2[i]]
        pred[i] = pair[int(np.argmax(prior[pair]))]
    return pred


def evaluate(prob: np.ndarray, y: np.ndarray, n_splits: int = 5, l2: float = 0.5) -> dict:
    """Nested evaluation of C2 and C3 against the plain argmax."""
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=C.SEED)
    pred_raw = prob.argmax(axis=1)
    pred_bias = np.zeros(len(y), dtype=int)
    pred_defer = {m: np.zeros(len(y), dtype=int) for m in (0.05, 0.1, 0.2, 0.35)}
    biases = []

    for tr, va in cv.split(prob, y):
        b = fit_bias(prob[tr], y[tr], l2=l2)
        biases.append(b)
        pred_bias[va] = (_logits(prob[va]) + b).argmax(axis=1)
        prior = np.bincount(y[tr], minlength=N_CLASSES) / len(tr)
        for m in pred_defer:
            pred_defer[m][va] = defer_within_pair(prob[va], m, prior)

    zoom_mask = np.isin(y, ZOOM)

    def summary(pred: np.ndarray) -> dict:
        return {
            "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro")),
            "zoom_pair_accuracy": float((pred[zoom_mask] == y[zoom_mask]).mean()),
            "n_changed_vs_raw": int((pred != pred_raw).sum()),
        }

    result = {
        "raw_argmax": summary(pred_raw),
        "C2_logit_bias_nested": summary(pred_bias),
        "C2_mean_bias": np.mean(biases, axis=0).round(3).tolist(),
        "C2_bias_labels": C.LABELS,
        "C2_bias_stability": np.std(biases, axis=0).round(3).tolist(),
    }
    for m, pred in pred_defer.items():
        result[f"C3_defer_margin_{m}"] = summary(pred)
    return result
