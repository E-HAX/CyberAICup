"""Hierarchical application x mode modelling with a derived Zoom decision rule.

The flat ten-class formulation hides the fact that the two label dimensions
have very different difficulty and very different failure modes. Measured on
the training set:

* the application is recoverable at roughly 0.91-0.92 accuracy;
* the mode is recoverable at roughly 0.89, but almost all of the loss sits in
  flows whose five packets are all below the audio/video band boundary.

Those "audio-only windows" are the first five packets of a flow that carries
only small packets, which is what a voice call looks like - and also what the
opening of a video call looks like before any video fragment appears. For
Discord, Google Meet and Messenger the two remain separable (grouped-CV AUC
0.83-0.88, presumably through codec and packetisation differences). For Zoom
they do not: the subset is 80 voice against 80 video flows and a gradient
boosting model scores AUC 0.551 on it, which at that sample size is
indistinguishable from a coin flip.

That measurement licenses one deterministic rule. On an evenly-split,
inseparable subset every assignment scores the same accuracy, so the choice is
free in accuracy and not free in macro-F1: sending the whole subset to
Zoom_voice maximises the macro average because it rescues the recall of the
smaller class at no cost to the larger one. The rule is derived from the class
balance rather than fitted, and uses the band boundary already declared in
`config.BAND_AUDIO`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import config as C
from models import registry

N_APPS = len(C.APPS)
N_CLASSES = len(C.LABELS)


def audio_only(df: pd.DataFrame) -> np.ndarray:
    """Flows whose every packet sits inside the audio band."""
    return df[C.LEN_COLS].to_numpy().max(axis=1) < C.BAND_AUDIO[1]


def _fit_predict(model: str, params: dict, X_tr, y_tr, X_va, n_out: int) -> np.ndarray:
    mode = params.get("class_weight_mode", "none")
    built = registry.build_model(
        model, registry.apply_class_weight(model, params), n_classes=n_out
    )
    sw = registry.sample_weights(model, mode, y_tr)
    if sw is not None:
        built.fit(X_tr.to_numpy(), y_tr, sample_weight=sw)
    else:
        built.fit(X_tr.to_numpy(), y_tr)
    proba = built.predict_proba(X_va.to_numpy())
    if proba.shape[1] != n_out:
        full = np.zeros((len(proba), n_out))
        full[:, built.classes_.astype(int)] = proba
        proba = full
    return proba


def heads_predict(
    model: str,
    params: dict,
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    mode_params: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (flat ten-class probabilities, hierarchical ten-class probabilities)."""
    app_tr, mode_tr = y_tr // 2, y_tr % 2
    flat = _fit_predict(model, params, X_tr, y_tr, X_va, N_CLASSES)
    p_app = _fit_predict(model, params, X_tr, app_tr, X_va, N_APPS)

    hier = np.zeros((len(X_va), N_CLASSES))
    for a in range(N_APPS):
        sel = np.flatnonzero(app_tr == a)
        p_video = _fit_predict(
            model, mode_params or params, X_tr.iloc[sel], mode_tr[sel], X_va, 2
        )[:, 1]
        hier[:, 2 * a] = p_app[:, a] * (1.0 - p_video)
        hier[:, 2 * a + 1] = p_app[:, a] * p_video
    return flat, hier


def mixture_predict(
    model: str,
    params: dict,
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    audio_tr: np.ndarray,
) -> np.ndarray:
    """Fifteen-class view: each video class is split by whether its window is
    audio-only, then the two video sub-classes are summed back together.

    A video class is a mixture of two visibly different populations - windows
    that contain a video fragment and windows that do not - and forcing one
    class boundary around both makes the model average over them. The split is
    a deterministic function of the features, so it adds no label information;
    what it changes is the loss decomposition, which is enough to move the
    decision boundary for the separable applications.
    """
    app_tr, mode_tr = y_tr // 2, y_tr % 2
    sub = np.where(mode_tr == 0, 0, np.where(audio_tr, 2, 1))
    y15 = app_tr * 3 + sub

    # Some sub-classes are empty (WhatsApp never opens a video call with an
    # audio-only window), and XGBoost requires contiguous labels, so fit on the
    # present sub-classes and scatter the columns back into the full layout.
    present = np.unique(y15)
    compact = np.searchsorted(present, y15)
    p_compact = _fit_predict(model, params, X_tr, compact, X_va, len(present))
    p15 = np.zeros((len(X_va), 3 * N_APPS))
    p15[:, present] = p_compact

    out = np.zeros((len(X_va), N_CLASSES))
    for a in range(N_APPS):
        out[:, 2 * a] = p15[:, 3 * a]
        out[:, 2 * a + 1] = p15[:, 3 * a + 1] + p15[:, 3 * a + 2]
    return out


def apply_zoom_rule(proba: np.ndarray, is_audio_only: np.ndarray) -> np.ndarray:
    """Send predicted-Zoom, audio-only flows to Zoom_voice. See module docstring."""
    pred = proba.argmax(axis=1)
    zoom = C.APPS.index("Zoom")
    target = C.LABEL_TO_ID[f"Zoom_{C.MODES[0]}"]
    pred[(pred // 2 == zoom) & is_audio_only] = target
    return pred


def score(y: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
    }
