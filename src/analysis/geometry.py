"""Label geometry of the RTC task: which flows are winnable, and by how much.

Round 3 stopped asking which estimator to use and asked what the labels can
possibly support. Every record is the first five packets of a flow, so a video
call whose opening packets carry no video fragment produces a record that is
indistinguishable from a voice call. This module measures that effect and turns
it into a ceiling.

Run it with `python -m analysis.geometry` (or the Modal function
`run_geometry`) to reproduce every number quoted in `ABLATION3.md` sections 1-4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import config as C
import io_utils as IO
from features import base as fbase

# The audio band's upper edge doubles as the definition of an audio-only
# window: a flow whose every packet is smaller than this carries no video
# fragment in the five packets we are shown.
AUDIO_MAX = C.BAND_AUDIO[1]


def audio_only(df: pd.DataFrame) -> np.ndarray:
    return df[C.LEN_COLS].to_numpy().max(axis=1) < AUDIO_MAX


def regime_table(train: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    """Accuracy split by packet-size regime, given any set of predictions."""
    y = IO.labels_to_ids(train["label"])
    small = audio_only(train)
    err = pred != y
    return pd.DataFrame(
        [
            {
                "regime": name,
                "flows": int(mask.sum()),
                "share": round(float(mask.mean()), 4),
                "accuracy": round(1.0 - float(err[mask].mean()), 4),
                "errors": int(err[mask].sum()),
            }
            for name, mask in (("audio-only window", small),
                               ("contains a larger packet", ~small))
        ]
    )


def separability(train: pd.DataFrame, n_repeats: int = 3) -> pd.DataFrame:
    """Per-application video-vs-voice AUC among audio-only windows.

    This is the decisive measurement of the round. An application whose two
    modes stay separable inside the audio band is a modelling problem; one
    whose modes do not is a ceiling.
    """
    import lightgbm as lgb

    small = audio_only(train)
    app, _ = IO.split_app_mode(train["label"])
    rows = []
    for name in C.APPS:
        sub = train[(app == name).to_numpy() & small].reset_index(drop=True)
        y = sub["label"].str.endswith("video").to_numpy().astype(int)
        if y.sum() < 10 or (1 - y).sum() < 10:
            rows.append({"application": name, "audio_only_flows": len(sub),
                         "of_which_video": int(y.sum()), "auc": np.nan})
            continue
        X = fbase.build(sub[C.FEATURE_COLS])
        groups = IO.group_keys(sub)
        oof = np.zeros(len(y))
        for rep in range(n_repeats):
            splitter = StratifiedGroupKFold(C.N_SPLITS, shuffle=True, random_state=rep)
            for tr_idx, va_idx in splitter.split(X, y, groups):
                model = lgb.LGBMClassifier(
                    n_estimators=400, num_leaves=15, learning_rate=0.05,
                    colsample_bytree=0.5, subsample=0.8, subsample_freq=1,
                    verbose=-1, random_state=C.SEED,
                )
                model.fit(X.iloc[tr_idx], y[tr_idx])
                oof[va_idx] += model.predict_proba(X.iloc[va_idx])[:, 1] / n_repeats
        rows.append({"application": name, "audio_only_flows": len(sub),
                     "of_which_video": int(y.sum()),
                     "auc": round(float(roc_auc_score(y, oof)), 3)})
    return pd.DataFrame(rows)


def threshold_stability(train: pd.DataFrame) -> pd.DataFrame:
    """Is the ambiguous Zoom subset an artefact of the 300-byte boundary?"""
    y = IO.labels_to_ids(train["label"])
    longest = train[C.LEN_COLS].to_numpy().max(axis=1)
    rows = []
    for thr in (200, 250, 300, 400, 500):
        mask = (y // 2 == C.APPS.index("Zoom")) & (longest < thr)
        rows.append({"threshold_bytes": thr, "flows": int(mask.sum()),
                     "voice": int((y[mask] == C.LABEL_TO_ID["Zoom_voice"]).sum()),
                     "video": int((y[mask] == C.LABEL_TO_ID["Zoom_video"]).sum())})
    return pd.DataFrame(rows)


def ceiling(train: pd.DataFrame) -> dict:
    """Best attainable accuracy and macro-F1 given the inseparable subset.

    Treats every flow outside the ambiguous Zoom subset as perfectly classified
    and sweeps the only remaining free choice: what fraction of the ambiguous
    subset is called voice. Accuracy is flat in that fraction - the subset is
    evenly split - and macro-F1 is not, which is what the Zoom rule exploits.
    """
    y = IO.labels_to_ids(train["label"])
    amb = (y // 2 == C.APPS.index("Zoom")) & audio_only(train)
    voice_id, video_id = C.LABEL_TO_ID["Zoom_voice"], C.LABEL_TO_ID["Zoom_video"]
    n_v, n_d = int((y[amb] == voice_id).sum()), int((y[amb] == video_id).sum())
    tot_v, tot_d = int((y == voice_id).sum()), int((y == video_id).sum())

    best = {"macro_f1": -1.0}
    for p in np.linspace(0.0, 1.0, 101):
        tp_v, fp_v = n_v * p, n_d * p
        prec_v = tp_v / max(tp_v + fp_v, 1e-9)
        rec_v = tp_v / tot_v
        f1_v = 0.0 if prec_v + rec_v == 0 else 2 * prec_v * rec_v / (prec_v + rec_v)
        tp_d = (tot_d - n_d) + n_d * (1 - p)
        fp_d = n_v * (1 - p)
        prec_d = tp_d / max(tp_d + fp_d, 1e-9)
        rec_d = tp_d / tot_d
        f1_d = 2 * prec_d * rec_d / (prec_d + rec_d)
        macro = (8.0 + f1_v + f1_d) / len(C.LABELS)
        if macro > best["macro_f1"]:
            best = {"macro_f1": round(macro, 4), "voice_fraction": round(float(p), 2),
                    "zoom_voice_f1": round(f1_v, 3), "zoom_video_f1": round(f1_d, 3)}
    best["accuracy"] = round((len(y) - min(n_v, n_d)) / len(y), 4)
    best["ambiguous_flows"] = int(amb.sum())
    best["ambiguous_split"] = [n_v, n_d]
    return best


def rule_gain(train: pd.DataFrame, proba: np.ndarray, n_boot: int = 2000) -> dict:
    """Paired bootstrap of the Zoom rule's macro-F1 gain on one probability matrix."""
    from models.hier import apply_zoom_rule

    y = IO.labels_to_ids(train["label"])
    small = audio_only(train)
    before = proba.argmax(axis=1)
    after = apply_zoom_rule(proba, small)
    rng = np.random.default_rng(C.SEED)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        deltas[b] = f1_score(y[idx], after[idx], average="macro") - f1_score(
            y[idx], before[idx], average="macro"
        )
    return {
        "macro_f1_before": round(float(f1_score(y, before, average="macro")), 4),
        "macro_f1_after": round(float(f1_score(y, after, average="macro")), 4),
        "accuracy_before": round(float(accuracy_score(y, before)), 4),
        "accuracy_after": round(float(accuracy_score(y, after)), 4),
        "delta_mean": round(float(deltas.mean()), 4),
        "delta_ci95": [round(float(np.percentile(deltas, 2.5)), 4),
                       round(float(np.percentile(deltas, 97.5)), 4)],
        "p_gain_positive": round(float((deltas > 0).mean()), 3),
    }


def report() -> dict:
    train = IO.load_train()
    return {
        "threshold_stability": threshold_stability(train).to_dict("records"),
        "separability": separability(train).to_dict("records"),
        "ceiling": ceiling(train),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(report(), indent=2))
