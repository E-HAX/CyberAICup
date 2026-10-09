"""RTC Traffic Classification --- CyberAI Cup 2026, Task 3 (winning solution).

FINAL BLENDED PIPELINE: a light ensemble (RandomForest + LightGBM + XGBoost)
blended with a TabICL tabular foundation model, followed by the parameter-free
Zoom audio-only-window rule.

Validation (10-fold grouped nested CV; blend weight chosen on train folds only,
scored on held-out folds):

    TabICL alone                    : macro-F1 0.8201
    Ensemble alone (raw)            : macro-F1 0.7900
    Blend (w = 0.40 ensemble / 0.60 TabICL) : macro-F1 0.8286
    Blend + Zoom rule               : macro-F1 0.8318   <-- shipped

Why the blend works:
    - The ensemble's contribution is concentrated almost entirely in Zoom_voice
      (recall 0.712 vs TabICL's 0.438 on that class --- the one class where the
      recall-bias genuinely wins).
    - TabICL is equal-or-better on every other class.
    - The Zoom rule (predicted Zoom + audio-only 5-packet window -> force
      Zoom_voice) is applied AFTER the blend; it is parameter-free and orthogonal
      to which model produced the probabilities.

Inputs (defaults, relative to the repository root):
    data/Training_set.csv
    data/Testing_set.csv
    artifacts/hier_tabicl_2803104002.npz
        TabICL's test_flat / test_hier probability matrices, produced by a
        separate GPU run. This script only CONSUMES them; it does not retrain
        TabICL. The arrays are 327 x 10 and use the fixed label order
        Discord, GoogleMeet, Messenger, WhatsApp, Zoom x voice, video.

Output (default): submission.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

RNG = 42
np.random.seed(RNG)

T_FIXED = 2.0          # temperature scaling
BLEND_W_MINE = 0.40    # weight on the ensemble; TabICL gets (1 - this)
BAND_AUDIO_MAX = 300   # bytes; a flow is "audio-only" if every packet is < this

# Fixed label order used by the TabICL artifact.
APPS = ["Discord", "GoogleMeet", "Messenger", "WhatsApp", "Zoom"]
MODES = ["voice", "video"]
TABICL_LABEL_ORDER = [f"{a}_{m}" for a in APPS for m in MODES]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """31 hand-crafted features over the five (length, time) pairs."""
    df = df.copy()
    feats = pd.DataFrame(index=df.index)

    t_cols = [f"relative_time_{i}" for i in range(5)]
    l_cols = [f"packet_length_{i}" for i in range(5)]

    for c in l_cols:
        feats[c] = df[c]
    for c in t_cols[1:]:
        feats[c] = df[c]

    lens = df[l_cols].values
    times = df[t_cols].values
    delta_len = np.zeros((len(df), 4))

    for i in range(1, 5):
        feats[f"delta_len_{i}"] = lens[:, i] - lens[:, i - 1]
        delta_len[:, i - 1] = lens[:, i] - lens[:, i - 1]
    for i in range(1, 5):
        feats[f"delta_t_{i}"] = times[:, i] - times[:, i - 1]

    sorted_lens = np.sort(lens, axis=1)[:, ::-1]
    for rank in range(5):
        feats[f"len_rank_{rank}"] = sorted_lens[:, rank]

    feats["argmax_pos"] = np.argmax(lens, axis=1)
    feats["argmin_pos"] = np.argmin(lens, axis=1)
    feats["len_range"] = lens.max(axis=1) - lens.min(axis=1)
    feats["len_mean"] = lens.mean(axis=1)
    feats["len_std"] = lens.std(axis=1)
    feats["len_median"] = np.median(lens, axis=1)
    feats["total_span"] = times[:, -1] - times[:, 0]
    feats["gp_log_pl3"] = np.log(np.clip(df["packet_length_3"].values, 1, None))
    feats["jump_pos"] = np.argmax(delta_len, axis=1) + 1

    return feats


def make_rf():
    return RandomForestClassifier(
        n_estimators=600, class_weight="balanced_subsample",
        random_state=RNG, n_jobs=1,
    )


def make_lgb(seed=RNG):
    return lgb.LGBMClassifier(
        n_estimators=400, max_depth=4, num_leaves=15, learning_rate=0.05,
        class_weight="balanced", random_state=seed, verbosity=-1, n_jobs=1,
    )


def make_xgb():
    return xgb.XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="mlogloss", random_state=RNG, verbosity=0, n_jobs=1,
    )


def apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
    logp = np.log(np.clip(probs, 1e-9, 1))
    scaled = np.exp(logp / T)
    return scaled / scaled.sum(axis=1, keepdims=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/Training_set.csv")
    parser.add_argument("--test", default="data/Testing_set.csv")
    parser.add_argument("--tabicl", default="artifacts/hier_tabicl_2803104002.npz")
    parser.add_argument("--output", default="submission.csv")
    parser.add_argument("--blend-weight", type=float, default=BLEND_W_MINE,
                        help="weight on the ensemble; TabICL gets (1 - w)")
    parser.add_argument("--temperature", type=float, default=T_FIXED)
    parser.add_argument("--band-audio-max", type=float, default=BAND_AUDIO_MAX)
    args = parser.parse_args(argv)

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)

    X = engineer_features(train)
    X_test = engineer_features(test)

    le = LabelEncoder()
    y = le.fit_transform(train["label"])
    classes = list(le.classes_)          # alphabetical label order
    n = len(y)
    pi_hat = np.bincount(y) / n

    print(f"Train: {X.shape}, Test: {X_test.shape}, classes: {len(classes)}")

    # --- Ensemble probabilities on the test set (no recall-bias here; those
    # were tuned for a recall objective and would distort the blend). ----------
    rf_full = make_rf(); rf_full.fit(X, y)
    lgb_full = make_lgb(); lgb_full.fit(X, y)
    xgb_full = make_xgb(); xgb_full.fit(X, y)

    test_ens = (
        rf_full.predict_proba(X_test)
        + lgb_full.predict_proba(X_test)
        + xgb_full.predict_proba(X_test)
    ) / 3
    test_scaled = apply_temperature(test_ens, args.temperature)
    test_corr = test_scaled / pi_hat[None, :]
    my_test_prob = test_corr / test_corr.sum(axis=1, keepdims=True)

    # --- TabICL test probabilities (flat and hierarchical, averaged). ---------
    tab = np.load(args.tabicl, allow_pickle=True)
    test_flat_raw = tab["test_flat"]
    test_hier_raw = tab["test_hier"]

    remap_idx = [TABICL_LABEL_ORDER.index(lab) for lab in classes]
    test_flat = test_flat_raw[:, remap_idx]
    test_hier = test_hier_raw[:, remap_idx]
    tabicl_test_prob = (test_flat + test_hier) / 2

    # --- Blend, then the Zoom rule. ------------------------------------------
    blend_prob = args.blend_weight * my_test_prob + (1 - args.blend_weight) * tabicl_test_prob
    pred = blend_prob.argmax(axis=1)

    zoom_video_idx = classes.index("Zoom_video")
    zoom_voice_idx = classes.index("Zoom_voice")

    L = [f"packet_length_{i}" for i in range(5)]
    audio_only = test[L].to_numpy().max(axis=1) < args.band_audio_max

    trigger = ((pred == zoom_video_idx) | (pred == zoom_voice_idx)) & audio_only
    final_pred = pred.copy()
    final_pred[trigger] = zoom_voice_idx

    print(f"Zoom rule flipped {trigger.sum()} of {len(pred)} test rows to Zoom_voice.")

    # --- Write submission. ----------------------------------------------------
    test_labels = [classes[p] for p in final_pred]
    submission = pd.DataFrame({
        "idx": np.arange(1, len(test_labels) + 1),
        "label": test_labels,
    })
    submission.to_csv(args.output, index=False, header=False)

    print(f"Saved {len(submission)} predictions to {args.output}")
    print(pd.Series(test_labels).value_counts())
    print("\nExpected macro-F1 (10-fold grouped nested CV on training data): ~0.8318")


if __name__ == "__main__":
    main()
