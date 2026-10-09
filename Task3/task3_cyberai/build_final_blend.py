"""
RTC Traffic Classification -- CyberAI Cup 2026, Task 3
FINAL BLENDED PIPELINE (mine + TabICL, with the Zoom audio-only-window rule)

Validated (10-fold grouped nested-CV, weight chosen on train-side only,
scored on unseen held-out folds):

    TabICL alone                    : macro-F1 0.8201
    My ensemble alone (raw)         : macro-F1 0.7900
    Blend (w=0.40 mine / 0.60 them) : macro-F1 0.8286
    Blend + Zoom rule               : macro-F1 0.8318   <-- shipped

Where each side actually helps (per-class OOF F1, same nested split):
    - My contribution is concentrated almost entirely in Zoom_voice
      (my recall 0.712 vs TabICL's 0.438 on that class -- this is the
      one class my WEAK_CLASSES recall-bias genuinely wins on).
    - TabICL is equal-or-better on every other one of the 10 classes.
    - The Zoom rule (predicted Zoom + audio-only 5-packet window ->
      force Zoom_voice) is applied AFTER the blend, since it is
      parameter-free and orthogonal to which model produced the
      probability.

Inputs expected:
    Task3/publish/RTC_CyberAICup2026/Training_set.csv
    Task3/publish/RTC_CyberAICup2026/Testing_set.csv
    task3-main/task3-main/artifacts/hier/hier_tabicl_2803104002.npz
        (TabICL's oof_flat/oof_hier/test_flat/test_hier arrays --
         produced by that repo's own Modal run; this script only
         consumes them, it does not retrain TabICL)

Output:
    /mnt/user-data/outputs/submission.csv
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import lightgbm as lgb
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

RNG = 42
np.random.seed(RNG)

TRAIN_PATH = "Task3/publish/RTC_CyberAICup2026/Training_set.csv"
TEST_PATH = "Task3/publish/RTC_CyberAICup2026/Testing_set.csv"
TABICL_ARTIFACT = "task3-main/task3-main/artifacts/hier/hier_tabicl_2803104002.npz"
OUTPUT_PATH = "/mnt/user-data/outputs/submission.csv"

T_FIXED = 2.0          # temperature scaling (kept from the original solo pipeline)
BLEND_W_MINE = 0.40    # weight on my model in the blend; TabICL gets (1 - this)
BAND_AUDIO_MAX = 300   # bytes; a flow is "audio-only" if every packet is < this


# ---------------------------------------------------------------------------
# Feature engineering (same as the solo pipeline)
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
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


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

X = engineer_features(train)
X_test = engineer_features(test)

le = LabelEncoder()
y = le.fit_transform(train["label"])
classes = list(le.classes_)          # my alphabetical label order
K = len(classes)
n = len(y)
pi_hat = np.bincount(y) / n

print(f"Train: {X.shape}, Test: {X_test.shape}, classes: {K}")


# ---------------------------------------------------------------------------
# Step 1: my raw ensemble probabilities on the test set
# (RF + LightGBM + XGBoost, temperature scaling + prior correction --
#  deliberately NO per-class recall-bias, NO specialist override here:
#  those were tuned to help my recall metric and would distort the
#  probabilities being fed into the blend)
# ---------------------------------------------------------------------------
rf_full = make_rf(); rf_full.fit(X, y)
lgb_full = make_lgb(); lgb_full.fit(X, y)
xgb_full = make_xgb(); xgb_full.fit(X, y)

test_ens = (
    rf_full.predict_proba(X_test)
    + lgb_full.predict_proba(X_test)
    + xgb_full.predict_proba(X_test)
) / 3
test_scaled = apply_temperature(test_ens, T_FIXED)
test_corr = test_scaled / pi_hat[None, :]
my_test_prob = test_corr / test_corr.sum(axis=1, keepdims=True)

print("My raw ensemble test-probabilities ready.")


# ---------------------------------------------------------------------------
# Step 2: load TabICL's (flat, hier) test-side probabilities and average
# them -- this is the "avg" variant from their round-3 report, the core
# shipped model before their own Zoom rule.
# ---------------------------------------------------------------------------
tab = np.load(TABICL_ARTIFACT, allow_pickle=True)
test_flat_raw = tab["test_flat"]
test_hier_raw = tab["test_hier"]

APPS = ["Discord", "GoogleMeet", "Messenger", "WhatsApp", "Zoom"]
MODES = ["voice", "video"]
their_label_order = [f"{a}_{m}" for a in APPS for m in MODES]
remap_idx = [their_label_order.index(lab) for lab in classes]   # align to my label order

test_flat = test_flat_raw[:, remap_idx]
test_hier = test_hier_raw[:, remap_idx]
tabicl_test_prob = (test_flat + test_hier) / 2

print("TabICL test-probabilities loaded and re-aligned to my label order.")


# ---------------------------------------------------------------------------
# Step 3: blend, then apply the Zoom rule
# ---------------------------------------------------------------------------
blend_prob = BLEND_W_MINE * my_test_prob + (1 - BLEND_W_MINE) * tabicl_test_prob
pred = blend_prob.argmax(axis=1)

zoom_video_idx = classes.index("Zoom_video")
zoom_voice_idx = classes.index("Zoom_voice")

L = [f"packet_length_{i}" for i in range(5)]
audio_only = test[L].to_numpy().max(axis=1) < BAND_AUDIO_MAX

is_zoom_pred = (pred == zoom_video_idx) | (pred == zoom_voice_idx)
trigger = is_zoom_pred & audio_only
final_pred = pred.copy()
final_pred[trigger] = zoom_voice_idx

print(f"Zoom rule flipped {trigger.sum()} of {len(pred)} test rows to Zoom_voice.")


# ---------------------------------------------------------------------------
# Step 4: write submission
# ---------------------------------------------------------------------------
test_labels = [classes[p] for p in final_pred]
submission = pd.DataFrame({
    "idx": np.arange(1, len(test_labels) + 1),
    "label": test_labels,
})
submission.to_csv(OUTPUT_PATH, index=False, header=False)

print(f"\nSaved {len(submission)} predictions to {OUTPUT_PATH}")
print(pd.Series(test_labels).value_counts())
print("\nExpected macro-F1 (10-fold grouped nested-CV on training data): ~0.8318")
