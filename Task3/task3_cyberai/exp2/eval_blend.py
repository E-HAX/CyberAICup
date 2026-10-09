"""Honest grouped-CV evaluation of build_final_blend.py's logic.

The script claims a validated macro-F1 of 0.8318 but contains no validation
code - it fits on the full training set and writes a submission. This module
reproduces its model and blend exactly, but scores it on the canonical grouped
folds (artifacts/folds.npz), the same scheme every number in this repo uses.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'src')
import config as C
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
import lightgbm as lgb
import xgboost as xgb
import warnings; warnings.filterwarnings("ignore")

RNG = 42
T_FIXED = 2.0
BLEND_W_MINE = 0.40
BAND_AUDIO_MAX = 300

def engineer_features(df):
    df = df.copy()
    feats = pd.DataFrame(index=df.index)
    t_cols = [f"relative_time_{i}" for i in range(5)]
    l_cols = [f"packet_length_{i}" for i in range(5)]
    for c in l_cols: feats[c] = df[c]
    for c in t_cols[1:]: feats[c] = df[c]
    lens = df[l_cols].values; times = df[t_cols].values
    delta_len = np.zeros((len(df), 4))
    for i in range(1,5):
        feats[f"delta_len_{i}"] = lens[:,i]-lens[:,i-1]
        delta_len[:,i-1] = lens[:,i]-lens[:,i-1]
    for i in range(1,5):
        feats[f"delta_t_{i}"] = times[:,i]-times[:,i-1]
    sorted_lens = np.sort(lens,axis=1)[:,::-1]
    for rank in range(5): feats[f"len_rank_{rank}"] = sorted_lens[:,rank]
    feats["argmax_pos"] = np.argmax(lens,axis=1)
    feats["argmin_pos"] = np.argmin(lens,axis=1)
    feats["len_range"] = lens.max(1)-lens.min(1)
    feats["len_mean"] = lens.mean(1); feats["len_std"] = lens.std(1)
    feats["len_median"] = np.median(lens,axis=1)
    feats["total_span"] = times[:,-1]-times[:,0]
    feats["gp_log_pl3"] = np.log(np.clip(df["packet_length_3"].values,1,None))
    feats["jump_pos"] = np.argmax(delta_len,axis=1)+1
    return feats

def make_rf(): return RandomForestClassifier(n_estimators=600, class_weight="balanced_subsample", random_state=RNG, n_jobs=1)
def make_lgb(): return lgb.LGBMClassifier(n_estimators=400, max_depth=4, num_leaves=15, learning_rate=0.05, class_weight="balanced", random_state=RNG, verbosity=-1, n_jobs=1)
def make_xgb(): return xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss", random_state=RNG, verbosity=0, n_jobs=1)

def apply_temperature(probs, T):
    logp = np.log(np.clip(probs,1e-9,1)); s = np.exp(logp/T)
    return s/s.sum(1,keepdims=True)

def zoom_rule(pred, audio_only):
    pred = pred.copy()
    zv, zvo = C.LABEL_TO_ID["Zoom_video"], C.LABEL_TO_ID["Zoom_voice"]
    pred[((pred==zv)|(pred==zvo)) & audio_only] = zvo
    return pred

train = pd.read_csv('Task3/RTC_CyberAICup2026/Training_set.csv')
X = engineer_features(train)
y = train.label.map(C.LABEL_TO_ID).to_numpy()
audio_only = train[C.LEN_COLS].to_numpy().max(1) < BAND_AUDIO_MAX
n = len(y)

folds = np.load('artifacts/folds.npz')
group_folds = [(folds[f"group_{i}_tr"], folds[f"group_{i}_va"]) for i in range(10)]

tab = np.load('artifacts/hier/hier_tabicl_2803104002.npz', allow_pickle=True)
tab_oof = (tab["oof_flat"] + tab["oof_hier"]) / 2   # their "avg" variant, repo label order

# "my" ensemble OOF, same grouped folds
my_oof = np.zeros((n, len(C.LABELS)))
for tr_idx, va_idx in group_folds:
    rf = make_rf(); rf.fit(X.iloc[tr_idx], y[tr_idx])
    lg = make_lgb(); lg.fit(X.iloc[tr_idx], y[tr_idx])
    xg = make_xgb(); xg.fit(X.iloc[tr_idx], y[tr_idx])
    ens = (rf.predict_proba(X.iloc[va_idx]) + lg.predict_proba(X.iloc[va_idx]) + xg.predict_proba(X.iloc[va_idx]))/3
    sc = apply_temperature(ens, T_FIXED)
    pi = np.bincount(y[tr_idx], minlength=len(C.LABELS))/len(tr_idx)
    corr = sc / pi[None,:]
    my_oof[va_idx] = corr / corr.sum(1,keepdims=True)

blend = BLEND_W_MINE * my_oof + (1-BLEND_W_MINE) * tab_oof

def sc(P, rule=False):
    p = P.argmax(1)
    if rule: p = zoom_rule(p, audio_only)
    return accuracy_score(y,p), f1_score(y,p,average="macro")

rows = {
  "mine alone (raw)": sc(my_oof, False),
  "tabicl avg alone": sc(tab_oof, False),
  "blend w=0.40/0.60": sc(blend, False),
  "blend + zoom rule": sc(blend, True),
  "tabicl avg + zoom (SHIPPED)": sc(tab_oof, True),
}
print(f"{'candidate':28s} {'acc':>7s} {'macroF1':>8s}")
for k,(a,f) in rows.items():
    print(f"{k:28s} {a:7.4f} {f:8.4f}")

# bootstrap: blend+rule vs shipped (tabicl avg + rule)
rng = np.random.default_rng(0)
pb = sc(blend, True)[1]; ps = sc(tab_oof, True)[1]
d = np.empty(2000)
for b in range(2000):
    i = rng.integers(0,n,n)
    d[b] = f1_score(y[i], zoom_rule(blend[i].argmax(1), audio_only[i]), average="macro") - \
           f1_score(y[i], zoom_rule(tab_oof[i].argmax(1), audio_only[i]), average="macro")
print(f"\nblend+rule vs shipped: dF1 mean {d.mean():+.4f}  95% CI [{np.percentile(d,2.5):+.4f},{np.percentile(d,97.5):+.4f}]  P(>0)={ (d>0).mean():.3f}")
