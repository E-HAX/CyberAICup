import sys, os, numpy as np, pandas as pd
os.environ['RTC_WORK']='/tmp/work'
sys.path.insert(0,'src')
import config as C
from features.base import build as build_base_features
from io_utils import group_keys
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import lightgbm as lgb

tr = pd.read_csv('Task3/RTC_CyberAICup2026/Training_set.csv')
X = build_base_features(tr)
y = tr.label.map(C.LABEL_TO_ID).to_numpy()
g = group_keys(tr)
print('X', X.shape, 'groups', len(set(g)))
oof = np.zeros((len(y), 10))
for rep in range(2):
    sk = StratifiedGroupKFold(5, shuffle=True, random_state=C.SEED+rep)
    for a,b in sk.split(X,y,g):
        m = lgb.LGBMClassifier(n_estimators=600, num_leaves=31, learning_rate=0.05,
                               colsample_bytree=0.5, subsample=0.8, subsample_freq=1,
                               verbose=-1, n_jobs=-1, random_state=0)
        m.fit(X.iloc[a], y[a])
        oof[b] += m.predict_proba(X.iloc[b])/2
p = oof.argmax(1)
print('acc', accuracy_score(y,p), 'macroF1', f1_score(y,p,average='macro'))
app_t = y//2; app_p = p//2; mode_t=y%2; mode_p=p%2
print('app acc', accuracy_score(app_t,app_p), 'mode acc', accuracy_score(mode_t,mode_p))
print('mode acc | app correct', accuracy_score(mode_t[app_t==app_p], mode_p[app_t==app_p]))
cm = confusion_matrix(y,p)
print(pd.DataFrame(cm, index=C.LABELS, columns=[l[:9] for l in C.LABELS]).to_string())
np.save('exp/oof_lgbm.npy', oof); np.save('exp/y.npy', y); np.save('exp/g.npy', np.array(g))
