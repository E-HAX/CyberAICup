import sys, numpy as np, pandas as pd
sys.path.insert(0,'src')
import config as C
from features.base import build as build_base
from io_utils import group_keys
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score
import lightgbm as lgb
tr=pd.read_csv('Task3/RTC_CyberAICup2026/Training_set.csv')
L=[f'packet_length_{i}' for i in range(5)]
ln=tr[L].to_numpy()
small = ln.max(1)<300
zoom = tr.label.str.startswith('Zoom')
sub = tr[zoom & small].reset_index(drop=True)
print('zoom small-packet rows:', len(sub), sub.label.value_counts().to_dict())
X=build_base(sub); y=(sub.label=='Zoom_video').astype(int).to_numpy(); g=group_keys(sub)
print('groups',len(set(g)))
oof=np.zeros(len(y))
for rep in range(3):
    sk=StratifiedGroupKFold(5,shuffle=True,random_state=rep)
    for a,b in sk.split(X,y,g):
        m=lgb.LGBMClassifier(n_estimators=400,num_leaves=15,learning_rate=0.05,
            colsample_bytree=0.5,subsample=0.8,subsample_freq=1,verbose=-1,random_state=0)
        m.fit(X.iloc[a],y[a]); oof[b]+=m.predict_proba(X.iloc[b])[:,1]/3
print('AUC %.4f  acc %.4f  majority %.4f'%(roc_auc_score(y,oof), accuracy_score(y,(oof>0.5).astype(int)), max(y.mean(),1-y.mean())))
# top exact length values by class
for lab in ['Zoom_voice','Zoom_video']:
    v=sub[sub.label==lab][L].to_numpy().ravel()
    print(lab, pd.Series(v).value_counts().head(10).to_dict())
