import sys,numpy as np,pandas as pd
sys.path.insert(0,'src'); import config as C
from features.base import build as build_base
from io_utils import group_keys
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
tr=pd.read_csv('Task3/RTC_CyberAICup2026/Training_set.csv')
L=[f'packet_length_{i}' for i in range(5)]
small=tr[L].to_numpy().max(1)<300
app=tr.label.str.split('_').str[0]
rows=[]
for a in ['Discord','GoogleMeet','Messenger','Zoom']:
    sub=tr[(app==a)&small].reset_index(drop=True)
    y=(sub.label.str.endswith('video')).astype(int).to_numpy()
    if y.sum()<10: continue
    X=build_base(sub); g=group_keys(sub); oof=np.zeros(len(y))
    for rep in range(3):
        for tr_i,va_i in StratifiedGroupKFold(5,shuffle=True,random_state=rep).split(X,y,g):
            m=lgb.LGBMClassifier(n_estimators=400,num_leaves=15,learning_rate=0.05,
              colsample_bytree=0.5,subsample=0.8,subsample_freq=1,verbose=-1,random_state=0)
            m.fit(X.iloc[tr_i],y[tr_i]); oof[va_i]+=m.predict_proba(X.iloc[va_i])[:,1]/3
    rows.append((a,len(sub),int(y.sum()),round(roc_auc_score(y,oof),3)))
print(pd.DataFrame(rows,columns=['app','n_small','n_video','AUC_video_vs_voice']).to_string(index=False))
