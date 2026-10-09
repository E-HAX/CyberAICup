import sys,numpy as np,pandas as pd
sys.path.insert(0,'src'); import config as C
from features.base import build as B
from io_utils import group_keys
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score,accuracy_score
import lightgbm as lgb
tr=pd.read_csv('Task3/RTC_CyberAICup2026/Training_set.csv')
X=B(tr); y=tr.label.map(C.LABEL_TO_ID).to_numpy(); g=group_keys(tr)
small=tr[C.LEN_COLS].to_numpy().max(1)<300
def lgbm(n):
    return lgb.LGBMClassifier(objective='multiclass',num_class=n,n_estimators=600,num_leaves=31,
      learning_rate=0.05,colsample_bytree=0.5,subsample=0.8,subsample_freq=1,verbose=-1,
      random_state=0,n_jobs=-1)
def run(balanced):
    Pf=np.zeros((len(y),10)); Ph=np.zeros((len(y),10))
    for rep in range(3):
        for a,b in StratifiedGroupKFold(5,shuffle=True,random_state=C.SEED+rep).split(X,y,g):
            w=compute_sample_weight('balanced',y[a]) if balanced else None
            m=lgbm(10); m.fit(X.iloc[a],y[a],sample_weight=w); Pf[b]+=m.predict_proba(X.iloc[b])/3
            app=y[a]//2; mo=y[a]%2
            wa=compute_sample_weight('balanced',app) if balanced else None
            ma=lgbm(5); ma.fit(X.iloc[a],app,sample_weight=wa); pa=ma.predict_proba(X.iloc[b])
            for k in range(5):
                sel=np.flatnonzero(app==k)
                wm=compute_sample_weight('balanced',mo[sel]) if balanced else None
                mm=lgbm(2); mm.fit(X.iloc[a].iloc[sel],mo[sel],sample_weight=wm)
                pv=mm.predict_proba(X.iloc[b])[:,1]
                Ph[b,2*k]+=pa[:,k]*(1-pv)/3; Ph[b,2*k+1]+=pa[:,k]*pv/3
    for nm,P in [('flat',Pf),('hier',Ph),('avg',0.5*Pf+0.5*Ph)]:
        p=P.argmax(1); q=p.copy(); q[(p//2==4)&small]=8
        print('balanced=%-5s %-5s acc %.4f mF1 %.4f | +zoom acc %.4f mF1 %.4f'%(balanced,nm,
          accuracy_score(y,p),f1_score(y,p,average='macro'),accuracy_score(y,q),f1_score(y,q,average='macro')))
run(False); run(True)
