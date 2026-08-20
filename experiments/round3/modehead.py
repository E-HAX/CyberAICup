import sys,numpy as np,pandas as pd
sys.path.insert(0,'src'); import config as C
from features.base import build as B
from io_utils import group_keys
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score,accuracy_score,roc_auc_score
import lightgbm as lgb
tr=pd.read_csv('Task3/RTC_CyberAICup2026/Training_set.csv')
X=B(tr); y=tr.label.map(C.LABEL_TO_ID).to_numpy(); g=group_keys(tr)
small=tr[C.LEN_COLS].to_numpy().max(1)<300
mode=y%2
def lgbm(**k):
    p=dict(n_estimators=600,num_leaves=31,learning_rate=0.05,colsample_bytree=0.5,
      subsample=0.8,subsample_freq=1,verbose=-1,random_state=0,n_jobs=-1); p.update(k)
    return lgb.LGBMClassifier(**p)
Pg=np.zeros(len(y))
for rep in range(3):
    for a,b in StratifiedGroupKFold(5,shuffle=True,random_state=C.SEED+rep).split(X,mode,g):
        m=lgbm(); m.fit(X.iloc[a],mode[a]); Pg[b]+=m.predict_proba(X.iloc[b])[:,1]/3
Pm=np.load('exp/P_mode.npy'); Pa=np.load('exp/P_app.npy'); Pf=np.load('exp/P_flat.npy'); Px=np.load('exp/P_mix15.npy')
per_app=Pm[np.arange(len(y)), y//2]   # oracle-app selection, for AUC comparison only
print('global mode AUC %.4f | per-app-head AUC (true app) %.4f'%(roc_auc_score(mode,Pg),roc_auc_score(mode,per_app)))
def comp(pv_matrix):
    P=np.zeros((len(y),10))
    for a in range(5): P[:,2*a]=Pa[:,a]*(1-pv_matrix[:,a]); P[:,2*a+1]=Pa[:,a]*pv_matrix[:,a]
    return P
Hg=comp(np.repeat(Pg[:,None],5,axis=1)); Hp=comp(Pm)
def rep_(n,P):
    p=P.argmax(1); q=p.copy(); q[(p//2==4)&small]=8
    print('%-24s acc %.4f mF1 %.4f | +zoom acc %.4f mF1 %.4f'%(n,accuracy_score(y,p),
      f1_score(y,p,average='macro'),accuracy_score(y,q),f1_score(y,q,average='macro')))
rep_('hier(global mode)',Hg); rep_('hier(per-app mode)',Hp)
rep_('avg4 global',(Pf+Hg+Hp+Px)/4)
rep_('avg3 per-app',(Pf+Hp+Px)/3)
