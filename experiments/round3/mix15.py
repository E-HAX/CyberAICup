import sys,numpy as np,pandas as pd
sys.path.insert(0,'src'); import config as C
from features.base import build as B
from io_utils import group_keys
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score,accuracy_score
import lightgbm as lgb
tr=pd.read_csv('Task3/RTC_CyberAICup2026/Training_set.csv')
X=B(tr); y=tr.label.map(C.LABEL_TO_ID).to_numpy(); g=group_keys(tr)
small=tr[C.LEN_COLS].to_numpy().max(1)<300
app=y//2; mode=y%2
sub=np.where(mode==0,0,np.where(small,2,1))     # 0 voice, 1 video-large, 2 video-small
y15=app*3+sub
def lgbm(**k):
    p=dict(n_estimators=600,num_leaves=31,learning_rate=0.05,colsample_bytree=0.5,
      subsample=0.8,subsample_freq=1,verbose=-1,random_state=0,n_jobs=-1); p.update(k)
    return lgb.LGBMClassifier(**p)
P15=np.zeros((len(y),15))
for rep in range(3):
    for a,b in StratifiedGroupKFold(5,shuffle=True,random_state=C.SEED+rep).split(X,y15,g):
        m=lgbm(); m.fit(X.iloc[a],y15[a]); pr=m.predict_proba(X.iloc[b])
        full=np.zeros((len(b),15)); full[:,m.classes_.astype(int)]=pr; P15[b]+=full/3
P10=np.zeros((len(y),10))
for a_ in range(5):
    P10[:,2*a_]=P15[:,3*a_]; P10[:,2*a_+1]=P15[:,3*a_+1]+P15[:,3*a_+2]
def rep_(n,P):
    p=P.argmax(1); q=p.copy(); q[(p//2==4)&small]=8
    print('%-16s acc %.4f mF1 %.4f | +zoom acc %.4f mF1 %.4f'%(n,accuracy_score(y,p),
      f1_score(y,p,average='macro'),accuracy_score(y,q),f1_score(y,q,average='macro')))
rep_('mixture-15',P10)
Pf=np.load('exp/P_flat.npy'); Ph=np.load('exp/P_app.npy'); Pm=np.load('exp/P_mode.npy')
Hi=np.zeros_like(Pf)
for a_ in range(5): Hi[:,2*a_]=Ph[:,a_]*(1-Pm[:,a_]); Hi[:,2*a_+1]=Ph[:,a_]*Pm[:,a_]
rep_('flat',Pf); rep_('avg(flat,hier)',0.5*Pf+0.5*Hi)
rep_('avg3',(Pf+Hi+P10)/3)
np.save('exp/P_mix15.npy',P10)
