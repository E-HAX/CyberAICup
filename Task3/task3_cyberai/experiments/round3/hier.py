"""Hierarchical app x mode model with a structured, macro-F1-aware decision layer."""
import sys, numpy as np, pandas as pd
sys.path.insert(0,'src'); import config as C
from features.base import build as build_base
from io_utils import group_keys
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score
import lightgbm as lgb

APPS=C.APPS
tr=pd.read_csv('Task3/RTC_CyberAICup2026/Training_set.csv')
X=build_base(tr); y=tr.label.map(C.LABEL_TO_ID).to_numpy(); g=group_keys(tr)
app=(y//2); mode=(y%2)   # mode 0=voice 1=video  (LABELS order app_voice, app_video)
N=len(y)

def lgbm(**kw):
    p=dict(n_estimators=600,num_leaves=31,learning_rate=0.05,colsample_bytree=0.5,
           subsample=0.8,subsample_freq=1,verbose=-1,random_state=0,n_jobs=-1); p.update(kw)
    return lgb.LGBMClassifier(**p)

REPS=3
P_app=np.zeros((N,5)); P_mode=np.zeros((N,5))   # P(video | app=a, x)
P_flat=np.zeros((N,10))
for rep in range(REPS):
    for a_i,b_i in StratifiedGroupKFold(5,shuffle=True,random_state=C.SEED+rep).split(X,y,g):
        m=lgbm(); m.fit(X.iloc[a_i],app[a_i]); P_app[b_i]+=m.predict_proba(X.iloc[b_i])/REPS
        mf=lgbm(); mf.fit(X.iloc[a_i],y[a_i]); P_flat[b_i]+=mf.predict_proba(X.iloc[b_i])/REPS
        for a in range(5):
            sel=a_i[app[a_i]==a]
            mm=lgbm(n_estimators=400,num_leaves=15)
            mm.fit(X.iloc[sel],mode[sel])
            P_mode[b_i,a]+=mm.predict_proba(X.iloc[b_i])[:,1]/REPS

np.save('exp/P_app.npy',P_app); np.save('exp/P_mode.npy',P_mode); np.save('exp/P_flat.npy',P_flat)

def compose(P_app,P_mode,bias=None):
    b=np.zeros(5) if bias is None else bias
    pv=1/(1+np.exp(-(np.log(np.clip(P_mode,1e-6,1-1e-6)/np.clip(1-P_mode,1e-6,1))+b)))
    out=np.zeros((len(P_app),10))
    for a in range(5):
        out[:,2*a]=P_app[:,a]*(1-pv[:,a]); out[:,2*a+1]=P_app[:,a]*pv[:,a]
    return out

def rep(name,P):
    p=P.argmax(1)
    print('%-22s acc %.4f  macroF1 %.4f'%(name,accuracy_score(y,p),f1_score(y,p,average='macro')))
rep('flat',P_flat)
rep('hierarchical',compose(P_app,P_mode))
rep('avg(flat,hier)',0.5*P_flat+0.5*compose(P_app,P_mode))
print('app-head acc %.4f (flat-implied %.4f)'%(accuracy_score(app,P_app.argmax(1)),
       accuracy_score(app,P_flat.argmax(1)//2)))

# structured decision layer: one mode-bias per app, nested-fit for macro-F1
def fit_bias(idx,P_app,P_mode,grid=np.linspace(-3,3,61)):
    b=np.zeros(5)
    for _ in range(6):
        改=False
        for a in range(5):
            cur=f1_score(y[idx],compose(P_app,P_mode,b)[idx].argmax(1),average='macro')
            best=b[a]
            for v in grid:
                bb=b.copy(); bb[a]=v
                s=f1_score(y[idx],compose(P_app,P_mode,bb)[idx].argmax(1),average='macro')
                if s>cur+1e-6: cur,best=s,v
            if best!=b[a]: b[a]=best; 改=True
        if not 改: break
    return b
b_all=fit_bias(np.arange(N),P_app,P_mode)
print('in-sample bias',np.round(b_all,2))
rep('hier+bias(in-sample)',compose(P_app,P_mode,b_all))
pred=np.zeros(N,dtype=int)
for a_i,b_i in StratifiedGroupKFold(5,shuffle=True,random_state=99).split(X,y,g):
    bb=fit_bias(a_i,P_app,P_mode); pred[b_i]=compose(P_app,P_mode,bb)[b_i].argmax(1)
print('%-22s acc %.4f  macroF1 %.4f'%('hier+bias(nested)',accuracy_score(y,pred),f1_score(y,pred,average='macro')))
