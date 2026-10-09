import sys,pickle,numpy as np,pandas as pd
sys.path.insert(0,'src'); import config as C
from sklearn.metrics import f1_score,accuracy_score,classification_report
from scipy.stats import binomtest
tr=pd.read_csv('Task3/RTC_CyberAICup2026/Training_set.csv')
te=pd.read_csv('Task3/RTC_CyberAICup2026/Testing_set.csv')
y=tr.label.map(C.LABEL_TO_ID).to_numpy()
au_tr=tr[C.LEN_COLS].to_numpy().max(1)<C.BAND_AUDIO[1]
au_te=te[C.LEN_COLS].to_numpy().max(1)<C.BAND_AUDIO[1]
cands=pickle.load(open('exp/cands.pkl','rb'))
def rule(P,au):
    p=P.argmax(1).copy(); p[(p//2==4)&au]=C.LABEL_TO_ID['Zoom_voice']; return p
base=cands['tabicl:flat'][0].argmax(1)                     # standing model
ship=rule(cands['tabicl:avg'][0],au_tr)                    # pre-specified new model
obs =rule(cands['tabicl2+lgbm'][0],au_tr)                  # observed maximum
rng=np.random.default_rng(0)
def boot(a,b):
    d=[]
    for _ in range(4000):
        i=rng.integers(0,len(y),len(y))
        d.append(f1_score(y[i],a[i],average='macro')-f1_score(y[i],b[i],average='macro'))
    d=np.array(d); return d.mean(),np.percentile(d,2.5),np.percentile(d,97.5),(d>0).mean()
for nm,p in [('tabicl:avg+zoom',ship),('tabicl2+lgbm+zoom',obs)]:
    m,lo,hi,pg=boot(p,base)
    n01=int(((p==y)&(base!=y)).sum()); n10=int(((p!=y)&(base==y)).sum())
    mc=binomtest(n01,n01+n10,0.5).pvalue if n01+n10 else 1.0
    print('%-20s acc %.4f mF1 %.4f | vs standing: dF1 %+.4f CI[%+.4f,%+.4f] P(>0)=%.3f | McNemar %d/%d p=%.3f'%(
        nm,accuracy_score(y,p),f1_score(y,p,average='macro'),m,lo,hi,pg,n01,n10,mc))
print()
print(classification_report(y,ship,target_names=C.LABELS,digits=3))
# submission from the shipped model
P_te=cands['tabicl:avg'][1]
pred=rule(P_te,au_te)
labels=[C.ID_TO_LABEL[int(i)] for i in pred]
pd.Series(labels).to_csv('submission.csv',index=False,header=False)
print('test rows',len(labels))
print('predicted Zoom & audio-only:',int(((P_te.argmax(1)//2==4)&au_te).sum()))
print(pd.Series(labels).value_counts().to_string())
old=pd.read_csv('submission_prev.csv',header=None)[0] if False else None
