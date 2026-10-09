import sys,numpy as np,pandas as pd
sys.path.insert(0,'src'); import config as C
from sklearn.metrics import f1_score, accuracy_score, classification_report
tr=pd.read_csv('Task3/RTC_CyberAICup2026/Training_set.csv')
L=[f'packet_length_{i}' for i in range(5)]
small=tr[L].to_numpy().max(1)<300
y=tr.label.map(C.LABEL_TO_ID).to_numpy()
Pf=np.load('exp/P_flat.npy'); Pa=np.load('exp/P_app.npy'); Pm=np.load('exp/P_mode.npy')
Ph=np.zeros_like(Pf)
for a in range(5):
    Ph[:,2*a]=Pa[:,a]*(1-Pm[:,a]); Ph[:,2*a+1]=Pa[:,a]*Pm[:,a]
def show(n,p):
    print('%-28s acc %.4f  macroF1 %.4f  ZoomVoiceF1 %.3f ZoomVideoF1 %.3f'%(n,
      accuracy_score(y,p),f1_score(y,p,average='macro'),
      f1_score(y==8,p==8),f1_score(y==9,p==9)))
for nm,P in [('flat',Pf),('hier',Ph),('avg',0.5*Pf+0.5*Ph)]:
    p=P.argmax(1); show(nm,p)
    q=p.copy(); q[(p//2==4)&small]=8   # predicted-Zoom & audio-only -> Zoom_voice
    show(nm+' +zoom-rule',q)
# bootstrap the gain on the avg model
P=0.5*Pf+0.5*Ph; p=P.argmax(1); q=p.copy(); q[(p//2==4)&small]=8
rng=np.random.default_rng(0); d=[]
for _ in range(2000):
    i=rng.integers(0,len(y),len(y))
    d.append(f1_score(y[i],q[i],average='macro')-f1_score(y[i],p[i],average='macro'))
d=np.array(d); print('\nzoom-rule macroF1 delta: mean %+.4f  95%% CI [%+.4f, %+.4f]  P(>0)=%.3f'%(
    d.mean(),np.percentile(d,2.5),np.percentile(d,97.5),(d>0).mean()))
print(classification_report(y,q,target_names=C.LABELS,digits=3))
