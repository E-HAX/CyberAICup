import sys,numpy as np,pandas as pd
sys.path.insert(0,'src'); import config as C
from sklearn.metrics import f1_score,accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from io_utils import group_keys
tr=pd.read_csv('Task3/RTC_CyberAICup2026/Training_set.csv')
L=[f'packet_length_{i}' for i in range(5)]; small=tr[L].to_numpy().max(1)<300
y=tr.label.map(C.LABEL_TO_ID).to_numpy(); g=group_keys(tr)
Pf=np.load('exp/P_flat.npy'); Pa=np.load('exp/P_app.npy'); Pm=np.load('exp/P_mode.npy')
def comp(b):
    lo=np.log(np.clip(Pm,1e-6,1-1e-6)/np.clip(1-Pm,1e-6,1)); pv=1/(1+np.exp(-(lo+b)))
    P=np.zeros((len(y),10))
    for a in range(5): P[:,2*a]=Pa[:,a]*(1-pv[:,a]); P[:,2*a+1]=Pa[:,a]*pv[:,a]
    return 0.5*Pf+0.5*P
def rule(P):
    p=P.argmax(1); p[(p//2==4)&small]=8; return p
grid=np.linspace(-2,2,41)
sc=[f1_score(y,rule(comp(b)),average='macro') for b in grid]
print('best global mode-bias %.2f -> in-sample macroF1 %.4f (b=0: %.4f)'%(grid[int(np.argmax(sc))],max(sc),sc[20]))
pred=np.zeros(len(y),dtype=int)
for a_i,b_i in StratifiedGroupKFold(5,shuffle=True,random_state=99).split(Pf,y,g):
    s=[f1_score(y[a_i],rule(comp(b))[a_i],average='macro') for b in grid]
    pred[b_i]=rule(comp(grid[int(np.argmax(s))]))[b_i]
print('nested one-parameter: acc %.4f macroF1 %.4f'%(accuracy_score(y,pred),f1_score(y,pred,average='macro')))
