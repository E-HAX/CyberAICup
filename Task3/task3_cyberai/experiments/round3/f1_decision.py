import numpy as np, sys
sys.path.insert(0,'src')
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
oof=np.load('exp/oof_lgbm.npy'); y=np.load('exp/y.npy'); g=np.load('exp/g.npy',allow_pickle=True)
lp=np.log(np.clip(oof,1e-9,1))
def apply(b): return (lp+b).argmax(1)
def mf1(b,idx): return f1_score(y[idx], apply(b)[idx], average='macro')
def fit(idx, iters=12, grid=np.linspace(-2,2,41)):
    b=np.zeros(10)
    for _ in range(iters):
        改=False
        for c in range(10):
            best=b[c]; bv=mf1(b,idx)
            for v in grid:
                bb=b.copy(); bb[c]=v; s=mf1(bb,idx)
                if s>bv+1e-6: bv, best = s, v
            if best!=b[c]: b[c]=best; 改=True
        if not 改: break
    return b
base_i=np.arange(len(y))
print('raw  acc %.4f  macroF1 %.4f'%(accuracy_score(y,lp.argmax(1)), f1_score(y,lp.argmax(1),average='macro')))
b_all=fit(base_i)
print('in-sample tuned macroF1 %.4f (bias %s)'%(mf1(b_all,base_i), np.round(b_all,2)))
# nested
sk=StratifiedGroupKFold(5,shuffle=True,random_state=7)
pred=np.zeros(len(y),dtype=int)
for a,bx in sk.split(lp,y,g):
    bb=fit(a); pred[bx]=apply(bb)[bx]
print('nested tuned  acc %.4f  macroF1 %.4f'%(accuracy_score(y,pred), f1_score(y,pred,average='macro')))
from sklearn.metrics import classification_report
import config as C
print(classification_report(y,pred,target_names=C.LABELS,digits=3))
