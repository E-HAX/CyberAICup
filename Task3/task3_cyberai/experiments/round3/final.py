import sys,glob,json,numpy as np,pandas as pd
sys.path.insert(0,'src'); import config as C
from sklearn.metrics import f1_score,accuracy_score,classification_report,confusion_matrix
tr=pd.read_csv('Task3/RTC_CyberAICup2026/Training_set.csv')
te=pd.read_csv('Task3/RTC_CyberAICup2026/Testing_set.csv')
y=tr.label.map(C.LABEL_TO_ID).to_numpy()
au_tr=tr[C.LEN_COLS].to_numpy().max(1)<C.BAND_AUDIO[1]
au_te=te[C.LEN_COLS].to_numpy().max(1)<C.BAND_AUDIO[1]
M={}
for f in glob.glob('exp/hier/*.npz'):
    name=f.split('hier_')[-1].split('_')[0]
    d=np.load(f,allow_pickle=True)
    M[name]={k:d[k] for k in d.files if k!='meta'}
def norm(P): return P/np.clip(P.sum(1,keepdims=True),1e-9,None)
def rule(P,au):
    p=P.argmax(1); p[(p//2==4)&au]=C.LABEL_TO_ID['Zoom_voice']; return p
def sc(p): return accuracy_score(y,p), f1_score(y,p,average='macro')
cands={}
for m in M:
    for v in ('flat','hier','mix'): cands[f'{m}:{v}']=(norm(M[m][f'oof_{v}']),norm(M[m][f'test_{v}']))
    cands[f'{m}:avg']=(norm(norm(M[m]['oof_flat'])+norm(M[m]['oof_hier'])),
                       norm(norm(M[m]['test_flat'])+norm(M[m]['test_hier'])))
cands['all-models:avg']=(norm(sum(norm(M[m]['oof_flat'])+norm(M[m]['oof_hier']) for m in M)),
                         norm(sum(norm(M[m]['test_flat'])+norm(M[m]['test_hier']) for m in M)))
strong=['tabicl','lgbm','xgb']
cands['top3:avg']=(norm(sum(norm(M[m]['oof_flat'])+norm(M[m]['oof_hier']) for m in strong)),
                   norm(sum(norm(M[m]['test_flat'])+norm(M[m]['test_hier']) for m in strong)))
cands['tabicl2+lgbm']=(norm(2*(norm(M['tabicl']['oof_flat'])+norm(M['tabicl']['oof_hier']))
                            +norm(M['lgbm']['oof_flat'])+norm(M['lgbm']['oof_hier'])),
                       norm(2*(norm(M['tabicl']['test_flat'])+norm(M['tabicl']['test_hier']))
                            +norm(M['lgbm']['test_flat'])+norm(M['lgbm']['test_hier'])))
rows=[]
for k,(O,T) in cands.items():
    a0,f0=sc(O.argmax(1)); a1,f1_=sc(rule(O,au_tr))
    rows.append((k,a0,f0,a1,f1_))
df=pd.DataFrame(rows,columns=['candidate','acc','macroF1','acc+zoom','macroF1+zoom']).sort_values('macroF1+zoom',ascending=False)
print(df.head(14).to_string(index=False,float_format=lambda v:'%.4f'%v))
np.save('exp/cand_keys.npy',np.array(list(cands)))
import pickle; pickle.dump(cands,open('exp/cands.pkl','wb'))
