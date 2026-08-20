import sys,numpy as np,pandas as pd
sys.path.insert(0,'src'); import config as C
tr=pd.read_csv('Task3/RTC_CyberAICup2026/Training_set.csv')
L=[f'packet_length_{i}' for i in range(5)]
ln=tr[L].to_numpy(); small=ln.max(1)<300
oof=np.load('exp/oof_lgbm.npy'); y=np.load('exp/y.npy'); p=oof.argmax(1)
err=(p!=y)
print('rows small=%d (%.1f%%)  errors in small=%d  errors in large=%d  total err=%d'%(
    small.sum(),100*small.mean(),err[small].sum(),err[~small].sum(),err.sum()))
print('\nacc small %.4f  acc large %.4f'%(1-err[small].mean(),1-err[~small].mean()))
# per app: how many video flows are all-small (ambiguous)
app=tr.label.str.split('_').str[0]; mode=tr.label.str.split('_').str[1]
t=pd.crosstab([app,mode], small)
t.columns=['large','small']; print('\n',t.to_string())
# error breakdown by true label within small
d=pd.DataFrame({'lab':tr.label,'small':small,'err':err})
print('\nerrors by (label,small):\n', d.groupby(['lab','small']).err.agg(['sum','count']).to_string())
