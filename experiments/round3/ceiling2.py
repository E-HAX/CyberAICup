import sys,numpy as np,pandas as pd
sys.path.insert(0,'src'); import config as C
from sklearn.metrics import f1_score
tr=pd.read_csv('Task3/RTC_CyberAICup2026/Training_set.csv')
y=tr.label.map(C.LABEL_TO_ID).to_numpy()
mx=tr[C.LEN_COLS].to_numpy().max(1)
for thr in (200,250,300,400,500):
    z=(y//2==4)&(mx<thr)
    print('thr %3d: zoom audio-only rows %3d  voice %3d video %3d'%(thr,z.sum(),(y[z]==8).sum(),(y[z]==9).sum()))
# macro-F1 ceiling: everything perfect except the inseparable Zoom subset
small=mx<300; amb=(y//2==4)&small
nv=(y[amb]==8).sum(); nd=(y[amb]==9).sum()
Nv=(y==8).sum(); Nd=(y==9).sum()
best=None
for p in np.linspace(0,1,101):        # fraction of the ambiguous subset sent to voice
    tp_v=nv*p; fp_v=nd*p
    prec_v=tp_v/max(tp_v+fp_v,1e-9); rec_v=tp_v/Nv
    f1v=0 if prec_v+rec_v==0 else 2*prec_v*rec_v/(prec_v+rec_v)
    tp_d=(Nd-nd)+nd*(1-p); fp_d=nv*(1-p)
    prec_d=tp_d/max(tp_d+fp_d,1e-9); rec_d=tp_d/Nd
    f1d=2*prec_d*rec_d/(prec_d+rec_d)
    m=(8*1.0+f1v+f1d)/10
    if best is None or m>best[0]: best=(m,p,f1v,f1d)
print('\nabsolute macro-F1 ceiling (all 8 other classes perfect): %.4f at p=%.2f  ZoomVoiceF1=%.3f ZoomVideoF1=%.3f'%best)
print('accuracy ceiling: %.4f'%((len(y)-min(nv,nd))/len(y)))
