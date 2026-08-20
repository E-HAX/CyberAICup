import sys,glob,numpy as np,pandas as pd
sys.path.insert(0,'src'); import config as C
from features.base import build as build_base
from sklearn.metrics import accuracy_score, roc_auc_score
import lightgbm as lgb
MAP={'Discord':'Discord','Meet':'GoogleMeet','Messenger':'Messenger','WhatsApp':'WhatsApp','Zoom':'Zoom'}
Ls,names=[],[]
for f in sorted(glob.glob('exp/mirage/*.npz')):
    d=np.load(f,allow_pickle=True); nm=d['label_names']; lc=d['label_code']
    lab=np.array([str(nm[i]) for i in lc])
    keep=np.isin(lab,list(MAP)) & d['mask'][:,:5].all(1) & d['is_udp']
    if keep.sum()==0: continue
    Ls.append((d['lengths'][keep][:,:5], d['gaps'][keep][:,:5])); names.append(lab[keep])
lens=np.vstack([a for a,_ in Ls]); gaps=np.vstack([b for _,b in Ls]); lab=np.concatenate(names)
print('mirage flows for the 5 apps:', lens.shape, pd.Series(lab).value_counts().to_dict())
# drop non-media (all-zero payload) and absurd
ok=(lens>0).all(1); lens,gaps,lab=lens[ok],gaps[ok],lab[ok]
print('after payload filter', lens.shape)
for offset in (0,28):
    t=np.cumsum(gaps,axis=1); t[:,0]=0.0
    df=pd.DataFrame({**{f'packet_length_{i}':lens[:,i]+offset for i in range(5)},
                     **{f'relative_time_{i}':t[:,i] for i in range(5)}})
    Xm=build_base(df); ym=pd.Series(lab).map(MAP).map({a:i for i,a in enumerate(C.APPS)}).to_numpy()
    tr=pd.read_csv('Task3/RTC_CyberAICup2026/Training_set.csv')
    Xc=build_base(tr); yc=(tr.label.map(C.LABEL_TO_ID).to_numpy())//2
    Xm=Xm[Xc.columns]
    m=lgb.LGBMClassifier(n_estimators=600,num_leaves=63,learning_rate=0.05,colsample_bytree=0.5,
        subsample=0.8,subsample_freq=1,verbose=-1,n_jobs=-1,random_state=0,class_weight='balanced')
    m.fit(Xm,ym)
    P=m.predict_proba(Xc)
    print('offset %2d  transfer app acc on competition train: %.4f  (chance-by-majority %.4f)'%(
        offset, accuracy_score(yc,P.argmax(1)), pd.Series(yc).value_counts(normalize=True).max()))
    np.save(f'exp/mirage_appP_{offset}.npy', P)
