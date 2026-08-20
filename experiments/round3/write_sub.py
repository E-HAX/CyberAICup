import sys,pickle,numpy as np,pandas as pd
sys.path.insert(0,'src'); import config as C
cands=pickle.load(open('exp/cands.pkl','rb'))
te=pd.read_csv('Task3/RTC_CyberAICup2026/Testing_set.csv')
au=te[C.LEN_COLS].to_numpy().max(1)<C.BAND_AUDIO[1]
P=cands['tabicl:avg'][1]
pred=P.argmax(1).copy(); pred[(pred//2==4)&au]=C.LABEL_TO_ID['Zoom_voice']
labels=[C.ID_TO_LABEL[int(i)] for i in pred]
sub=pd.DataFrame({"index":np.arange(1,len(labels)+1),"label":labels})
sub.to_csv('submission.csv',header=False,index=False)
print(len(sub)); print(sub.head(3).to_string(index=False,header=False))
print(sub.label.value_counts().to_string())
