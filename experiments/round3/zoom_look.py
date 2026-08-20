import pandas as pd, numpy as np
tr=pd.read_csv('Task3/RTC_CyberAICup2026/Training_set.csv')
L=[f'packet_length_{i}' for i in range(5)]; T=[f'relative_time_{i}' for i in range(5)]
for lab in ['Zoom_voice','Zoom_video','Discord_voice','Discord_video']:
    d=tr[tr.label==lab]
    ln=d[L].to_numpy(); dur=d[T].to_numpy()[:,-1]
    print(f'{lab:16s} n={len(d):4d} meanlen={ln.mean():7.1f} maxlen={np.median(ln.max(1)):6.1f} minlen={np.median(ln.min(1)):6.1f} med_dur={np.median(dur)*1e3:8.3f}ms  frac_all_small={np.mean(ln.max(1)<300):.2f} frac_any_mtu={np.mean(ln.max(1)>1100):.2f}')
print()
z=tr[tr.label.str.startswith('Zoom')]
ln=z[L].to_numpy()
z2=z.assign(mx=ln.max(1), mn=ln.min(1), mean=ln.mean(1), dur=z[T].to_numpy()[:,-1])
print(z2.groupby('label')[['mx','mn','mean','dur']].describe().T.to_string())
