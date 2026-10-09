import numpy as np, pandas as pd
from collections import Counter
tr = pd.read_csv('Task3/RTC_CyberAICup2026/Training_set.csv')
te = pd.read_csv('Task3/RTC_CyberAICup2026/Testing_set.csv')
L=[f'packet_length_{i}' for i in range(5)]; T=[f'relative_time_{i}' for i in range(5)]
print(tr.label.value_counts().to_string())
print('\ntrain',tr.shape,'test',te.shape)

# exact duplicate length tuples
key = tr[L].astype(int).apply(lambda r: tuple(r), axis=1)
g = pd.DataFrame({'k':key,'y':tr.label})
amb = g.groupby('k').y.agg(['nunique','count'])
dup = amb[amb['count']>1]
print('\nunique len-tuples', amb.shape[0], 'rows in dup tuples', dup['count'].sum(),
      'dup tuples with >1 label', (dup['nunique']>1).sum())
# best achievable if only length tuple known
maj = g.groupby('k').y.agg(lambda s: s.value_counts().iloc[0]).sum()
print('majority-oracle acc on length tuple alone:', maj/len(g))

# with times too (full 10-dim exact)
key2 = tr[L+T].round(9).apply(lambda r: tuple(r), axis=1)
g2=pd.DataFrame({'k':key2,'y':tr.label})
maj2=g2.groupby('k').y.agg(lambda s:s.value_counts().iloc[0]).sum()
print('unique full rows', g2.k.nunique(), 'majority-oracle acc:', maj2/len(g2))

# overlap train/test on length tuple
kt = te[L].astype(int).apply(lambda r: tuple(r), axis=1)
print('test rows whose length-tuple appears in train:', kt.isin(set(key)).sum(), '/', len(te))

# constant-length flows
print('\nrows where all 5 lengths equal:', (tr[L].nunique(axis=1)==1).sum())
sub=tr[tr[L].nunique(axis=1)==1]
print(sub.label.value_counts().head(12).to_string())
