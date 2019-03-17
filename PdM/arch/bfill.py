import pandas as pd
import numpy as np

df = pd.read_csv('test/test.csv', index_col='datetime', parse_dates=True)
df1 = pd.pivot_table(df, values=['machineID'], index=['datetime', 'machineID'], columns='failure', aggfunc=len)

rng = pd.date_range(start='2015-01-25 01', end = '2015-01-25 06', freq='H', name='datetime')
machines = np.arange(1, 6)
idx = [(d, m) for d in rng for m in machines]
midx = pd.MultiIndex.from_tuples(idx, names=('datetime', 'machineID'))
dfx = pd.DataFrame(index=midx)

dfx = dfx.join(df1, on=['datetime', 'machineID'],how='left')
print(dfx)
dfx.sort_index(inplace=True)
dfx.reset_index('machineID', inplace=True)
dfx = dfx.groupby('machineID').bfill(limit=2)
for name, g in dfx.groupby('machineID'):
    print(name)
    print(g)

