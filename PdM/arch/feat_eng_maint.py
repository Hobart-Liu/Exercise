import pandas as pd
import numpy as np

df_main = pd.read_csv('data/pdm_maint_data.csv', index_col='datetime', parse_dates=True, encoding='utf-8')
df_main.sort_index(inplace=True)

df1 = pd.pivot_table(df_main, values='machineID', index=['datetime', 'machineID'], columns='comp', aggfunc=len).fillna(0)

rng = pd.date_range(start=df_main.index.min().date(),end=df_main.index.max().date(),freq='H', name='datetime')
machines = np.arange(1, 1001)
ids = [(d, m) for d in rng for m in machines]
idx = pd.MultiIndex.from_tuples(ids, names=('datetime', 'machineID'))
df1 = df1.join(pd.DataFrame(index=idx), on=['datetime', 'machineID'],how='right').fillna(0)

df1.reset_index('machineID', inplace=True) # move machineID from index to normal column
df1.sort_index(inplace=True)


comp_feat = df_main['comp'].unique()
for feat in comp_feat:
    col_name = "sincelast" + feat
    df1[col_name] = df1.index.where(df1[feat].eq(1))
    df1[col_name] = (df1.index - df1.groupby('machineID')[col_name].ffill()).fillna(pd.Timedelta(0)).dt.days
    print("Done "+ feat)

rng = pd.date_range(start='2015-01-01 00', end='2016-01-01 00', freq='12H', name='datetime')
df1 = df1.join(pd.DataFrame(index=rng), how='inner').fillna(0)
df1.to_csv('out_maint_feat.csv')