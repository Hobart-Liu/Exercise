import pandas as pd
import numpy as np
import gc

df_erro = pd.read_csv('data/pdm_errors_data.csv', index_col=['datetime'], parse_dates=['datetime'])
df_erro.sort_index(inplace=True)

rng = pd.date_range(start='2015-01-01 00', end='2016-01-01 00', freq='H', name='datetime')
rng_final = pd.date_range(start='2015-01-01 00', end='2016-01-01 00', freq='12H', name='datetime')
'''
remove multiple index 
machines = np.arange(1, 1001)
ids = [(d, m) for d in rng for m in machines]
idx = pd.MultiIndex.from_tuples(ids, names=('datetime', 'machineID'))
df1 = df1.join(pd.DataFrame(index=rng), how='right').fillna(0)
df1.to_csv('out_erros_full.csv')
'''

error_feats = df_erro['errorID'].unique()
df1 = pd.pivot_table(df_erro, values='machineID', index=['datetime', 'machineID'], columns='errorID', aggfunc=len).fillna(0)
df_long_index = pd.DataFrame(index=rng)
df_short_index = pd.DataFrame(index=rng_final)

del df_erro
gc.collect()

df2 = None
print_i = 0
grouped = df1.groupby('machineID')
for mid, group in grouped:
    tmp = group.copy()
    tmp = tmp.join(df_long_index, on='datetime', how='right')
    tmp.fillna(0, inplace=True)
    tmp.sort_index(inplace=True)
    for feat in error_feats:
        col_name = feat + "_rollingmean_24"
        tmp[col_name] = tmp[feat].rolling(24).mean()

    tmp = tmp.join(df_short_index, how='right')

    if df2 is None:
        df2 = tmp
    else:
        df2 = pd.concat((df2, tmp), axis=0)

    if print_i % 10 == 0:
        print("{} > {}".format(print_i, df2.shape))
    print_i += 1

    del tmp
    gc.collect()

df2.fillna(0, inplace=True)
df2.drop(error_feats, inplace=True, axis=1)
df2.to_csv("out_errors_feature.csv")


