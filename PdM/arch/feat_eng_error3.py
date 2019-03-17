import pandas as pd
import gc
from  multiprocessing import Pool
from arch import time

df_erro = pd.read_csv('data/pdm_errors_data.csv', index_col=['datetime'], parse_dates=['datetime'])
df_erro.sort_index(inplace=True)

rng = pd.date_range(start='2015-01-01 00', end='2016-01-01 00', freq='H', name='datetime')
rng_final = pd.date_range(start='2015-01-01 00', end='2016-01-01 00', freq='12H', name='datetime')

error_feats = df_erro['errorID'].unique()
df1 = pd.pivot_table(df_erro, values='machineID', index=['datetime', 'machineID'], columns='errorID', aggfunc=len).fillna(0)
df_long_index = pd.DataFrame(index=rng)
df_short_index = pd.DataFrame(index=rng_final)

del df_erro
gc.collect()

cpu_count = 6

def tmpFunc(params):
    (name, df) = params
    dfx = df.join(pd.DataFrame(index=rng), how='right')
    dfx['machineID'] = name
    dfx.fillna(0, inplace=True)
    dfx.sort_index(inplace=True)
    for feat in error_feats:
        col_name = feat + "_rollingmean24"
        dfx[col_name] = dfx[feat].rolling(24).mean()
    return dfx

def applyParallel(dfGrouped, func):
    with Pool(cpu_count) as p:
        ret_list = p.map(func, [params for params in dfGrouped])
    return pd.concat(ret_list, axis=0)

df1.reset_index('machineID', inplace=True)

start_time = time.time()
grouped = df1.groupby('machineID')
df_tmp = applyParallel(grouped, tmpFunc)

elapsed_time = time.time() - start_time


print("\nComplete", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

print(len(df_tmp.groupby('machineID').get_group(1)))




