import pandas as pd
from  multiprocessing import Pool
from arch import time

# enhanced by multi processing

cpu_count = 6

df_read = pd.read_csv('data/pdm_telemetry_data.csv',index_col='datetime', parse_dates=['datetime'], encoding='utf-8')
rolling_features = ['volt','rotate', 'pressure', 'vibration']
lags = [12, 24, 36]
rng = pd.date_range(start=df_read.index.min().date(), end=df_read.index.max().date(), freq='12H')


def tmpFunc(df):
    df.sort_index(inplace=True)
    dfx = pd.DataFrame(index=df.index)
    dfx['machineID'] = df['machineID']
    for lag in lags:
        for feat in rolling_features:
            col_name = feat + "_rollingmean_" + str(lag)
            dfx[col_name] = df[feat].rolling(lag).mean()
            col_name = feat + "_rollingstd_" + str(lag)
            dfx[col_name] = df[feat].rolling(lag).std(ddof=0)
    return dfx

def applyParallel(dfGrouped, func):
    with Pool(cpu_count) as p:
        ret_list = p.map(func, [group for name, group in dfGrouped])
    return pd.concat(ret_list, axis=0)


start_time = time.time()
grouped = df_read.groupby('machineID')
df_tmp = applyParallel(grouped, tmpFunc)

elapsed_time = time.time() - start_time


print("\nComplete", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
df_tmp.to_csv('out_reading_full.csv', index_label='datetime')

df_feat = pd.DataFrame(index=rng)
df_feat = df_feat.join(df_tmp, how='inner')
df_feat.fillna(0, inplace=True)
df_feat.to_csv('out_reading_feat.csv', index_label='datetime')