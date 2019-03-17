import pandas as pd
import numpy as np


df_read = pd.read_csv('data/pdm_telemetry_data.csv',index_col='datetime', parse_dates=['datetime'], encoding='utf-8')
rolling_features = ['volt','rotate', 'pressure', 'vibration']
lags = [12, 24, 36]
rng = pd.date_range(start=df_read.index.min().date(), end=df_read.index.max().date(), freq='12H')

df_tmp = None
grouped = df_read.groupby('machineID')
print_i = 0
for name, group in grouped:
    group = group.sort_index()
    dfx = pd.DataFrame(index=group.index)
    dfx['machineID'] = group['machineID']
    for lag in lags:
        for feat in rolling_features:
            col_name = feat + "_rollingmean_" + str(lag)
            dfx[col_name] = group[feat].rolling(lag).mean()
            col_name = feat + "_rollingstd_" + str(lag)
            dfx[col_name] = group[feat].rolling(lag).std(ddof=0)
    if df_tmp is None:
        df_tmp = dfx
    else:
        df_tmp = pd.concat([df_tmp, dfx], axis=0)

    print_i += 1
    if print_i % 100 == 0:
        print("> {} ".format(print_i), end="", flush=True)

print("\nComplete")
# df_tmp.to_csv('out_reading_full.csv', index_label='datetime')

df_feat = pd.DataFrame(index=rng)
df_feat = df_feat.join(df_tmp, how='inner')
df_feat.fillna(0, inplace=True)
df_feat.to_csv('out_reading_feat.csv', index_label='datetime')