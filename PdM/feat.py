import numpy as np
import pandas as pd
from multiprocessing import Pool

import gc
import time

pd.set_option('display.max_columns', 999)

file_failure = 'data/pdm_failures_data.csv'
file_error = 'data/pdm_errors_data.csv'
file_maintenance = 'data/pdm_maint_data.csv'
file_machine = 'data/pdm_machines_data.csv'
file_telemetry = 'data/pdm_telemetry_data.csv'

cpu_count = 6 # use 6 cpu core

start, end = '2015-01-01 00:00:00', '2016-01-02 00:00:00'
rng_H = pd.date_range(start=start, end=end, freq='H', closed='left', name='datetime')
rng_12H = pd.date_range(start=start, end=end, freq='12H', closed='left', name='datetime')
machines = np.arange(1, 1001)
idx = [(d, m) for d in rng_H for m in machines]
print("{} days, {} machines, {} cross records".format(len(rng_H), len(machines), len(idx)))


# processing telemetry data set
df_telemetry = pd.read_csv(file_telemetry, index_col='datetime', parse_dates=True, encoding='utf-8')
df_telemetry.sort_index(inplace=True)
rolling_features = ['volt','rotate', 'pressure', 'vibration']
lags = [12, 24, 36]


def telemetryFunc(params):
    (name, df) = params
    df.sort_index(inplace=True)
    dfx = pd.DataFrame(index=df.index)
    dfx['machineID'] = name
    for lag in lags:
        for feat in rolling_features:
            col_name = feat + "_rollingmean_" + str(lag)
            dfx[col_name] = df[feat].rolling(lag).mean()
            col_name = feat + "_rollingstd_" + str(lag)
            col_name2 = col_name
            dfx[col_name] = df[feat].rolling(lag).std(ddof=0)
            col_name = feat + "_rollingstd2_" + str(lag)
            dfx[col_name] = dfx[col_name2].rolling(lag).std(ddof=0)
    return dfx

def applyTelemetryParallel(dfGrouped, func):
    with Pool(cpu_count) as p:
        ret_list = p.map(func, [params for params in dfGrouped])
    return pd.concat(ret_list, axis=0)


start_time = time.time()  # count time
grouped = df_telemetry.groupby('machineID')
df_telemetry_feat = applyTelemetryParallel(grouped, telemetryFunc)
elapsed_time = time.time() - start_time # count time

print("\nTelemetry data processing completed", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))


df_telemetry_feat = df_telemetry_feat.join(pd.DataFrame(index=rng_12H), how='right').fillna(0)

del df_telemetry, grouped
gc.collect()

# processing error data set
df_error = pd.read_csv(file_error, index_col='datetime', parse_dates=['datetime'], encoding='utf-8')
df_error.sort_index(inplace=True)
error_feats = df_error['errorID'].unique()
df1 = pd.pivot_table(df_error, values='machineID', index=['datetime', 'machineID'], columns='errorID', aggfunc=len).fillna(0)
df1.reset_index('machineID', inplace=True)

def errorFunc(params):
    (name, df) = params
    dfx = df.join(pd.DataFrame(index=rng_H), how='right')
    dfx['machineID'] = name
    dfx.fillna(0, inplace=True)
    dfx.sort_index(inplace=True)
    for feat in error_feats:
        col_name = feat + "_rollingmean24"
        dfx[col_name] = dfx[feat].rolling(24).mean()
    return dfx

def applyErrorParallel(dfGrouped, func):
    with Pool(cpu_count) as p:
        ret_list = p.map(func, [params for params in dfGrouped])
    return pd.concat(ret_list, axis=0)


start_time = time.time()
grouped = df1.groupby('machineID')
df_error_feat = applyErrorParallel(grouped, errorFunc)

elapsed_time = time.time() - start_time

print("\nError data processing completed", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
df_error_feat = df_error_feat.join(pd.DataFrame(index=rng_12H), on='datetime', how='right').fillna(0)

del df1, df_error
gc.collect()

# processing maintenance data set
df_maint = pd.read_csv(file_maintenance, index_col='datetime', parse_dates=True, encoding='utf-8')
df_maint.sort_index(inplace=True)
df1 = pd.pivot_table(df_maint, values='machineID', index=['datetime', 'machineID'], columns='comp', aggfunc=len).fillna(0)
df1.sort_index(inplace=True)


start_time = time.time()

comp_feat = df_maint['comp'].unique()
midx = pd.MultiIndex.from_tuples(idx, names=('datetime', 'machineID'))
df1 = df1.join(pd.DataFrame(index=midx), on=['datetime', 'machineID'], how='right').fillna(0)
df1.reset_index('machineID', inplace=True)  # move machineID from index to normal column
df1.sort_index(inplace=True)

for feat in comp_feat:
    col_name = "sincelast" + feat
    df1[col_name] = df1.index.where(df1[feat].eq(1))
    df1[col_name] = (df1.index - df1.groupby('machineID')[col_name].ffill()).fillna(pd.Timedelta(0)).dt.days
    print("Done "+ feat)

elapsed_time = time.time() - start_time
print("\nMaintenance data processing completed", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

df_maint_feat = df1.join(pd.DataFrame(index=rng_12H), how='inner').fillna(0)
del df1, df_maint
gc.collect()


# processing label data set
df_failure = pd.read_csv(file_failure, index_col='datetime', parse_dates=True, encoding='utf-8')
df_label = pd.pivot_table(df_failure, values='machineID', index=['datetime', 'machineID'], columns='failure', aggfunc=len)
df_label = df_label.join(pd.DataFrame(index=midx), on=['datetime', 'machineID'],how='right')
df_label.sort_index(inplace=True)
df_label.reset_index('machineID', inplace=True) # move machineID from index to normal column

# backfill 7 days
win = 7* 24
df_label = df_label.groupby('machineID').bfill(limit=win)
df_label.fillna(0, inplace=True)
df_label.rename(columns={"comp1": "failure_comp1", "comp2": "failure_comp2", "comp3": "failure_comp3", "comp4": "failure_comp4"}, inplace=True)
df_label = df_label.join(pd.DataFrame(index=rng_12H), how='right')
df_label.reset_index(inplace=True)

print("\nLabeling completed")


del df_failure
gc.collect()


# Merge features and labels
df_error_feat.reset_index(inplace=True)
df_maint_feat.reset_index(inplace=True)
df_telemetry_feat.reset_index(inplace=True)

df_feat = pd.merge(df_error_feat, df_maint_feat, how='left', left_on=['datetime', 'machineID'], right_on = ['datetime', 'machineID'])
df_feat = pd.merge(df_feat, df_telemetry_feat, how='right', left_on=['datetime', 'machineID'], right_on = ['datetime', 'machineID'])
df_feat = pd.merge(df_feat, df_label, how='left', left_on=['datetime', 'machineID'], right_on = ['datetime', 'machineID'])

print("\n Merging data completed")

print("Sanity Check")
l1 = len(df_feat)
df_feat.dropna(inplace=True)
l2 = len(df_feat)
print("drop {} rows, data set has {} rows now".format(l1-l2, l2))
print(df_feat.isnull().any())
df_feat.to_csv('featured_data.csv', index=False)
