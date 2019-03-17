import pandas as pd
import numpy as np

df_erro = pd.read_csv('data/pdm_errors_data.csv',index_col='datetime', parse_dates=['datetime'], encoding='utf-8')

def get_table(start='2015-01-01 00', end='2016-01-01 00', cols=[], force_rebuild=False):
    if force_rebuild:
        rng = pd.date_range(start=start, end=end, freq='H')
        machineID = np.arange(1, 1001)

        idx = [(d, m) for d in rng for m in machineID]

        df = pd.DataFrame(index=pd.MultiIndex.from_tuples(idx), columns = cols).fillna(0)
        df.index.names =['datetime', 'machineID']
        df.sort_index(inplace=True)
        df.to_csv("tmp_error_empty.csv")
    else:
        df = pd.read_csv('tmp_error_empty.csv', index_col=['datetime', 'machineID'], parse_dates=['datetime'])


    return df


# df = get_table(start='2015-01-01 00', end='2016-01-01 00', force_rebuild=True, cols = df_erro['errorID'].unique())
df = get_table()

l = len(df_erro)
print("len = ", l)

print_i = 0
for row in df_erro.itertuples():
    tid, mid, eid = row[0], row[1], row[2]
    df.loc[(tid, mid), eid] = 1

    if print_i % 100  == 0:
        print("{} > ".format(print_i), end="", flush=True)

    print_i += 1

df.to_csv('record.csv')

print("\nComplete")
