import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os.path import expanduser, join
from pandas.stats.moments import rolling_mean, rolling_std

def symbol_to_path(symbol):
    return join(expanduser("~/github/python/finance"), "{}.csv".format(symbol))

def get_data(symbols, dates):

    df_final = pd.DataFrame(index=dates)

    if 'hs300' not in symbols:
        symbols.insert(0, 'hs300')

    for symbol in symbols:
        file_path = symbol_to_path(symbol)
        df_tmp = pd.read_csv(file_path, parse_dates=True, index_col='date',
                             usecols=['date','close'], na_values=['nan'])
        df_tmp = df_tmp.rename(columns={'close': symbol})
        df_final = df_final.join(df_tmp)
        if symbol == 'hs300':
            df_final = df_final.dropna(subset=['hs300'])

    return df_final

def fill_missing_values(df_data):
    df_data.fillna(method='ffill', inplace=True)
    df_data.fillna(method='bfill', inplace=True)

def compute_daily_returns(df):
    df = df/df.shift(1) -1
    df.iloc[0, :] = 0
    return df


# define data range, use hs300 as indication to mark market working day
start_date = '2010-01-01'
end_date = '2010-12-31'
dates = pd.date_range(start_date, end_date)

# define stock list, hs300 will be added by default.
symbol_list= ['601318', '000538']
df = get_data(symbol_list, dates)

# check if there is any nan data in df
# fill na with first forward and then backward if any
if df.isnull().values.any():
    fill_missing_values(df)

df = compute_daily_returns(df)


# plot
ax = df[['601318', 'hs300', '000538']].plot(title='daily returns', label='601318')

# ax = df[['000538','hs300']].plot.hist(bins=100, label='601318', grid=False, alpha=0.5)

# df['601318'].hist(bins=20, grid=False, label='601318', alpha=0.5)
# df['000538'].hist(bins=20, grid=False, label='000538', alpha=0.5)
#
# plt.legend(loc='best')
# set labels
ax.set_xlabel('date')
ax.set_ylabel('price')
ax.legend(loc='best')
plt.show()