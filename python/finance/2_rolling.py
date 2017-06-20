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


# rolling 20 days
r = df['hs300'].rolling(window=20)
rm_hs300, rstd_hs300 = r.mean(), r.std()
# Bollinger Bands
up_band, low_band = rm_hs300 + 2* rstd_hs300, rm_hs300 - 2*rstd_hs300


# plot
ax = df['hs300'].plot(title='Bollinger Bands', label='hs300')
rm_hs300.plot(label='Rolling mean', ax=ax)
up_band.plot(label='up', ax=ax)
low_band.plot(label='low', ax=ax)
# set labels
ax.set_xlabel('date')
ax.set_ylabel('price')
ax.legend(loc='lower left')
plt.show()