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
# beta is slop, alpha is intersection of axis
# axis x is hs300
# axis y is 601318
beta_601318, alpha_601318 = np.polyfit(df['601318'], df['000538'], 1)
print("slop (beta) is: ", beta_601318)
print("axis intersection (alpha) is: ", alpha_601318)
fig = plt.figure()
ax = plt.subplot()
ax.plot(df['000538'], beta_601318*df['000538'] + alpha_601318, '-', color='r')
df.plot(kind='scatter', x='000538', y='601318', ax = ax)

ax.legend(loc='best')

print(df.corr(method='pearson'))

# set labels
# ax.set_xlabel('date')
# ax.set_ylabel('price')
# ax.legend(loc='best')
plt.show()