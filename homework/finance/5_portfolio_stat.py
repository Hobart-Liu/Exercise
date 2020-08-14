import pandas as pd
from os.path import expanduser, join
import datetime as dt

import matplotlib.pyplot as plt

'''
reference link: http://quantsoftware.gatech.edu/MC1-Project-1
test case: http://quantsoftware.gatech.edu/MC1-Project-1-Test-Cases-spr2016

(reference code: 
https://stackoverflow.com/questions/9622163/save-plot-to-image-file-instead-of-displaying-it-using-matplotlib)

'''


def symbol_to_path(symbol):
    return join(expanduser("~/github/python/finance"), "{}.csv".format(symbol))


def get_trade_dates(start, end):
    dates = pd.date_range(start, end)
    df = pd.DataFrame(index=dates)
    index_file = symbol_to_path('SPY')
    df_tmp = pd.read_csv(index_file, parse_dates=True, index_col='Date',
                         usecols=['Date','Adj Close'], na_values=['nan'])
    df = df.join(df_tmp)
    df.dropna(inplace=True)
    return df.index


def get_data(symbols, dates):

    df_final = pd.DataFrame(index=dates)

    for symbol in symbols:
        file_path = symbol_to_path(symbol)
        df_tmp = pd.read_csv(file_path, parse_dates=True, index_col='Date',
                             usecols=['Date','Adj Close'], na_values=['nan'])
        df_tmp = df_tmp.rename(columns={'Adj Close': symbol})
        df_final = df_final.join(df_tmp)

    if df_final.isnull().values.any():
        fill_missing_values(df_final)

    return df_final

def fill_missing_values(df_data):
    df_data.fillna(method='ffill', inplace=True)
    df_data.fillna(method='bfill', inplace=True)

def normalize_data(df):
    return df/df.iloc[0]

def compute_daily_returns(df):
    daily_rets = (df/df.shift(1)) - 1
    daily_rets.iloc[0] = 0
    return daily_rets


def assess_portfolio(sd, ed, syms, allocs, sv, rfr=0.0, sf=252, gen_plot=False):
    '''

    :param sd: A datetime object that represents the start date
    :param ed: A datetime object that represents the end date
    :param syms: A list of 2 or more symbols that make up the portfolio
    :param allocs: A list of 2 or more allocations to the stocks, must sum to 1.0
    :param sv: start value of the portfolio
    :param rfr: The risk free return per sample period (single value, not array)
                that does not change for the entire date range
    :param sf: Sampleing frequency per year
    :param gen_plot: if False, do not create any output,
                     if True, it is OK to output a plot such as plot.png
    :return: Cumulative return, Average daily return, standard deviation of daily return,
             Sharp ratio, End value of portfolio
              (cr, adr, sddr, sr, ev)
    '''

    dates = get_trade_dates(sd, ed)

    prices = get_data(syms, dates)

    norm = normalize_data(prices)

    pos_val = norm * allocs * sv

    port_val = pos_val.sum(axis=1)

    daily_rets = compute_daily_returns(port_val)
    daily_rets = daily_rets[1:]

    cr = port_val[-1]/port_val[0] - 1

    adr, sddr  = daily_rets.mean(), daily_rets.std()

    dailyrfr = ((1.0+rfr)**(1./sf))-1
    sr = ((daily_rets - dailyrfr).mean()/sddr)*(sf**(1./2))
    ev = port_val[-1]

    if gen_plot:
        ax = normalize_data(port_val).plot(title='Daily portfolio value vs. S&P 500', label='Portfolio')
        SPY = get_data(['SPY'], dates = dates)
        normed_SPY = normalize_data(SPY)
        normed_SPY.plot(label='SPY', ax=ax)
        ax.set_xlabel('date')
        ax.set_ylabel('price')
        ax.legend(loc='best')
        plt.savefig('plot.png')
        # plt.show()
    
    print("Start Date:", sd)
    print("End Date:", ed)
    print("Symbols:", syms)
    print("Allocations:", allocs)
    print("Starting Portfolio Value:", sv)
    print("Cumulative Return:", cr)
    print("Average Daily Return:", adr)
    print("Sharpe Ratio:", sr)
    print("Volatility (stdev of daily returns):", sddr)
    print("Ending Portfolio Value:", ev)
    

    return cr, adr, sddr, sr, ev


def test_run():
    assess_portfolio(sd=dt.datetime(2006, 1, 3), ed=dt.datetime(2008, 1, 2), \
                     syms=['MMM', 'MO', 'MSFT', 'INTC'], \
                     allocs=[0.0, 0.9, 0.1, 0.0], \
                     sv=1000000, rfr=0.0, sf=252.0, \
                     gen_plot=True)

if __name__ == "__main__":
    test_run()