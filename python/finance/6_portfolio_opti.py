import pandas as pd
from os.path import expanduser, join
import datetime as dt

import matplotlib.pyplot as plt
import scipy.optimize as sco
import numpy as np

'''
reference:
http://quantsoftware.gatech.edu/MC1-Project-2-archive
'''


def symbol_to_path(symbol):
    return join(expanduser("~/github/python/finance"), "{}.csv".format(symbol))


def get_trade_dates(start, end):
    dates = pd.date_range(start, end)
    df = pd.DataFrame(index=dates)
    index_file = symbol_to_path('SPY')
    df_tmp = pd.read_csv(index_file, parse_dates=True, index_col='Date',
                         usecols=['Date', 'Adj Close'], na_values=['nan'])
    df = df.join(df_tmp)
    df.dropna(inplace=True)
    return df.index


def get_data(symbols, dates):
    df_final = pd.DataFrame(index=dates)

    for symbol in symbols:
        file_path = symbol_to_path(symbol)
        df_tmp = pd.read_csv(file_path, parse_dates=True, index_col='Date',
                             usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_tmp = df_tmp.rename(columns={'Adj Close': symbol})
        df_final = df_final.join(df_tmp)

    if df_final.isnull().values.any():
        fill_missing_values(df_final)

    return df_final


def fill_missing_values(df_data):
    df_data.fillna(method='ffill', inplace=True)
    df_data.fillna(method='bfill', inplace=True)


def normalize_data(df):
    return df / df.iloc[0]


def compute_daily_returns(df):
    daily_rets = (df / df.shift(1)) - 1
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
    :param gen_plot: if True, crate a plot named plot.png
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

    cr = port_val[-1] / port_val[0] - 1

    adr, sddr = daily_rets.mean(), daily_rets.std()

    dailyrfr = ((1.0 + rfr) ** (1. / sf)) - 1
    sr = ((daily_rets - dailyrfr).mean() / sddr) * (sf ** (1. / 2))
    ev = port_val[-1]

    if gen_plot:
        ax = normalize_data(port_val).plot(title='Daily portfolio value vs. S&P 500', label='Portfolio')
        SPY = get_data(['SPY'], dates=dates)
        normed_SPY = normalize_data(SPY)
        normed_SPY.plot(label='SPY', ax=ax)
        ax.set_xlabel('date')
        ax.set_ylabel('price')
        ax.legend(loc='best')
        plt.savefig('plot.png')

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

def sharp_ratio(allocs, normed, rfr=0, sf=252):
    alloced = normed * allocs
    port_val = alloced.sum(axis=1)
    daily_rets = compute_daily_returns(port_val)
    daily_rets = daily_rets[1:]
    sddr = daily_rets.std()
    dailyrfr = ((1.0 + rfr) ** (1. / sf)) - 1
    sr = ((daily_rets - dailyrfr).mean() / sddr) * (sf ** (1. / 2))
    return sr * -1



def optimize_portfolio(sd, ed, syms, gen_plot=False):
    '''

    :param sd: A datetime object that represents the start date
    :param ed: A datetime object that represents the end date
    :param syms: A list of symbols that make up the portfolio
    :param gen_plot: if Ture, create a plot named plot.png
    :return allocs: A 1-d Numpy ndarray of allocations to the stocks.
                     All the allocations must be between 0.0 and 1.0 and
                     they must sum to 1.0
    :return cr: Cumulative return
    :return adr: Average daily return
    :return sddr: Standard deviation of daily return
    :return sr: Sharpe ratio
    '''
    dates = get_trade_dates(sd, ed)
    prices = get_data(symbols=syms, dates=dates)
    normed = normalize_data(prices)

    guess_allocs = [1./len(syms)] * len(syms)
    bnds = ((0., 1.), ) * len(syms)
    constraints = ({'type': "eq", 'fun': lambda allocs: 1.0 - np.sum(allocs)})

    allocs = sco.minimize(fun=sharp_ratio,
                          x0=guess_allocs,
                          args=(normed, ),
                          method='SLSQP',
                          bounds= bnds,
                          constraints=constraints,
                          options={'disp':True})

    cr, adr, sddr, sr, er = assess_portfolio(sd=sd,
                                             ed=ed,
                                             syms=syms,
                                             allocs=allocs.x,
                                             sv=1,
                                             rfr=0.0,
                                             sf=252,
                                             gen_plot=gen_plot)
    return allocs.x, cr, adr, sddr, sr

def test_run():
    allocs, cr, adr, sddr, sr = \
        optimize_portfolio(sd=dt.datetime(2004,1,1), ed=dt.datetime(2006,1,1), \
        syms=['AXP', 'HPQ', 'IBM', 'HNZ'], gen_plot=True)
    print(allocs, cr, adr, sddr, sr)

if __name__ == "__main__":
    test_run()