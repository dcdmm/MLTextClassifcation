import numpy as np


def calc_wap1(df):
    """Function to calculate first WAP"""
    wap = (df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']) / (
            df['bid_size1'] + df['ask_size1'])
    return wap


def calc_wap2(df):
    """Function to calculate second WAP"""
    wap = (df['bid_price2'] * df['ask_size2'] + df['ask_price2'] * df['bid_size2']) / (
            df['bid_size2'] + df['ask_size2'])
    return wap


def log_return(series):
    """
    Function to calculate the log of the return
    Remember that logb(x / y) = logb(x) - logb(y)
    """
    return np.log(series).diff()


def realized_volatility(series):
    """Calculate the realized volatility"""
    return np.sqrt(np.sum(series ** 2))


def count_unique(series):
    """Function to count unique elements of a series"""
    return len(np.unique(series))


def linear_return(series):
    """Function to calculate diff of a series"""
    return series.diff()


def calc_taus(df):
    """Function to calculate Tau features"""
    df['size_tau'] = np.sqrt(1 / df['trade_seconds_in_bucket_count_unique'])
    df['size_tau2'] = np.sqrt(1 / df['trade_order_count_sum'])
    return df
