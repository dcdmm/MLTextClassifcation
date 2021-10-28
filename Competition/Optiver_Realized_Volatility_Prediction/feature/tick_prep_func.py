import pandas as pd
from basic_function import *


def tick_prep_func(file_path_book):
    """
    Function to calculate reconstructed stock prices from standardized book prices
    时间逆向特征
    参考:https://en.wikipedia.org/wiki/Tick_size
    """
    stock = file_path_book.split('=')[1]
    df = pd.read_parquet(file_path_book,
                         columns=['time_id', 'bid_price1', 'ask_price1', 'bid_price2', 'ask_price2'])
    df['bid_price1'] = np.abs(df.groupby('time_id')['bid_price1'].apply(linear_return))
    df['bid_price2'] = np.abs(df.groupby('time_id')['bid_price2'].apply(linear_return))
    df['ask_price1'] = np.abs(df.groupby('time_id')['ask_price1'].apply(linear_return))
    df['ask_price2'] = np.abs(df.groupby('time_id')['ask_price2'].apply(linear_return))
    df = df[['bid_price1', 'bid_price2', 'ask_price1', 'ask_price2']].groupby(df['time_id']).agg(
        lambda x: np.min(x[x > 0])).reset_index()

    # tick:是报价中最小的价格增量
    df['tick'] = df[['bid_price1', 'bid_price2', 'ask_price1', 'ask_price2']].min(axis=1)
    df['price'] = 0.01 / df['tick']

    df['row_id'] = df['time_id'].apply(lambda x: f'{stock}-{x}')
    df.drop(['time_id', 'bid_price1', 'bid_price2', 'ask_price1', 'ask_price2'], axis=1, inplace=True)
    return df
