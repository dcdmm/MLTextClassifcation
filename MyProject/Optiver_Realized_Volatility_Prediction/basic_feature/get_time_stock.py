def get_time_stock(df):
    """对stock_id和time_id进一步的特征构造"""

    vol_cols = ['log_return1_realized_volatility',
                'log_return2_realized_volatility',
                'log_return1_realized_volatility_400',
                'log_return2_realized_volatility_400',
                'log_return1_realized_volatility_300',
                'log_return2_realized_volatility_300',
                'log_return1_realized_volatility_200',
                'log_return2_realized_volatility_200',
                'trade_log_return_realized_volatility',
                'trade_log_return_realized_volatility_400',
                'trade_log_return_realized_volatility_300',
                'trade_log_return_realized_volatility_200']

    # Group by the stock id
    df_stock_id = df.groupby(['stock_id'])[vol_cols].agg(['mean', 'std', 'max', 'min', ]).reset_index()
    # Rename columns joining suffix
    df_stock_id.columns = ['_'.join(col) for col in df_stock_id.columns]
    df_stock_id = df_stock_id.add_suffix('_' + 'stock')

    # Group by the stock id
    df_time_id = df.groupby(['time_id'])[vol_cols].agg(['mean', 'std', 'max', 'min', ]).reset_index()
    # Rename columns joining suffix
    df_time_id.columns = ['_'.join(col) for col in df_time_id.columns]
    df_time_id = df_time_id.add_suffix('_' + 'time')

    # Merge with original dataframe
    df = df.merge(df_stock_id, how='left', left_on=['stock_id'], right_on=['stock_id__stock'])
    df = df.merge(df_time_id, how='left', left_on=['time_id'], right_on=['time_id__time'])
    df.drop(['stock_id__stock', 'time_id__time'], axis=1, inplace=True)
    return df
