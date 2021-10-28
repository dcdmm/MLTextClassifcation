import pandas as pd
from basic_function import *


def rv_from_square(df, col, drop=True):
    """Function to speed up realized volatility calculation from squared sums of returns"""
    df[col + '_realized_volatility'] = np.sqrt(df[col + '_sq_sum'])
    df[col + '_realized_volatility_150'] = np.sqrt(df[col + '_sq_sum_150'])
    df[col + '_realized_volatility_300'] = np.sqrt(df[col + '_sq_sum_300'])
    df[col + '_realized_volatility_450'] = np.sqrt(df[col + '_sq_sum_450'])

    if drop:
        df = df.drop([col + '_sq_sum', col + '_sq_sum_150',
                      col + '_sq_sum_300', col + '_sq_sum_450'], axis=1)
    return df


def get_stats_window(df, fe_dict, seconds_in_bucket, add_suffix=False):
    """Function to get group stats for different windows (seconds in bucket)"""
    # Group by the window
    df_feature = df[df['seconds_in_bucket'] >= seconds_in_bucket].groupby(['time_id']).agg(fe_dict).reset_index()
    # Rename columns joining suffix
    df_feature.columns = ['_'.join(col) for col in df_feature.columns]
    # Add a suffix to differentiate windows
    if add_suffix:
        df_feature = df_feature.add_suffix('_' + str(seconds_in_bucket))
    return df_feature


def book_preprocessor(file_path):
    """
    book表特征数据集构造
    """
    df = pd.read_parquet(file_path)

    # Calculate Wap
    df['wap1'] = calc_wap1(df)
    df['wap2'] = calc_wap2(df)

    # Calculate log returns
    df['log_return1'] = df.groupby(['time_id'])['wap1'].apply(log_return)
    df['log_return2'] = df.groupby(['time_id'])['wap2'].apply(log_return)

    # Calculate square of log returns
    df['log_return1_sq'] = np.square(df['log_return1'])
    df['log_return2_sq'] = np.square(df['log_return2'])

    # Calculate spread features
    df['bid_ask_spread'] = (df['ask_price1'] - df['bid_price1'])
    df['bid_ask_spread2'] = (df['ask_price2'] - df['bid_price2'])
    df['bid_spread'] = df['bid_price1'] - df['bid_price2']
    df['ask_spread'] = df['ask_price2'] - df['ask_price1']
    df["total_spread"] = df['bid_spread'] + df['ask_spread']

    # Calculate depth features
    df['total_depth'] = (df['ask_size1'] + df['ask_size2']) + (df['bid_size1'] + df['bid_size2'])
    df['depth_imbalance'] = abs((df['ask_size1'] + df['ask_size2']) - (df['bid_size1'] + df['bid_size2']))

    # 聚合字典(所有时间)
    create_feature_dict = {
        'wap1': [np.sum, np.std],
        'wap2': [np.sum, np.std],
        'log_return1_sq': [np.sum],
        'log_return2_sq': [np.sum],
        'bid_ask_spread': [np.sum, np.max],
        'bid_ask_spread2': [np.sum],
        'bid_spread': [np.sum, np.max],
        'ask_spread': [np.sum, np.max],
        'total_spread': [np.sum, np.max],
        'total_depth': [np.sum],
        'depth_imbalance': [np.sum, np.max],
    }

    # 聚合字典(时间片)
    create_feature_dict_time = {
        'log_return1_sq': [np.sum],
        'log_return2_sq': [np.sum],
    }

    # Get the stats for different windows
    df_feature = get_stats_window(df, create_feature_dict, seconds_in_bucket=0, add_suffix=False)
    df_feature_450 = get_stats_window(df, create_feature_dict_time, seconds_in_bucket=450, add_suffix=True)
    df_feature_300 = get_stats_window(df, create_feature_dict_time, seconds_in_bucket=300, add_suffix=True)
    df_feature_150 = get_stats_window(df, create_feature_dict_time, seconds_in_bucket=150, add_suffix=True)

    # Merge all
    df_feature = df_feature.merge(df_feature_450, how='left', left_on='time_id_', right_on='time_id__450')
    df_feature = df_feature.merge(df_feature_300, how='left', left_on='time_id_', right_on='time_id__300')
    df_feature = df_feature.merge(df_feature_150, how='left', left_on='time_id_', right_on='time_id__150')
    # Drop unnecesary time_ids
    df_feature.drop(['time_id__450', 'time_id__300', 'time_id__150'], axis=1, inplace=True)

    # Calculate realized volatility from squared returns
    df_feature = rv_from_square(df_feature, 'log_return1', drop=True)
    df_feature = rv_from_square(df_feature, 'log_return2', drop=True)

    # Create row_id so we can merge
    stock_id = file_path.split('=')[1]
    df_feature['row_id'] = df_feature['time_id_'].apply(lambda x: f'{stock_id}-{x}')
    df_feature.drop(['time_id_'], axis=1, inplace=True)
    return df_feature


def trade_preprocessor(file_path):
    """trade表特征构造"""
    df = pd.read_parquet(file_path)

    df['log_return'] = df.groupby('time_id')['price'].apply(log_return)
    df['log_return_sq'] = np.square(df['log_return'])
    df['amount'] = df['price'] * df['size']

    # 聚合字典(所有时间)
    create_feature_dict = {
        'log_return_sq': [np.sum],
        'seconds_in_bucket': [count_unique],
        'size': [np.sum, np.max, np.min],
        'order_count': [np.sum, np.max],
        'amount': [np.sum, np.max, np.min],
    }
    # 聚合字典(时间片)
    create_feature_dict_time = {
        'log_return_sq': [np.sum],
        'seconds_in_bucket': [count_unique],
        'size': [np.sum],
        'order_count': [np.sum],
    }

    # Get the stats for different windows
    df_feature = get_stats_window(df, create_feature_dict, seconds_in_bucket=0, add_suffix=False)
    df_feature_450 = get_stats_window(df, create_feature_dict_time, seconds_in_bucket=450, add_suffix=True)
    df_feature_300 = get_stats_window(df, create_feature_dict_time, seconds_in_bucket=300, add_suffix=True)
    df_feature_150 = get_stats_window(df, create_feature_dict_time, seconds_in_bucket=150, add_suffix=True)

    # Merge all
    df_feature = df_feature.merge(df_feature_450, how='left', left_on='time_id_', right_on='time_id__450')
    df_feature = df_feature.merge(df_feature_300, how='left', left_on='time_id_', right_on='time_id__300')
    df_feature = df_feature.merge(df_feature_150, how='left', left_on='time_id_', right_on='time_id__150')
    # Drop unnecesary time_ids
    df_feature.drop(['time_id__450', 'time_id__300', 'time_id__150'], axis=1, inplace=True)

    df_feature = rv_from_square(df_feature, 'log_return', drop=True)

    df_feature = df_feature.add_prefix('trade_')
    stock_id = file_path.split('=')[1]
    df_feature['row_id'] = df_feature['trade_time_id_'].apply(lambda x: f'{stock_id}-{x}')
    df_feature.drop(['trade_time_id_'], axis=1, inplace=True)
    return df_feature
