import pandas as pd
from function import *


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
    df['wap3'] = calc_wap3(df)
    df['wap4'] = calc_wap4(df)

    # Calculate log returns
    df['log_return1'] = df.groupby(['time_id'])['wap1'].apply(log_return)
    df['log_return2'] = df.groupby(['time_id'])['wap2'].apply(log_return)
    df['log_return3'] = df.groupby(['time_id'])['wap3'].apply(log_return)
    df['log_return4'] = df.groupby(['time_id'])['wap4'].apply(log_return)

    # Calculate wap balance
    df['wap_balance'] = abs(df['wap1'] - df['wap2'])

    # Calculate spread
    df['price_spread'] = (df['ask_price1'] - df['bid_price1']) / ((df['ask_price1'] + df['bid_price1']) / 2)
    df['price_spread2'] = (df['ask_price2'] - df['bid_price2']) / ((df['ask_price2'] + df['bid_price2']) / 2)
    df['bid_spread'] = df['bid_price1'] - df['bid_price2']
    df['ask_spread'] = df['ask_price1'] - df['ask_price2']
    df["bid_ask_spread"] = abs(df['bid_spread'] - df['ask_spread'])
    df['total_volume'] = (df['ask_size1'] + df['ask_size2']) + (df['bid_size1'] + df['bid_size2'])
    df['volume_imbalance'] = abs((df['ask_size1'] + df['ask_size2']) - (df['bid_size1'] + df['bid_size2']))

    # 聚合字典(所有时间)
    create_feature_dict = {
        'wap1': [np.sum, np.std],
        'wap2': [np.sum, np.std],
        'wap3': [np.sum, np.std],
        'wap4': [np.sum, np.std],
        'log_return1': [realized_volatility],
        'log_return2': [realized_volatility],
        'log_return3': [realized_volatility],
        'log_return4': [realized_volatility],
        'wap_balance': [np.sum, np.max],
        'price_spread': [np.sum, np.max],
        'price_spread2': [np.sum, np.max],
        'bid_spread': [np.sum, np.max],
        'ask_spread': [np.sum, np.max],
        'total_volume': [np.sum, np.max],
        'volume_imbalance': [np.sum, np.max],
        "bid_ask_spread": [np.sum, np.max],
    }

    # 聚合字典(时间片)
    create_feature_dict_time = {
        'log_return1': [realized_volatility],
        'log_return2': [realized_volatility],
        'log_return3': [realized_volatility],
        'log_return4': [realized_volatility],
    }

    # Get the stats for different windows
    df_feature = get_stats_window(df, create_feature_dict, seconds_in_bucket=0, add_suffix=False)
    df_feature_500 = get_stats_window(df, create_feature_dict_time, seconds_in_bucket=500, add_suffix=True)
    df_feature_400 = get_stats_window(df, create_feature_dict_time, seconds_in_bucket=400, add_suffix=True)
    df_feature_300 = get_stats_window(df, create_feature_dict_time, seconds_in_bucket=300, add_suffix=True)
    df_feature_200 = get_stats_window(df, create_feature_dict_time, seconds_in_bucket=200, add_suffix=True)
    df_feature_100 = get_stats_window(df, create_feature_dict_time, seconds_in_bucket=100, add_suffix=True)

    # Merge all
    df_feature = df_feature.merge(df_feature_500, how='left', left_on='time_id_', right_on='time_id__500')
    df_feature = df_feature.merge(df_feature_400, how='left', left_on='time_id_', right_on='time_id__400')
    df_feature = df_feature.merge(df_feature_300, how='left', left_on='time_id_', right_on='time_id__300')
    df_feature = df_feature.merge(df_feature_200, how='left', left_on='time_id_', right_on='time_id__200')
    df_feature = df_feature.merge(df_feature_100, how='left', left_on='time_id_', right_on='time_id__100')
    # Drop unnecesary time_ids
    df_feature.drop(['time_id__500', 'time_id__400', 'time_id__300', 'time_id__200', 'time_id__100'], axis=1,
                    inplace=True)

    # Create row_id so we can merge
    stock_id = file_path.split('=')[1]
    df_feature['row_id'] = df_feature['time_id_'].apply(lambda x: f'{stock_id}-{x}')
    df_feature.drop(['time_id_'], axis=1, inplace=True)
    return df_feature


def trade_preprocessor(file_path):
    """trade表特征构造"""
    df = pd.read_parquet(file_path)

    df['log_return'] = df.groupby('time_id')['price'].apply(log_return)
    df['amount'] = df['price'] * df['size']  # add

    # 聚合字典(所有时间)
    create_feature_dict = {
        'log_return': [realized_volatility],
        'seconds_in_bucket': [count_unique],
        'size': [np.sum, np.max, np.min],
        'order_count': [np.sum, np.max],
        'amount': [np.sum, np.max, np.min],
    }

    # 聚合字典(时间片)
    create_feature_dict_time = {
        'log_return': [realized_volatility],
        'seconds_in_bucket': [count_unique],
        'size': [np.sum],
        'order_count': [np.sum],
    }

    # Get the stats for different windows
    df_feature = get_stats_window(df, create_feature_dict, seconds_in_bucket=0, add_suffix=False)
    df_feature_500 = get_stats_window(df, create_feature_dict_time, seconds_in_bucket=500, add_suffix=True)
    df_feature_400 = get_stats_window(df, create_feature_dict_time, seconds_in_bucket=400, add_suffix=True)
    df_feature_300 = get_stats_window(df, create_feature_dict_time, seconds_in_bucket=300, add_suffix=True)
    df_feature_200 = get_stats_window(df, create_feature_dict_time, seconds_in_bucket=200, add_suffix=True)
    df_feature_100 = get_stats_window(df, create_feature_dict_time, seconds_in_bucket=100, add_suffix=True)

    # ***********************************************************************************************************
    # 衡量波动率的一个指标
    def tendency(price, vol):
        df_diff = np.diff(price)
        val = (df_diff / price[1:]) * 100
        power = np.sum(val * vol[1:])
        return power

    lis = []
    for n_time_id in df['time_id'].unique():
        # 计算每一个n_time_id下的df_id,tendencyV,f_max,......,energy_v,iqr_p_v等指标
        df_id = df[df['time_id'] == n_time_id]  # time_id == n_time_id的样本

        tendencyV = tendency(df_id['price'].values, df_id['size'].values)

        f_max = np.sum(df_id['price'].values > np.mean(df_id['price'].values))
        f_min = np.sum(df_id['price'].values < np.mean(df_id['price'].values))
        df_max = np.sum(np.diff(df_id['price'].values) > 0)
        df_min = np.sum(np.diff(df_id['price'].values) < 0)

        abs_diff = np.median(np.abs(df_id['price'].values - np.mean(df_id['price'].values)))
        energy = np.mean(df_id['price'].values ** 2)
        iqr_p = np.percentile(df_id['price'].values, 75) - np.percentile(df_id['price'].values, 25)  # 75%分位数 - 25%分位数

        abs_diff_v = np.median(np.abs(df_id['size'].values - np.mean(df_id['size'].values)))
        energy_v = np.sum(df_id['size'].values ** 2)
        iqr_p_v = np.percentile(df_id['size'].values, 75) - np.percentile(df_id['size'].values, 25)

        lis.append({'time_id': n_time_id,
                    'tendency': tendencyV,
                    'f_max': f_max,
                    'f_min': f_min,
                    'df_max': df_max,
                    'df_min': df_min,
                    'abs_diff': abs_diff,
                    'energy': energy,
                    'iqr_p': iqr_p,
                    'abs_diff_v': abs_diff_v,
                    'energy_v': energy_v,
                    'iqr_p_v': iqr_p_v})

    df_lr = pd.DataFrame(lis)  # 所有time_id的指标转换为DataFrame
    # ***********************************************************************************************************

    # Merge all
    df_feature = df_feature.merge(df_lr, how='left', left_on='time_id_', right_on='time_id')
    df_feature = df_feature.merge(df_feature_500, how='left', left_on='time_id_', right_on='time_id__500')
    df_feature = df_feature.merge(df_feature_400, how='left', left_on='time_id_', right_on='time_id__400')
    df_feature = df_feature.merge(df_feature_300, how='left', left_on='time_id_', right_on='time_id__300')
    df_feature = df_feature.merge(df_feature_200, how='left', left_on='time_id_', right_on='time_id__200')
    df_feature = df_feature.merge(df_feature_100, how='left', left_on='time_id_', right_on='time_id__100')

    # Drop unnecesary time_ids
    df_feature.drop(['time_id__500', 'time_id__400', 'time_id__300', 'time_id__200', 'time_id', 'time_id__100'], axis=1,
                    inplace=True)

    df_feature = df_feature.add_prefix('trade_')
    stock_id = file_path.split('=')[1]
    df_feature['row_id'] = df_feature['trade_time_id_'].apply(lambda x: f'{stock_id}-{x}')
    df_feature.drop(['trade_time_id_'], axis=1, inplace=True)
    return df_feature
