import os
import sys
from preprocessor import *

path = os.path.abspath('../util')
sys.path.append(path)  # 添加第三方模块路径到临时path变量中

from read_train_test import *


def get_time_agg(df):
    """对time_id进一步的特征构造"""
    vol_cols = ['log_return1_realized_volatility',
                'log_return2_realized_volatility',
                'trade_log_return_realized_volatility',
                'log_return1_realized_volatility_150',
                'log_return2_realized_volatility_150',
                'trade_log_return_realized_volatility_150',
                'log_return1_realized_volatility_300',
                'log_return2_realized_volatility_300',
                'trade_log_return_realized_volatility_300',
                'log_return1_realized_volatility_450',
                'log_return2_realized_volatility_450',
                'trade_log_return_realized_volatility_450']

    # Group by the time_id
    df_time_id = df.groupby(['time_id'])[vol_cols].agg(['mean', 'std', 'max', 'min']).reset_index()
    # Rename columns joining suffix
    df_time_id.columns = ['_'.join(col) for col in df_time_id.columns]
    df_time_id = df_time_id.add_suffix('_' + 'time')

    # Merge with original dataframe
    df = df.merge(df_time_id, how='left', left_on=['time_id'], right_on=['time_id__time'])
    df.drop(['time_id__time'], axis=1, inplace=True)

    return df


def calc_features_from_raw_data(data_dir, train, test):
    # Get unique stock ids
    train_stock_ids = train['stock_id'].unique()
    test_stock_ids = test['stock_id'].unique()

    # Preprocess them using Parallel and our single stock id functions
    train_ = preprocessor(data_dir, train_stock_ids, is_train=True)
    train = train.merge(train_, on=['row_id'], how='left')

    test_ = preprocessor(data_dir, test_stock_ids, is_train=False)
    test = test.merge(test_, on=['row_id'], how='left')
    return train, test


if __name__ == '__main__':
    data_path = "../data_init/"
    train_data, test_data = read_train_test(data_path)
    train_data, test_data = calc_features_from_raw_data(data_path, train_data, test_data)
    train_data = get_time_agg(train_data)
    test_data = get_time_agg(test_data)
    train_data = calc_taus(train_data)
    test_data = calc_taus(test_data)
    train_data.to_csv("../data_feature/train_raw.csv")
    test_data.to_csv("../data_feature/test_raw.csv")
