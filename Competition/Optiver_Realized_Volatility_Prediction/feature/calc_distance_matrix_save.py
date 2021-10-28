from joint_preprocessor import *
import os
import sys
from tick_prep_func import *
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import pairwise_distances

path = os.path.abspath('../util')
sys.path.append(path)  # 添加第三方模块路径到临时path变量中

from read_train_test import *


def calc_distance_matrix(data_dir):
    """Function to calculate distances from reconstructed stock prices"""
    train, test = read_train_test(data_dir)

    train_stock_ids = train['stock_id'].unique()
    test_stock_ids = test['stock_id'].unique()

    # Estimate stock prices
    df_prices = joint_preprocessor(data_dir, train_stock_ids, tick_prep_func)
    df_prices = train.merge(df_prices, how='left', on='row_id')
    df_prices = df_prices.pivot('time_id', 'stock_id', 'price')  # 重塑
    df_prices_test = joint_preprocessor(data_dir, test_stock_ids, tick_prep_func, is_train=False)
    df_prices_test = test.merge(df_prices_test, how='left', on='row_id')
    df_prices_test = df_prices_test.pivot('time_id', 'stock_id', 'price')

    df_prices = pd.concat([df_prices, df_prices_test])
    df_prices = df_prices.fillna(df_prices.mean())

    # Minmax scale
    df_prices = pd.DataFrame(minmax_scale(df_prices), index=df_prices.index)
    # Calculate distance matrix
    d_mat = pairwise_distances(df_prices.values)

    # Sort each row so we have time_ids sorted by distance to the rows time_id
    for row in range(d_mat.shape[0]):
        d_mat[row, :] = df_prices.index[np.argsort(d_mat[row, :]).astype(int)]

    d_df = pd.DataFrame(d_mat, index=df_prices.index).astype(int)
    return d_df

if __name__ == "__main__":
    data_dir = "../data_init/"
    d_df = calc_distance_matrix(data_dir)
    d_df.to_pickle("../data_feature/d_df.pkl")