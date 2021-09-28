from data_gene import *
import pandas as pd

nnn = ['time_id',
       'log_return1_realized_volatility_0c1',
       'log_return1_realized_volatility_1c1',
       'log_return1_realized_volatility_3c1',
       'log_return1_realized_volatility_4c1',
       'log_return1_realized_volatility_6c1',
       'total_volume_sum_0c1',
       'total_volume_sum_1c1',
       'total_volume_sum_3c1',
       'total_volume_sum_4c1',
       'total_volume_sum_6c1',
       'trade_size_sum_0c1',
       'trade_size_sum_1c1',
       'trade_size_sum_3c1',
       'trade_size_sum_4c1',
       'trade_size_sum_6c1',
       'trade_order_count_sum_0c1',
       'trade_order_count_sum_1c1',
       'trade_order_count_sum_3c1',
       'trade_order_count_sum_4c1',
       'trade_order_count_sum_6c1',
       'price_spread_sum_0c1',
       'price_spread_sum_1c1',
       'price_spread_sum_3c1',
       'price_spread_sum_4c1',
       'price_spread_sum_6c1',
       'bid_spread_sum_0c1',
       'bid_spread_sum_1c1',
       'bid_spread_sum_3c1',
       'bid_spread_sum_4c1',
       'bid_spread_sum_6c1',
       'ask_spread_sum_0c1',
       'ask_spread_sum_1c1',
       'ask_spread_sum_3c1',
       'ask_spread_sum_4c1',
       'ask_spread_sum_6c1',
       'volume_imbalance_sum_0c1',
       'volume_imbalance_sum_1c1',
       'volume_imbalance_sum_3c1',
       'volume_imbalance_sum_4c1',
       'volume_imbalance_sum_6c1',
       'bid_ask_spread_sum_0c1',
       'bid_ask_spread_sum_1c1',
       'bid_ask_spread_sum_3c1',
       'bid_ask_spread_sum_4c1',
       'bid_ask_spread_sum_6c1',
       'size_tau2_0c1',
       'size_tau2_1c1',
       'size_tau2_3c1',
       'size_tau2_4c1',
       'size_tau2_6c1']

if __name__ == "__main__":
    train = pd.read_pickle("../basic_feature/basic_train.pkl")
    test = pd.read_pickle("../basic_feature/basic_test.pkl")

    mat1 = pd.read_pickle('../clustering_feature/mat1.pkl')
    mat2 = pd.read_pickle('../clustering_feature/mat2.pkl')

    train_ck, test_ck = lgb_data(train, test, mat1, mat2, nnn)
    train_nn_ck, test_nn_ck = nn_data(train, test, mat1, mat2, nnn)

    print(train_ck.shape, test_ck.shape)
    print(train_nn_ck.shape, test_nn_ck.shape)  # train_nn_ck没有"row_id"列

    # 保存最后用于模型训练的数据
    train_ck.to_pickle("train_ck.pkl")
    test_ck.to_pickle("test_ck.pkl")
    train_nn_ck.to_pickle("train_nn_ck.pkl")
    test_nn_ck.to_pickle("test_nn_ck.pkl")
