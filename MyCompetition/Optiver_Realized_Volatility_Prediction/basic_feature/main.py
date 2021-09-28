from read_train_test import *
from preprocessor import *
from get_time_stock import *

data_dir = "../init_data/"  # 数据路径

if __name__ == '__main__':
    train, test = read_train_test(data_dir=data_dir)
    # Get unique stock ids
    train_stock_ids = train['stock_id'].unique()
    # Preprocess them using Parallel and our single stock id functions
    train_ = preprocessor(data_dir, train_stock_ids, is_train=True)
    train = train.merge(train_, on=['row_id'], how='left')

    # Get unique stock ids
    test_stock_ids = test['stock_id'].unique()
    # Preprocess them using Parallel and our single stock id functions
    test_ = preprocessor(data_dir, test_stock_ids, is_train=False)
    test = test.merge(test_, on=['row_id'], how='left')

    # Get group stats of time_id and stock_id
    train = get_time_stock(train)
    test = get_time_stock(test)
    print(train.shape)
    print(test.shape)

    train['size_tau'] = np.sqrt(1 / train['trade_seconds_in_bucket_count_unique'])
    test['size_tau'] = np.sqrt(1 / test['trade_seconds_in_bucket_count_unique'])
    train['size_tau_400'] = np.sqrt(1 / train['trade_seconds_in_bucket_count_unique_400'])
    test['size_tau_400'] = np.sqrt(1 / test['trade_seconds_in_bucket_count_unique_400'])
    train['size_tau_300'] = np.sqrt(1 / train['trade_seconds_in_bucket_count_unique_300'])
    test['size_tau_300'] = np.sqrt(1 / test['trade_seconds_in_bucket_count_unique_300'])
    train['size_tau_200'] = np.sqrt(1 / train['trade_seconds_in_bucket_count_unique_200'])
    test['size_tau_200'] = np.sqrt(1 / test['trade_seconds_in_bucket_count_unique_200'])

    train['size_tau2'] = np.sqrt(1 / train['trade_order_count_sum'])
    test['size_tau2'] = np.sqrt(1 / test['trade_order_count_sum'])
    train['size_tau2_400'] = np.sqrt(0.33 / train['trade_order_count_sum'])
    test['size_tau2_400'] = np.sqrt(0.33 / test['trade_order_count_sum'])
    train['size_tau2_300'] = np.sqrt(0.5 / train['trade_order_count_sum'])
    test['size_tau2_300'] = np.sqrt(0.5 / test['trade_order_count_sum'])
    train['size_tau2_200'] = np.sqrt(0.66 / train['trade_order_count_sum'])
    test['size_tau2_200'] = np.sqrt(0.66 / test['trade_order_count_sum'])

    train['size_tau2_d'] = train['size_tau2_400'] - train['size_tau2']
    test['size_tau2_d'] = test['size_tau2_400'] - test['size_tau2']

    # 保存基本特征train,test
    train.to_pickle("basic_train.pkl")
    test.to_pickle("basic_test.pkl")
