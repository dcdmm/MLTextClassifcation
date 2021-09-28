import pandas as pd
from sklearn.preprocessing import QuantileTransformer


def lgb_data(train, test, mat1, mat2, clus_feature):
    """lightgbm/catboost/tabnet模型的训练和测试数据集"""
    train_ck = pd.merge(train, mat1[clus_feature], how='left', on='time_id')
    test_ck = pd.merge(test, mat2[clus_feature], how='left', on='time_id')
    return train_ck, test_ck


def nn_data(train, test, mat1, mat2, clus_feature):
    """ffnn模型的训练和测试数据集"""
    colNames = [col for col in list(train.columns)
                if col not in {"stock_id", "time_id", "target", "row_id"}]

    train_nn = train[colNames].copy()  # 基本特征进一步加工后的特征数据的复制
    test_nn = test[colNames].copy()
    for col in colNames:
        qt = QuantileTransformer(random_state=21, n_quantiles=2000, output_distribution='normal')  # 执行正态分布数据缩放
        train_nn[col] = qt.fit_transform(train_nn[[col]])
        test_nn[col] = qt.transform(test_nn[[col]])

    train_nn[['stock_id', 'time_id', 'target']] = train[['stock_id', 'time_id', 'target']]
    test_nn[['stock_id', 'time_id']] = test[['stock_id', 'time_id']]

    train_nn_ck = pd.merge(train_nn, mat1[clus_feature], how='left', on='time_id')
    test_nn_ck = pd.merge(test_nn, mat2[clus_feature], how='left', on='time_id')

    return train_nn_ck, test_nn_ck
