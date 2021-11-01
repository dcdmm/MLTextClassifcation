import os
import sys
import pandas as pd
import numpy as np
from model import *

path = os.path.abspath('../../util')
sys.path.append(path)  # 添加第三方模块路径到临时path变量中

from evaluation_index import *  # pycharm显示报错不用理会


def feval_rmspe(y_pred, lgb_train):
    """自定义lgb评估函数"""
    y_true = lgb_train.get_label()
    return 'RMSPE', rmspe(y_true, y_pred), False


def weight_func(y):
    """均方根百分比误差权重"""
    return 1 / np.square(y)


params1 = {
    'learning_rate': 0.05,
    'lambda_l1': 1,
    'lambda_l2': 1,
    'num_leaves': 500,
    'min_sum_hessian_in_leaf': 30,
    'feature_fraction': 0.6,
    'feature_fraction_bynode': 0.8,
    'bagging_fraction': 0.97,
    'bagging_freq': 46,
    'min_data_in_leaf': 600,
    'num_boost_round': 1300,
    'categorical_column': [0],
    'objective': 'rmse',
    'boosting': 'gbdt',
    'verbosity': -1,
    'n_jobs': -1,
    'seed': 43,
    'feature_fraction_seed': 43,
    'bagging_seed': 43,
    'drop_seed': 43,
    'data_random_seed': 43
}

params2 = params1.copy()
params2['seed'] = 23
params2['feature_fraction_seed'] = 23
params2['bagging_seed'] = 23,
params2['drop_seed'] = 23
params2['data_random_seed'] = 23

early_stopping_rounds = 70
verbose_eval = 70

if __name__ == "__main__":
    train_data = pd.read_pickle("../../data_feature/train_last.pkl")
    test_data = pd.read_pickle("../../data_feature/test_last.pkl")
    kfolds = pd.read_pickle("../../data_feature/train_fold.pkl")['kfold']

    features = [col for col in train_data.columns if col not in {"time_id", "target", "row_id"}]
    X_train_data = train_data[features]
    y_train_data = train_data['target']
    X_test_data = test_data[features]
    lgb_train_pre1, lgb_test_pre1, lgb_model_list1 = MyLightGBM(X_train_data=X_train_data, y_train_data=y_train_data,
                                                                X_test_data=X_test_data, kfolds=kfolds,
                                                                params=params1,
                                                                early_stopping_rounds=early_stopping_rounds,
                                                                verbose_eval=verbose_eval,
                                                                feval=feval_rmspe, fweight=weight_func)

    # 保存模型训练结果
    np.save("../../result/lgb1.npy", lgb_test_pre1)

    lgb_train_pre2, lgb_test_pre2, lgb_model_list2 = MyLightGBM(X_train_data=X_train_data, y_train_data=y_train_data,
                                                                X_test_data=X_test_data, kfolds=kfolds,
                                                                params=params2,
                                                                early_stopping_rounds=early_stopping_rounds,
                                                                verbose_eval=verbose_eval,
                                                                feval=feval_rmspe, fweight=weight_func)

    np.save("../../result/lgb2.npy", lgb_test_pre2)
