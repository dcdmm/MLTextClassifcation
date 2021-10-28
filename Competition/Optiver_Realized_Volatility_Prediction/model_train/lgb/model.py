import sys
import os
import numpy as np
from sklearn.model_selection import KFold
import lightgbm as lgb

path = os.path.abspath('../../util')
sys.path.append(path)  # 添加第三方模块路径到临时path变量中

from evaluation_index import *  # pycharm显示报错不用理会


def feval_rmspe(y_pred, lgb_train):
    """自定义lgb评估函数"""
    y_true = lgb_train.get_label()
    return 'RMSPE', rmspe(y_true, y_pred), False


def train_and_evaluate_lgb(X_train_lgb, y_train_lgb, X_test_lgb, params, kf_seed, num_boost_round):
    """lgb模型训练评估和预测"""
    # Create out of folds array
    oof_predictions = np.zeros(X_train_lgb.shape[0])
    # Create test array to store predictions
    test_predictions = np.zeros(X_test_lgb.shape[0])

    # Create a KFold object
    kfold = KFold(n_splits=5, random_state=kf_seed, shuffle=True)  # 设置kflod的随机数种子,实现seed层次的模型融合
    # Iterate through each fold
    for fold, (trn_ind, val_ind) in enumerate(kfold.split(X_train_lgb)):
        print(f'Training fold {fold + 1}')
        x_train, x_val = X_train_lgb.iloc[trn_ind], X_train_lgb.iloc[val_ind]
        y_train, y_val = y_train_lgb.iloc[trn_ind], y_train_lgb.iloc[val_ind]
        # Root mean squared percentage error weights
        train_weights = 1 / np.square(y_train)
        val_weights = 1 / np.square(y_val)
        train_dataset = lgb.Dataset(x_train, y_train, weight=train_weights)
        val_dataset = lgb.Dataset(x_val, y_val, weight=val_weights)
        model = lgb.train(params=params,
                          num_boost_round=num_boost_round,
                          train_set=train_dataset,
                          valid_sets=[train_dataset, val_dataset],
                          verbose_eval=250,
                          early_stopping_rounds=100,
                          feval=feval_rmspe)
        # Add predictions to the out of folds array
        oof_predictions[val_ind] = model.predict(x_val)
        # Predict the test set
        test_predictions += model.predict(X_test_lgb) / 5  # 5折交叉验证
    rmspe_score = rmspe(y_train_lgb, oof_predictions)
    print(f'Our out of folds RMSPE is {rmspe_score}')
    # Return test predictions
    return oof_predictions, test_predictions
