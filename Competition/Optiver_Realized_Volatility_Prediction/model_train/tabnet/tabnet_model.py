import numpy as np
from pytorch_tabnet.tab_model import TabNetRegressor
import os
import sys

path = os.path.abspath('../../util')
sys.path.append(path)  # 添加第三方模块路径到临时path变量中

from evaluation_index import *  # pycharm显示报错不用理会


def MyTabnet(X_train, y_train, X_test, kfolds, tabnet_params, fit_params):
    """Function to train TabNet model"""
    oof_predictions = np.zeros((X_train.shape[0], 1))
    test_predictions = np.zeros(X_test.shape[0])

    for fold in range(5):
        print(f'Training fold {fold + 1}')

        trn_ind = kfolds != fold
        val_ind = kfolds == fold

        clf = TabNetRegressor(**tabnet_params)

        fit_params["X_train"] = X_train[trn_ind].values
        fit_params["y_train"] = y_train[trn_ind].values.reshape(-1, 1)
        fit_params["eval_set"] = [(X_train[val_ind].values, y_train[val_ind].values.reshape(-1, 1))]

        clf.fit(**fit_params)

        oof_predictions[val_ind] = clf.predict(X_train[val_ind].values)
        test_predictions += clf.predict(X_test.values).flatten() / 5

    print(f'OOF score across folds: {rmspe(y_train, oof_predictions.flatten())}')

    return test_predictions, oof_predictions
