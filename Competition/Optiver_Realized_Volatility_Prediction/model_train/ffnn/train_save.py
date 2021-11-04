import numpy as np
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
import tensorflow as tf
from ffnn_model import MyFFNN
from tensorflow import keras
import pandas as pd
import os
import sys
import gc

path = os.path.abspath('../../util')
sys.path.append(path)  # 添加第三方模块路径到临时path变量中

from evaluation_index import *  # pycharm显示报错不用理会


def process_nn_data(train, test, colNames):
    """Function to process features as input to ffnn model"""
    train.replace([np.inf, -np.inf], np.nan, inplace=True)
    test.replace([np.inf, -np.inf], np.nan, inplace=True)
    train_nn = train[colNames].copy()
    test_nn = test[colNames].copy()

    # 分位数正态分布缩放
    for col in colNames:
        qt = QuantileTransformer(random_state=21, n_quantiles=2000, output_distribution='normal')
        train_nn[col] = qt.fit_transform(train_nn[[col]])
        test_nn[col] = qt.transform(test_nn[[col]])

    # 缺失值处理
    train_nn[colNames] = train_nn[colNames].fillna(train_nn[colNames].mean())
    test_nn[colNames] = test_nn[colNames].fillna(train_nn[colNames].mean())

    train_nn[['stock_id', 'time_id', 'target']] = train[['stock_id', 'time_id', 'target']]
    test_nn[['stock_id', 'time_id']] = test[['stock_id', 'time_id']]

    return train_nn, test_nn


def root_mean_squared_per_error(y_true, y_pred):
    """Function to calculate the root mean squared percentage error in TF"""
    return tf.math.sqrt(tf.math.reduce_mean(tf.math.square((y_true - y_pred) / y_true)))


def train_and_evaluate_nn(train_nn, test_nn, kfolds, colNames, hidden_units, output_dim):
    """Function to train FFNN"""
    oof_predictions_nn = np.zeros(train_nn.shape[0])
    test_predictions_nn = np.zeros(test_nn.shape[0])

    es = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=20, verbose=0,
        mode='min', restore_best_weights=True)

    plateau = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=7, verbose=0,
        mode='min')

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(train_nn[colNames].values)

    for fold in range(5):
        print('CV {}/{}'.format(fold + 1, 5))

        trn_ind = train_nn[kfolds != fold].index
        val_ind = train_nn[kfolds == fold].index

        y_train = train_nn.loc[trn_ind, 'target']
        y_test = train_nn.loc[val_ind, 'target']

        num_data = scaler.transform(train_nn.loc[trn_ind, colNames].values)
        num_data_test = scaler.transform(train_nn.loc[val_ind, colNames].values)

        cat_data = train_nn['stock_id'][trn_ind]
        cat_data_test = train_nn['stock_id'][val_ind]

        input_dim = max(cat_data) + 1
        input_shape = len(colNames)

        # 3 NN models per fold
        for ff in range(3):
            model = MyFFNN(input_shape, hidden_units, input_dim, output_dim)

            model.compile(keras.optimizers.Adam(learning_rate=0.006),
                          loss=root_mean_squared_per_error)

            model.fit([cat_data, num_data],
                      y_train,
                      batch_size=2048,
                      epochs=1000,
                      validation_data=([cat_data_test, num_data_test], y_test),
                      callbacks=[es, plateau],
                      validation_batch_size=len(y_test),
                      shuffle=True,
                      verbose=0)

            preds = model.predict([cat_data_test, num_data_test]).reshape(1, -1)[0]
            oof_predictions_nn[val_ind] += preds

            score = round(rmspe(y_true=y_test, y_pred=preds), 5)
            print('Fold {}/{}: {}'.format(fold, ff, score))

            test_predictions_nn += \
                model.predict([test_nn['stock_id'], scaler.transform(test_nn[colNames].values)]).reshape(1, -1)[0].clip(
                    0,
                    1e10)
            gc.collect()

        del num_data, num_data_test, cat_data, cat_data_test, y_train, y_test
        gc.collect()

    test_predictions_nn = test_predictions_nn / 15.0
    oof_predictions_nn = oof_predictions_nn / 3.0
    rmspe_score = rmspe(train_nn['target'], oof_predictions_nn)
    print(f'Our out of folds RMSPE is {rmspe_score}')

    return test_predictions_nn, oof_predictions_nn


if __name__ == "__main__":
    train_data = pd.read_csv("../../data_feature/train_last.csv", index_col=0, header=0)
    test_data = pd.read_csv("../../data_feature/test_last.csv", index_col=0, header=0)
    kfolds = pd.read_csv("../../data_feature/train_fold.csv", index_col=0, header=0)['kfold']
    colNames = [col for col in list(train_data.columns)
                if col not in {"stock_id", "time_id", "target", "row_id"}]  # 特征列
    train_nn, test_nn = process_nn_data(train_data, test_data, colNames)
    hidden_units = (128, 64, 32)
    test_predictions_nn, oof_predictions_nn = train_and_evaluate_nn(train_nn, test_nn, kfolds,
                                                                    colNames, hidden_units=hidden_units, output_dim=24)
    print(test_predictions_nn)

    np.save("../../result/ffnn.npy", test_predictions_nn)
