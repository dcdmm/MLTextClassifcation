from pytorch_tabnet.metrics import Metric
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tabnet_model import *
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


class RMSPE(Metric):
    """自定义评估指标"""

    def __init__(self):
        self._name = "rmspe"
        self._maximize = False

    def __call__(self, y_true, y_score):
        return np.sqrt(np.mean(np.square((y_true - y_score) / y_true)))


def RMSPELoss(y_pred, y_true):
    """自定义损失函数"""
    return torch.sqrt(torch.mean(((y_true - y_pred) / y_true) ** 2)).clone()


def process_tabnet_data(train, test):
    """Function to process features as input to TabNet model"""
    train.replace([np.inf, -np.inf], np.nan, inplace=True)
    test.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 缺失值处理
    for col in train.columns.to_list()[4:]:
        train[col] = train[col].fillna(train[col].mean())
        train = train.fillna(0)
    for col in test.columns.to_list()[3:]:
        test[col] = test[col].fillna(test[col].mean())
        test = test.fillna(0)

    X_train = train.drop(['row_id', 'target', 'time_id'], axis=1)  # 训练数据集特征
    y_train = train['target']  # 训练数据集标签

    X_test = test.drop(['time_id', 'row_id'], axis=1)

    categorical_columns = []
    categorical_dims = {}

    # 数据预处理:标签编码与数据缩放
    for col in X_train.columns:
        if col == 'stock_id':
            l_enc = LabelEncoder()
            X_train[col] = l_enc.fit_transform(X_train[col].values)
            X_test[col] = l_enc.transform(X_test[col].values)
            categorical_columns.append(col)
            categorical_dims[col] = len(l_enc.classes_)
        else:
            scaler = StandardScaler()
            X_train[col] = scaler.fit_transform(X_train[col].values.reshape(-1, 1))
            X_test[col] = scaler.transform(X_test[col].values.reshape(-1, 1))

    cat_idxs = [i for i, f in enumerate(X_train.columns.tolist()) if f in categorical_columns]

    cat_dims = [categorical_dims[f] for i, f in enumerate(X_train.columns.tolist()) if f in categorical_columns]

    return X_train, y_train, X_test, cat_idxs, cat_dims


if __name__ == "__main__":
    train_data = pd.read_csv("../../data_feature/train_last.csv", header=0, index_col=0)
    test_data = pd.read_csv("../../data_feature/test_last.csv", header=0, index_col=0)
    kfolds = pd.read_csv("../../data_feature/train_fold.csv", header=0, index_col=0)['kfold']
    X_train, y_train, X_test, cat_idxs, cat_dims = process_tabnet_data(train_data, test_data)
    tabnet_params = dict(
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=8,
        n_d=16,
        n_a=16,
        n_steps=2,
        gamma=1.3,
        n_independent=2,
        n_shared=2,
        lambda_sparse=0,
        optimizer_fn=Adam,
        optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
        mask_type="entmax",
        scheduler_params=dict(T_0=200, T_mult=1, eta_min=1e-4, last_epoch=-1, verbose=False),
        scheduler_fn=CosineAnnealingWarmRestarts,
        seed=23,
        verbose=10)
    tabnet_fit_params = dict(max_epochs=200,
                             patience=50,
                             batch_size=1024 * 10,
                             virtual_batch_size=128 * 10,
                             num_workers=8,
                             drop_last=False,
                             eval_metric=[RMSPE],
                             loss_fn=RMSPELoss)
    test_predictions, oof_predictions = MyTabnet(X_train, y_train, X_test, kfolds, tabnet_params, tabnet_fit_params)
    print(test_predictions)
    np.save("../../result/tabnet.npy", test_predictions)
