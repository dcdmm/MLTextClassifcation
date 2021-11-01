import numpy as np
import lightgbm as lgb


def MyLightGBM(X_train_data, y_train_data, X_test_data, kfolds,
               params, early_stopping_rounds=None, verbose_eval=True, feval=None, fweight=None,
               categorical_feature='auto'):
    """
    原生lightgbm模型封装(回归问题请微调)
    Parameters
    ---------
    X_train_data : numpy array (n_sample, n_feature)
        训练数据集
    y_train_data : numpy array (n_sample, )
        训练数据集标签
    X_test_data : numpy array (n_sample, n_feature)
        测试数据集
    kfold :
        k折交叉验证对象
    params : dict
        lightgbm模型train方法params参数
    early_stopping_rounds:
        lightgbm模型train方法early_stopping_rounds参数
    verbose_eval :
        lightgbm模型train方法verbose_eval参数
    feval :
        lightgbm模型train方法feval参数
    fweight : 函数(返回训练数据集的权重)
        返回值为lightgbm模型Dataset方法weight参数
    categorical_feature : list(分类特征的索引) or 'auto'
        lightgbm模型Dataset方法categorical_feature参数

    return
    ---------
    train_predictions : array
        训练数据集预测结果
    test_predictions : array
        测试数据集预测结果
    model_list : list
        训练模型组成的列表
    """
    num_class = params.get('num_class')  # 多分类问题的判别
    train_predictions = np.zeros(
        X_train_data.shape[0] if num_class is None else [X_train_data.shape[0], num_class])  # 训练数据集预测结果
    test_predictions = np.zeros(
        X_test_data.shape[0] if num_class is None else [X_test_data.shape[0], num_class])  # 测试数据集预测结果
    model_list = list()  # k折交叉验证模型结果

    for fold in range(5):
        print(f'Training fold {fold + 1}')

        trn_ind = X_train_data[kfolds != fold].index
        val_ind = X_train_data[kfolds == fold].index

        print(trn_ind.shape)
        print(val_ind.shape)

        x_train, x_val = X_train_data.iloc[trn_ind], X_train_data.iloc[val_ind]
        y_train, y_val = y_train_data.iloc[trn_ind], y_train_data.iloc[val_ind]

        train_weights = None if fweight is None else fweight(y_train)
        val_weights = None if fweight is None else fweight(y_val)

        train_dataset = lgb.Dataset(x_train, y_train, weight=train_weights, categorical_feature=categorical_feature)
        val_dataset = lgb.Dataset(x_val, y_val, weight=val_weights, categorical_feature=categorical_feature)

        # 警告的避免
        if 'num_boost_round' in params:
            num_boost_round = params.pop('num_boost_round')
        else:
            num_boost_round = 1000

        model = lgb.train(params=params,
                          train_set=train_dataset,
                          valid_sets=[train_dataset, val_dataset],
                          num_boost_round=num_boost_round,
                          early_stopping_rounds=early_stopping_rounds,
                          verbose_eval=verbose_eval,
                          feval=feval)
        model_list.append(model)
        train_predictions[val_ind] = model.predict(x_val)
        test_predictions += model.predict(X_test_data) / 5

    return train_predictions, test_predictions, model_list
