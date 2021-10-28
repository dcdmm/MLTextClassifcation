import numpy as np


def rmspe(y_true, y_pred):
    """比赛评估指标"""
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))
