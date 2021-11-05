import numpy as np
from util.read_train_test import *


data_dir = "data_init/"
train, test = read_train_test(data_dir)

ffnn_result = np.load('result/ffnn.npy')
lbg1_result = np.load('result/lgb1.npy')
lgb2_result = np.load('result/lgb2.npy')
tabnet_result = np.load('result/tabnet.npy')
lgb_result = 0.5 * lbg1_result + 0.5* lgb2_result

# 最终预测结果
print((ffnn_result + lgb_result + tabnet_result) / 3)
