import pandas as pd
from sklearn.cluster import KMeans
import numpy as np


def cluster_kmeans(data_dir, train_data, test_data):
    """使用kemean聚类增添新的特征,这里聚类簇为7类"""
    train_p = pd.read_csv(data_dir + 'train.csv')

    # train_p存在有缺失值(28个),每个stoick的time_id不是完全一致
    train_p = train_p.pivot(index='time_id', columns='stock_id', values='target')  # time_id和stock_id的交叉表
    train_p_corr = train_p.corr()  # 交叉表train_p关于stock_id的相关系数矩阵(112 * 112)

    ids = train_p_corr.index  # 相关系数矩阵的列名(即所有的stock_id)

    kmeans = KMeans(n_clusters=7, random_state=0).fit(train_p_corr.values)  # 聚类结果

    l = []  # 聚类结果列表(len(l) = 7)
    for n in range(7):
        l.append([(x - 1) for x in ((ids + 1) * (kmeans.labels_ == n)) if x > 0])

    mat = []
    matTest = []
    n = 0
    for ind in l:
        newDf = train_data.loc[train_data['stock_id'].isin(ind)]  # train中属于同一聚类簇的数据
        newDf = newDf.groupby(['time_id']).agg(np.nanmean)
        newDf.loc[:, 'stock_id'] = str(n) + 'c1'
        mat.append(newDf)

        newDf = test_data.loc[test_data['stock_id'].isin(ind)]
        newDf = newDf.groupby(['time_id']).agg(np.nanmean)
        newDf.loc[:, 'stock_id'] = str(n) + 'c1'
        matTest.append(newDf)
        n += 1

    mat1 = pd.concat(mat).reset_index()
    mat1.drop(columns=['target'], inplace=True)  # mat1相比train没有"row_id"和"target"列

    mat2 = pd.concat(matTest).reset_index()
    mat2 = pd.concat([mat2, mat1.loc[mat1.time_id == 5]])

    mat1 = mat1.pivot(index='time_id', columns='stock_id')
    mat1.columns = ["_".join(x) for x in mat1.columns.ravel()]
    mat1.reset_index(inplace=True)

    mat2 = mat2.pivot(index='time_id', columns='stock_id')
    mat2.columns = ["_".join(x) for x in mat2.columns.ravel()]
    mat2.reset_index(inplace=True)

    return mat1, mat2
