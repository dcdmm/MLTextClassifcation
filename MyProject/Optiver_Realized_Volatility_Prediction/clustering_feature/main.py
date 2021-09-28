from feature_structure import *

data_dir = "../init_data/"  # 数据路径

if __name__ == "__main__":
    # 读取基本特征构造的结果
    train = pd.read_pickle("../basic_feature/basic_train.pkl")
    test = pd.read_pickle("../basic_feature/basic_test.pkl")

    mat1, mat2 = cluster_kmeans(data_dir, train, test)
    print(mat1.shape)
    print(mat2.shape)

    # 保存聚类特征mat1,mat2
    mat1.to_pickle("mat1.pkl")
    mat2.to_pickle("mat2.pkl")
