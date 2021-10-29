import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


def calc_cluster_features(data_dir, train, test):
    """Function to calculate Cluster features"""
    # Pivot target vs (time_id, stock_id)
    train_p = pd.read_csv(data_dir)
    train_p = train_p.pivot(index='time_id', columns='stock_id', values='target')

    # Correlation of targets
    corr = train_p.corr()
    # Kmeans clustering
    ac = KMeans(n_clusters=7, random_state=1)
    ac.fit(corr)
    kmeans_clusters_target = pd.Series(ac.labels_, index=corr.index)
    print('Clusters: ')
    print([kmeans_clusters_target[kmeans_clusters_target == c].index for c in range(7)])

    train['target_clusters'] = train.stock_id.map(kmeans_clusters_target)
    test['target_clusters'] = test.stock_id.map(kmeans_clusters_target)
    print(train)
    create_feature_dict = {
        'log_return1_realized_volatility': np.mean,
        'log_return1_realized_volatility_mean_ma5': np.mean,
        'log_return1_realized_volatility_mean_ma10': np.mean,
        'log_return1_realized_volatility_mean_ma20': np.mean,
        'log_return1_realized_volatility_mean_ma500': np.mean,
        'trade_order_count_sum': np.mean,
        'depth_imbalance_sum': np.mean,
        'size_tau2': np.mean}

    # Aggregate train features
    target_cluster_agg_train = train.groupby(['time_id', 'target_clusters']).agg(create_feature_dict).reset_index()

    print(target_cluster_agg_train)

    # Aggregate features calculated above by clusters
    # Not all clusters are used. One cluster removed consists only of stock 80, while
    # The other removed cluster has significantly smaller average correlation between its
    # members than the rest of the clusters
    for cluster in [1, 3, 4, 5, 6]:
        sub_agg = target_cluster_agg_train[target_cluster_agg_train.target_clusters == cluster].set_index(
            'time_id').drop('target_clusters', axis=1)
        for col in sub_agg.columns:
            train[col + '_cluster{}'.format(cluster)] = train.time_id.map(sub_agg[col])


if __name__ == "__main__":
    data_path = "../data_init/train.csv"
    train_data = pd.read_pickle("../data_feature/train_tick.pkl")
    test_data = pd.read_pickle("../data_feature/test_tick.pkl")
    calc_cluster_features(data_path, train_data, test_data)