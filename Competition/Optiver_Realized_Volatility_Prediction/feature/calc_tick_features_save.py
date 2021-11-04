import pandas as pd
import numpy as np


def calc_tick_features(train, test, d_df):
    """Function to calculate Tick features from distance matrix obtained from reconstructed stock prices"""
    # Features to aggregate
    feats = ['log_return1_realized_volatility']

    # Concatenate train and test time_ids
    tick_df = pd.concat([train[['stock_id', 'time_id'] + feats], test[['stock_id', 'time_id'] + feats]])

    def merge_feature(tr, te, ma_list, mas):
        """新的特征合并到tr中"""
        for ma_idx, ma in zip(ma_list, mas):
            ma = ma.copy()
            for i, idx in enumerate(d_df.index):
                if ma[i].columns.__class__.__name__ == "Index":
                    ma[i].columns = ['_'.join(col) + '_ma{}'.format(ma_idx) for col in ma[i].columns]
                else:
                    ma[i].columns = [col[0] + '_' + str(col[1]) + '_ma{}'.format(ma_idx) for col in ma[i].columns]
                ma[i]['time_id'] = idx
                ma[i] = ma[i].reset_index()

            ma = pd.concat(ma).rename(columns={f: f + '_ma{}'.format(ma_idx) for f in feats})
            tr = tr.merge(ma, on=['stock_id', 'time_id'], how='left')
            te = te.merge(ma, on=['stock_id', 'time_id'], how='left')
        return tr, te

    ma_list_0 = [5]
    mas_0 = [[pd.DataFrame(
        tick_df[tick_df.time_id.isin(list(d_df.iloc[:, range(ma)].loc[idx].values))].groupby('stock_id')[feats].quantile(
            [0.25, 0.5, 0.75]).unstack()) for idx in d_df.index] for ma in ma_list_0]

    train, test = merge_feature(train, test, ma_list_0, mas_0)
    del ma_list_0, mas_0

    ma_list_1 = [5, 10, 20, 50]
    mas_1 = [[pd.DataFrame(
        tick_df[tick_df.time_id.isin(list(d_df.iloc[:, range(ma)].loc[idx].values))].groupby('stock_id')[feats].agg(
            [np.mean, np.max, np.min])) for idx in d_df.index] for ma in ma_list_1]

    train, test = merge_feature(train, test, ma_list_1, mas_1)
    del ma_list_1, mas_1

    ma_list_2 = [100, 200, 500]
    mas_2 = [[pd.DataFrame(
        tick_df[tick_df.time_id.isin(list(d_df.iloc[:, range(ma)].loc[idx].values))].groupby('stock_id')[feats].agg([np.mean]))
        for idx in d_df.index] for ma in ma_list_2]

    train, test = merge_feature(train, test, ma_list_2, mas_2)
    del ma_list_2, mas_2

    print('Tick features done')
    return train, test


if __name__ == "__main__":
    data_path = "../data_init/"
    d_df_data = pd.read_csv("../data_feature/d_df.csv", index_col=0, header=1)
    train_data = pd.read_csv("../data_feature/train_raw.csv", index_col=0, header=0)
    test_data = pd.read_csv("../data_feature/test_raw.csv", index_col=0, header=0)
    train_data, test_data = calc_tick_features(train_data, test_data, d_df_data)
    train_data.to_csv("../data_feature/train_tick.csv")
    test_data.to_csv("../data_feature/test_tick.csv")