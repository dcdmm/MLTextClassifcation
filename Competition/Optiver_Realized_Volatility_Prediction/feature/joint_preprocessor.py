from joblib import Parallel, delayed
import pandas as pd


def joint_preprocessor(data_dir, list_stock_ids, prep_func, is_train=True):
    """Can be used for joint features using trade and book data"""

    # Parrallel for loop
    def for_joblib(stock_id):
        # Train
        if is_train:
            file_path_book = data_dir + "/book_train.parquet/stock_id=" + str(stock_id)
        # Test
        else:
            file_path_book = data_dir + "/book_test.parquet/stock_id=" + str(stock_id)

        # Preprocess book and trade data and merge them
        df_tmp = prep_func(file_path_book)

        # Return the merge dataframe
        return df_tmp

    # Use parallel api to call parallel for loop
    df = Parallel(n_jobs=-1, verbose=1)(delayed(for_joblib)(stock_id) for stock_id in list_stock_ids)
    # Concatenate all the dataframes that return from Parallel
    df = pd.concat(df, ignore_index=True)
    return df
