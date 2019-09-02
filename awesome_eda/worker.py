import os
import sys
import pandas as pd
import numpy as np

def load_dataset(file_path, verbose=True):
    if not os.path.isfile(file_path):
        raise IOError(f'Invalid {file_path} file path.')

    df = pd.read_csv(file_path)
    df = reduce_mem_usage(df, verbose=verbose)
    if verbose:
        file_name = file_path.split('/')[-1]
        print(
            f'''
        -------------------- SHAPE ---------------------
        DF {file_name}: {df.shape}
        ------------------------------------------------
        ''')
    return df


def save_data_frame(df=None, filename=None):
    """
    Saves data frame to csv format
    Parameters
    ----------
    df: Pandas DataFrame
    filename: File path or name
    Returns
    -------
    csv file
    """
    try:
        if not filename.endswith('.csv'):
            filename += '.csv'
        df.to_csv(filename, index=False)
        print(f"Data Frame saved @:{filename}")
    except Exception as e:
        print("Data Frame couldn't be saved: ", sys.exc_info()[0])
        raise

def reduce_mem_usage(df, verbose=True):
    """ Function iterates through all the columns of a dataframe and modify the data type
        to reduce memory usage.
        Credit to: https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
        Parameters
        ----------
        df : Pandas DataFrame
        verbose: (True) by default, prints out before and after memory usage
        Returns
        -------
        df : Reduced Memory Pandas DataFrame
    """

    if verbose:
        start_mem = df.memory_usage().sum() / 1024**2
        print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    if verbose:
        end_mem = df.memory_usage().sum() / 1024**2
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(
            100 * (start_mem - end_mem) / start_mem))

    return df