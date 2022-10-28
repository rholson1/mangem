from sklearn.preprocessing import scale
import numpy as np


def preprocess(X, method):
    """Apply selected data preprocessing step to a Numpy array
     or Pandas dataframe"""
    if method == 'scale':
        if hasattr(X, 'iloc'):
            return scale_df(X)
        return scale(X)
    elif method == 'log':
        return np.log2(X + 1)
    else:
        return X


def scale_df(X):
    X.iloc[:, 0:-1] = X.iloc[:, 0:-1].apply(lambda x: (x - x.mean()) / x.std(), axis=0)
    return X
