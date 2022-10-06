from sklearn.preprocessing import scale
import numpy as np


def preprocess(X, method):
    """Apply selected data preprocessing step to a Numpy array"""
    if method == 'scale':
        return scale(X)
    elif method == 'log':
        return np.log2(X + 1)
    else:
        return X
