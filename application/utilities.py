from pathvalidate import sanitize_filename
import numpy as np


def safe_filenames(label_1, label_2):
    """ Return safe and distinct filenames from the labels provided for the uploaded datasets"""

    sf_1 = sanitize_filename(label_1)
    sf_2 = sanitize_filename(label_2)
    if sf_1 == sf_2:
        return f'{sf_1}_1', f'{sf_1}_2'
    else:
        return sf_1, sf_2


def cache_key(session_id, file_type):
    """ Generate a cache key in a consistent way from session id and file type"""
    return '_'.join([session_id, file_type])


def df_to_data(df):
    """ Extract the data from a Pandas dataframe, dropping metadata columns"""
    return np.array(df.drop(columns=['cluster', 'ttype']))

