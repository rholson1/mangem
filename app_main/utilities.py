from pathvalidate import sanitize_filename
import numpy as np
import re
from collections import OrderedDict

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


def short_ephys_labels(label: str) -> str:
    """Given an excessively verbose ephysiology label, return an abbreviated one"""
    updates = dict([
              ('_', ' '),
              ('upstroke', 'up'),
              ('downstroke', 'down'),
              ('square', 'sq'),
              ('ratio', 'rat'),
              ('resistance', 'resist'),
              ('polarization', 'polar'),
              ('amplitude', 'amp'),
              ('number', 'num'),
              ('frequency', 'freq'),
              ('adaptation', 'adapt'),
              ('potential', 'pot'),
              ('..', ' '),
              ('.', ' ')
    ])
    for k, v in updates.items():
        label = re.sub(r'(?i)'+re.escape(k), v, label)  # case-insensitive matching
    return label


def short_morph_labels(label: str) -> str:
    """Shorten morphology feature labels"""
    updates = dict([
        ('axon', 'ax'),
        ('dendrite', 'dend'),
        ('number', 'num'),
        ('angle', 'ang'),
        ('"apical"', 'ap'),
        ('bifurcation', 'bifurc'),
        ('fraction', 'frac'),
        ('tortuosity', 'tort'),
        ('distance', 'dist'),
    ])
    for k, v in updates.items():
        label = re.sub(r'(?i)'+re.escape(k), v, label)  # case-insensitive matching
    return label
