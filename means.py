'''

File: means.py
Author: Hadayat Seddiqi
Date: 12-30-2013
Description: Keeps all mean functions.

'''

import numpy as np

def zero(hypmean, x):
    """
    Returns a zero-mean function.
    """
    return np.zeros((x.shape[0],1))
