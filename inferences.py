'''

File: inferences.py
Author: Hadayat Seddiqi
Date: 12-30-2013
Description: Keeps all inference functions.

'''

import numpy as np
from scipy import linalg as sln

def exact(cov, mean, hyp, x, y, pred=True):
    """
    This is the exact inference method which is solved partially
    analytically. See GPML textbook Ch. 2 for more information,
    especially the equation references in the comments and algorithm
    2.1.
    """
    n, D = x.shape
    K = cov(hyp['cov'], x, x)
    m = mean(hyp['mean'], x)
    sn2 = np.exp(2*hyp['lik'])  # noise
    lower = False
    # Basically the correlation matrix
    L = sln.cholesky(K/sn2 + np.eye(n))
    # This is just (K + I*noise)^-1 * y, see GPML Eq. (2.27)
    alpha = np.atleast_2d(sln.cho_solve((L,lower), y-m) / sn2)
    # Sqrt of noise precision vector
    sW = np.ones((n,1)) / np.sqrt(sn2)
    # Training phase
    if not pred:
        # negative log marginal likelihood, GPML Eq. (2.30)
        nlZ = np.dot( (y-m).T, alpha) / 2 + np.sum(np.log(np.diag(L))) + \
            n*np.log(2*np.pi*sn2) / 2
        return nlZ[0,0]  # Make sure we're giving a number
    # Prediction phase
    else:
        return alpha, L, sW
