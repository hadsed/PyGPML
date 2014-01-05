'''

File: likelihoods.py
Author: Hadayat Seddiqi
Date: 12-30-2013
Description: Keeps all likelihood functions.

'''

import numpy as np
from scipy import linalg as sln

def gaussian(hyp=None, y=None, mu=None, s2=None):
    """
    Compute a Gaussian predictive distribution on target points,
    return the negative log probability of the target along with
    its means and variances. It can be expressed as

    p(t|D,xs) = exp(-(t-f(xs))^2/(2*sn^2)) / sqrt(2*pi*sn^2)

    where t is the target data points, f(xs) is the mean and sn 
    is the standard deviation. See GPML Eq. (2.34).
    """
    sn2 = np.exp(2*hyp['lik'])
    if not y:
        y = np.zeros(mu.shape)
    # Calculate the [negative] log probability of the target point's
    # [Gaussian] distribution (also where the mean and variance come from)
    if sln.norm(s2) <= 0:
        lp = -np.power(y-mu,2)/(2*sn2) - np.log(2*np.pi*sn2)/2
        s2 = 0
    else:
        lp = -np.power(y-mu, 2)/((s2+sn2))/2 - np.log(2*np.pi*(s2+sn2))/2
    return lp, mu, s2+sn2
