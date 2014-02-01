'''

File: core.py
Author: Hadayat Seddiqi
Date: 12-27-2013
Description: Any abstract classes and helper functions.

'''

import numpy as np
from scipy import signal as spsig
from scipy import spatial as spat
import collections

def initSMParamsFourier(Q, x, y, sn, samplingFreq, nPeaks, relMaxOrder=2):
    """
    Initialize hyperparameters for the spectral-mixture kernel. Weights are
    all set to be uniformly distributed, means are given as the peaks in the
    frequency spectrum, and variances are given by a random sample from a 
    uniform distribution with a max equal to the max distance.
    """
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    n, D = x.shape
    w = np.zeros(Q)
    m = np.zeros((D,Q))
    s = np.zeros((D,Q))
    w[:] = np.std(y) / Q
    hypinit = {
        'cov': np.zeros(Q+2*D*Q),
        'lik': np.atleast_1d(np.log(sn)),
        'mean': np.array([])
        }

    # Assign hyperparam weights
    hypinit['cov'][0:Q] = np.log(w)

    # Assign hyperparam frequencies (mu's)
    signal = np.array(y.ravel()).ravel()  # Make into 1D array
    n = x.shape[0]
    k = np.arange(n)
    ts = n/samplingFreq
    frqx = k/float(ts)
    frqx = frqx[range(n/2)]
    frqy = np.fft.fft(signal)/n
    frqy = abs(frqy[range(n/2)])
    # Find the peaks in the frequency spectrum
    peakIdx = np.array([])
    while not peakIdx.any() and relMaxOrder > 0:
        peakIdx = spsig.argrelmax(np.log(frqy**2), order=relMaxOrder)[0]
        relMaxOrder -= 1
    if not peakIdx.any():
        raise ValueError("Data doesn't have any detectable peaks in Fourier space."
                         " Switching to a different kernel besides the spectral "
                         "mixture is recommended.")
    # Find specified number (nPeaks) largest peaks
    sortedIdx = frqy[peakIdx].argsort()[::-1][:nPeaks]
    sortedPeakIdx = peakIdx[sortedIdx]
    hypinit['cov'][Q + np.arange(0,Q*D)] = np.log(frqx[sortedPeakIdx])

    # Assign hyperparam length scales (sigma's)
    for i in range(0,D):
        xslice = np.atleast_2d(x[:,i]).T
        d2 = spat.distance.cdist(xslice, xslice, 'sqeuclidean')
        if n > 1:
            d2[d2 == 0] = d2[0,1]
        else:
            d2[d2 == 0] = 1
        maxshift = np.max(np.max(np.sqrt(d2)))
        s[i,:] = 1./np.abs(maxshift*np.random.ranf((1,Q)))
    hypinit['cov'][Q + Q*D + np.arange(0,Q*D)] = np.log(s[:]).T
    
    return hypinit


def initSMParams(Q, x, y, sn):
    """
    Initialize hyperparameters for the spectral-mixture kernel. Weights are
    all set to be uniformly distributed, means are given by a random sample
    from a uniform distribution scaled by the Nyquist frequency, and variances 
    are given by a random sample from a uniform distribution scaled by the max 
    distance.
    """
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    n, D = x.shape
    w = np.zeros(Q)
    m = np.zeros((D,Q))
    s = np.zeros((D,Q))
    w[:] = np.std(y) / Q
    hypinit = {
        'cov': np.zeros(Q+2*D*Q),
        'lik': np.atleast_1d(np.log(sn)),
        'mean': np.array([])
        }

    for i in range(0,D):
        # Calculate distances
        xslice = np.atleast_2d(x[:,i]).T
        d2 = spat.distance.cdist(xslice, xslice, 'sqeuclidean')
        if n > 1:
            d2[d2 == 0] = d2[0,1]
        else:
            d2[d2 == 0] = 1
        minshift = np.min(np.min(np.sqrt(d2)))
        nyquist = 0.5/minshift
        m[i,:] = nyquist*np.random.ranf((1,Q))
        maxshift = np.max(np.max(np.sqrt(d2)))
        s[i,:] = 1./np.abs(maxshift*np.random.ranf((1,Q)))

    hypinit['cov'][0:Q] = np.log(w)
    hypinit['cov'][Q + np.arange(0,Q*D)] = np.log(m[:]).T
    hypinit['cov'][Q + Q*D + np.arange(0,Q*D)] = np.log(s[:]).T
    return hypinit


def initBoundedParams(bounds, sn=[]):
    """
    Takes in @bounds and returns hyperparameters as an array of the same 
    length. The elements of @bounds are pairs [lower, upper], and the 
    corresponding hyperparameter is sampled from a uniform distribution
    in the interval [lower, upper]. If instead of a pair we have a number
    in @bounds, then we assign that value as the appropriate hyperparameter.
    """
    hypinit = {
        'cov': np.zeros(len(bounds)),
        'lik': np.atleast_1d(np.log(sn)),
        'mean': np.array([])
        }
    # Sample from a uniform distribution
    for idx, pair in enumerate(bounds):
        # Randomize only if bounds are specified
        if isinstance(pair, collections.Iterable):
            hypinit['cov'][idx] = np.random.uniform(pair[0], pair[1])
        # If no bounds, then keep default value always
        else:
            hypinit['cov'][idx] = pair
    return hypinit
