'''

File: core.py
Author: Hadayat Seddiqi
Date: 12-27-2013
Description: Any abstract classes and helper functions.

'''

import numpy as np
from scipy import signal as spsig

def sq_dist(A, B=None):
    """
    Calculate the squared-distance sq_dist(x,x') = (x-x')^2. If only
    one argument is supplied, then it's just sq_dist(x,x).
    """
    D, n = A.shape

    if B is None:
        mu = np.mean(A, axis=1)
        a = A - np.tile(mu, (1,A.shape[1]))
        b = a
        m = n
    else:
        d, m = B.shape
        if d is not D:
            raise ValueError("sq_dist(): Both matrices must have same"
                             "number of columns.")
        mu = (m/(n+m))*np.mean(B, axis=1) + (n/(n+m))*np.mean(A, axis=1)
        a = A - np.tile(mu, (1,n))
        b = B - np.tile(mu, (1,m))

    C = np.tile(np.sum(np.multiply(a,a), axis=0).T, (1,m)) + \
        np.tile(np.sum(np.multiply(b,b), axis=0), (n,1)) - 2*a.T*b
    # Make sure we're staying positive :)
    C = C.clip(min=0)
    return C

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
    if sn is None:
        hypinit = np.zeros(Q+2*D*Q)
    else:
        hypinit = np.zeros(Q+2*D*Q+1)
        hypinit[-1] = np.log(sn)

    # Assign hyperparam weights
    hypinit[0:Q] = np.log(w)

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
    hypinit[Q + np.arange(0,Q*D)] = np.log(frqx[sortedPeakIdx])

    # Assign hyperparam length scales (sigma's)
    for i in range(0,D):
        d2 = np.sqrt(sq_dist(x[:,i].T))
        if n > 1:
            d2[d2 == 0] = d2[0,1]
        else:
            d2[d2 == 0] = 1
        maxshift = np.max(np.max(d2))
        s[i,:] = 1./np.abs(maxshift*np.random.ranf((1,Q)))
    hypinit[Q + Q*D + np.arange(0,Q*D)] = np.log(s[:]).T
    
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
    if sn is None:
        hypinit = np.zeros(Q+2*D*Q)
    else:
        hypinit = np.zeros(Q+2*D*Q+1)
        hypinit[-1] = np.log(sn)

    for i in range(0,D):
        # Calculate distances
        d2 = np.sqrt(sq_dist(x[:,i].T))
        if n > 1:
            d2[d2 == 0] = d2[0,1]
        else:
            d2[d2 == 0] = 1
        minshift = np.min(np.min(d2))
        nyquist = 0.5/minshift
        m[i,:] = nyquist*np.random.ranf((1,Q))
        maxshift = np.max(np.max(d2))
        s[i,:] = 1./np.abs(maxshift*np.random.ranf((1,Q)))

    hypinit[0:Q] = np.log(w)
    hypinit[Q + np.arange(0,Q*D)] = np.log(m[:]).T
    hypinit[Q + Q*D + np.arange(0,Q*D)] = np.log(s[:]).T
    return hypinit


def initBoundedParams(bounds, sn=None):
    """
    Takes in @bounds and returns hyperparameters as an array of the same 
    length. The elements of @bounds are pairs [lower, upper], and the 
    corresponding hyperparameter is sampled from a uniform distribution
    in the interval [lower, upper]. If instead of a pair we have a number
    in @bounds, then we assign that value as the appropriate hyperparameter.
    """
    if sn is None:
        hypinit = np.empty(len(bounds))
    else:
        hypinit = np.empty(len(bounds)+1)
        hypinit[-1] = sn
    # Sample from a uniform distribution
    for idx, pair in enumerate(bounds):
        # Randomize only if bounds are specified
        if isinstance(pair, collections.Iterable):
            hypinit[idx] = np.random.uniform(pair[0], pair[1])
        # If no bounds, then keep default value always
        else:
            hypinit[idx] = pair
    return hypinit
