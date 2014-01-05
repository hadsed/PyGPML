'''

File: kernels.py
Author: Hadayat Seddiqi
Date: 12-27-2013
Description: Keeps all kernel functions for the GP.

'''

import numpy as np
import core

def radial_basis(hyp, x=None, z=None, diag=False):
    """
    Radial-basis function (also known as squared-exponential and 
    Gaussian kernels) takes the following form,

    k(t) = \sigma^2_f * exp(-t^2/(2L^2))

    where \sigma and L are the adjustable hyperparameters giving
    hyp = [ \sigma, L ].
    """
    sf2 = np.exp(2*hyp[0])
    ell2 = np.exp(2*hyp[1])
    if diag:
        K = np.zeros((x.shape[0],1))
    else:
        if x is z:
            K = core.sq_dist(x.T/ell2)
        else:
            K = core.sq_dist(x.T/ell2, z.T/ell2)
    K = sf2*np.exp(-K/2)
    return K


def rational_quadratic(hyp, x=None, z=None, diag=False):
    """
    Rational-quadratic kernel has the following form,

    k(t) = hscale^2 * (1 + t^2/(alpha*lamb^2))^{-alpha}

    where hscale, alpha, and lamb are hyperparameters that give

    hyp = [ hscale, alpha, lamb ].
    """
    hscale = np.exp(hyp[0])
    alpha = np.exp(hyp[1])
    lamb = np.exp(hyp[2])
    if diag:
        K = np.zeros((x.shape[0],1))
    else:
        if x is z:
            K = core.sq_dist(x.T/lamb**2)
        else:
            K = core.sq_dist(x.T/lamb**2, z.T/lamb**2)
    K = hscale**2 * np.power(1 + K/alpha, -alpha)
    return K


def periodic(hyp, x=None, z=None, diag=False):
    """
    The periodic kernel has a form

    k(x,x') = sigma^2 * exp(-2/ell^2 * sin^2(pi*|x-x'|/per))

    where sigma, ell, and per are hyperparameters giving

    hyp = [ sigma, ell, per ].
    """
    sigma = np.exp(hyp[0])
    ell = np.exp(hyp[1])
    per = np.exp(hyp[2])
    if diag:
        K = np.zeros((x.shape[0],1))
    else:
        if x is z:
            K = core.sq_dist(x.T)
        else:
            K = core.sq_dist(x.T, z.T)
    K = np.sqrt(K)  # need the absolute distance, not squared
    K = sigma**2 * np.exp(-2/ell**2 * np.power(np.sin(np.pi*K/per), 2))
    return K


def spectral_mixture(hyp, x=None, z=None, diag=False):
    """
    Spectral Mixture kernel takes the following form [1],

    k(t) = \sum^Q_{q=0} w_q \prod^P_{p=0} exp(-2pi^2*v^2_{p,q}*t_p^2)
           * cos(2pi*\mu_{p,q}*t_p)

    It's corresponding hyperparameters are constructed according to

    [ [w_0, w_1, ..., w_q],
      [mu_0, mu_1, ..., mu_q],
      [v_0, v_1, ..., v_q] ]

    and then flattened to give hyp = [ w_0, ..., w_q, mu_0, ..., v_q ].
    So then P is the dimensionality of the data and Q is the number of
    Gaussians in the Gaussian mixture model (roughly speaking, Q is the
    number of peaks we attempt to model).

    [1] Wilson, A. G., & Adams, R. P. (2013). Gaussian process covariance
        kernels for pattern discovery and extrapolation. arXiv preprint 
        arXiv:1302.4245.
    
    """
    n, D = x.shape
    hyp = np.array(hyp).flatten()
    Q = hyp.size/(1+2*D)
    w = np.exp(hyp[0:Q])
    m = np.exp(hyp[Q+np.arange(0,Q*D)]).reshape(D,Q)
    v = np.exp(2*hyp[Q+Q*D+np.arange(0,Q*D)]).reshape(D,Q)
    if diag:
        d2 = np.zeros((n,1,D))
    else:
        if x is z:
            d2 = np.zeros((n,n,D))
            for j in np.arange(0,D):
                d2[:,:,j] = core.sq_dist(x[:,j].T)
        else:
            d2 = np.zeros((n,z.shape[0],D))
            for j in np.arange(0,D):
                d2[:,:,j] = core.sq_dist(x[:,j].T, z[:,j].T)

    # Define kernel functions
    k = lambda d2v, dm: np.multiply(np.exp(-2*np.pi**2 * d2v),
                                    np.cos(2*np.pi * dm))
    # Calculate correlation matrix
    K = 0
    d = np.sqrt(d2)
    for q in range(0,Q):
        C = w[q]**2
        for j in range(0,D):
            C = C*k(d2[:,:,j]*v[j,q], d[:,:,j]*m[j,q])
        K = K + C

    return K
