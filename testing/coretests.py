"""
"""

import numpy as np
import scipy as sp
from scipy import spatial
import addpath
import gaussian_process as gp

def test_sqdist():
    a = np.matrix(np.arange(0,1,0.1))
    b = np.matrix(np.arange(0.5,1.5,0.1))
    sd_ab = gp.core.sq_dist(a,b)
    assert sd_ab.all() == sd_ab.T.all()
    assert sd_ab.all() == sp.spatial.distance.cdist(a.T,b.T, 'sqeuclidean').all()
    sd_aa = gp.core.sq_dist(a,a)
    assert sd_aa.all() == sd_aa.T.all()
    assert sd_aa.all() == sp.spatial.distance.cdist(a.T,a.T, 'sqeuclidean').all()

def test_initboundedparams():
    bounds = [0., [0,1], 3.]
    sn = 0.5
    hyps = gp.core.initBoundedParams(bounds, sn)
    assert hyps['cov'][0] == 0.0
    assert hyps['cov'][2] == 3.0
    assert len(hyps['cov']) == 3
    assert hyps['lik'][0] == np.log(sn)
    assert len(hyps['mean']) == 0

def test_initsmparams():
    q = 3
    f = lambda x: np.sin(2*np.pi*10.0*x)
    x = np.atleast_2d(np.arange(0,1,0.1)).T
    y = np.atleast_2d(f(x).ravel()).T
    sn = 0.5
    hyps = gp.core.initSMParams(q, x, y, sn)
    assert hyps['cov'].size == 9
    assert hyps['cov'][0] == hyps['cov'][1] == hyps['cov'][2]
    assert len(hyps['mean']) == 0
    assert hyps['lik'][0] == np.log(sn)

def test_initsmparamsfourier():
    q = 3
    f = lambda x: np.sin(2*np.pi*10.*x)
    x = np.atleast_2d(np.arange(0,1,1/150.)).T
    y = np.atleast_2d(f(x).ravel()).T
    sn = 0.5
    samplingFreq = 1.
    nPeaks = 1
    relMaxOrder = 2
    hyps = gp.core.initSMParamsFourier(q, x, y, sn, samplingFreq, 
                                       nPeaks, relMaxOrder)
    assert np.exp(hyps['cov'])[3:6].all() == np.array([0.0666666666]*3).all()
    assert hyps['cov'].size == 9
    assert hyps['cov'][0] == hyps['cov'][1] == hyps['cov'][2]
    assert len(hyps['mean']) == 0
    assert hyps['lik'][0] == np.log(sn)
