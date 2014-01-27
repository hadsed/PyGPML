"""

File: kerneltests.py
Author: Hadayat Seddiqi
Date: 1-26-2014
Description: Holds all tests for kernel functions in kernels.py.

"""

import numpy as np
import scipy as sp
from scipy import spatial
import addpath
import gaussian_process as gp

def test_rbf():
    f = lambda x: np.sin(2*np.pi*10.*x)
    step = 0.4
    x = np.atleast_2d(np.arange(0,1,step)).T
    y = np.atleast_2d(f(x).ravel()).T
    hypcov = np.array(np.log([0.5, 0.25]))
    xx = np.array([[  2.500000000000000000000000000000e-01,
                      3.188519073815099012395337897745e-10,
                      6.615000000000001201934677979330e-37],
                   [  3.188519073815099012395337897745e-10,
                      2.500000000000000000000000000000e-01,
                      3.188519073815099012395337897745e-10],
                   [  6.615000000000001201934677979330e-37,
                      3.188519073815099012395337897745e-10,
                      2.500000000000000000000000000000e-01]])
    xy = np.array([[  2.500000000000000000000000000000e-01,
                      2.500000000000000000000000000000e-01,
                      2.500000000000000000000000000000e-01],
                   [  3.188519073815099012395337897745e-10,
                      3.188519073814781581835270673374e-10,
                      3.188519073814464668263086294647e-10],
                   [  6.615000000000001201934677979330e-37,
                      6.615000000000001201934677979330e-37,
                      6.615000000000001201934677979330e-37]])
    diag = np.matrix([0.25]*3).T
    xxreal =  np.around(gp.kernels.radial_basis(hypcov, x, x, False), 40)
    xyreal =  np.around(gp.kernels.radial_basis(hypcov, x, y, False), 40)
    diagreal = np.around(gp.kernels.radial_basis(hypcov, x, None, True), 40)
    assert (xxreal == xx).all()
    assert (xyreal == xy).all()
    assert (diagreal == diag).all()

def test_rq():
    f = lambda x: np.sin(2*np.pi*10.*x)
    step = 0.5
    x = np.atleast_2d(np.arange(0,1,step)).T
    y = np.atleast_2d(f(x).ravel()).T
    hypcov = np.array([1., 2., 3.])
    xx = np.array([[ 7.3890561,   7.38904883,  7.38902704],
                   [ 7.38904883,  7.3890561,   7.38904883],
                   [ 7.38902704,  7.38904883,  7.3890561 ]])
    xy = np.array([[ 7.3890561,   7.3890561,   7.3890561 ],
                   [ 7.38904883,  7.38904883,  7.38904883],
                   [ 7.38902704,  7.38902704,  7.38902704]])
    diag = np.matrix([7.3890561]*3).T
    assert ( gp.kernels.rational_quadratic(hypcov, x, x, False).all() == 
             xx.all() )
    assert ( gp.kernels.rational_quadratic(hypcov, x, y, False).all() == 
             xy.all() )
    assert ( gp.kernels.rational_quadratic(hypcov, x, None, True).all() == 
             diag.all() )

def test_per():
    f = lambda x: np.sin(2*np.pi*10.*x)
    step = 0.4
    x = np.atleast_2d(np.arange(0,1,step)).T
    y = np.atleast_2d(f(x).ravel()).T
    hypcov = np.array([1., 2., 3.])
    xx = np.array([[ 7.389056098930649518763402738841,
                     7.387998073641419694013166008517,
                     7.384841442457948268440759420628],
                   [ 7.387998073641419694013166008517,
                     7.389056098930649518763402738841,
                     7.387998073641419694013166008517],
                   [ 7.384841442457948268440759420628,
                     7.387998073641419694013166008517,
                     7.389056098930649518763402738841]])
    xy = np.array([[ 7.389056098930649518763402738841,
                     7.389056098930649518763402738841,
                     7.389056098930649518763402738841],
                   [ 7.387998073641419694013166008517,
                     7.387998073641419694013166008517,
                     7.387998073641419694013166008517],
                   [ 7.384841442457948268440759420628,
                     7.384841442457948268440759420628,
                     7.384841442457948268440759420628]])
    diag = np.matrix([7.389056098930649518763402738841]*3).T
    xxreal = gp.kernels.periodic(hypcov, x, x, False)
    xyreal = gp.kernels.periodic(hypcov, x, y, False)
    diagreal = gp.kernels.periodic(hypcov, x, None, True)
    assert (xxreal == xx).all()
    assert (xyreal == xy).all()
    assert (diagreal == diag).all()

def test_sm():
    f = lambda x: np.sin(2*np.pi*10.*x)
    step = 0.4
    x = np.atleast_2d(np.arange(0,1,step)).T
    y = np.atleast_2d(f(x).ravel()).T
    x = np.matrix(np.arange(0,1,step)).T
    y = np.matrix(f(x).ravel()).T
    hypcov = np.array([-0.34657359, -2.7080502, 0.08456656])
    xx = np.array([[  5.000000002799727116808981008944e-01,
                      1.170752896709145556009001865050e-02,
                      1.501742198183448997964587389928e-07],
                   [  1.170752896709145556009001865050e-02,
                      5.000000002799727116808981008944e-01,
                      1.170752896709145556009001865050e-02],
                   [  1.501742198183448997964587389928e-07,
                      1.170752896709145556009001865050e-02,
                      5.000000002799727116808981008944e-01]])
    xy = np.array([[  5.000000002799727116808981008944e-01,
                      5.000000002799727116808981008944e-01,
                      5.000000002799727116808981008944e-01],
                   [  1.170752896709145556009001865050e-02,
                      1.170752896709124565854942545684e-02,
                      1.170752896709102708339145237915e-02],
                   [  1.501742198183448997964587389928e-07,
                      1.501742198183392882031831792530e-07,
                      1.501742198183341795357200517541e-07]])
    diag = np.matrix([0.500000000279972711680898100894]*3).T
    xxreal = np.around(gp.kernels.spectral_mixture(hypcov, x, x, False), 16)
    xyreal = gp.kernels.spectral_mixture(hypcov, x, y, False)
    diagreal = gp.kernels.spectral_mixture(hypcov, x, None, True)
    assert (np.around(xxreal, 16) == np.around(xx, 16)).all()
    assert (np.around(xyreal, 16) == np.around(xy, 16)).all()
    assert (np.around(diagreal, 16) == np.around(diag, 16)).all()
