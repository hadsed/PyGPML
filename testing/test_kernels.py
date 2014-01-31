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

class TestKernels:
    def setUp(self):
        f = lambda x: np.sin(2*np.pi*10.*x)
        step = 0.4
        # The spectral mixture kernel doesn't yet work with ndarrays
        self.x = np.atleast_2d(np.arange(0,1,step)).T
        self.y = np.atleast_2d(f(self.x).ravel()).T
        # self.x = np.matrix(np.arange(0,1,step)).T
        # self.y = np.matrix(f(self.x).ravel()).T

    def test_rbf(self):
        x = self.x
        y = self.y
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
        xxreal =  gp.kernels.radial_basis(hypcov, x, x, False)
        xyreal =  gp.kernels.radial_basis(hypcov, x, y, False)
        diagreal = gp.kernels.radial_basis(hypcov, x, None, True)
        assert np.allclose(xxreal, xx)
        assert np.allclose(xyreal, xy)
        assert np.allclose(diagreal, diag)

    def test_rq(self):
        x = self.x
        y = self.y
        hypcov = np.array([1., 2., 3.])
        xx = np.array([[ 7.3890561,   7.38904883,  7.38902704],
                       [ 7.38904883,  7.3890561,   7.38904883],
                       [ 7.38902704,  7.38904883,  7.3890561 ]])
        xy = np.array([[ 7.3890561,   7.3890561,   7.3890561 ],
                       [ 7.38904883,  7.38904883,  7.38904883],
                       [ 7.38902704,  7.38902704,  7.38902704]])
        diag = np.matrix([7.3890561]*3).T
        assert np.allclose(gp.kernels.rational_quadratic(hypcov, x, x, False), xx)
        assert np.allclose(gp.kernels.rational_quadratic(hypcov, x, y, False), xy)
        assert np.allclose(gp.kernels.rational_quadratic(hypcov, x, None, True), diag)

    def test_per(self):
        x = self.x
        y = self.y
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
        assert np.allclose(xxreal, xx)
        assert np.allclose(xyreal, xy)
        assert np.allclose(diagreal, diag)

    def test_sm(self):
        x = self.x
        y = self.y
        hypcov = np.array([-0.34657359, -0.34657359, -0.34657359,
                           -2.7080502, -2.7080502, -2.7080502,
                            0.08456656, 0.08456656, 0.08456656])
        xx = np.array([[  1.50000000e+00,   3.51225869e-02,   4.50522659e-07],
                       [  3.51225869e-02,   1.50000000e+00,   3.51225869e-02],
                       [  4.50522659e-07,   3.51225869e-02,   1.50000000e+00]])
        xy = np.array([[  1.50000000e+00,   1.50000000e+00,   1.50000000e+00],
                       [  3.51225869e-02,   3.51225869e-02,   3.51225869e-02],
                       [  4.50522659e-07,   4.50522659e-07,   4.50522659e-07]])
        diag = np.matrix([1.5]*3).T
        xxreal = gp.kernels.spectral_mixture(hypcov, x, x, False)
        xyreal = gp.kernels.spectral_mixture(hypcov, x, y, False)
        diagreal = gp.kernels.spectral_mixture(hypcov, x, None, True)
        assert np.allclose(xxreal, xx)
        assert np.allclose(xyreal, xy)
        assert np.allclose(diagreal, diag)
