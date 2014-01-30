"""

File: gptests.py
Author: Hadayat Seddiqi
Date: 1-29-2014
Description: Holds tests for gaussian process functions.

"""

import numpy as np
import scipy as sp
import addpath
import gaussian_process as gp

class TestGP:
    def setUp(self):
        f = lambda x: np.sin(2*np.pi*10.*x) + np.sin(2*np.pi*13.*x)
        Ts = 1/150.0
        self.x = np.matrix(np.arange(0,1,Ts)).T
        self.xt = np.matrix(np.arange(0,2,Ts)).T
        self.y = np.matrix(f(self.x).ravel()).T
        self.yt = np.matrix(f(self.xt).ravel()).T
        self.hyps = {
            'mean': np.array([]), 
            'lik': [np.log(0.5)], 
            'cov': np.log([2.54846959e-01, 5.24927544e-01, 2.15436686e-01, 
                           2.86117061e-01, 1.81531436e-01, 1.44032999e-01,
                           1.29493734e+01, 9.92190150e+00, 1.85152636e+02,
                           7.49453588e+01, 1.81532400e+00, 2.28497943e+01,
                           1.03092807e-01, 4.88398945e-02, 1.94985263e+01,
                           3.19829864e+01, 2.12244665e+00, 8.39227875e-01])
            }
        self.hypGP = gp.GaussianProcess(hyp=self.hyps, 
                                        inf=gp.inferences.exact, 
                                        mean=gp.means.zero, 
                                        cov=gp.kernels.spectral_mixture,
                                        lik=gp.likelihoods.gaussian, 
                                        xtrain=self.x, ytrain=self.y, 
                                        xtest=self.xt, ytest=self.yt)

    def test_train(self):
        hypGP = self.hypGP
        hypGP.hyp, hypGP.nlml = hypGP.train(method='COBYLA', 
                                            options={'maxiter':100})
        self.hypGP = hypGP
        assert np.around(hypGP.nlml, 8) == np.around(80.6742912434, 8)

    def test_predict(self):
        pred = self.hypGP.predict()
        assert (pred['post'][2] == np.matrix(2*np.ones(150)).T).all()
