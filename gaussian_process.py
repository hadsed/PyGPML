'''

File: gaussian_process.py
Author: Hadayat Seddiqi
Date: 12-27-2013
Description: Gaussian process object class defined here with
             helper functions for initialization.

'''

import numpy as np
from scipy import spatial as spatial
from scipy import linalg as sln
from scipy import signal as spsig
import collections
# GP imports
import kernels
import core


class GaussianProcess(object):
    """
    """
    def __init__(self, xTrain=None, yTrain=None, xTest=None, yTest=None, hyp=None,
                 fixHypLik=False, hypLik=None, cov='covSM', inf='infExact', 
                 lik='likGauss', mean='meanZero'):
        self.xt = np.atleast_2d(xTrain)
        self.yt = np.atleast_2d(yTrain)
        self.xs = np.atleast_2d(xTest) if xTest is not None else None
        self.ys = np.atleast_2d(yTest) if yTest is not None else None
        # Check if we're being given a custom functions or a string
        # which corresponds to a built-in function
        if isinstance(cov, basestring):
            self.cov = eval('kernels.' + cov)
        else:
            self.cov = cov
        if isinstance(inf, basestring):
            self.inf = eval('self._' + inf)
        else:
            self.inf = inf
        if isinstance(lik, basestring):
            self.lik = eval('self._' + lik)
        else:
            self.lik = lik
        if isinstance(mean, basestring):
            self.mean = eval('self._' + mean)
        else:
            self.mean = mean
        self.hyp = hyp
        self.hypLik = hypLik
        self.fixHypLik = fixHypLik

    def train(self, hyp):
        """
        Return the negative log-likelihood of Z. This routine
        is used for optimization.
        """
        # Last parameter is always the noise variable
        if self.fixHypLik:
            hypLik = hyp[-1]
            hyp = hyp[0:-1]
            self.hypLik = hypLik
        else:
            hypLik = self.hypLik
        return self.inf(hyp, self.xt, self.yt, False, hypLik)


    def predict(self):
        x = self.xt
        xs = self.xs
        hyp = self.hyp
        hypLik = self.hypLik
        alpha, L, sW = self.inf(self.hyp, self.xt, self.yt, pred=True)
        ones = np.arange(alpha.shape[0], dtype=int) # Well, in MATLAB it's all ones
        # If for some reason L isn't provided
        if L is None:
            nz = np.where(alpha != 0)[0]  # this is really to determine sparsity
            K = self.cov(hyp, x[nz,:])
            L = sln.cholesky(np.eye(np.sum(nz)) + (sW*sW.T)*K)
        # Initialize some parameters
        isLtri = (np.tril(L,-1) == 0).all()
        nPoints = xs.shape[0]
        nProcessed = 0
        nBatch = 1000
        ymu = np.empty((nPoints,1))
        ys2 = np.empty((nPoints,1))
        fmu = np.empty((nPoints,1))
        fs2 = np.empty((nPoints,1))
        lp = np.empty((nPoints,1))
        # Loop through points
        while nProcessed < nPoints:
            rng = range(nProcessed, min(nProcessed+nBatch, nPoints))
            Kdiag = self.cov(self.hyp, xs[rng,:], diag=True)
            Koff = self.cov(self.hyp, x[ones,:], xs[rng,:], diag=False)
            ms = self.mean(xs[rng,:])
            N = alpha.shape[1]
            # Conditional mean fs|f, GPML Eqs. (2.25), (2.27)
            Fmu = np.tile(ms, (1,N)) + Koff.T*alpha[ones,:]
            # Predictive means, GPML Eqs. (2.25), (2.27)
            fmu[rng] = np.sum(Fmu, axis=1) / N
            # Calculate the predictive variances, GPML Eq. (2.26)
            if isLtri:
                # Use Cholesky parameters (L, alpha, sW) if L is triangular
                V = np.linalg.solve(L.T, 
                                    np.multiply(np.tile(sW, (1,len(rng))),Koff))
                # Predictive variances
                fs2[rng] = (Kdiag - 
                            np.matrix(np.multiply(V,V)).sum(axis=0).T)
            else:
                # Use alternative parameterization incase L is not triangular
                # Predictive variances
                fs2[rng] = (Kdiag + 
                            np.matrix(np.multiply(Koff,(L*Koff))).sum(axis=0).T)
            # No negative elements allowed (it's numerical noise)
            fs2[rng] = fs2[rng].clip(min=0)
            # In case of sampling (?)
            Fs2 = np.matrix(np.tile(fs2[rng], (1,N)))
            # 
            if self.ys is None:
                Lp, Ymu, Ys2 = self.lik(hyp, [], Fmu, Fs2, hypLik)
            else:
                Ys = np.tile(ys[rng], (1,N))
                Lp, Ymu, Ys2 = self.lik(hyp, Ys, Fmu, Fs2, hypLik)

            # Log probability
            lp[rng] = np.sum(Lp.reshape(Lp.size/N,N), axis=1) / N
            # Predictive mean ys|y
            ymu[rng] = np.sum(Ymu.reshape(Ymu.size/N,N), axis=1) / N
            # Predictive variance ys|y
            ys2[rng] = np.sum(Ys2.reshape(Ys2.size/N,N), axis=1) / N
            # Iterate batch
            nProcessed = rng[-1] + 1
        
        test = False
        if test:
            return {'ymu': Ymu,
                    'ys2': Ys2,
                    'fmu': Fmu,
                    'fs2': Fs2,
                    'lp': None if self.ys is None else Lp,
                    'post': [alpha, L, sW] }
        else:
            return {'ymu': ymu,
                    'ys2': ys2,
                    'fmu': fmu,
                    'fs2': fs2,
                    'lp': None if self.ys is None else lp,
                    'post': [alpha, L, sW] }


    def _likGauss(self, hyp=None, y=None, mu=None, s2=None, hypLik=0):
        """
        Compute a Gaussian predictive distribution on target points,
        return the negative log probability of the target along with
        its means and variances. It can be expressed as

        p(t|D,xs) = exp(-(t-f(xs))^2/(2*sn^2)) / sqrt(2*pi*sn^2)

        where t is the target data points, f(xs) is the mean and sn 
        is the standard deviation. See GPML Eq. (2.34).
        """
        sn2 = np.exp(2*hypLik)
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

    def _infExact(self, hyp, x, y, pred=True, hypLik=0):
        n, D = x.shape
        K = self.cov(hyp, x, x)
        m = self.mean(x)
        sn2 = np.exp(2*hypLik) # noise
        lower = False
        # Basically the correlation matrix
        L = sln.cholesky(K/sn2 + np.eye(n))
        # This is just (K + I*noise)^-1 * y, see GPML Eq. (2.27)
        alpha = np.matrix(sln.cho_solve((L,lower), y-m) / sn2, dtype=np.float64)
        # Sqrt of noise precision vector
        sW = np.ones((n,1)) / np.sqrt(sn2)
        # Training phase
        if not pred:
            # negative log marginal likelihood, GPML Eq. (2.30)
            nlZ = ( (y-m).T*(alpha) / 2 + np.sum(np.log(np.diag(L))) +
                    n*np.log(2*np.pi*sn2)/2 )
            return nlZ[0,0]  # Make sure we're giving a number
        # Prediction phase
        else:
            return alpha, L, sW

    def _meanZero(self, x):
        """
        """
        return np.zeros((x.shape[0],1))
