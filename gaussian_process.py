'''

File: gaussian_process.py
Author: Hadayat Seddiqi
Date: 12-27-2013
Description: Gaussian process object class defined here with
             helper functions for initialization.

'''

import collections
import numpy as np
from scipy import spatial as spatial
from scipy import linalg as sln
from scipy import signal as spsig
from scipy import optimize as sopt

# GP imports
import core
import inferences
import kernels
import likelihoods
import means

class GaussianProcess(object):
    """
    """
    def __init__(self, xtrain=None, ytrain=None, xtest=None, ytest=None, hyp=None,
                 cov='radial_basis', inf='exact', lik='gaussian', mean='zero'):
        """
        @hyp is a dict comprised of..
        """
        self.xtrain = np.atleast_2d(xtrain)
        self.ytrain = np.atleast_2d(ytrain)
        self.xtest = np.atleast_2d(xtest) if xtest is not None else None
        self.ytest = np.atleast_2d(ytest) if ytest is not None else None
        self.cov = cov
        self.inf = inf
        self.lik = lik
        self.mean = mean
        self.nlml = np.inf
        self.hyp = hyp
        # Make sure we have each of 'cov', 'lik', and 'mean' in the
        # hyperparameters dict, make sure they are iterable or fail,
        # and ensure they are numpy arrays, not lists or otherwise
        if 'cov' not in self.hyp:
            self.hyp['cov'] = np.array([])
        elif not isinstance(self.hyp['cov'], collections.Iterable):
            raise ValueError("Covariance kernel hyperparameters is not "
                             "iterable. Must be list or Numpy array.")
        else:
            self.hyp['cov'] = np.array(self.hyp['cov'])

        if 'lik' not in self.hyp:
            self.hyp['lik'] = np.array([])
        elif not isinstance(self.hyp['lik'], collections.Iterable):
            raise ValueError("Likelihood hyperparameters is not iterable. "
                             "Must be list or Numpy array.")
        else:
            self.hyp['lik'] = np.array(self.hyp['lik'])

        if 'mean' not in self.hyp:
            self.hyp['mean'] = np.array([])
        elif not isinstance(self.hyp['mean'], collections.Iterable):
            raise ValueError("Mean hyperparameters is not iterable. "
                             "Must be list or Numpy array.")
        else:
            self.hyp['mean'] = np.array(self.hyp['mean'])
        # Keep a standard flattened version of the hyperparams dict
        self.hypflat = self._hypDict2Flat(self.hyp)

    def _hypFlat2Dict(self, hypflat):
        """
        Reconstruct a hyperparameter dictionary from a flattened
        hyperparameter list using the structure of the hyperparameter
        dict attribute but using values from the incoming flattened
        hyperparameters, @hypflat.
        """
        hypdict = self.hyp
        ncov = len(hypdict['cov'])
        nlik = ncov + len(hypdict['lik'])
        nmean = ncov + nlik + len(hypdict['mean'])
        return {
            'cov': hypflat[0:ncov] if ncov > 0 else np.array([]),
            'lik': hypflat[ncov:nlik] if nlik > 0 else np.array([]),
            'mean': hypflat[nlik:nmean] if nmean > 0 else np.array([])
            }

    def _hypDict2Flat(self, hypdict):
        """
        Create a flattened version of @hypdict, which is
        the structured hyperparameters dictionary with elements
        'cov', 'lik', and 'mean', which we will unpack in that order
        into a flat list.
        """
        hypflat = np.array([])
        if np.atleast_1d(hypdict['cov']).size > 0:
            hypflat = np.atleast_1d(hypdict['cov'])
        if np.atleast_1d(hypdict['lik']).size > 0:
            hypflat = np.concatenate([hypflat, np.atleast_1d(hypdict['lik'])])
        if np.atleast_1d(hypdict['mean']).size > 0:
            hypflat = np.concatenate([hypflat, np.atleast_1d(hypdict['mean'])])
        return hypflat

    def train(self, method='COBYLA', options={}, write=True):
        """
        Train the Gaussian process model using @method, where
        @options are the optimization routine arguments and @hyp
        is the initialization. Optimization is done over the negative
        log marginal likelihood (see GPML section 2.7.1).

        @write is a flag that tells whether to write the optimized
        hyperparameters and likelihood to the object attributes.
        """
        # A wrapper function (the optimized variable needs to be first)
        def infwrapper(hypflat, cov, mean, xt, yt, p):
            hyp = self._hypFlat2Dict(hypflat)
            return self.inf(cov=cov, mean=mean, hyp=hyp, x=xt, y=yt, pred=p)
        # Do the actual optimization
        optparams = sopt.minimize(fun=infwrapper,
                                  args=(self.cov, self.mean, self.xtrain,
                                        self.ytrain, False),
                                  x0=self.hypflat,
                                  method=method,
                                  options=options)
        # Record these results in the relevant attributes
        if write:
            self.hypflat = optparams.x
            self.hyp = self._hypFlat2Dict(optparams.x)
            self.nlml = optparams.fun
        # Just return the optimized hyperparams and log-likelihood
        return (self._hypFlat2Dict(optparams.x), optparams.fun)

    def predict(self):
        """
        """
        x = self.xtrain
        xs = self.xtest
        hyp = self.hyp
        alpha, L, sW = self.inf(cov=self.cov, mean=self.mean, hyp=self.hyp, 
                                x=self.xtrain, y=self.ytrain, pred=True)
        ones = np.arange(alpha.shape[0], dtype=int) # Well, in MATLAB it's all ones
        # If for some reason L isn't provided
        if L is None:
            nz = np.where(alpha != 0)[0]  # this is really to determine sparsity
            K = self.cov(hyp['cov'], x[nz,:])
            L = sln.cholesky(np.eye(np.sum(nz)) + (sW*sW.T)*K)
        # Initialize some parameters
        isLtri = (np.tril(L,-1) == 0).all()
        nPoints = xs.shape[0]
        nProcessed = 0
        nBatch = 1000
        ymu = np.empty(nPoints)
        ys2 = np.empty(nPoints)
        fmu = np.empty(nPoints)
        fs2 = np.empty(nPoints)
        lp = np.empty(nPoints)
        # Loop through points
        while nProcessed < nPoints:
            rng = range(nProcessed, min(nProcessed+nBatch, nPoints))
            xsrng = xs[rng,:]
            xones = x[ones,:]
            Kdiag = self.cov(self.hyp['cov'], xsrng, diag=True)
            Koff = self.cov(self.hyp['cov'], xones, xsrng, diag=False)
            ms = self.mean(hyp['mean'], xsrng)
            N = alpha.shape[1]
            # Conditional mean fs|f, GPML Eqs. (2.25), (2.27)
            Fmu = np.tile(ms, (1,N)) + np.dot(Koff.T, alpha[ones,:])
            # Predictive means, GPML Eqs. (2.25), (2.27)
            fmu[rng] = np.atleast_2d(np.sum(Fmu, axis=1) / N).T
            # Calculate the predictive variances, GPML Eq. (2.26)
            if isLtri:
                # Use Cholesky parameters (L, alpha, sW) if L is triangular
                V = np.linalg.solve(L.T, np.tile(sW, (1,len(rng)))*Koff)
                # Predictive variances
                fs2[rng] = Kdiag - (V*V).sum(axis=0).T
            else:
                # Use alternative parameterization incase L is not triangular
                # Predictive variances
                fs2[rng] = Kdiag + (Koff*(L*Koff)).sum(axis=0).T
            # No negative elements allowed (it's numerical noise)
            fs2[rng] = fs2[rng].clip(min=0)
            # In case of sampling (? this was in the original code ?)
            Fs2 = np.atleast_2d(np.tile(fs2[rng], (1,N)))
            if self.ytest is None:
                Lp, Ymu, Ys2 = self.lik(hyp=hyp, y=None, mu=Fmu, s2=Fs2)
            else:
                Ys = np.tile(self.ytest[rng], (1,N))
                Lp, Ymu, Ys2 = self.lik(hyp=hyp, y=Ys, mu=Fmu, s2=Fs2)

            # Log probability
            lp[rng] = np.sum(Lp.reshape(Lp.size/N,N), axis=1) / N
            # Predictive mean ys|y
            ymu[rng] = np.sum(Ymu.reshape(Ymu.size/N,N), axis=1) / N
            # Predictive variance ys|y
            ys2[rng] = np.sum(Ys2.reshape(Ys2.size/N,N), axis=1) / N
            # Iterate batch
            nProcessed = rng[-1] + 1
        
        return {'ymu': np.atleast_2d(ymu).T,
                'ys2': np.atleast_2d(ys2).T,
                'fmu': np.atleast_2d(fmu).T,
                'fs2': np.atleast_2d(fs2).T,
                'lp': None if self.ytest is None else np.atleast_2d(lp).T,
                'post': [np.atleast_2d(alpha).T,
                         np.atleast_2d(L).T, 
                         np.atleast_2d(sW).T] 
                }
