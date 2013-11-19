'''
Gaussian process
'''

import numpy as np
from scipy import spatial as spatial
from scipy import linalg as sln

def sq_dist(A, B=None):
    """
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

def initHyperParams(Q, x, y, sn):
    """
    """
    n, D = x.shape
    w = np.zeros(Q)
    m = np.zeros((D,Q))
    s = np.zeros((D,Q))
    hypinit = np.zeros(Q+2*D*Q+1)
    w[:] = np.std(y) / Q
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

class GaussianProcess(object):
    def __init__(self, xTrain=None, yTrain=None, xTest=None, yTest=None, hyp=None,
                 hypLik=None, cov='covSM', inf='infExact', lik='likGauss', 
                 mean='meanZero'):
        self.xt = xTrain
        self.yt = yTrain
        self.xs = xTest
        self.ys = yTest
        self.cov = eval('self._' + cov)
        self.inf = eval('self._' + inf)
        self.lik = eval('self._' + lik)
        self.mean = eval('self._' + mean)
        self.hyp = hyp
        self.hypLik = hypLik
        
    def train(self, hyp):
        """
        Return the negative log-likelihood of Z. This routine
        is used for optimization.
        """
        # Last parameter is always the noise variable
        hypLik = hyp[-1]
        hyp = hyp[0:-1]
        self.hypLik = hyp
        return self.inf(hyp, self.xt, self.yt, False, hypLik)


    def predict(self):
        x = self.xt
        xs = self.xs
        hyp = self.hyp
        hypLik = self.hypLik
        alpha, L, sW = self.inf(self.hyp, self.xt, self.yt, pred=True)
        ones = np.arange(alpha.shape[0], dtype=int) # Well, in MATLAB it's all ones

        if L is None:
            K = self.cov(hyp, x[nz,:])
            L = sln.cholesky(np.eye(np.sum(nz)) + (sW*sW.T)*K)

        isLtri = (np.tril(L,-1) == 0).all()
        nPoints = xs.shape[0]
        nProcessed = 0
        nBatch = 1000
        ymu = np.empty((nPoints,1))
        ys2 = np.empty((nPoints,1))
        fmu = np.empty((nPoints,1))
        fs2 = np.empty((nPoints,1))
        lp = np.empty((nPoints,1))

        while nProcessed < nPoints:
            rng = range(nProcessed, min(nProcessed+nBatch, nPoints))
            Kdiag = self.cov(self.hyp, xs[rng,:], diag=True)
            Koff = self.cov(self.hyp, x[ones,:], xs[rng,:], diag=False)
            ms = self.mean(xs[rng,:])
            N = alpha.shape[1]
            # Conditional mean fs|f
            Fmu = np.tile(ms, (1,N)) + Koff.T*alpha[ones,:]
            # Predictive means
            fmu[rng] = np.sum(Fmu, axis=1) / N

            if isLtri:
                V = np.linalg.solve(L.T, 
                                    np.multiply(np.tile(sW, (1,len(rng))),Koff))
                # This is pretty ugly...
                fs2[rng] = (Kdiag - 
                            np.matrix(np.multiply(V,V)).sum(axis=0).T)
            else:
                # This too.
                fs2[rng] = (Kdiag + 
                            np.matrix(np.multiply(Koff,(L*Koff))).sum(axis=0).T)

            # No negative elements allowed
            fs2[rng] = fs2[rng].clip(min=0)

            Fs2 = np.matrix(np.tile(fs2[rng], (1,N)))

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
#            print("Points processed (prediction mode): ", nProcessed)
        
        return {'ymu': ymu,
                'ys2': ys2,
                'fmu': fmu,
                'fs2': fs2,
                'lp': None if self.ys is None else lp,
                'post': [alpha, L, sW] }


    def _likGauss(self, hyp=None, y=None, mu=None, s2=None, hypLik=0):
        sn2 = np.exp(2*hypLik)
        if not y:
            y = np.zeros(mu.shape)
        if sln.norm(s2) <= 0:
            lp = -np.power(y-mu,2)/(2*sn2) - np.log(2*np.pi*sn2)/2
            s2 = 0
        else:
            lp = self._infEP(hyp, y, mu, s2, sn2)['lZ']

        return lp, mu, s2+sn2


    def _infEP(self, hyp, y, mu, s2, sn2):
        """
        Helper function for calculating the log-likelihood of
        Z, the partition function, and its derivatives for the
        Gaussian likelihood function.
        """
        lZ = -np.power(y-mu, 2)/((s2+sn2))/2 - np.log(2*np.pi*(s2+sn2))/2
        dlZ = (y - mu) / (s2+sn2)
        d2lZ = -1./(s2+sn2)

        return {'lZ': lZ, 'dlZ': dlZ, 'd2lZ': d2lZ }


    def _infExact(self, hyp, x, y, pred=True, hypLik=0):
        n, D = x.shape
        K = self.cov(hyp, x, x)
        m = self.mean(x)

        sn2 = np.exp(2*hypLik) # noise
        lower = False
        L = sln.cholesky(K/sn2 + np.eye(n))
        alpha = np.matrix(sln.cho_solve((L,lower), y-m) / sn2, dtype=np.float64)
        sW = np.ones((n,1)) / np.sqrt(sn2) # Sqrt of noise precision vector

        if not pred:
            # Training phase
            nlZ = ( (y-m).T*(alpha) / 2 + np.sum(np.log(np.diag(L))) +
                    n*np.log(2*np.pi*sn2)/2 )
            return nlZ[0,0] # Some optimizers freak if you return a 1-element matrix
        else:
            # Prediction phase
            return alpha, L, sW


    def _covSM(self, hyp, x=None, z=None, diag=False):
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
                    d2[:,:,j] = sq_dist(x[:,j].T)
            else:
                d2 = np.zeros((n,z.shape[0],D))
                for j in np.arange(0,D):
                    d2[:,:,j] = sq_dist(x[:,j].T, z[:,j].T)

        # Define kernel functions and derivatives
        k = lambda d2v, dm: np.multiply(np.exp(-2*np.pi**2 * d2v),
                                        np.cos(2*np.pi * dm))

        # Calculate correlation matrix
        K = 0
        d = np.sqrt(d2)

        for q in range(0,Q):
            C = w[q]
            for j in range(0,D):
                C = C*k(d2[:,:,j]*v[j,q], d[:,:,j]*m[j,q])
            K = K + C

        return K


    def _covSE(self, hyp, x=None, z=None, diag=False):
        """
        """
        ell = np.exp(hyp[0])
        sf2 = np.exp(2*hyp[1])

        if diag:
           K = np.zeros((x.shape[0],1))
        else:
            if x is z:
                K = sq_dist(x.T/ell)
            else:
                K = sq_dist(x.T/ell, z.T/ell)
        K = sf2*np.exp(-K/2)
        return K


    def _meanZero(self, x):
        """
        """
        return np.zeros((x.shape[0],1))
