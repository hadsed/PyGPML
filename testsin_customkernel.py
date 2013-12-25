import numpy as np
from scipy import optimize as sopt
from scipy import io as sio
import pylab as pl
import GaussianProcess as gp

# Generate some test data
ff = 10
f = lambda x: np.sin(2*np.pi*ff*x) + np.sin(2*np.pi*(ff+3)*x)/3.
Fs = 150.0
Ts = 1.0/Fs
x = np.matrix(np.arange(0,1,Ts)).T
xt = np.matrix(np.arange(0,2,Ts)).T
y = np.matrix(f(x).ravel()).T
dy = 0.5 + 1.e-1 * np.random.random(y.shape)
noise = np.random.normal(0, dy)
y += noise

# Number of Gaussians in the mixture model
Q = 4
skipSM = False
negLogML = np.inf
nItr = 1

def rqkernel(hyp, x=None, z=None, diag=False):
    """
    Implements a rational-quadratic kernel:
    k(x,x') = hscale^2 * (1 + (x-x')^2/(alpha*lamb^2))^{-alpha}
    """
    hscale = np.exp(hyp[0])
    alpha = np.exp(hyp[1])
    lamb = np.exp(hyp[2])
    if diag:
        K = np.zeros((x.shape[0],1))
    else:
        if x is z:
            K = gp.sq_dist(x.T/lamb**2)
        else:
            K = gp.sq_dist(x.T/lamb**2, z.T/lamb**2)
    K = hscale**2 * np.power(1 + K/alpha, -alpha)
    return K

def perkernel(hyp, x=None, z=None, diag=False):
    """
    Implements a periodic kernel:
    k(x,x') = sigma^2 * exp(-2/ell^2 * sin^2(pi*|x-x'|/per))
    """
    sigma = np.exp(hyp[0])
    ell = np.exp(hyp[1])
    per = np.exp(hyp[2])
    if diag:
        K = np.zeros((x.shape[0],1))
    else:
        if x is z:
            K = gp.sq_dist(x.T)
        else:
            K = gp.sq_dist(x.T, z.T)
    K = np.sqrt(K)  # need the absolute distance, not squared
    K = sigma**2 * np.exp(-2/ell**2 * np.power(np.sin(np.pi*K/per), 2))
    return K

# Define core functions
likFunc = 'likGauss'
meanFunc = 'meanZero'
infFunc = 'infExact'
# covFunc = rqkernel
covFunc = perkernel

l1Optimizer = 'COBYLA'
l1Options = {'maxiter':100 if not skipSM else 1}
l2Optimizer = 'CG'
l2Options = {'maxiter':100 if not skipSM else 1}

# Noise std. deviation
fixHypLik = False
sn = 1.0

# Random starts
for itr in range(nItr):
    initArgs = {'Q':Q,'x':x,'y':y, 'samplingFreq':150, 'nPeaks':4}
    initArgs['sn'] = None if fixHypLik else sn
    # Initialize hyperparams
    # hypGuess = np.log([0.1, 0.1, 0.1])  # for the rqkernel
    hypGuess = np.log([0.05, 0.1, 1.0])
    # Optimize the guessed hyperparams
    hypGP = gp.GaussianProcess(hyp=hypGuess, inf=infFunc, mean=meanFunc,
                               cov=covFunc, lik=likFunc, hypLik=np.log(sn),
                               fixHypLik=fixHypLik, xTrain=x, yTrain=y)
    try:
        optOutput = sopt.minimize(fun=hypGP.train, x0=hypGuess, method=l1Optimizer,
                                  options=l1Options)
    except Exception, e:
        print "Iteration: ", itr, "FAILED"
        print "\tError message:", e
        continue
    hypTrained = optOutput.x
    newNegLogML = optOutput.fun
    # Update
    if newNegLogML < negLogML:
        hypInit = hypTrained
        negLogML = newNegLogML
    print "Iteration: ", itr, newNegLogML

# Best random initialization
hypGP = gp.GaussianProcess(hyp=hypTrained, inf=infFunc, mean=meanFunc, cov=covFunc,
                           lik=likFunc, hypLik=np.log(sn), fixHypLik=fixHypLik,
                           xTrain=x, yTrain=y)

# Optimize the best hyperparams even more
optOutput = sopt.minimize(fun=hypGP.train, x0=hypInit, method=l2Optimizer,
                          options=l2Options)

hypTrained = optOutput.x
newNegLogML = optOutput.fun
print "Final hyperparams likelihood: ", negLogML
print "Noise parameter: ", np.exp(hypTrained[-1]) if not fixHypLik else sn
print "Reoptimized: ", newNegLogML
print np.exp(hypTrained)

# Fit the GP
fittedGP = gp.GaussianProcess(hyp=hypTrained, inf=infFunc, mean=meanFunc,
                              cov=covFunc, lik=likFunc, hypLik=np.log(sn),
                              xTrain=x, yTrain=y, xTest=xt)

prediction = fittedGP.predict()
mean = prediction['ymu']
sigma2 = prediction['ys2']

# Plot the stuff
pl.plot(x, y, 'b', label=u'Training Data')
pl.plot(xt[x.size:], f(xt)[x.size:], 'k', label=u'Test Data')
pl.plot(xt, mean, 'r', label=u'SM Prediction')
sigma = np.power(sigma2, 0.5)
fillx = np.concatenate([np.array(xt.ravel()).ravel(), 
                        np.array(xt.ravel()).ravel()[::-1]])
filly = np.concatenate([(np.array(mean.ravel()).ravel() - 1.9600 * 
                         np.array(sigma.ravel()).ravel()),
                        (np.array(mean.ravel()).ravel() + 1.9600 * 
                         np.array(sigma.ravel()).ravel())[::-1]])
pl.fill(fillx, filly, alpha=.5, fc='0.5', ec='None', 
        label='95% confidence interval')
pl.legend()
pl.show()
