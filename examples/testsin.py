import numpy as np
from scipy import optimize as sopt
import pylab as pl

# Adds the GP code directory to the system path
# so we can call the library from this subdir.
import addpath
import gaussian_process as gp

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
hypInit = None
nItr = 10

# Define core functions
likFunc = 'gaussian'
meanFunc = 'zero'
infFunc = 'exact'
covFunc = 'spectral_mixture'

l1Optimizer = 'COBYLA'
l1Options = {'maxiter':100 if not skipSM else 1}
l2Optimizer = 'CG'
# l2Optimizer = 'COBYLA'
l2Options = {'maxiter':100 if not skipSM else 1}

# Noise std. deviation
fixHypLik = False
sn = 1.0

# Initialize hyperparams
initArgs = {'Q':Q,'x':x,'y':y, 'samplingFreq':150, 'nPeaks':Q}
initArgs['sn'] = None if fixHypLik else sn
hypGuess = gp.core.initSMParamsFourier(**initArgs)
# Initialize GP object
hypGP = gp.GaussianProcess(hyp=hypGuess, inf=infFunc, mean=meanFunc,
                           cov=covFunc, lik=likFunc, hypLik=np.log(sn),
                           fixHypLik=fixHypLik, xtrain=x, ytrain=y, xtest=xt)
# Random starts
for itr in range(nItr):
    # Start over
    hypGuess = gp.core.initSMParamsFourier(**initArgs)
    hypGP.hyp = hypGuess
    # Optimize the guessed hyperparams
    try:
        hypGP.train(l1Optimizer, l1Options)
    except Exception as e:
        print "Iteration: ", itr, "FAILED"
        print "\t", e
        continue
    # Best random initialization
    if hypGP.nlml < negLogML:
        hypInit = hypGP.hyp
        negLogML = hypGP.nlml
    print "Iteration: ", itr, hypGP.nlml

# Give the GP object the proper params
hypGP.hyp = hypInit
hypGP.nlml = negLogML
# Optimize the best hyperparams even more
hypGP.hyp, hypGP.nlml = hypGP.train(method=l2Optimizer, options=l2Options)
print "Final hyperparams likelihood: ", negLogML
print "Noise parameter: ", np.exp(hypGP.hyp[-1]) if not fixHypLik else sn
print "Reoptimized: ", hypGP.nlml
if fixHypLik:
    print np.exp(hypGP.hyp.reshape(3,Q))
else:
    print np.exp(hypGP.hyp[0:-1].reshape(3,Q))

# Do the extrapolation
prediction = hypGP.predict()
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
pl.fill(fillx, filly, alpha=.5, fc='0.5', ec='None', label='95% confidence interval')
pl.legend()
pl.show()
