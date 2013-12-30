import numpy as np
from scipy import optimize as sopt
import pylab as pl
import gaussian_process as gp

from scipy import io as sio

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

# Define core functions
likFunc = 'likGauss'
meanFunc = 'meanZero'
infFunc = 'infExact'
covFunc = 'spectral_mixture'

l1Optimizer = 'COBYLA'
l1Options = {'maxiter':100 if not skipSM else 1}
l2Optimizer = 'CG'
#l2Optimizer = 'COBYLA'
l2Options = {'maxiter':100 if not skipSM else 1}

# Noise std. deviation
fixHypLik = False
sn = 1.0

# Random starts
for itr in range(nItr):
    initArgs = {'Q':Q,'x':x,'y':y, 'samplingFreq':150, 'nPeaks':4}
    initArgs['sn'] = None if fixHypLik else sn
    # Initialize hyperparams
    hypGuess = gp.core.initSMParamsFourier(**initArgs)
    # Optimize the guessed hyperparams
    hypGP = gp.GaussianProcess(hyp=hypGuess, inf=infFunc, mean=meanFunc,
                               cov=covFunc, lik=likFunc, hypLik=np.log(sn),
                               fixHypLik=fixHypLik, xTrain=x, yTrain=y)
    try:
        optOutput = sopt.minimize(fun=hypGP.train, x0=hypGuess, method=l1Optimizer,
                                  options=l1Options)
    except:
        print "Iteration: ", itr, "FAILED"
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
if fixHypLik:
    print np.exp(hypTrained.reshape(3,Q))
else:
    print np.exp(hypTrained[0:-1].reshape(3,Q))

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
pl.fill(fillx, filly, alpha=.5, fc='0.5', ec='None', label='95% confidence interval')
pl.legend()
pl.show()
