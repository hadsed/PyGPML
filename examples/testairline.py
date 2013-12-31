import numpy as np
import pylab as pl
from scipy import io as sio
from scipy import optimize as sopt

# Adds the GP code directory to the system path
# so we can call the library from this subdir.
import addpath
import gaussian_process as gp

data = sio.loadmat('data/airlinedata.mat')

x = np.matrix(data['xtrain'])
y = np.matrix(data['ytrain'])
xt = np.matrix(data['xtest'])
yt = np.matrix(data['ytest'])
# To get interpolation too
#xt = np.concatenate((x,xt))
#yt = np.concatenate((y,yt))

skipSM = False
Q = 10

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
l2Optimizer = 'L-BFGS-B'
l2Optimizer = 'COBYLA'
l2Options = {'maxiter':100 if not skipSM else 1}

# Noise std. deviation
fixHypLik = False
sn = 0.1

# Initialize hyperparams
initArgs = {'Q':Q, 'x':x, 'y':y, 'samplingFreq':1, 'nPeaks':Q}
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
pl.plot(xt, yt, 'k', label=u'Test Data')
pl.plot(xt, mean, 'r', label=u'SM Prediction')
fillx = np.concatenate([np.array(xt.ravel()).ravel(), 
                        np.array(xt.ravel()).ravel()[::-1]])
filly = np.concatenate([(np.array(mean.ravel()).ravel() - 1.9600 * 
                         np.array(sigma2.ravel()).ravel()),
                        (np.array(mean.ravel()).ravel() + 1.9600 * 
                         np.array(sigma2.ravel()).ravel())[::-1]])
pl.fill(fillx, filly, alpha=.5, fc='0.5', ec='None', 
        label='95% confidence interval')

# Now try to do a vanilla isotropic Gaussian kernel
seOptimizer = 'COBYLA'
covFunc = 'radial_basis'
sn = 0.1
hypSEInit = np.log([np.std(y), 40., sn])
seGP = gp.GaussianProcess(hyp=hypSEInit, inf=infFunc, mean=meanFunc, 
                          cov=covFunc, lik=likFunc, hypLik=np.log(sn), 
                          xtrain=x, ytrain=y, xtest=xt)
seGP.train(seOptimizer, {'maxiter':1000})
sePred = seGP.predict()
seMean = sePred['ymu']
seSig2 = sePred['ys2']

print "Optimized SE likelihood: ", seGP.nlml
print "Noise parameter: ", seGP.hyp[-1]
print "SE hyperparams: ", seGP.hyp[0:-1]

pl.plot(xt, seMean, 'g', label=u'SE Prediction')
pl.legend()
pl.show()
