import numpy as np
import pylab as pl
from scipy import io as sio
from scipy import optimize as sopt

import gaussian_process as gp

data = sio.loadmat('data/CO2data.mat')

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
nItr = 10

# Define core functions
likFunc = 'likGauss'
meanFunc = 'meanZero'
infFunc = 'infExact'
covFunc = 'spectral_mixture'

l1Optimizer = 'COBYLA'
l1Options = {'maxiter':100 if not skipSM else 1}
l2Optimizer = 'L-BFGS-B'
#l2Optimizer = 'COBYLA'
l2Options = {'maxiter':100 if not skipSM else 1}

# Noise std. deviation
fixHypLik = False
sn = 1.0
initArgs = {'Q':Q, 'x':x, 'y':y, 'samplingFreq':1, 'nPeaks':Q}
initArgs['sn'] = None if fixHypLik else sn

# Random starts
for itr in range(nItr):
    # Initialize hyperparams
    hypGuess = gp.core.initSMParamsFourier(**initArgs)
    # Optimize the guessed hyperparams
    hypGP = gp.GaussianProcess(hyp=hypGuess, inf=infFunc, mean=meanFunc, 
                               cov=covFunc, lik=likFunc, hypLik=np.log(sn),
                               xTrain=x, yTrain=y)
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
print "Final", hypInit
# Optimize the best hyperparams even more
hypGP = gp.GaussianProcess(hyp=hypTrained, inf=infFunc, mean=meanFunc, cov=covFunc,
                           lik=likFunc, hypLik=np.log(sn), xTrain=x, yTrain=y)
optOutput = sopt.minimize(fun=hypGP.train, x0=hypInit, method=l2Optimizer,
                          options=l2Options)
hypTrained = optOutput.x
newNegLogML = optOutput.fun
print "Final hyperparams likelihood: ", negLogML
print "Noise parameter: ", hypTrained[-1]
print "Reoptimized: ", newNegLogML
print hypTrained[0:-1].reshape(3,10)

# Fit the GP
fittedGP = gp.GaussianProcess(hyp=hypTrained, inf=infFunc, mean=meanFunc, 
                              cov=covFunc, lik=likFunc, hypLik=np.log(sn), 
                              xTrain=x, yTrain=y, xTest=xt)

prediction = fittedGP.predict()
mean = prediction['ymu']
sigma2 = prediction['ys2']

# Plot the stuff
pl.plot(x, y, 'b', label=u'Training Data')
pl.plot(xt, yt, 'r', label=u'Test Data')
pl.plot(xt, mean, 'k', label=u'SM Prediction')
sigma = np.power(sigma2, 0.5)
fillx = np.concatenate([np.array(xt.ravel()).ravel(), 
                        np.array(xt.ravel()).ravel()[::-1]])
filly = np.concatenate([(np.array(mean.ravel()).ravel() - 1.9600 * 
                         np.array(sigma.ravel()).ravel()),
                        (np.array(mean.ravel()).ravel() + 1.9600 * 
                         np.array(sigma.ravel()).ravel())[::-1]])
pl.fill(fillx, filly, alpha=.5, fc='0.5', ec='None', label='95% confidence interval')

# Now try to do a vanilla isotropic Gaussian kernel
seOptimizer = 'COBYLA'
covFunc = 'radial_basis'
sn = 0.1
hypSEInit = np.log([np.std(y), 40., sn])
seGP = gp.GaussianProcess(hyp=hypSEInit, inf=infFunc, mean=meanFunc, 
                          cov=covFunc, lik=likFunc, hypLik=np.log(sn), 
                          xTrain=x, yTrain=y)
optSE = sopt.minimize(fun=seGP.train, x0=hypSEInit, method=seOptimizer,
                      options={'maxiter':1000})
seFitted = gp.GaussianProcess(hyp=optSE.x, inf=infFunc, mean=meanFunc, 
                              cov=covFunc, lik=likFunc, hypLik=np.log(sn), 
                              xTrain=x, yTrain=y, xTest=xt)
sePred = seFitted.predict()
seMean = sePred['ymu']
seSig2 = sePred['ys2']

print "Optimized SE likelihood: ", optSE.fun
print "Noise parameter: ", optSE.x[-1]
print "SE hyperparams: ", optSE.x[0:-1]

pl.plot(xt, seMean, 'g', label=u'SE Prediction')
fillx = np.concatenate([np.array(xt.ravel()).ravel(), 
                        np.array(xt.ravel()).ravel()[::-1]])
filly = np.concatenate([(np.array(seMean.ravel()).ravel() - 1.9600 * 
                         np.array(seSig2.ravel()).ravel()),
                        (np.array(seMean.ravel()).ravel() + 1.9600 * 
                         np.array(seSig2.ravel()).ravel())[::-1]])
pl.fill(fillx, filly, alpha=.5, fc='b', ec='None', label='95% confidence interval')
pl.show()
