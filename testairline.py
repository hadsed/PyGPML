import numpy as np
import pylab as pl
from scipy import io as sio
from scipy import optimize as sopt

import GaussianProcess as gp

data = sio.loadmat('airlinedata.mat')

x = np.matrix(data['xtrain'])
y = np.matrix(data['ytrain'])
xt = np.matrix(data['xtest'])
yt = np.matrix(data['ytest'])
# To get interpolation too
#xt = np.concatenate((x,xt))
#yt = np.concatenate((y,yt))

skipSM = False
n = 10 if not skipSM else 1
D = 1
Q = 10

negLogML = np.inf
nItr = 10
hypInit = []
hypTrained = []

# Define core functions
likFunc = 'likGauss'
meanFunc = 'meanZero'
infFunc = 'infExact'
covFunc = 'covSM'

l1Optimizer = 'COBYLA'
l1Options = {'maxiter':100 if not skipSM else 1}
l2Optimizer = 'L-BFGS-B'
#l2Optimizer = 'COBYLA'
l2Options = {'maxiter':100 if not skipSM else 1}

# Noise std. deviation
sn = 1

# Random starts
for itr in range(nItr):
    # Initialize hyperparams
    hypGuess = gp.initHyperParams(Q,x,y,sn)
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

# Now try to do a vanilla isotropic Gaussian kernel
seOptimizer = 'COBYLA'
covFunc = 'covSE'
sn = 0.1
hypSEInit = np.log([40., np.std(y), sn])
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
pl.show()
