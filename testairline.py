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

n = 10
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

method = 'COBYLA'

# Noise std. deviation
sn = 1

# Random starts
for itr in range(nItr):
    # Initialize hyperparams
    hypGuess = gp.initHyperParams(Q,x,y)
    # Optimize the guessed hyperparams
    hypGP = gp.GaussianProcess(hyp=hypGuess, inf=infFunc, mean=meanFunc, 
                               cov=covFunc, lik=likFunc, hypLik=np.log(sn),
                               xTrain=x, yTrain=y)
    try:
        optOutput = sopt.minimize(fun=hypGP.train, x0=hypGuess, method=method,
                                  options={'maxiter':100})
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
optOutput = sopt.minimize(fun=hypGP.train, x0=hypInit, method='L-BFGS-B',
                          options={'maxiter':10})
hypTrained = optOutput.x
newNegLogML = optOutput.fun
print "Final hyperparams: ", negLogML
print "Reoptimized: ", newNegLogML
print hypTrained.reshape(3,10)

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
pl.plot(xt, mean, 'k', label=u'Prediction')
pl.show()
