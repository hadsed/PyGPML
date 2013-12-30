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
xt = np.concatenate((x,xt))
yt = np.concatenate((y,yt))
# Define core functions
likFunc = 'likGauss'
meanFunc = 'meanZero'
infFunc = 'infExact'
covFunc = 'radial_basis'
# Now try to do a vanilla isotropic Gaussian kernel
seOptimizer = 'COBYLA'
sn = 0.1
hypSEInit = np.log([np.std(y), 40., sn])
seGP = gp.GaussianProcess(hyp=hypSEInit, inf=infFunc, mean=meanFunc, 
                          cov=covFunc, lik=likFunc, hypLik=np.log(sn), 
                          xTrain=x, yTrain=y)
optSE = sopt.minimize(fun=seGP.train, x0=hypSEInit, method=seOptimizer,
                      options={'maxiter':1000})
optSEx = optSE.x
optSEfun = optSE.fun
seFitted = gp.GaussianProcess(hyp=optSEx, inf=infFunc, mean=meanFunc, 
                              cov=covFunc, lik=likFunc, hypLik=np.log(sn), 
                              xTrain=x, yTrain=y, xTest=xt)
sePred = seFitted.predict()
seMean = sePred['ymu']
seSig2 = sePred['ys2']

print "Optimized SE likelihood: ", optSEfun
print "SE hyperparams: ", optSEx

# Plot the stuff
pl.plot(xt, yt, 'r', label=u'Test Data')
pl.plot(x, y, 'b', label=u'Training Data')
pl.plot(xt, seMean, 'g', label=u'SE Prediction')
sigma = np.power(seSig2, 0.5)
fillx = np.concatenate([np.array(xt.ravel()).ravel(), 
                        np.array(xt.ravel()).ravel()[::-1]])
filly = np.concatenate([(np.array(seMean.ravel()).ravel() - 1.9600 * 
                         np.array(sigma.ravel()).ravel()),
                        (np.array(seMean.ravel()).ravel() + 1.9600 * 
                         np.array(sigma.ravel()).ravel())[::-1]])
pl.fill(fillx, filly, alpha=.5, fc='0.5', ec='None', label='95% confidence interval')
pl.show()
