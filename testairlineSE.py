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
xt = np.concatenate((x,xt))
yt = np.concatenate((y,yt))
# Define core functions
likFunc = 'likGauss'
meanFunc = 'meanZero'
infFunc = 'infExact'
covFunc = 'covSE'

# Now try to do a vanilla isotropic Gaussian kernel
seOptimizer = 'COBYLA'
sn = 0.1
hypSEInit = np.log([40., np.std(y), sn])
seGP = gp.GaussianProcess(hyp=hypSEInit, inf=infFunc, mean=meanFunc, 
                          cov=covFunc, lik=likFunc, hypLik=np.log(sn), 
                          xTrain=x, yTrain=y)
#seGP.train(hypSEInit)
optSE = sopt.minimize(fun=seGP.train, x0=hypSEInit, method=seOptimizer,
                      options={'maxiter':1000})
optSEx = optSE.x
optSEfun = optSE.fun
#optSEx = [ 5.0796, 5.9429 ]
#optSEfun = seGP.train(optSEx)
seFitted = gp.GaussianProcess(hyp=optSEx, inf=infFunc, mean=meanFunc, 
                              cov=covFunc, lik=likFunc, hypLik=np.log(sn), 
                              xTrain=x, yTrain=y, xTest=xt)
sePred = seFitted.predict()
seMean = sePred['ymu']
seSig2 = sePred['ys2']

print "Optimized SE likelihood: ", optSEfun
print "SE hyperparams: ", optSEx

# Plot the stuff
pl.plot(x, y, 'b', label=u'Training Data')
pl.plot(xt, yt, 'r', label=u'Test Data')
pl.plot(xt, seMean, 'g', label=u'SE Prediction')
pl.show()
