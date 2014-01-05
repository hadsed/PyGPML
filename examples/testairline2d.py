import numpy as np
import pylab as pl
from scipy import io as sio
from scipy import optimize as sopt

# Adds the GP code directory to the system path
# so we can call the library from this subdir.
import addpath
import gaussian_process as gp

data = sio.loadmat('data/airlinedata.mat')

x1 = np.matrix(data['xtrain'])
x2 = np.matrix(data['xtrain'])
x = np.hstack((x1,x2))

y1 = np.matrix(data['ytrain'])
y2 = np.matrix(data['ytrain']) + 500
y = np.hstack((y1,y2))

x1t = np.matrix(data['xtest'])
x2t = np.matrix(data['xtest'])
xt = np.hstack((x1t,x2t))

y1t = np.matrix(data['ytest'])
y2t = np.matrix(data['ytest']) + 500
yt = np.hstack((y1t,y2t))

# To get interpolation too
#xt = np.concatenate((x,xt))
#yt = np.concatenate((y,yt))

negLogML = np.inf
hypInit = None
nItr = 10
Q = 10

# Define core functions
likFunc = 'gaussian'
meanFunc = 'zero'
infFunc = 'exact'
covFunc = 'spectral_mixture'

# Set the optimizer types and options
# l1 is for the random starts, l2 does more
# optimization for the best one from l1.
l1Optimizer = 'COBYLA'
l1Options = {'maxiter':100}
l2Optimizer = 'L-BFGS-B'
l2Options = {'maxiter':100}

# Noise std. deviation
sn = 1

# Initialize hyperparams
initArgs = {'Q':Q,'x':x,'y':y, 'sn':sn}
hypGuess = gp.core.initSMParams(**initArgs)
# Initialize GP object
hypGP = gp.GaussianProcess(hyp=hypGuess, inf=infFunc, mean=meanFunc, cov=covFunc, 
                           lik=likFunc, xtrain=x, ytrain=y, xtest=xt)
# Random starts
for itr in range(nItr):
    # Start over
    hypGuess = gp.core.initSMParams(**initArgs)
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
print "Noise parameter: ", np.exp(hypGP.hyp['lik'])
print "Reoptimized: ", hypGP.nlml
print np.exp(hypGP.hyp['cov'])

# Do the extrapolation
prediction = hypGP.predict()
mean = prediction['ymu']
sigma2 = prediction['ys2']

# Plot the stuff
pl.plot(x, y, 'b', label=u'Training Data')
pl.plot(xt, yt, 'k', label=u'Test Data')
pl.plot(xt, mean, 'r', label=u'SM Prediction')
sigma = np.power(sigma2, 0.5)
# fillx = np.concatenate([np.array(xt.ravel()).ravel(), 
#                         np.array(xt.ravel()).ravel()[::-1]])
# filly = np.concatenate([(np.array(mean.ravel()).ravel() - 1.9600 * 
#                          np.array(sigma.ravel()).ravel()),
#                         (np.array(mean.ravel()).ravel() + 1.9600 * 
#                          np.array(sigma.ravel()).ravel())[::-1]])
# pl.fill(fillx, filly, alpha=.5, fc='0.5', ec='None', 
#         label='95% confidence interval')
pl.show()
