PyGPML
======

Python version of Carl Rasmussen and Hannes Nickisch for Gaussian Processes. Their code can be found here: 

http://www.gaussianprocess.org/gpml/code/matlab/doc/

This repo, so far, is implementing a small subset of the original MATLAB code from above. It implements a spectral mixture (SM) kernel for pattern recognition given by Andrew G. Wilson and Ryan P. Adams in the following references:

[1] http://arxiv.org/abs/1302.4245

[2] http://arxiv.org/abs/1310.5288

And a resource page for this work is given here:

http://mlg.eng.cam.ac.uk/andrew/pattern/

Briefly, it implements a typical gaussian process with a covariance kernel function that is a mixture of Gaussians:

k(t) = sum_{q=1}^Q w_q prod_{p=1}^P exp(-2pi^2 t_p^2 v_{p,q}^2) cos(2pi t_p m_{p,q})

where t = x-x', q = ith out of Q Gaussians in the mixture, p = jth out of P dimensions, w = weighting of qth Gaussian mixture, v2 = v^2 = std. deviation, and m = means.

This form allows the SM kernel to approximate any kernel with enough Gaussians in the mixture and to capture any consistent pattern in the data. An example problem using the airline data from [1] is given here.

Code has been tested with Python 2.7 only. Requires SciPy, NumPy, and matplotlib.
