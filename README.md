PyGPML
======

Python version of Carl Rasmussen and Hannes Nickisch for Gaussian Processes. Their code can be found here: 

http://www.gaussianprocess.org/gpml/code/matlab/doc/

This repo, so far, is implementing a small subset of the original MATLAB code from above. It mainly implements a Gaussian process with Gaussian noise, making the maximum likelihood integral analytically solveable exactly. The corresponding function is given in inferences.py.

There are a few standard built-in kernels, but this code also implements a spectral mixture (SM) kernel for pattern recognition (which was the original motivation for this code) given by Andrew G. Wilson and Ryan P. Adams in the following references:

[1] http://arxiv.org/abs/1302.4245

[2] http://arxiv.org/abs/1310.5288

And a resource page for this work is given here:

http://mlg.eng.cam.ac.uk/andrew/pattern/

Briefly, it implements a typical gaussian process with a covariance kernel function that is a mixture of Gaussians:

k(t) = sum_{q=1}^Q w_q prod_{p=1}^P exp(-2pi^2 t_p^2 v_{p,q}^2) cos(2pi t_p m_{p,q})

where t = x-x', q = ith out of Q Gaussians in the mixture, p = jth out of P dimensions, w = weighting of qth Gaussian mixture, v2 = v^2 = std. deviation, and m = means.

This form allows the SM kernel to approximate any kernel with enough Gaussians in the mixture and to capture any consistent pattern in the data. An example problem using the airline data from [1] is given here as well as the atmospheric CO2 problem.

For any model functions, like inference, likelihood, mean, and kernel/covariance, the user can specify a built-in function included with the code, or define their own (using the built-in functions as a template as the arguments and return types obviously must be the same).

Code has been tested with Python 2.7 only. Requires SciPy, NumPy, and matplotlib.


## Testing

To test the code, simply navigate to the testing/ directory, edit addpath.py accordingly with the absolute path of PyGPML (it will add it to your python path), and then simply run 'nosetests', or 'nosetests -sv' to see what's going on. Obviously this requires nosetests to be installed.


## Usage

The example scripts are probably the best place to start understanding how to use the code, and they're found in the examples/ dir. Some of them have data, which is included in examples/data/. There is a file called addpath.py that requires you to edit in the absolute path of PyGPML folder so that it can add it to your python-path. Same goes for the tests.

Feel free to ping me for help: had sed (at) google mail
