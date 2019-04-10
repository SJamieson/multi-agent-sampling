# Author: Vincent Dubourg <vincent.dubourg@gmail.com>
#         Jake Vanderplas <vanderplas@astro.washington.edu>
# License: BSD style
# Adapted by: Stewart Jamieson

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from matplotlib import pyplot as pl
from matplotlib import animation

np.random.seed(1)
fps = 1
filename = 'gp_demo'


def f(x):
    """The function to predict."""
    return np.abs(x + 2 * np.sin(x))


# ----------------------------------------------------------------------
#  First the noiseless case
X = np.atleast_2d([3., 4., 5., 6., 7., 8.]).T

# Observations
y = f(X).ravel()

# Mesh the input space for evaluations of the real function, the prediction and
# its MSE
x = np.atleast_2d(np.linspace(0, 10, 1000)).T

# Instanciate a Gaussian Process model
gp = GaussianProcessRegressor(kernel=RBF(length_scale=.25))

fig = pl.figure()
pl.plot(x, f(x), 'r:', label=r'Seafloor Depth ($|x\,+\, 2\sin(x)|$)')
pl.xlabel('$x$')
pl.ylabel('$f(x)$')
pl.ylim(0, 20)

lines = []

def draw(frame):
    global lines
    for l in lines:
        l.remove()
    lines = []
    num_pts = frame + 1
    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(X[:num_pts, :], y[:num_pts] / np.max(f(x)))

    # Make the prediction on the meshed x-axis (ask for MSE as well)
    y_pred, sigma = gp.predict(x, return_std=True)
    y_pred, sigma = y_pred * np.max(f(x)), sigma * np.max(f(x))

    # Plot the function, the prediction and the 95% confidence interval based on
    # the MSE
    lines.extend(pl.plot(X[:num_pts, :], y[:num_pts], 'r.', markersize=10, label=u'Observations'))
    lines.extend(pl.plot(x, y_pred, 'b-', label=u'Prediction'))
    lines.extend(pl.fill(np.concatenate([x, x[::-1]]),
                         np.concatenate([y_pred - 1 * sigma,
                                         (y_pred + 1 * sigma)[::-1]]),
                         alpha=.5, fc='b', ec='None', label=r'1$\sigma$ confidence interval'))
    pl.legend(loc='upper left')


anim = animation.FuncAnimation(fig, draw, frames=len(y), blit=False)
anim.save(filename + '.gif', writer='imagemagick', fps=fps)
