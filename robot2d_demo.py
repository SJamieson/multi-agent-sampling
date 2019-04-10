# Author: Vincent Dubourg <vincent.dubourg@gmail.com>
#         Jake Vanderplas <vanderplas@astro.washington.edu>
# License: BSD style
# Adapted by: Stewart Jamieson

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from matplotlib import pyplot as pl
from matplotlib import animation
from bayes_opt import UtilityFunction
from bayes_opt.util import acq_max

np.random.seed(1)
fps = .5
start = 3
num_pts = 8
kappa = float(1)
acq_func = UtilityFunction('ucb', kappa=kappa, xi=0)
lookahead = 5

filename = '{}_demo_k={}_steps={}'.format('robot2d', kappa, lookahead)


def f(x):
    """The function to predict."""
    return np.abs(x + np.sin(x) * 4)


# ----------------------------------------------------------------------
# Mesh the input space for evaluations of the real function, the prediction and
# its MSE
x = np.atleast_2d(np.linspace(0, 10, 1000)).T

fig = pl.figure()
pl.plot(x, f(x), 'r:', label=r'Seafloor Depth ($|x\,+\, 4 \sin(x)|$)')
pl.xlabel('$x$')
pl.ylabel('$f(x)$')
pl.ylim(0, 20)

lines = []
points = []
bayesopt, gp = None, None

def fit_gp(X, gp):
    Xfit = np.array(list(set(list(X[:, 0])))).reshape(-1, 1)
    gp.fit(Xfit, f(Xfit) / np.max(f(x)))


def draw(frame):
    global lines, points, bayesopt, gp
    num_pts = frame + 1
    if frame == 0:
        points = [[start]]
        # Instanciate a Gaussian Process model
        # bayesopt = BayesianOptimization(f, {'x': (0, 10)}, random_state=100)
        # bayesopt.set_gp_params(kernel=RBF(length_scale=.25))
        gp = GaussianProcessRegressor(kernel=RBF(length_scale=.25))
    for l in lines:
        l.remove()

    X = np.array(points, dtype=np.float64).reshape(-1, 1)
    # bayesopt.register(points[-1], f(points[-1][0]))
    lines = []
    # Fit to data using Maximum Likelihood Estimation of the parameters
    fit_gp(X, gp)

    # Make the prediction on the meshed x-axis (ask for MSE as well)
    y_pred, sigma = gp.predict(x, return_std=True)
    y_pred, sigma = y_pred * np.max(f(x)), sigma * np.max(f(x))
    sigma = sigma.reshape(y_pred.shape)
    acq = acq_func.utility(x, gp, np.max(f(x))) * np.max(f(x))

    current_x = X[-1, 0]

    # Plot the function, the prediction and the 95% confidence interval based on
    # the MSE
    lines.extend(pl.plot(X, f(X), 'r.', markersize=10))
    lines.extend(pl.plot(x, y_pred, 'b-'))
    lines.extend(pl.plot(x, acq, color='orange', label=r'UCB ($\kappa = {}$)'.format(kappa)))
    xfill = np.concatenate([x, x[::-1]])
    yfill = np.concatenate([y_pred - 1 * sigma, (y_pred + 1 * sigma)[::-1]])
    lines.extend(pl.fill(xfill, yfill,
                         alpha=.5, fc='b', ec='None'))
    lines.extend(pl.plot([X[-1, 0], X[-1, 0]], [0, 20], '--m', label='Robot Location'))
    pl.legend(loc='upper left')
    left_score, right_score = 0, 0
    left_lookahead, right_lookahead = int(np.min([lookahead, current_x])), int(np.min([lookahead, 10 - current_x]))
    if current_x + right_lookahead <= 10:
        lines.extend(pl.plot(current_x + right_lookahead, f(current_x + right_lookahead), 'g*'))
        for i in range(1, 1 + right_lookahead):
            right_score += acq_func.utility(np.array(current_x + i).reshape(-1, 1), gp, 0)
            fake_X = X + [current_x + i]
            fit_gp(fake_X, gp)
    if current_x - left_lookahead >= 0:
        lines.extend(pl.plot(current_x - left_lookahead, f(current_x - left_lookahead), 'g*'))
        for i in range(1, 1 + left_lookahead):
            left_score += acq_func.utility(np.array(current_x - i).reshape(-1, 1), gp, 0)
            fake_X = X + [current_x - i]
            fit_gp(fake_X, gp)
    fit_gp(X, gp)

    points.append([current_x + (1 if right_score >= left_score else -1)])


anim = animation.FuncAnimation(fig, draw, frames=num_pts, blit=False)
anim.save(filename + '.gif', writer='imagemagick', fps=fps)
