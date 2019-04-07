from bayes_opt import UtilityFunction, BayesianOptimization
from plotting import HeatmapPlot, ContourPlot, draw_caldera_maxima, caldera_sim_function
import numpy as np
from tqdm import tqdm
import sys
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import animation


def get_param(num, default):
    return sys.argv[num] if len(sys.argv) > num else default


video_length = 8  # in seconds
debug = False

## Sim parameters
pbounds = {'x': (0, 100), 'y': (0, 100)}
kappa = float(get_param(1, 2.576))
xi = 5
num_samples = int(get_param(2, 8))
num_steps = int(get_param(3, 1))
fps = num_samples // video_length
acq = 'ucb'
filename = 'bayesopt-acq={}.{}-ns{}-st{}'.format(acq, kappa if acq == 'ucb' else xi, num_samples, num_steps)

## bayesopt setup
acq_func = UtilityFunction(acq, kappa=kappa, xi=xi)

## Plotting setup
delta = 1
x = np.arange(0, 101.0, delta)
y = np.arange(0, 101.0, delta)
X, Y = np.meshgrid(x, y)

fig = plt.figure()

ax1 = fig.add_subplot(1, 2, 1, adjustable='box', aspect=1.0)
contour_plot = ContourPlot(ax1, 'Depth Map')
contour_plot.draw_contours(X, Y, caldera_sim_function(X, Y), label=True, colorbar=True, levels=12, cmap='Blues')
draw_caldera_maxima(ax1)

ax2 = fig.add_subplot(1, 2, 2, adjustable='box', aspect=1.0)
acq_plot = HeatmapPlot(ax2, 'Acquisition Function')
draw_caldera_maxima(ax2)
xs = np.array([])
ys = np.array([])

t = tqdm(total=num_samples + 1, file=sys.stdout)

optimizer = None


def update(frame):
    global xs, ys, optimizer
    if frame == 0:
        optimizer = BayesianOptimization(
            f=caldera_sim_function,
            pbounds=pbounds,
            verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            random_state=1,
        )
    reset_loc = False
    next_point = optimizer.suggest(acq_func)
    if frame % (num_steps) == 0:
        next_point = optimizer._space.array_to_params(optimizer._space.random_sample())
        reset_loc = True
    target = caldera_sim_function(**next_point)
    optimizer.register(params=next_point, target=target)

    x, y = next_point['x'], next_point['y']

    # plt.subplot(211)
    fig_changes = list()

    xs = np.concatenate((xs, [x]))
    ys = np.concatenate((ys, [y]))
    fig_changes.extend(contour_plot.draw_robot((x, y), connect=(not reset_loc)))
    fig_changes.extend(acq_plot.draw_robot((x, y), connect=(not reset_loc)))
    # plt.pause(0.1)

    # plt.subplot(212)
    scores = acq_func.utility(np.vstack([X.ravel(), Y.ravel()]).transpose(), optimizer._gp, 0)
    scores = scores.reshape(X.shape)
    fig_changes.extend(acq_plot.draw_heatmap(scores, colorbar=True, cmap='hot', vmin=0))

    t.update(n=1)
    plt.tight_layout()
    return fig_changes


if debug:
    for i in range(num_samples):
        update(i)
        plt.pause(0.1)
    plt.show()
else:
    anim = animation.FuncAnimation(fig, update, save_count=num_samples, frames=num_samples, blit=True)
    anim.save(filename + '.gif', writer='imagemagick', fps=fps, savefig_kwargs={'bbox_inches': 'tight'})
