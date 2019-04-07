from bayes_opt import UtilityFunction, BayesianOptimization
from caldera_mcts import caldera_sim_function
import numpy as np
from tqdm import tqdm
import sys
from plotting import draw_caldera_maxima, HeatmapPlot, ContourPlot
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import animation


def get_param(num, default):
    return sys.argv[num] if len(sys.argv) > num else default


video_length = 20  # in seconds
debug = True

## Sim parameters
pbounds = {'x': (0, 100), 'y': (0, 100)}
kappa = float(get_param(1, 2.576))
xi = float(get_param(2, 0))
acq = get_param(3, 'ucb')

type = 'world-model'
# type = 'acq-func'
filename = 'lawnmower' + '-acq={}.{}'.format(acq, kappa if acq == 'ucb' else xi) if type == 'acq-func' else ''
acq_func = UtilityFunction(acq, kappa=kappa, xi=xi)

## Plotting setup
delta = 1
x = np.arange(0, 101.0, delta)
y = np.arange(0, 101.0, delta)
X, Y = np.meshgrid(x, y)

fig = plt.figure()

ax1 = fig.add_subplot(1, 2, 1, adjustable='box', aspect=1.0)
contour_plot = ContourPlot(ax1, 'Depth Map')
contour_plot.draw_contours(X, Y, caldera_sim_function(X, Y), label=True, colorbar=(type == 'acq-func'), levels=12, cmap='Blues')
draw_caldera_maxima(ax1)

ax2 = fig.add_subplot(1, 2, 2, adjustable='box', aspect=1.0)
if type == 'acq-func':
    ax2_plot = HeatmapPlot(ax2, 'Acquisition Function')
else:
    ax2_plot = ContourPlot(ax2, 'World Model')
draw_caldera_maxima(ax2)
xs = np.array([])
ys = np.array([])


def generate_lawnmower_points(step_size):
    def get_row(y, dir):
        if dir == 'left':
            return zip(range(90, 5, -5), iter(lambda: y, -1))
        return zip(range(10, 95, 5), iter(lambda: y, -1))

    def get_skip(x, y1, y2, step_size):
        step_size = 5 * (1 if y2 > y1 else -1)
        return zip(iter(lambda: x, -1), range(y1 + step_size, y2, step_size))

    y1 = 10
    dir = 'left'
    points = list(get_row(y1, dir))
    for y2 in [25, 40, 55, 70, 85]:
        dir = 'left' if dir is 'right' else 'right'
        points.extend(get_skip(points[-1][0], y1, y2, step_size))
        points.extend(get_row(y2, dir))
        y1 = y2
    points = [{'x': p[0], 'y': p[1]} for p in points]
    return points


points = generate_lawnmower_points(15)
fps = len(points) // video_length
t = tqdm(total=len(points), file=sys.stdout)

optimizer = None


def update(frame):
    global points, cbar, pos1, marker1, pos2, marker2, xs, ys, t, optimizer, im, pos, lab
    if frame == 0:
        optimizer = BayesianOptimization(
            f=caldera_sim_function,
            pbounds=pbounds,
            verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            random_state=1,
        )
    reset_loc = False
    next_point = points[frame]
    optimizer.suggest(acq_func)
    target = caldera_sim_function(**next_point)
    optimizer.register(params=next_point, target=target)

    x, y = next_point['x'], next_point['y']

    fig_changes = list()

    xs = np.concatenate((xs, [x]))
    ys = np.concatenate((ys, [y]))
    fig_changes.extend(contour_plot.draw_robot((x, y), connect=(not reset_loc)))
    fig_changes.extend(ax2_plot.draw_robot((x, y), connect=(not reset_loc)))

    if type == 'acq-func':
        scores = acq_func.utility(np.vstack([X.ravel(), Y.ravel()]).transpose(), optimizer._gp, 0).reshape(X.shape)
        fig_changes.extend(ax2_plot.draw_heatmap(scores, colorbar=True, cmap='hot', vmin=0))
    else:
        depth = optimizer._gp.predict(np.vstack([X.ravel(), Y.ravel()]).transpose()).reshape(X.shape)
        fig_changes.extend(ax2_plot.draw_contours(X, Y, depth, label=True, colorbar=False, levels=12, cmap='Blues'))

    t.update(n=1)
    plt.tight_layout()
    return fig_changes


if debug:
    for i in range(len(points)):
        update(i)
        plt.pause(0.05)
    plt.show()
else:
    anim = animation.FuncAnimation(fig, update, save_count=len(points), frames=len(points) - 1,
                                   blit=True if type == 'acq-func' else False)
    # anim.save(filename + '.mp4', fps=fps)
    anim.save(filename + '.gif', writer='imagemagick', fps=fps)
