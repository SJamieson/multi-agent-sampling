from bayes_opt import UtilityFunction, BayesianOptimization
from caldera_mcts import caldera_sim_function
import numpy as np
from tqdm import tqdm
import sys
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import animation

def get_param(num, default):
    return sys.argv[num] if len(sys.argv) > num else default

video_length = 12  # in seconds

## Sim parameters
pbounds = {'x': (0, 100), 'y': (0, 100)}
kappa = float(get_param(1, 2.576))
xi = float(get_param(2, 0))
acq = get_param(3, 'ucb')

filename = 'lawnmower-acq={}.{}'.format(acq, kappa if acq == 'ucb' else xi)
## bayesopt setup
acq_func = UtilityFunction(acq, kappa=kappa, xi=xi)

## Plotting setup
delta = 1
x = np.arange(0, 101.0, delta)
y = np.arange(0, 101.0, delta)
X, Y = np.meshgrid(x, y)
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1, adjustable='box', aspect=1.0)
ax1.title.set_text('Depth Map')
CS = plt.contour(X, Y, caldera_sim_function(X, Y), levels=12, cmap='Blues')
plt.clabel(CS, inline=1, fontsize=8, fmt='%.3g')
plt.colorbar(CS, ax=ax1)
ax1.plot(20, 46, 'bx')
ax1.plot(79, 79, 'bx')
ax2 = fig.add_subplot(2, 1, 2, adjustable='box', aspect=1.0)
ax2.plot(20, 46, 'bx')
ax2.plot(79, 79, 'bx')
ax2.title.set_text('Acquisition Function')
xs = np.array([])
ys = np.array([])
cbar, pos1, marker1, pos2, marker2 = None, None, None, None, None

sample_xs = np.arange(80, 15, -5)
step_size = 10
def get_row(y, dir):
    if dir == 'left':
        return zip(range(80, 15, -5), iter(lambda: y, -1))
    return zip(range(20, 85, 5), iter(lambda: y, -1))
def get_skip(x, y1, y2):
    step_size = 5 * (1 if y2 > y1 else -1)
    return zip(iter(lambda: x, -1), range(y1 + step_size, y2, step_size))
y1 = 20
dir = 'left'
points = list(get_row(y1, dir))
for y2 in [35, 50, 65, 80]:
    dir = 'left' if dir is 'right' else 'right'
    points.extend(get_skip(points[-1][0], y1, y2))
    points.extend(get_row(y2, dir))
    y1 = y2
fps = len(points) // video_length
print(points)
points = [{'x': p[0], 'y': p[1]} for p in points]

t = tqdm(total=len(points), file=sys.stdout)

optimizer = None
def update(frame):
    global points, cbar, pos1, marker1, pos2, marker2, xs, ys, t, optimizer
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

    # plt.subplot(211)
    xs = np.concatenate((xs, [x]))
    ys = np.concatenate((ys, [y]))
    if marker1 is not None:
        ax1.lines.pop()
    if frame == 0 or not reset_loc:
        pos1, = ax1.plot(xs[-2:], ys[-2:], color='k')
    marker1, = ax1.plot(x, y, '*m')
    # plt.pause(0.1)

    # plt.subplot(212)
    scores = acq_func.utility(np.vstack([X.ravel(), Y.ravel()]).transpose(), optimizer._gp, 0)
    scores = scores.reshape(X.shape)
    if marker2 is not None:
        ax2.lines.pop()
    im = ax2.imshow(scores, cmap='hot', interpolation='nearest')
    im.set_clim(vmin=0)
    if cbar is not None:
        cbar.remove()
    cbar = plt.colorbar(im, ax=ax2)
    if frame == 0 or not reset_loc:
        pos2, = ax2.plot(xs[-2:], ys[-2:], color='k')
    marker2, = ax2.plot(x, y, '*m')
    ax2.invert_yaxis()
    # plt.pause(0.1)
    t.update(n=1)
    plt.tight_layout()
    return marker1, marker2, pos1, pos2, im


anim = animation.FuncAnimation(fig, update, save_count=len(points), frames=len(points)-1, blit=True)
# anim.save(filename + '.mp4', fps=fps)
anim.save(filename + '.gif', writer='imagemagick', fps=fps)
# plt.show()