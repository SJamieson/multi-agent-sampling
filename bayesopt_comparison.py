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

video_length = 8  # in seconds

## Sim parameters
pbounds = {'x': (0, 100), 'y': (0, 100)}
kappa = float(get_param(1, 2.576))
xi = 5
num_samples = int(get_param(2, 40))
num_steps = int(get_param(3, 1))
fps = num_samples // video_length
acq = 'ucb'
filename = 'bayesopt-acq={}.{}-ns{}-st{}.mp4'.format(acq, kappa if acq == 'ucb' else xi, num_samples, num_steps)

## bayesopt setup
acq_func = UtilityFunction(acq, kappa=kappa, xi=xi)
optimizer = BayesianOptimization(
    f=caldera_sim_function,
    pbounds=pbounds,
    verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)

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

t = tqdm(total=num_samples + 1, file=sys.stdout)

counter = 0
def update(_):
    global cbar, pos1, marker1, pos2, marker2, xs, ys, t, counter
    reset_loc = False
    next_point = optimizer.suggest(acq_func)
    if counter % (num_steps) == 0:
        next_point = optimizer._space.array_to_params(optimizer._space.random_sample())
        reset_loc = True
    target = caldera_sim_function(**next_point)
    optimizer.register(params=next_point, target=target)

    x, y = next_point['x'], next_point['y']

    # plt.subplot(211)
    xs = np.concatenate((xs, [x]))
    ys = np.concatenate((ys, [y]))
    if marker1 is not None:
        ax1.lines.pop()
    if counter == 0 or not reset_loc:
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
    if counter == 0 or not reset_loc:
        pos2, = ax2.plot(xs[-2:], ys[-2:], color='k')
    marker2, = ax2.plot(x, y, '*m')
    ax2.invert_yaxis()
    # plt.pause(0.1)
    t.update(n=1)
    plt.tight_layout()
    counter += 1
    return marker1, marker2, pos1, pos2, im


anim = animation.FuncAnimation(fig, update, frames=num_samples, blit=True)
anim.save(filename, fps=fps)
# plt.show()