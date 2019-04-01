from caldera_mcts import *
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
start = [80, 20]
kappa = float(get_param(1, 2.576))
xi = 0
max_depth = int(get_param(2, 5))
num_actions = 180
fps = num_actions // video_length
step_size = int(get_param(3, 4))
can_backtrack = False
acq = 'ucb'
filename = 'single-acq={}.{}-{}x{}y-d{}-na{}-ss{}.mp4'.format(acq, kappa if acq == 'ucb' else xi, start[0],
                                                              start[1], max_depth, num_actions, step_size)

## MCTS setup
acq_func = UtilityFunction(acq, kappa=kappa, xi=xi)
mcts = mcts(timeLimit=max_depth * 300)
samples = dict()
robot_state = RobotState(*start, pbounds, samples, acq_func, max_depth, step_size)

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
cbar, marker1, marker2 = None, None, None

t = tqdm(total=num_actions + 1, file=sys.stdout)


def update(_):
    global samples, cbar, marker1, marker2, xs, ys, robot_state, t
    robot_state = mcts_state_update(mcts, robot_state, samples, sample_func=caldera_sim_function)

    # plt.subplot(211)
    xs = np.concatenate((xs, [robot_state.x]))
    ys = np.concatenate((ys, [robot_state.y]))
    if marker1 is not None:
        ax1.lines.pop()
    pos1, = ax1.plot(xs[-2:], ys[-2:], color='k')
    marker1, = ax1.plot(robot_state.x, robot_state.y, '*m')
    # plt.pause(0.1)

    # plt.subplot(212)
    scores = acq_func.utility(np.vstack([X.ravel(), Y.ravel()]).transpose(), robot_state.gp, 0)
    scores = scores.reshape(X.shape) * robot_state.y_max
    if marker2 is not None:
        ax2.lines.pop()
    im = ax2.imshow(scores, cmap='hot', interpolation='nearest')
    im.set_clim(vmin=0)
    if cbar is not None:
        cbar.remove()
    cbar = plt.colorbar(im, ax=ax2)
    pos2, = ax2.plot(xs[-2:], ys[-2:], color='k')
    marker2, = ax2.plot(robot_state.x, robot_state.y, '*m')
    ax2.invert_yaxis()
    # plt.pause(0.1)
    t.update(n=1)
    plt.tight_layout()
    return marker1, marker2, pos1, pos2, im


anim = animation.FuncAnimation(fig, update, frames=num_actions, blit=True)
anim.save(filename, fps=fps)
