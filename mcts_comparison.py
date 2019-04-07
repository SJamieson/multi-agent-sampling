from caldera_mcts import *
from tqdm import tqdm
import sys
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import animation
from plotting import ContourPlot, HeatmapPlot, draw_caldera_maxima, caldera_sim_function


def get_param(num, default):
    return sys.argv[num] if len(sys.argv) > num else default


video_length = 12  # in seconds
debug = False

## Sim parameters
pbounds = {'x': (0, 100), 'y': (0, 100)}
start = [80, 20]
kappa = float(get_param(1, 2.576))
xi = 0
max_tree_depth = int(get_param(2, 5))
num_actions = int(get_param(4, 60))
fps = num_actions // video_length
step_size = int(get_param(3, 4))
can_backtrack = False
acq = 'ucb'
filename = 'single-acq={}.{}-{}x{}y-d{}-na{}-ss{}'.format(acq, kappa if acq == 'ucb' else xi, start[0],
                                                          start[1], max_tree_depth, num_actions, step_size)

## MCTS setup
acq_func = UtilityFunction(acq, kappa=kappa, xi=xi)
# mcts = mcts(timeLimit=max_depth * 600)
mcts = mcts(iterationLimit=16 * (max_tree_depth ** 2))
samples, robot_state = None, None

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

t = tqdm(total=num_actions + 1, file=sys.stdout)


def update(frame):
    global samples, xs, ys, robot_state
    if frame == 0:
        samples = dict()
        robot_state = RobotState(*start, pbounds, samples, acq_func, max_tree_depth, step_size)
    else:
        robot_state = mcts_state_update(mcts, robot_state, samples, sample_func=caldera_sim_function)

    fig_changes = list()

    xs = np.concatenate((xs, [robot_state.x]))
    ys = np.concatenate((ys, [robot_state.y]))
    fig_changes.extend(contour_plot.draw_robot((robot_state.x, robot_state.y), connect=True))
    fig_changes.extend(acq_plot.draw_robot((robot_state.x, robot_state.y), connect=True))

    scores = acq_func.utility(np.vstack([X.ravel(), Y.ravel()]).transpose(), robot_state.gp, 0)
    scores = scores.reshape(X.shape) * robot_state.y_max
    fig_changes.extend(acq_plot.draw_heatmap(scores, colorbar=True, cmap='hot', vmin=0))

    t.update(n=1)
    plt.tight_layout()
    return fig_changes


if debug:
    for i in range(num_actions):
        update(i)
        plt.pause(0.1)
    plt.show()
else:
    anim = animation.FuncAnimation(fig, update, save_count=num_actions, frames=num_actions, blit=True)
    # anim.save(filename + '.mp4', fps=fps)
    anim.save(filename + '.gif', writer='imagemagick', fps=fps)
