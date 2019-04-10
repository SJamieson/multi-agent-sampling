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
start = [[25, 20], [85, 25], [50, 75]]
kappa = float(get_param(1, 10))  # 2.576
xi = 0
max_tree_depth = int(get_param(2, 4))
num_actions = int(get_param(4, 60))
num_agents = 3
fps = num_actions // video_length
step_size = int(get_param(3, 8))
can_backtrack = False
acq = 'ucb'
# type = 'indep'
# type = 'serial'
type = 'partition'
filename = '{}-acq={}.{}-{}x{}y-d{}-na{}-ss{}'.format(type, acq, kappa if acq == 'ucb' else xi, start[0],
                                                      start[1], max_tree_depth, num_actions, step_size)

## MCTS setup
acq_func = UtilityFunction(acq, kappa=kappa, xi=xi)
# mcts = mcts(timeLimit=max_depth * 600)
mcts = mcts(iterationLimit=16 * (max_tree_depth ** 2))
samples, robot_state = None, [None] * num_agents

pbounds_1 = {'x': (0, 50), 'y': (0, 50)}
pbounds_2 = {'x': (50, 100), 'y': (0, 50)}
pbounds_3 = {'x': (0, 100), 'y': (50, 100)}
pbounds = {0: pbounds_1, 1: pbounds_2, 2: pbounds_3} if type == 'partition' else pbounds

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
    global samples, robot_state
    if frame == 0:
        samples = dict()
        for i in range(num_agents):
            if type == 'partition':
                bounds = pbounds[i]
            else:
                bounds = pbounds
            robot_state[i] = RobotState(*start[i], bounds, samples, acq_func, max_tree_depth, step_size)
    elif type == 'serial':
        sim_samples = deepcopy(samples)  # will store simulated samples of previous agents
        for i in range(num_agents):
            if i > 0:
                # Figure out where the previous robot chose to go
                previous_robot_target = robot_state[i - 1].x, robot_state[i - 1].y
                # Predict what they will sample at that location
                predicted_sample = robot_state[i].gp.predict(np.array(previous_robot_target).reshape(1, -1))
                # Add that simulated sample to the list of sim samples
                sim_samples[previous_robot_target] = predicted_sample
            # Replace the current robot's samples with the simulated samples generated so far
            sim_state = robot_state[i].renew(sim_samples)
            # Tell the current robot to actually perform their sample
            # IMPORTANT: Note that `samples` is passed here instead of `sim_samples`. This means that the result
            # of the real sample will be put into `samples`, and will not affect other robots' simulations
            robot_state[i] = mcts_state_update(mcts, sim_state, samples, sample_func=caldera_sim_function)
        # We no longer care about sim_samples, so let's clean it up
        del sim_samples
    elif type == 'indep' or type == 'partition':
        for i in range(num_agents):
            if i not in samples:
                samples[i] = dict()
            robot_state[i] = mcts_state_update(mcts, robot_state[i], samples[i], sample_func=caldera_sim_function)
    else:
        raise RuntimeError('Unknown type')

    fig_changes = list()

    for i in range(num_agents):
        fig_changes.extend(contour_plot.draw_robot((robot_state[i].x, robot_state[i].y), connect=True, index=i))
        fig_changes.extend(acq_plot.draw_robot((robot_state[i].x, robot_state[i].y), connect=True, index=i))

    # Use the last robot to get the updated model
    scores = acq_func.utility(np.vstack([X.ravel(), Y.ravel()]).transpose(), robot_state[-1].gp, 0)
    scores = scores.reshape(X.shape) * robot_state[-1].y_max
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
