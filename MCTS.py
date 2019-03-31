from mcts import mcts
from bayes_opt.bayesian_optimization import *
from copy import deepcopy
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor

import warnings
warnings.filterwarnings("ignore", category=matplotlib.cbook.MatplotlibDeprecationWarning)

pbounds = {'x': (0, 100), 'y': (0, 100)}
start = [80, 50]
kappa = 2.576
xi = 0
max_depth = 7
num_actions = 150
fps = num_actions / 10
step_size = 4
can_backtrack = False
acq = 'ucb'
filename = 'singleagent-acq={}.{}-{}x{}y-d{}-na{}-ss{}.mp4'.format(acq, kappa if acq == 'ucb' else xi, start[0],
                                                                   start[1], max_depth, num_actions, step_size)

def caldera_sim_function(x, y):
    x, y = x / 10.0, y / 10.0
    z0 = mlab.bivariate_normal(x, y, 10.0, 5.0, 5.0, 0.0)
    z1 = mlab.bivariate_normal(x, y, 1.0, 2.0, 2.0, 5.0)
    z2 = mlab.bivariate_normal(x, y, 1.7, 1.7, 8.0, 8.0)
    return 50000.0 * z0 + 2500.0 * z1 + 5000.0 * z2

ucb = UtilityFunction('ucb', kappa=kappa, xi=0)
ei = UtilityFunction('ei', kappa=0, xi=0)
acq_func = ucb if acq == 'ucb' else ei

class RobotState:
    def __init__(self, x, y, samples, depth=0, base_reward=0,
                 dir='north', kernel=RBF(length_scale=5)):
        self.x = x
        self.y = y
        self.samples = deepcopy(samples)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-3, normalize_y=False, n_restarts_optimizer=25,
            random_state=1)
        self.depth = depth
        self.base_reward = base_reward
        self.dir = dir
        self.y_max = 1
        if len(samples) > 0:
            xdata = np.vstack(list(self.samples.keys()))
            ydata = np.vstack(list(self.samples.values()))
            self.y_max = np.max(ydata)
            self.gp.fit(xdata, ydata.reshape(-1) / self.y_max)

    @staticmethod
    def _invert_dir(dir):
        return {'north': 'south', 'south': 'north',
                'west':'east', 'east':'west'}[dir]

    def renew(self, samples):
        return RobotState(self.x, self.y, samples, dir=self.dir, kernel=self.gp.kernel)

    def getPossibleActions(self):
        actions = []
        if self.x >= pbounds['x'][0] + step_size:
            actions.append('west')
        if self.x <= pbounds['x'][1] - step_size:
            actions.append('east')
        if self.y >= pbounds['y'][0] + step_size:
            actions.append('south')
        if self.y <= pbounds['y'][1] - step_size:
            actions.append('north')
        if not can_backtrack and len(actions) > 1 and self._invert_dir(self.dir) in actions:
            actions.remove(self._invert_dir(self.dir))
        return actions

    def takeAction(self, action):
        new_samples = deepcopy(self.samples)
        new_samples[(self.x, self.y)] = self.getReward()
        new_state = RobotState(self.x, self.y, new_samples,
                               depth=self.depth + 1,
                               base_reward=self.base_reward + self.getReward(),
                               dir=action)
        if action == 'west':
            new_state.x -= step_size
        if action == 'east':
            new_state.x += step_size
        if action == 'north':
            new_state.y += step_size
        if action == 'south':
            new_state.y -= step_size
        return new_state

    def isTerminal(self):
        global max_depth
        return self.depth >= max_depth

    def getReward(self):
        global acq_func
        reward, = acq_func.utility(np.array([self.x, self.y]).reshape(1, -1),
                                   self.gp, y_max=1)
        return reward * self.y_max


delta = 1
x = np.arange(0, 101.0, delta)
y = np.arange(0, 101.0, delta)
X, Y = np.meshgrid(x, y)
fig = plt.figure()
ax1 = fig.add_subplot(2,1,1,adjustable='box', aspect=1.0)
ax1.title.set_text('Depth Map')
CS = plt.contour(X, Y, caldera_sim_function(X, Y), levels=12)
plt.clabel(CS, inline=1, fontsize=8, fmt='%.3g')
plt.colorbar(CS, ax=ax1)
# fig2 = plt.figure()

mcts = mcts(timeLimit=max_depth*300)
samples = dict()
robot_state = RobotState(*start, samples)
ax2 = fig.add_subplot(2,1,2,adjustable='box', aspect=1.0)
ax2.title.set_text('Acquisition Function')

xs = np.array([])
ys = np.array([])
cbar, marker1, marker2 = None, None, None
t = tqdm(total=num_actions+1, file=sys.stdout)
def update(frame):
    global samples, cbar, marker1, marker2, xs, ys, robot_state, t
    bestAction = mcts.search(initialState=robot_state)
    robot_state = mcts.getBestChild(mcts.root, 0).state
    if (robot_state.x, robot_state.y) not in samples:
        samples[(robot_state.x, robot_state.y)] = caldera_sim_function(robot_state.x, robot_state.y)
    robot_state = robot_state.renew(samples)

    # plt.subplot(211)
    xs = np.concatenate((xs, [robot_state.x]))
    ys = np.concatenate((ys, [robot_state.y]))
    if marker1 is not None:
        ax1.lines.pop()
    pos1, = ax1.plot(xs[-2:], ys[-2:], color='k')
    marker1, = ax1.plot(robot_state.x, robot_state.y, '*b')
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
    marker2, = ax2.plot(robot_state.x, robot_state.y, '*b')
    ax2.invert_yaxis()
    # plt.pause(0.1)
    t.update(n=1)
    plt.tight_layout()
    return marker1, marker2, pos1, pos2, im

from matplotlib import animation
anim = animation.FuncAnimation(fig, update, frames=num_actions, blit=True)
anim.save(filename, fps=fps)

