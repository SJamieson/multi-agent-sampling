from mcts import mcts
from bayes_opt.bayesian_optimization import *
from copy import deepcopy
import matplotlib
import matplotlib.mlab as mlab
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor
import warnings


def caldera_sim_function(x, y):
    warnings.filterwarnings("ignore", category=matplotlib.cbook.MatplotlibDeprecationWarning)
    x, y = x / 10.0, y / 10.0
    z0 = mlab.bivariate_normal(x, y, 10.0, 5.0, 5.0, 0.0)
    z1 = mlab.bivariate_normal(x, y, 1.0, 2.0, 2.0, 5.0)
    z2 = mlab.bivariate_normal(x, y, 1.7, 1.7, 8.0, 8.0)
    return 50000.0 * z0 + 2500.0 * z1 + 5000.0 * z2


class RobotState:
    def __init__(self, x, y, bounds, samples, acq_func, max_depth, step_size, depth=0, base_reward=0, dir='north',
                 kernel=RBF(length_scale=1)):
        self.x = x
        self.y = y
        self.bounds = bounds
        self.samples = deepcopy(samples)
        self.acq_func = acq_func
        self.depth = depth
        self.base_reward = base_reward
        self.max_depth = max_depth
        self.step_size = step_size
        self.dir = dir
        self.history = set()
        self.actions = self._generate_actions()

        self.y_max = 1
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-3, normalize_y=False, n_restarts_optimizer=25,
                                           random_state=1)

        if len(samples) > 0:
            xdata = np.vstack(list(self.samples.keys()))
            ydata = np.vstack(list(self.samples.values()))
            self.y_max = np.max(ydata)
            self.gp.fit(xdata, ydata.reshape(-1) / self.y_max)


    def set_backtracking(self, val):
        self.can_backtrack = val

    @staticmethod
    def _invert_dir(dir):
        return {'north': 'south', 'south': 'north',
                'west': 'east', 'east': 'west'}[dir]

    def renew(self, new_samples):
        return RobotState(self.x, self.y, self.bounds, new_samples, self.acq_func, depth=0, base_reward=0,
                          max_depth=self.max_depth, step_size=self.step_size, dir=self.dir, kernel=self.gp.kernel)

    def _generate_actions(self):
        actions = []
        if self.x >= (self.bounds['x'][0] + self.step_size):
            actions.append('west')
        if self.x <= (self.bounds['x'][1] - self.step_size):
            actions.append('east')
        if self.y >= (self.bounds['y'][0] + self.step_size):
            actions.append('south')
        if self.y <= (self.bounds['y'][1] - self.step_size):
            actions.append('north')
        if self._invert_dir(self.dir) in actions:
            actions.remove(self._invert_dir(self.dir))
        for action in self.history:
            if len(actions) > 1 and action in actions:
                actions.remove(action)
        return actions

    def getPossibleActions(self):
        return self.actions

    def takeAction(self, action):
        new_samples = deepcopy(self.samples)
        new_samples[(self.x, self.y)] = self.getReward()
        new_state = self.renew(new_samples)
        new_state.depth = self.depth + 1
        new_state.base_reward = self.base_reward + self.getReward()
        new_state.history = self.history.copy()
        new_state.history.add(action)

        if action == 'west':
            new_state.x -= self.step_size
        if action == 'east':
            new_state.x += self.step_size
        if action == 'north':
            new_state.y += self.step_size
        if action == 'south':
            new_state.y -= self.step_size
        return new_state

    def isTerminal(self):
        global max_depth
        return self.depth >= self.max_depth

    def getReward(self):
        reward, = self.acq_func.utility(np.array([self.x, self.y]).reshape(1, -1),
                                        self.gp, y_max=1)
        return reward * self.y_max


def mcts_state_update(mcts, state, samples, sample_func):
    mcts.search(initialState=state)
    state = mcts.getBestChild(mcts.root, 0, bestOnly=True).state
    if (state.x, state.y) not in samples:
        samples[(state.x, state.y)] = sample_func(state.x, state.y)
    new_robot_state = state.renew(samples)
    return new_robot_state


if __name__ == "__main__":
    bounds = {'x': (0, 100), 'y': (0, 100)}
    start = [80, 50]
    kappa = 2.576
    xi = 0
    max_depth = 7
    num_actions = 150
    step_size = 4
    can_backtrack = False
    acq = 'ucb'

    ucb = UtilityFunction('ucb', kappa=kappa, xi=0)
    ei = UtilityFunction('ei', kappa=0, xi=0)
    acq_func = ucb if acq == 'ucb' else ei

    mcts = mcts(timeLimit=max_depth * 300)
    samples = dict()
    robot_state = RobotState(*start, bounds, samples, acq_func, max_depth=max_depth)
    for _ in range(num_actions):
        mcts_state_update(mcts, robot_state, samples, sample_func=caldera_sim_function)
