from mcts import mcts
from bayes_opt.bayesian_optimization import *
from copy import deepcopy
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor


# import matplotlib
# import matplotlib.mlab as mlab
# import warnings
#
#
# def caldera_sim_function(x, y):
#     warnings.filterwarnings("ignore", category=matplotlib.cbook.MatplotlibDeprecationWarning)
#     x, y = x / 10.0, y / 10.0
#     z0 = mlab.bivariate_normal(x, y, 10.0, 5.0, 5.0, 0.0)
#     z1 = mlab.bivariate_normal(x, y, 1.0, 2.0, 2.0, 5.0)
#     z2 = mlab.bivariate_normal(x, y, 1.7, 1.7, 8.0, 8.0)
#     return 50000.0 * z0 + 2500.0 * z1 + 5000.0 * z2


class RobotState:
    def __init__(self, x, y, bounds, samples, acq_func, max_tree_depth, step_size,
                 current_direction=None, kernel=RBF(length_scale=1)):
        # Creates a new state that represents the root of an action tree
        self.x = x
        self.y = y
        self.bounds = bounds
        self.samples = deepcopy(samples)
        self.acq_func = acq_func
        self.max_tree_depth = max_tree_depth
        self.step_size = step_size

        self.tree_depth = 0
        self.base_reward = 0
        self.current_direction = current_direction
        self.action_history = list()
        self.available_actions = None

        self.y_max = 1
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-3, normalize_y=False, n_restarts_optimizer=25,
                                           random_state=1)

        if len(samples) > 0:
            xdata = np.vstack(list(self.samples.keys()))
            ydata = np.vstack(list(self.samples.values()))
            self.y_max = np.max(ydata)
            self.gp.fit(xdata, ydata.reshape(-1) / (self.y_max if self.y_max > 0 else 1))

    def set_backtracking(self, val):
        self.can_backtrack = val

    @staticmethod
    def _invert_dir(dir):
        return {None: None, 'north': 'south', 'south': 'north',
                'west': 'east', 'east': 'west'}[dir]

    def renew(self, new_samples):
        # Returns a new robot state that has new_samples instead of the existing samples
        # This resets internal variables so that the returned state is the root of a tree
        return RobotState(self.x, self.y, self.bounds, new_samples, self.acq_func, max_tree_depth=self.max_tree_depth,
                          step_size=self.step_size, current_direction=self.current_direction, kernel=self.gp.kernel)

    def _generate_actions(self, prevent_backtracking=True, direct_paths_only=False):
        actions = []
        if self.x >= (self.bounds['x'][0] + self.step_size):
            actions.append('west')
        if self.x <= (self.bounds['x'][1] - self.step_size):
            actions.append('east')
        if self.y >= (self.bounds['y'][0] + self.step_size):
            actions.append('south')
        if self.y <= (self.bounds['y'][1] - self.step_size):
            actions.append('north')

        if prevent_backtracking:
            # Prevent the robot from moving backwards
            if self._invert_dir(self.current_direction) in actions:
                actions.remove(self._invert_dir(self.current_direction))
        if direct_paths_only:
            for action in self.action_history:
                if len(actions) > 1 and self._invert_dir(action) in actions:
                    actions.remove(self._invert_dir(action))
        self.available_actions = actions

    def getPossibleActions(self):
        if self.available_actions is None:
            self._generate_actions()
        return self.available_actions[:]

    def takeAction(self, action):
        # Add the predicted sample to the set of samples
        new_samples = deepcopy(self.samples)
        new_samples[(self.x, self.y)] = self.gp.predict(np.array([self.x, self.y]).reshape(1, -1))

        # Create a new state representing the child
        new_state = self.renew(new_samples)  # initially a root note
        new_state.tree_depth = self.tree_depth + 1  # Set depth in tree (no longer a root node)
        new_state.base_reward = self.getReward()  # Add accumulated reward
        new_state.current_direction = action  # Update direction
        new_state.action_history = self.action_history.copy()  # Pass on action history
        new_state.action_history.append(action)

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
        global max_tree_depth
        assert (self.tree_depth == len(self.action_history))
        return self.tree_depth >= self.max_tree_depth

    def getReward(self):
        reward, = self.acq_func.utility(np.array([self.x, self.y]).reshape(1, -1),
                                        self.gp, y_max=1)
        return self.base_reward + reward * self.y_max


def mcts_state_update(mcts, state, samples, sample_func):
    mcts.search(initialState=state)
    state = mcts.getBestChild(mcts.root, 0).state  # bestOnly = True
    if (state.x, state.y) not in samples:
        samples[(state.x, state.y)] = sample_func(state.x, state.y)
    new_robot_state = state.renew(samples)
    return new_robot_state


if __name__ == "__main__":
    bounds = {'x': (0, 100), 'y': (0, 100)}
    start = [80, 50]
    kappa = 2.576
    max_tree_depth = 7
    num_actions = 150
    step_size = 4

    ucb = UtilityFunction('ucb', kappa=kappa, xi=0)
    acq_func = ucb

    mcts = mcts(iterationLimit=16 * (max_tree_depth ** 2))
    samples = dict()
    robot_state = RobotState(*start, bounds, samples, acq_func, max_tree_depth=max_tree_depth)
    for _ in range(num_actions):
        mcts_state_update(mcts, robot_state, samples, sample_func=caldera_sim_function)
