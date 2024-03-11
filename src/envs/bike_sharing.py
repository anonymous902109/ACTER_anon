import gymnasium as gym

from src.envs.abs_env import AbstractEnv
from maro.simulator import Env


class BikeSharing(AbstractEnv):
    ''' Abstract class for defining an environment '''

    def __init__(self):
        self.gym_env = Env(scenario="citi_bike", topology="toy.3s_4t", start_tick=0, durations=1440, snapshot_resolution=30)

    def step(self, action):
        obs, rew, done, trunc, inf = self.gym_env.step(action)
        return None

    def close(self):
        pass

    def render(self):
        pass

    def reset(self):
        return None

    def render_state(self, x):
        ''' Renders single state x '''
        pass

    def realistic(self, x):
        ''' Returns a boolean indicating if x is a valid state in the environment (e.g. chess state without kings is not valid)'''
        return True

    def actionable(self, x, fact):
        ''' Returns a boolean indicating if all immutable features remain unchanged between x and fact states'''
        return True

    def get_actions(self, x):
        ''' Returns a list of actions available in state x'''
        return []

    def set_stochastic_state(self, state, env_state):
        ''' Changes the environment's current state to x '''
        pass

    def set_nonstoch_state(self, state, env_state):
        pass

    def check_done(self, x):
        ''' Returns a boolean indicating if x is a terminal state in the environment'''
        return False

    def equal_states(self, x1, x2):
        ''' Returns a boolean indicating if x1 and x2 are the same state'''
        return False

    def writable_state(self, x):
        ''' Returns a string with all state information to be used for writing results'''
        return None

    def check_failure(self):
        return self.failure