import gym
from gym import spaces
import numpy as np


class HomeEnv(gym.Env):
    """HoME basic starter environment
    TODO: explain env basics, reward, other important details
    """
    def __init__(self):
        self.action_space = spaces.Box(low=np.array([-self.bounds]), high=np.array([self.bounds]))
        self.observation_space = spaces.Discrete(4)

        self.observation = None # TODO: temporary

        self._seed()
        self._reset()

    def _seed(self, seed=None):
        # self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action)

        pass #TODO
        reward = 0
        done = False
        misc = None

        return self.observation, reward, done, misc

    def _reset(self):
        pass # TODO
        return self.observation

