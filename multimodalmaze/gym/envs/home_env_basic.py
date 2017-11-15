import gym
from gym import spaces
import numpy as np


class HomeEnv(gym.Env):
    """HoME basic starter environment
    TODO: explain env basics, reward, other important details

    Actions: (can do both move & look at the same time)

        1) Moving: Discrete 5 - NOOP[0],
                                UP[1],
                                RIGHT[2],
                                DOWN[3],
                                LEFT[4]
            - params: min: 0, max: 4
        2) Looking: Discrete 5 -NOOP[0],
                                UP[1],
                                RIGHT[2],
                                DOWN[3],
                                LEFT[4]
            - params: min: 0, max: 4

    """

    def __init__(self):
        # "human" render creates a Tk window, returns None
        # "rgb_array" creates NO window,
        #             returns ndarray containing the img
        self.metadata = {'render.modes': ['human', 'rgb_array']}

        self.action_space = spaces.MultiDiscrete([[0, 4], [0, 4]])
        self.observation_space = spaces.Discrete(1)

        self.observation = 1  # TODO: temporary

        self._seed()
        self._reset()

    def _seed(self, seed=None):
        # self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action)

        pass  # TODO
        reward = 0
        done = False
        misc = None

        return self.observation, reward, done, misc

    def _reset(self):
        pass  # TODO
        return self.observation

    def _render(self, mode='human', close=False):
        pass  # TODO
        return


if __name__ == '__main__':
    import multimodalmaze.gym  # to make the environment available in gym.make()

    env = gym.make("Home-v0")
    env.reset()

    env.render("human")

    action = env.action_space.sample()
    print ("action:", action)

    obs, rew, done, misc = env.step(action)
    print ("obs:", obs, "rew:", rew, "done:", done, "misc:", misc)
