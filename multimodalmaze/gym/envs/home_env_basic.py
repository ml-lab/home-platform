import gym
from gym import spaces
import numpy as np

from multimodalmaze.suncg import data_dir


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
        self.metadata = {'render.modes': ['human', 'rgb_array']}

        # Gym environments have no parameters, so we have to
        # make sure the user first creates a symlink
        # to their SUNCG dataset in ~/.suncg
        # so that there are folders ~/.suncg/[room|house|...]
        self.data_path = data_dir()
        print ("SUNCG DATA DIRECTORY:", self.data_path)

        self.action_space = spaces.MultiDiscrete([[0, 4], [0, 4]])
        self.observation_space = spaces.Discrete(1)

        self.observation = 1  # TODO: temporary

        # TODO get actual list of available houses from data dir
        # (or precompute CSV)
        # list must be sorted

        # self.list_of_houses = get_house_list(DATA_DIR)
        self.list_of_houses = ["0004d52d1aeeb8ae6de39d6bd993e992"]

        # for determinism we have to load
        # the houses in specific order
        self.next_house = 0

        self._seed()
        self._reset()

    def _seed(self, seed=None):
        # self.np_random, seed = seeding.np_random(seed)
        self.next_house = 0
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action)

        pass  # TODO
        reward = 0
        done = False
        misc = None

        return self.observation, reward, done, misc

    def _reset(self):
        # TODO load house, based on self.list_of_houses[self.next_house]


        self.next_house += 1

        # if we went to the end of the house list, loop around
        if (self.next_house == len(self.list_of_houses)):
            self.next_house = 0

        return self.observation

    def _render(self, mode='human', close=False):
        """
        "human" render creates a Tk window, returns None
        "rgb_array" creates NO window,
                    returns ndarray containing the img
        """
        # TODO
        pass
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
