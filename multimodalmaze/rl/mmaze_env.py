# Copyright (c) 2017, IGLU Consortium
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 
#  - Redistributions of source code must retain the above copyright notice, 
#    this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright notice, 
#    this list of conditions and the following disclaimer in the documentation 
#    and/or other materials provided with the distribution.
#  - Neither the name of the NECOTIS research group nor the names of its contributors 
#    may be used to endorse or promote products derived from this software 
#    without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND 
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT 
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, 
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
# POSSIBILITY OF SUCH DAMAGE.

import logging
import numpy as np
import gym

from gym import spaces
from gym.utils import seeding

logger = logging.getLogger(__name__)

from gym.spaces import prng

class Sentence(gym.Space):

    def __init__(self, dictionary):
        self.dictionary = np.array(dictionary, dtype='object')

    def sample(self):
        """
        Uniformly randomly sample a random element of this space
        """
        
        endProbability = 0.1
        indices = [prng.np_random.randint(0, len(self.dictionary)),]
        while prng.np_random.rand() > endProbability:
            indices.append(prng.np_random.randint(0, len(self.dictionary)))

        strings = self.dictionary[indices]
        sentence = strings.join(' ')
        raise sentence

    def contains(self, x):
        """
        Return boolean specifying if x is a valid
        member of this space
        """
        
        # Split sentence into words
        words = x.split(' ')
        
        # Make sure all words appear in the dictionary
        outOfVocabulary = False
        for word in words:
            word = word.lower().strip()
            if word not in self.dictionary:
                outOfVocabulary = True
                break
        return not outOfVocabulary

    def __repr__(self):
        return "Sentence" + str(self.dictionary)

class MultimodalMazeEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    actions = {
        0 : "NOOP",
        1 : "MOVE-FORWARD",
        3 : "MOVE-BACK",
        4 : "SLIDE-LEFT",
        5 : "SLIDE-RIGHT",
        6 : "TILT-CAMERA-UP",
        7 : "TILT-CAMERA-DOWN",
        8 : "TURN-CAMERA-LEFT",
        9 : "TURN-CAMERA-RIGHT",
        10: "POINT-OBJECT",
        11: "DIALOGUE",
    }

    def __init__(self):

        self._seed()

        #TODO: define action and observation spaces

        self.action_space = spaces.Discrete(len(MultimodalMazeEnv.actions))

        self.observation_space = spaces.Dict({"position": spaces.Box(0.0, 100.0, (3,)),             # x-y-z position [m]
                                              "orientation": spaces.Box(-np.pi, np.pi, (3,)),       # yaw-pitch-roll orientation [rad]
                                              "velocity": spaces.Box(-1.0, 1.0, (3,)),              # x-y-z velocity [m/sec]
                                              "collision": spaces.Discrete(2),                      # collision sensor
                                              "rgb-image": spaces.Box(0, 255, (256,256,3)),         # 8-bit RGB image
                                              "depth": spaces.Box(0.0, 100.0, (256,256,)),})        # Depth image [m]

    def _seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        return [seed1,]

    def _step(self, a):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the environment

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """

        ob = self._get_obs()
        reward = self._get_reward()
        
        #TODO: implement additional information and episode ending flag
        info = {}
        done = False

        return ob, reward, done, info

    def _get_reward(self):
        #TODO: implement reward
        return 0.0

    def _get_obs(self):
        
        ob = dict()
        ob["position"] = np.zeros((3,))
        ob["orientation"] = np.zeros((3,))
        ob["velocity"] = np.zeros((3,))
        ob["collision"] = 0
        ob["rgb-image"] = np.zeros((256,256,3), dtype=np.int8)
        ob["depth"] = np.zeros((256,256,3))
        
        #TODO: implement observation gathering
        
        return ob

    def _reset(self):
        """Resets the state of the environment and returns an initial observation.

        Returns: observation (object): the initial observation of the
            space.
        """
        
        #TODO: implement reset function
        
        return self._get_obs()

    def _get_image(self):
        
        #TODO: implement scene rendering function
        img = np.zeros((256,256,3))

        return img

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        img = self._get_image()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)

