# Copyright (c) 2017, simon
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

import os
import numpy as np

from multimodalmaze.env import BasicEnvironment
from multimodalmaze.core import House

TEST_SUNCG_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "suncg")

class BenchmarkEnvironment(BasicEnvironment):
    
    def __init__(self, activeEngines=['physics', 'render', 'acoustics']):
    
        # NOTE: set the rendered image size here (this will greatly impact framerate!)
        super(BenchmarkEnvironment, self).__init__(size=(256, 256))
    
        self.activeEngines = activeEngines
    
        houseFilename = os.path.join(TEST_SUNCG_DATA_DIR, "house", "0004d52d1aeeb8ae6de39d6bd993e992", "house.json")
        house = House.loadFromJson(houseFilename, TEST_SUNCG_DATA_DIR)
        self.loadHouse(house)
        
        self.agent.setPosition((42, -39, 1))
        self.agent.setOrientation((0.0, 0.0, -np.pi/3))
    
        self.linearVelocity = np.zeros(3)
        self.angularVelocity = np.zeros(3)
        
        self._initialize()
        
    def _initialize(self):
        # NOTE: we need to initialize all engines by running a single step, otherwise
        #       there are execution errors.
        super(BenchmarkEnvironment, self).step()
        
    def step(self):
        for engine in self.activeEngines:
            self.worlds[engine].step()
        
    def simulate(self, nbSteps):
        
        rotationStepCounter = -1 
        rotationsStepDuration = 40
        for _ in range(nbSteps):
        
            # Constant speed forward (Y-axis)
            self.linearVelocity = np.array([0.0, 1.0, 0.0])
            collision = self.agent.isCollision()
            if collision:
                self.linearVelocity *= -1.0
            self.agent.setLinearVelocity(self.linearVelocity)
            
            if rotationStepCounter > rotationsStepDuration:
                # End of rotation
                rotationStepCounter = -1
                self.angularVelocity = np.zeros(3)
            elif rotationStepCounter >= 0:
                # During rotation
                rotationStepCounter += 1
            else:
                # No rotation, initiate at random
                if np.random.random() > 0.5:
                    self.angularVelocity = np.zeros(3)
                    self.angularVelocity[2] = np.random.uniform(low=-np.pi, high=np.pi)
                    rotationStepCounter = 0
            
            # Randomly change angular velocity (rotation around Z-axis)
            self.agent.setAngularVelocity(self.angularVelocity)
            
            # Simulate
            self.step()
            
            # Grab some observations
            _ = self.getObservation()
            