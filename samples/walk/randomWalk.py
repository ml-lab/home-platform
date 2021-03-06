# Copyright (c) 2017, IGLU consortium
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without 
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright 
#    notice, this list of conditions and the following disclaimer.
#   
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT 
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR 
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY 
# OF SUCH DAMAGE.

import os
import sys
import logging
import numpy as np
from matplotlib import pyplot as plt

from home_platform.core import House
from home_platform.env import BasicEnvironment

TEST_SUNCG_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "tests", "data", "suncg")

logger = logging.getLogger(__name__)

def main():
    
    env = BasicEnvironment()
    agent = env.agent
    
    houseFilename = os.path.join(TEST_SUNCG_DATA_DIR, "house", "0004d52d1aeeb8ae6de39d6bd993e992", "house.json")
    house = House.loadFromJson(houseFilename, TEST_SUNCG_DATA_DIR)
    env.loadHouse(house)
    
    agent.setPosition((42, -39, 1))
    agent.setOrientation((0.0, 0.0, -np.pi/3))
    
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.ion()
    ax = plt.subplot(111)
    im = ax.imshow(np.zeros(env.renderWorld.size))
    fig.show()
    
    linearVelocity = np.zeros(3)
    angularVelocity = np.zeros(3)
    rotationStepCounter = -1 
    rotationsStepDuration = 40
    try:
        while True:
            
            # Constant speed forward (Y-axis)
            linearVelocity = np.array([0.0, 1.0, 0.0])
            agent.setLinearVelocity(linearVelocity)
            
            if rotationStepCounter > rotationsStepDuration:
                # End of rotation
                rotationStepCounter = -1
                angularVelocity = np.zeros(3)
            elif rotationStepCounter >= 0:
                # During rotation
                rotationStepCounter += 1
            else:
                # No rotation, initiate at random
                if np.random.random() > 0.5:
                    angularVelocity = np.zeros(3)
                    angularVelocity[2] = np.random.uniform(low=-np.pi, high=np.pi)
                    rotationStepCounter = 0
            
            # Randomly change angular velocity (rotation around Z-axis)
            agent.setAngularVelocity(angularVelocity)
            
            # Simulate
            env.step()
            
            # Grab some observations
            position = agent.getPosition()
            orientation = agent.getOrientation()
            image = env.renderWorld.getRgbImage()
            collision = agent.isCollision()
            
            if collision:
                linearVelocity *= -1.0
            
            print 'Position: %s (x,y,z)' % (str(position))
            print 'Orientation: %s (h,p,r)' % (str(orientation))
            print 'Collision detected: %s' % (str(collision))
            
            im.set_data(image)
            fig.canvas.draw()
            
    except KeyboardInterrupt:
        pass

    return 0

if __name__ == "__main__":
    sys.exit(main())
