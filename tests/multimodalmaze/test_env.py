# Copyright (c) 2017, IGLU consortium
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
#  - Neither the name of the copyright holder nor the names of its contributors 
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
import time
import logging
import numpy as np
import unittest

import matplotlib.pyplot as plt

from multimodalmaze.core import House
from multimodalmaze.env import BasicEnvironment

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data")
TEST_SUNCG_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "suncg")

class TestBasicEnvironment(unittest.TestCase):
    
    def testRender(self):
    
        house = House.loadFromJson(os.path.join(TEST_SUNCG_DATA_DIR, "house", "0004d52d1aeeb8ae6de39d6bd993e992", "house.json"),
                                   TEST_SUNCG_DATA_DIR)
        
        env = BasicEnvironment()
        env.loadHouse(house)
    
        env.agent.setPosition((42, -39, 1))
        env.agent.setOrientation((0.0, 0.0, -np.pi/3))
    
        env.renderWorld.addDefaultLighting()
        
        env.step()
        
        image = env.renderWorld.getRgbImage()
        depth = env.renderWorld.getDepthImage(mode='distance')
        
        fig = plt.figure(figsize=(16,8))
        plt.axis("off")
        ax = plt.subplot(121)
        ax.imshow(image)
        ax = plt.subplot(122)
        ax.imshow(depth/np.max(depth), cmap='binary')
        plt.show(block=False)
        time.sleep(1.0)
        plt.close(fig)
    
    def testGenerateSpawnPositions(self):
        
        house = House.loadFromJson(os.path.join(TEST_SUNCG_DATA_DIR, "house", "0004d52d1aeeb8ae6de39d6bd993e992", "house.json"),
                                   TEST_SUNCG_DATA_DIR)
        
        env = BasicEnvironment()
        env.loadHouse(house)
        
        occupancyMap, occupancyMapCoord, positions = env.generateSpawnPositions(n=10)
        
        xmin, ymin = np.min(occupancyMapCoord, axis=(0,1))
        xmax, ymax = np.max(occupancyMapCoord, axis=(0,1))
        
        fig = plt.figure()
        plt.axis("on")
        ax = plt.subplot(111)
        ax.imshow(occupancyMap, cmap='gray', extent=[xmin,xmax,ymin,ymax])
        ax.scatter(positions[:,0], positions[:,1], s=40, c=[1.0,0.0,0.0])
        plt.show(block=False)
        time.sleep(1.0)
        plt.close(fig)
        
if __name__ == '__main__':
    logging.basicConfig(level=logging.WARN)
    np.seterr(all='raise')
    unittest.main()
    