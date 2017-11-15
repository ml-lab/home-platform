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

from multimodalmaze.core import House, Object, Agent
from multimodalmaze.rendering import Panda3dRenderWorld

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data")
TEST_SUNCG_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "suncg")

class TestHouse(unittest.TestCase):
    
    #FIXME: find out why not working correctly with shadowing (shaders)
    def setUp(self):
        self.render = Panda3dRenderWorld(shadowing=False, showCeiling=False, mode='offscreen')
    
    def tearDown(self):
        self.render.destroy()
    
    def testRender(self):
        
        house = House.loadFromJson(os.path.join(TEST_SUNCG_DATA_DIR, "house", "0004d52d1aeeb8ae6de39d6bd993e992", "house.json"),
                                   TEST_SUNCG_DATA_DIR)

        self.render.addHouseToScene(house)
        self.render.addDefaultLighting()
        
        mat = np.array([0.999992, 0.00394238, 0, 0,
                        -0.00295702, 0.750104, -0.661314, 0,
                        -0.00260737, 0.661308, 0.75011, 0,
                        43.621, -55.7499, 12.9722, 1])
        self.render.setCamera(mat)
        self.render.step()
        image = self.render.getRgbImage()
        depth = self.render.getDepthImage(mode='distance')
        self.assertTrue(np.min(depth) >= self.render.zNear)
        self.assertTrue(np.max(depth) <= self.render.zFar)
        
        fig = plt.figure(figsize=(16,8))
        plt.axis("off")
        ax = plt.subplot(121)
        ax.imshow(image)
        ax = plt.subplot(122)
        ax.imshow(depth/np.max(depth), cmap='binary')
        plt.show(block=False)
        time.sleep(1.0)
        plt.close(fig)
        
class TestRoom(unittest.TestCase):
    
    def setUp(self):
        self.render = Panda3dRenderWorld(shadowing=False, showCeiling=False, mode='offscreen')
    
    def tearDown(self):
        self.render.destroy()
    
    def testRender(self):
        
        house = House.loadFromJson(os.path.join(TEST_SUNCG_DATA_DIR, "house", "0004d52d1aeeb8ae6de39d6bd993e992", "house.json"),
                                   TEST_SUNCG_DATA_DIR)
        room = house.rooms[1]
        self.assertTrue(room.instanceId == 'fr_0rm_1')
        self.assertTrue(len(room.objects) == 18)
        
        self.render.addRoomToScene(room)
        self.render.addDefaultLighting()
        
        mat = np.array([0.999992, 0.00394238, 0, 0,
                        -0.00295702, 0.750104, -0.661314, 0,
                        -0.00260737, 0.661308, 0.75011, 0,
                        43.621, -55.7499, 12.9722, 1])
        self.render.setCamera(mat)
        self.render.step()
        image = self.render.getRgbImage()
        depth = self.render.getDepthImage()
        
        fig = plt.figure()
        plt.axis("off")
        ax = plt.subplot(121)
        ax.imshow(image)
        ax = plt.subplot(122)
        ax.imshow(depth, cmap='binary')
        plt.show(block=False)
        time.sleep(1.0)
        plt.close(fig)
        
class TestObject(unittest.TestCase):
    
    def setUp(self):
        self.render = Panda3dRenderWorld(shadowing=False, showCeiling=False, mode='offscreen')
    
    def tearDown(self):
        self.render.destroy()
    
    def testRender(self):
        
        modelId = '83'
        modelFilename = os.path.join(TEST_SUNCG_DATA_DIR, "object", str(modelId), str(modelId) + ".egg")
        assert os.path.exists(modelFilename)
        instanceId = str(modelId) + '-0'
        obj = Object(instanceId, modelId, modelFilename)
        
        self.render.addObjectToScene(obj)
        obj.setPosition((0,0,0))
        
        self.render.addDefaultLighting()
        
        mat = np.array([1, 0, 0, 0,
                        0, 1, 0, 0,
                        0, 0, 1, 0,
                        0, -5, 0, 1])
        self.render.setCamera(mat)
        self.render.step()
        image = self.render.getRgbImage()
        depth = self.render.getDepthImage()
        
        fig = plt.figure()
        plt.axis("off")
        ax = plt.subplot(121)
        ax.imshow(image)
        ax = plt.subplot(122)
        ax.imshow(depth, cmap='binary')
        plt.show(block=False)
        time.sleep(1.0)
        plt.close(fig)
        
class TestAgent(unittest.TestCase):
    
    def setUp(self):
        self.render = Panda3dRenderWorld(shadowing=False, showCeiling=False, mode='offscreen')
        
    def tearDown(self):
        self.render.destroy()
    
    def testRender(self):
        
        modelFilename = os.path.join(TEST_DATA_DIR, "models", "sphere.egg")
        assert os.path.exists(modelFilename)
        agent = Agent('agent-0', modelFilename)
        
        self.render.addAgentToScene(agent)
        agent.renderInstance.nodePath.setColor(1,0,0,1)
        agent.setPosition((0,0,0))
        
        self.render.addDefaultLighting()
        
        mat = np.array([1, 0, 0, 0,
                        0, 1, 0, 0,
                        0, 0, 1, 0,
                        0, -5, 0, 1])
        self.render.setCamera(mat)
        self.render.step()
        image = self.render.getRgbImage()
        depth = self.render.getDepthImage()
        
        fig = plt.figure()
        plt.axis("off")
        ax = plt.subplot(121)
        ax.imshow(image)
        ax = plt.subplot(122)
        ax.imshow(depth, cmap='binary')
        plt.show(block=False)
        time.sleep(1.0)
        plt.close(fig)
        
if __name__ == '__main__':
    logging.basicConfig(level=logging.WARN)
    np.seterr(all='raise')
    unittest.main()
