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
import time
import logging
import numpy as np
import unittest

from direct.showbase.ShowBase import ShowBase
from multimodalmaze.core import House, Room, Object
from multimodalmaze.rendering import Panda3dRenderWorld

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data")
TEST_SUNCG_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "suncg")

class TestHouse(unittest.TestCase):
    
    def setUp(self):
        self.base = ShowBase()
        self.render = Panda3dRenderWorld(shadowing=False, showCeiling=False)
    
    def tearDown(self):
        self.base.destroy()
        self.base.graphicsEngine.removeAllWindows()
    
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
        for _ in range(20):
            self.render.step(0.0)
        time.sleep(1.0)
        self.render.resetScene()
        
class TestRoom(unittest.TestCase):
    
    def setUp(self):
        self.base = ShowBase()
        self.render = Panda3dRenderWorld(shadowing=False, showCeiling=False)
    
    def tearDown(self):
        self.base.destroy()
        self.base.graphicsEngine.removeAllWindows()
    
    def testRender(self):
        
        room = Room.loadFromJson(os.path.join(TEST_SUNCG_DATA_DIR, "house", "0004d52d1aeeb8ae6de39d6bd993e992", "house.json"),
                               TEST_SUNCG_DATA_DIR,
                               modelId="fr_0rm_1")
        
        self.render.addRoomToScene(room)
        self.render.addDefaultLighting()
        
        mat = np.array([0.999992, 0.00394238, 0, 0,
                        -0.00295702, 0.750104, -0.661314, 0,
                        -0.00260737, 0.661308, 0.75011, 0,
                        43.621, -55.7499, 12.9722, 1])
        self.render.setCamera(mat)
        for _ in range(20):
            self.render.step(0.0)
        time.sleep(1.0)
        self.render.resetScene()
        
class TestObject(unittest.TestCase):
    
    def setUp(self):
        self.base = ShowBase()
        self.render = Panda3dRenderWorld(shadowing=False, showCeiling=False)
    
    def tearDown(self):
        self.base.destroy()
        self.base.graphicsEngine.removeAllWindows()
    
    def testRender(self):
        
        modelId = '209'
        modelFilename = os.path.join(TEST_SUNCG_DATA_DIR, "object", str(modelId), str(modelId) + ".obj")
        assert os.path.exists(modelFilename)
        instanceId = str(modelId) + '-0'
        obj = Object(instanceId, modelId, modelFilename)
        
        self.render.addObjectToScene(obj)
        self.render.addDefaultLighting()
        
        mat = np.array([0.999992, 0.00394238, 0, 0,
                        -0.00295702, 0.750104, -0.661314, 0,
                        -0.00260737, 0.661308, 0.75011, 0,
                        43.621, -55.7499, 12.9722, 1])
        self.render.setCamera(mat)
        for _ in range(20):
            self.render.step(0.0)
        time.sleep(1.0)
        self.render.resetScene()
        
if __name__ == '__main__':
    logging.basicConfig(level=logging.WARN)
    np.seterr(all='raise')
    unittest.main()
    