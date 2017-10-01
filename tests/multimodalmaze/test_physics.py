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

import matplotlib.pyplot as plt

from panda3d.core import Mat4, Vec3
from direct.showbase.ShowBase import ShowBase
from direct.task.TaskManagerGlobal import taskMgr
from multimodalmaze.core import House, Agent
from multimodalmaze.physics import Panda3dBulletPhysicWorld
from multimodalmaze.rendering import Panda3dRenderWorld

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data")
TEST_SUNCG_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "suncg")

class TestAgent(unittest.TestCase):

    def setUp(self):
        self.engine = Panda3dBulletPhysicWorld(debug=True)
    
    def tearDown(self):
        pass
    
    def testRender(self):
        
        self.base = ShowBase()
        self.base.disableMouse()
        
        agent = Agent('agent')
        nodePath = self.engine.addAgentToScene(agent, height=1.5)
        nodePath.setPos(Vec3(0,0,1))
        
        nodePath.node().setLinearMovement(Vec3(1,0,0), True)
        
        mat = np.array([1, 0, 0, 0,
                        0, 1, 0, 0,
                        0, 0, 1, 0,
                        0, -10, 0, 1])
        mat = Mat4(*mat.ravel())
        self.base.camera.setMat(mat)

        self.engine.render.reparentTo(self.base.render)

        # Update
        def update(task):
            self.engine.step()
            return task.cont
        
        taskMgr.add(update, 'update')
        
        for _ in range(50):
            taskMgr.step()
        time.sleep(1.0)
        
        self.base.destroy()
        self.base.graphicsEngine.removeAllWindows()

class TestHouse(unittest.TestCase):
    
    def setUp(self):
        self.engine = Panda3dBulletPhysicWorld(debug=True)
    
    def tearDown(self):
        pass
    
    def testRender(self):
        
        self.base = ShowBase()
        self.base.disableMouse()
        
        house = House.loadFromJson(os.path.join(TEST_SUNCG_DATA_DIR, "house", "0004d52d1aeeb8ae6de39d6bd993e992", "house.json"),
                                   TEST_SUNCG_DATA_DIR)
        self.engine.addHouseToScene(house)
        
        agent = Agent('agent')
        self.engine.addAgentToScene(agent)
        
        mat = np.array([0.999992, 0.00394238, 0, 0,
                        -0.00295702, 0.750104, -0.661314, 0,
                        -0.00260737, 0.661308, 0.75011, 0,
                        43.621, -55.7499, 12.9722, 1])

        mat = Mat4(*mat.ravel())
        self.base.camera.setMat(mat)

        self.engine.render.reparentTo(self.base.render)

        # Update
        def update(task):
            self.engine.step()
            return task.cont
        
        taskMgr.add(update, 'update')
        
        for _ in range(20):
            taskMgr.step()
        time.sleep(1.0)
        
        self.base.destroy()
        self.base.graphicsEngine.removeAllWindows()
        
    def testConnectToRenderWorld(self):
        
        house = House.loadFromJson(os.path.join(TEST_SUNCG_DATA_DIR, "house", "0004d52d1aeeb8ae6de39d6bd993e992", "house.json"),
                                   TEST_SUNCG_DATA_DIR)
        self.engine.addHouseToScene(house)
        
        renderWorld = Panda3dRenderWorld(shadowing=False, showCeiling=False)
        renderWorld.addHouseToScene(house)
        renderWorld.addDefaultLighting()
        
        mat = np.array([0.999992, 0.00394238, 0, 0,
                        -0.00295702, 0.750104, -0.661314, 0,
                        -0.00260737, 0.661308, 0.75011, 0,
                        43.621, -55.7499, 12.9722, 1])
        renderWorld.setCamera(mat)
        
        self.engine.connectToRenderWorld(renderWorld)
        
        # Update
        def update(task):
            self.engine.step()
            return task.cont
        
        taskMgr.add(update, 'update')
        
        for _ in range(20):
            taskMgr.step()
        
        renderWorld.step()
        image = renderWorld.getRgbImage()
        
        fig = plt.figure()
        plt.axis("off")
        ax = plt.subplot(111)
        ax.imshow(image)
        plt.show(block=False)
        time.sleep(1.0)
        plt.close(fig)
        renderWorld.destroy()
        
if __name__ == '__main__':
    logging.basicConfig(level=logging.WARN)
    np.seterr(all='raise')
    unittest.main()
        