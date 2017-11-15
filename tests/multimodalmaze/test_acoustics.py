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
from panda3d.core import LVector3f, SceneGraphAnalyzer

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data")
TEST_SUNCG_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "suncg")

from multimodalmaze.core import House, Object, Agent
from multimodalmaze.acoustics import EvertAcousticWorld, CipicHRTF, FilterBank, \
                                     MaterialAbsorptionTable, AirAttenuationTable

class TestMaterialAbsorptionTable(unittest.TestCase):

    def testGetAbsorptionCoefficients(self):
        coefficientsDb, frequencies = MaterialAbsorptionTable.getAbsorptionCoefficients(category='hard surfaces', material='marble floor')
        self.assertTrue(len(frequencies) == 7)
        self.assertTrue(len(coefficientsDb) == 7)
        self.assertTrue(np.all(coefficientsDb <= 0.0))
        
        coefficientsDb, frequencies = MaterialAbsorptionTable.getAbsorptionCoefficients(category='floor coverings', material='carpet on hair felt or foam rubber')

class TestAirAttenuationTable(unittest.TestCase):

    def testGetAttenuations(self):
        
        for distance in [1.0, 24.0, 300.0]:
            for temperature in [10.0, 20.0, 35.0]:
                for relativeHumidity in [30.0, 55.0, 75.0]:
                    attenuationsDb, frequencies = AirAttenuationTable.getAttenuations(distance, temperature, relativeHumidity)
                    self.assertTrue(len(frequencies) == 7)
                    self.assertTrue(len(attenuationsDb) == 7)
                    self.assertTrue(np.all(attenuationsDb <= 0.0))

class TestCipicHRTF(unittest.TestCase):
    
    def testInit(self):
        hrtf = CipicHRTF(filename=os.path.join(TEST_DATA_DIR, 'hrtf', 'cipic_hrir.mat'),
                         samplingRate=16000.0)
        impulse = hrtf.getImpulseResponse(azimut=50.0, elevation=75.0)
        self.assertTrue(np.array_equal(hrtf.impulses.shape, [25,50,2,200]))
        self.assertTrue(impulse.shape[0] == 2)
        
class TestFilterBank(unittest.TestCase):
    
    def testInit(self):
        n = 256
        centerFrequencies = np.array([125, 500, 1000, 2000, 4000], dtype=np.float)
        samplingRate = 16000
        filterbank = FilterBank(n, centerFrequencies, samplingRate)
        impulse = filterbank.getScaledImpulseResponse()
        self.assertTrue(impulse.ndim == 1)
        self.assertTrue(impulse.shape[0] == n + 1)
        
        n = 511
        centerFrequencies = np.array([125, 500, 1000, 2000, 4000], dtype=np.float)
        samplingRate = 16000
        filterbank = FilterBank(n, centerFrequencies, samplingRate)
        impulse = filterbank.getScaledImpulseResponse()
        self.assertTrue(impulse.ndim == 1)
        self.assertTrue(impulse.shape[0] == n)
        
    def testDisplay(self):
        n = 257
        centerFrequencies = np.array([125, 500, 1000, 2000, 4000], dtype=np.float)
        samplingRate = 16000
        filterbank = FilterBank(n, centerFrequencies, samplingRate)
        fig = filterbank.display(merged=False)
        plt.show(block=False)
        time.sleep(1.0)
        plt.close(fig)
        fig = filterbank.display(merged=True)
        plt.show(block=False)
        time.sleep(1.0)
        plt.close(fig)
        
        n = 257
        scales = np.array([1.0, 0.5, 0.25, 0.5, 0.05])
        centerFrequencies = np.array([125, 500, 1000, 2000, 4000], dtype=np.float)
        samplingRate = 16000
        filterbank = FilterBank(n, centerFrequencies, samplingRate)
        fig = filterbank.display(scales, merged=False)
        plt.show(block=False)
        time.sleep(1.0)
        plt.close(fig)
        fig = filterbank.display(scales, merged=True)
        plt.show(block=False)
        time.sleep(1.0)
        plt.close(fig)
        
        impulse = filterbank.getScaledImpulseResponse(scales)
        fig = plt.figure()
        plt.plot(impulse)
        plt.show(block=False)
        time.sleep(1.0)
        plt.close(fig)
        
class TestEvertAcousticWorld(unittest.TestCase):
        
    def testInit(self):
        engine = EvertAcousticWorld(samplingRate=16000, maximumOrder=3)
        engine.destroy()
            
    def testRenderSimpleCubeRoom(self):
        
        engine = EvertAcousticWorld(samplingRate=16000, maximumOrder=4, materialAbsorption=False, frequencyDependent=False)

        # Define a simple cube (10 x 10 x 10 m) as room geometry
        roomSize = 10.0
        instanceId = 'room'
        modelId = '0'
        modelFilename = os.path.join(TEST_DATA_DIR, 'models', 'cube.egg')
        transform = np.array([[roomSize, 0.0, 0.0, 0.0],
                              [0.0, roomSize, 0.0, 0.0],
                              [0.0, 0.0, roomSize, 0.0],
                              [0.0, 0.0, 0.0, 1.0]])
        obj = Object(instanceId, modelId, modelFilename)
        obj.setTransform(transform)
        roomNode = engine.addObjectToScene(obj, mode='exact')
        roomNode.setRenderModeWireframe()
        
        # Define a listening agent
        agentSize = 0.1
        instanceId = 'agent'
        modelFilename = os.path.join(TEST_DATA_DIR, 'models', 'sphere.egg')
        transform = np.array([[agentSize, 0.0, 0.0, 0.0],
                              [0.0, agentSize, 0.0, 0.0],
                              [0.0, 0.0, agentSize, 0.0],
                              [0.0, 0.0, 0.0, 1.0]])
        agent = Agent(instanceId, modelFilename)
        agent.setTransform(transform)
        agentNode = engine.addAgentToScene(agent, interauralDistance=0.25)

        # Define a sound source
        sourceSize = 0.25
        instanceId = 'source'
        modelId = '0'
        modelFilename = os.path.join(TEST_DATA_DIR, 'models', 'sphere.egg')
        transform = np.array([[sourceSize, 0.0, 0.0, 0.0],
                              [0.0, sourceSize, 0.0, 0.0],
                              [0.0, 0.0, sourceSize, 0.0],
                              [0.0, 0.0, 0.0, 1.0]])
        source = Object(instanceId, modelId, modelFilename)
        source.setTransform(transform)
        sourceNode = engine.addStaticSourceToScene(source)
        sourceNode.setPos(0.0, 0.0, 0.0)
        
        center = engine.world.getCenter()
        self.assertTrue(np.allclose(engine.world.getMaxLength(), roomSize * 1000.0))
        self.assertTrue(np.allclose([center.x, center.y, center.z],[0.0, 0.0, 0.0]))
        self.assertTrue(engine.world.numElements() == 12)
        self.assertTrue(engine.world.numConvexElements() == 12)
        
        engine.updateBSP()
        
        # Configure the camera
        #NOTE: in Panda3D, the X axis points to the right, the Y axis is forward, and Z is up
        mat = np.array([0.999992, 0.00394238, 0, 0,
                        -0.00295702, 0.750104, -0.661314, 0,
                        -0.00260737, 0.661308, 0.75011, 0,
                        0.0, -25.0, 22, 1])
        engine.setCamera(mat)
        
        agentNode.setPos(0.25 * roomSize, -0.25 * roomSize, 0.3 * roomSize)
        engine.step()
        image1 = engine.getRgbImage()
        
        agentNode.setPos(0.35 * roomSize, -0.35 * roomSize, 0.4 * roomSize)
        engine.step()
        image2 = engine.getRgbImage()
        
        agentNode.setPos(-0.25 * roomSize, 0.25 * roomSize, -0.3 * roomSize)
        engine.step()
        image3 = engine.getRgbImage()
        
        fig = plt.figure(figsize=(16,8))
        ax = plt.subplot(131)
        ax.imshow(image1)
        ax = plt.subplot(132)
        ax.imshow(image2)
        ax = plt.subplot(133)
        ax.imshow(image3)
        plt.show(block=False)
        time.sleep(1.0)
        plt.close(fig)
        
        # Calculate and show impulse responses
        impulseLeft = engine.calculateImpulseResponse(engine.solutions[0], maxImpulseLength=1.0, threshold=120.0)
        impulseRight = engine.calculateImpulseResponse(engine.solutions[1], maxImpulseLength=1.0, threshold=120.0)
        
        fig = plt.figure()
        plt.plot(impulseLeft, color='b', label='Left channel')
        plt.plot(impulseRight, color='g', label='Right channel')
        plt.legend()
        plt.show(block=False)
        time.sleep(1.0)
        plt.close(fig)
        
        engine.destroy()
            
    def testRenderHouse(self):
        
        engine = EvertAcousticWorld(samplingRate=16000, maximumOrder=8, materialAbsorption=False, frequencyDependent=False, showCeiling=False)
        
        house = House.loadFromJson(os.path.join(TEST_SUNCG_DATA_DIR, "house", "0004d52d1aeeb8ae6de39d6bd993e992", "house.json"),
                                   TEST_SUNCG_DATA_DIR)

        engine.addHouseToScene(house)
        
        # Configure the camera
        #NOTE: in Panda3D, the X axis points to the right, the Y axis is forward, and Z is up
        mat = np.array([0.999992, 0.00394238, 0, 0,
                        -0.00295702, 0.750104, -0.661314, 0,
                        -0.00260737, 0.661308, 0.75011, 0,
                        43.621, -55.7499, 12.9722, 1])
        engine.setCamera(mat)
        engine.step()
        image = engine.getRgbImage()
        
        fig = plt.figure()
        plt.axis("off")
        ax = plt.subplot(111)
        ax.imshow(image)
        plt.show(block=False)
        time.sleep(1.0)
        plt.close(fig)
        
        engine.destroy()

    def testRenderRoom(self):

        engine = EvertAcousticWorld(samplingRate=16000, maximumOrder=2, materialAbsorption=True, frequencyDependent=True, showCeiling=False)
        
        # Define the scene geometry
        house = House.loadFromJson(os.path.join(TEST_SUNCG_DATA_DIR, "house", "0004d52d1aeeb8ae6de39d6bd993e992", "house.json"),
                                   TEST_SUNCG_DATA_DIR)
        room = house.rooms[1]
        self.assertTrue(room.instanceId == 'fr_0rm_1')
        self.assertTrue(len(room.objects) == 18)
        
        roomNode = engine.addRoomToScene(room)
        
        minRefBounds, maxRefBounds = roomNode.getTightBounds()
        refCenter = minRefBounds + (maxRefBounds - minRefBounds) / 2.0
        
        # Define a listening agent
        agentSize = 0.1
        instanceId = 'agent'
        modelFilename = os.path.join(TEST_DATA_DIR, 'models', 'sphere.egg')
        transform = np.array([[agentSize, 0.0, 0.0, 0.0],
                              [0.0, agentSize, 0.0, 0.0],
                              [0.0, 0.0, agentSize, 0.0],
                              [0.0, 0.0, 0.0, 1.0]])
        agent = Agent(instanceId, modelFilename)
        agent.setTransform(transform)
        agentNode = engine.addAgentToScene(agent, interauralDistance=0.5)
        agentNode.setHpr(90,0,0)
        agentNode.setPos(LVector3f(-2.0, 1.0, -0.75) + refCenter)
        
        # Define a sound source
        sourceSize = 0.25
        instanceId = 'source'
        modelId = '0'
        modelFilename = os.path.join(TEST_DATA_DIR, 'models', 'sphere.egg')
        transform = np.array([[sourceSize, 0.0, 0.0, 0.0],
                              [0.0, sourceSize, 0.0, 0.0],
                              [0.0, 0.0, sourceSize, 0.0],
                              [0.0, 0.0, 0.0, 1.0]])
        source = Object(instanceId, modelId, modelFilename)
        source.setTransform(transform)
        sourceNode = engine.addStaticSourceToScene(source)
        sourceNode.setPos(LVector3f(1.5, -1.0, -0.25) + refCenter)

        engine.updateBSP()
        
        # Configure the camera
        #NOTE: in Panda3D, the X axis points to the right, the Y axis is forward, and Z is up
        mat = np.array([[1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [refCenter.x-0.5, refCenter.y-0.5, 15, 1]])
        engine.setCamera(mat)
        
        engine.step()
        image = engine.getRgbImage()
        
        fig = plt.figure()
        plt.axis("off")
        ax = plt.subplot(111)
        ax.imshow(image)
        plt.show(block=False)
        time.sleep(1.0)
        plt.close(fig)
        
        # Calculate and show impulse responses
        impulseLeft = engine.calculateImpulseResponse(engine.solutions[0], maxImpulseLength=1.0, threshold=120.0)
        impulseRight = engine.calculateImpulseResponse(engine.solutions[1], maxImpulseLength=1.0, threshold=120.0)
        
        fig = plt.figure()
        plt.plot(impulseLeft, color='b', label='Left channel')
        plt.plot(impulseRight, color='g', label='Right channel')
        plt.legend()
        plt.show(block=False)
        time.sleep(1.0)
        plt.close(fig)
        
        engine.destroy()

    def testRenderRoomPolygons(self):

        engine = EvertAcousticWorld(samplingRate=16000, maximumOrder=1, materialAbsorption=False, frequencyDependent=False, showCeiling=False)
        
        # Define the scene geometry
        house = House.loadFromJson(os.path.join(TEST_SUNCG_DATA_DIR, "house", "0004d52d1aeeb8ae6de39d6bd993e992", "house.json"),
                                   TEST_SUNCG_DATA_DIR)
        room = house.rooms[1]
        self.assertTrue(room.instanceId == 'fr_0rm_1')
        self.assertTrue(len(room.objects) == 18)
        
        roomNode = engine.addRoomToScene(room)
        
        minRefBounds, maxRefBounds = roomNode.getTightBounds()
        refCenter = minRefBounds + (maxRefBounds - minRefBounds) / 2.0
        
        engine.updateBSP()
        
        sga = SceneGraphAnalyzer()
        sga.addNode(engine.render.node())
        self.assertTrue(engine.world.numElements() <= sga.get_num_triangles_in_strips())
        self.assertTrue(engine.world.numConvexElements() <= sga.get_num_triangles_in_strips())
        
        # Configure the camera
        #NOTE: in Panda3D, the X axis points to the right, the Y axis is forward, and Z is up
        mat = np.array([[1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [refCenter.x-0.5, refCenter.y-0.5, 15, 1]])
        engine.setCamera(mat)
        
        engine.step()
        image = engine.getRgbImage()
        
        fig = plt.figure()
        plt.axis("off")
        ax = plt.subplot(111)
        ax.imshow(image)
        plt.show(block=False)
        time.sleep(1.0)
        plt.close(fig)
        
        engine.destroy()

if __name__ == '__main__':
    logging.basicConfig(level=logging.WARN)
    np.seterr(all='raise')
    unittest.main()
    
