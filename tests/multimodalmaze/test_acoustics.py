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
from panda3d.core import TransformState, LVecBase3f, LMatrix4f
from home_platform.suncg import SunCgSceneLoader, loadModel
from home_platform.core import Scene
from home_platform.utils import Viewer

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data")
TEST_SUNCG_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "suncg")

from home_platform.acoustics import EvertAcoustics, CipicHRTF, FilterBank, \
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
                         samplingRate=44100.0)
        impulse = hrtf.getImpulseResponse(azimut=50.0, elevation=75.0)
        self.assertTrue(np.array_equal(hrtf.impulses.shape, [25,50,2,200]))
        self.assertTrue(np.array_equal(impulse.shape, [2,200]))
        
        hrtf = CipicHRTF(filename=os.path.join(TEST_DATA_DIR, 'hrtf', 'cipic_hrir.mat'),
                         samplingRate=16000.0)
        impulse = hrtf.getImpulseResponse(azimut=50.0, elevation=75.0)
        self.assertTrue(np.array_equal(hrtf.impulses.shape, [25,50,2,72]))
        self.assertTrue(np.array_equal(impulse.shape, [2,72]))
        
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
        
class TestEvertAcoustics(unittest.TestCase):
        
    def testInit(self):
        
        samplingRate = 16000.0
        scene = SunCgSceneLoader.loadHouseFromJson("0004d52d1aeeb8ae6de39d6bd993e992", TEST_SUNCG_DATA_DIR)
        hrtf = CipicHRTF(os.path.join(TEST_DATA_DIR, 'hrtf', 'cipic_hrir.mat'), samplingRate)
            
        engine = EvertAcoustics(scene, hrtf, samplingRate, maximumOrder=3, debug=True)
        engine.destroy()
            
    def testRenderSimpleCubeRoom(self):
         
        samplingRate = 16000.0
        scene = Scene()
        hrtf = CipicHRTF(os.path.join(TEST_DATA_DIR, 'hrtf', 'cipic_hrir.mat'), samplingRate)
        
        viewer = Viewer(scene)
        viewer.disableMouse()
    
        # Define a simple cube (10 x 10 x 10 m) as room geometry
        roomSize = 10.0
        modelId = 'room-0'
        modelFilename = os.path.join(TEST_DATA_DIR, 'models', 'cube.egg')
        layoutNp = scene.scene.attachNewNode('layouts')
        objectNp = layoutNp.attachNewNode('object-' + modelId)
        objectNp.setTag('acoustics-mode', 'obstacle')
        model = loadModel(modelFilename)
        model.setName('model-' + modelId)
        model.setTransform(TransformState.makeScale(roomSize))
        model.setRenderModeWireframe()
        model.reparentTo(objectNp)
        objectNp.setPos(LVecBase3f(0.0, 0.0, 0.0))
        
        agentNp = scene.agents[0]
        
        # Define a sound source
        sourceSize = 0.25
        modelId = 'source-0'
        modelFilename = os.path.join(TEST_DATA_DIR, 'models', 'sphere.egg')
        objectsNp = scene.scene.attachNewNode('objects')
        objectNp = objectsNp.attachNewNode('object-' + modelId)
        objectNp.setTag('acoustics-mode', 'source')
        model = loadModel(modelFilename)
        model.setName('model-' + modelId)
        model.setTransform(TransformState.makeScale(sourceSize))
        model.reparentTo(objectNp)
        objectNp.setPos(LVecBase3f(0.0, 0.0, 0.0))
        
        acoustics = EvertAcoustics(scene, hrtf, samplingRate, maximumOrder=3, materialAbsorption=False, frequencyDependent=False, debug=True)
        acoustics.updateGeometry()
        center = acoustics.world.getCenter()
        self.assertTrue(np.allclose(acoustics.world.getMaxLength()/1000.0, roomSize))
        self.assertTrue(np.allclose([center.x, center.y, center.z],[0.0, 0.0, 0.0]))
        self.assertTrue(acoustics.world.numElements() == 12)
        self.assertTrue(acoustics.world.numConvexElements() == 12)
        
        # Configure the camera
        #NOTE: in Panda3D, the X axis points to the right, the Y axis is forward, and Z is up
        mat = np.array([0.999992, 0.00394238, 0, 0,
                        -0.00295702, 0.750104, -0.661314, 0,
                        -0.00260737, 0.661308, 0.75011, 0,
                        0.0, -25.0, 22, 1])
        mat = LMatrix4f(*mat.ravel())
        viewer.cam.setMat(mat)
        
        agentNp.setPos(LVecBase3f(0.25 * roomSize, -0.25 * roomSize, 0.3 * roomSize))
        for _ in range(10):
            viewer.step()
        time.sleep(1.0)
         
        agentNp.setPos(LVecBase3f(0.35 * roomSize, -0.35 * roomSize, 0.4 * roomSize))
        for _ in range(10):
            viewer.step()
        time.sleep(1.0)
        
        agentNp.setPos(LVecBase3f(-0.25 * roomSize, 0.25 * roomSize, -0.3 * roomSize))
        for _ in range(10):
            viewer.step()
        time.sleep(1.0)
        
        # Calculate and show impulse responses
        impulse = acoustics.calculateImpulseResponses()[0]
        
        fig = plt.figure()
        plt.plot(impulse.impulse[0], color='b', label='Left channel')
        plt.plot(impulse.impulse[1], color='g', label='Right channel')
        plt.legend()
        plt.show(block=False)
        time.sleep(1.0)
        plt.close(fig)
        
        acoustics.destroy()
        viewer.destroy()
        viewer.graphicsEngine.removeAllWindows()
      
    def testRenderHouse(self):
         
        scene = SunCgSceneLoader.loadHouseFromJson("0004d52d1aeeb8ae6de39d6bd993e992", TEST_SUNCG_DATA_DIR)
            
        samplingRate = 16000.0
        hrtf = CipicHRTF(os.path.join(TEST_DATA_DIR, 'hrtf', 'cipic_hrir.mat'), samplingRate)
        acoustics = EvertAcoustics(scene, hrtf, samplingRate, maximumOrder=3, debug=True)
        
        # Hide ceilings
        for nodePath in scene.scene.findAllMatches('**/layouts/*/acoustics/*c'):
            nodePath.hide()
         
        viewer = Viewer(scene)
        viewer.disableMouse()
         
        # Configure the camera
        #NOTE: in Panda3D, the X axis points to the right, the Y axis is forward, and Z is up
        mat = np.array([0.999992, 0.00394238, 0, 0,
                        -0.00295702, 0.750104, -0.661314, 0,
                        -0.00260737, 0.661308, 0.75011, 0,
                        43.621, -55.7499, 12.9722, 1])
        mat = LMatrix4f(*mat.ravel())
        viewer.cam.setMat(mat)
        
        for _ in range(20):
            acoustics.step(dt=0.1)
            viewer.step()
        time.sleep(1.0)
         
        acoustics.destroy()
        viewer.destroy()
        viewer.graphicsEngine.removeAllWindows()

    def testRenderHouseWithAcousticsPath(self):

        scene = SunCgSceneLoader.loadHouseFromJson("0004d52d1aeeb8ae6de39d6bd993e992", TEST_SUNCG_DATA_DIR)
        
        agentNp = scene.agents[0]
        agentNp.setPos(LVecBase3f(45, -42.5, 1.6))
        agentNp.setHpr(45,0,0)
        
        # Define a sound source
        sourceSize = 0.25
        modelId = 'source-0'
        modelFilename = os.path.join(TEST_DATA_DIR, 'models', 'sphere.egg')
        objectsNp = scene.scene.attachNewNode('objects')
        objectNp = objectsNp.attachNewNode('object-' + modelId)
        objectNp.setTag('acoustics-mode', 'source')
        model = loadModel(modelFilename)
        model.setName('model-' + modelId)
        model.setTransform(TransformState.makeScale(sourceSize))
        model.reparentTo(objectNp)
        objectNp.setPos(LVecBase3f(39, -40.5, 1.5))
        
        samplingRate = 16000.0
        hrtf = CipicHRTF(os.path.join(TEST_DATA_DIR, 'hrtf', 'cipic_hrir.mat'), samplingRate)
        acoustics = EvertAcoustics(scene, hrtf, samplingRate, maximumOrder=2, debug=True)
        
        # Hide ceilings
        for nodePath in scene.scene.findAllMatches('**/layouts/*/acoustics/*c'):
            nodePath.hide()
         
        viewer = Viewer(scene)
        viewer.disableMouse()
         
        # Configure the camera
        #NOTE: in Panda3D, the X axis points to the right, the Y axis is forward, and Z is up
        center = agentNp.getNetTransform().getPos()
        mat = np.array([[1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [center.x, center.y, 20, 1]])
        mat = LMatrix4f(*mat.ravel())
        viewer.cam.setMat(mat)
        
        for _ in range(10):
            viewer.step()
        time.sleep(1.0)
         
        viewer.destroy()
        viewer.graphicsEngine.removeAllWindows()
         
        # Calculate and show impulse responses
        impulse = acoustics.calculateImpulseResponses()[0]
         
        fig = plt.figure()
        plt.plot(impulse.impulse[0], color='b', label='Left channel')
        plt.plot(impulse.impulse[1], color='g', label='Right channel')
        plt.legend()
        plt.show(block=False)
        time.sleep(1.0)
        plt.close(fig)
         
        acoustics.destroy()

if __name__ == '__main__':
    logging.basicConfig(level=logging.WARN)
    np.seterr(all='raise')
    unittest.main()
