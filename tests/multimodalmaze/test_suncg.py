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
from multimodalmaze.core import Scene
from panda3d.core import TransformState, LVector3f
from multimodalmaze.utils import Viewer

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data")
TEST_SUNCG_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "suncg")

from multimodalmaze.suncg import ModelCategoryMapping, ModelInformation, ObjectVoxelData,\
    SunCgSceneLoader, SunCgModelLights, loadModel

class TestModelCategoryMapping(unittest.TestCase):

    def testInit(self):
        _ = ModelCategoryMapping(os.path.join(TEST_SUNCG_DATA_DIR, "metadata", "ModelCategoryMapping.csv"))

class TestModelInformation(unittest.TestCase):

    def testInit(self):
        _ = ModelInformation(os.path.join(TEST_SUNCG_DATA_DIR, "metadata", "models.csv"))
    
    def testGetModelInfo(self):
        info = ModelInformation(os.path.join(TEST_SUNCG_DATA_DIR, "metadata", "models.csv"))
        _ = info.getModelInfo('261')
        
class TestObjectVoxelData(unittest.TestCase):
    
    def testGetFilledVolume(self):
        
        for modelId in ['83', '81', '561', '441', '317']:
            voxelData = ObjectVoxelData.fromFile(os.path.join(TEST_SUNCG_DATA_DIR, "object_vox", "object_vox_data", modelId, modelId + ".binvox"))
            volume = voxelData.getFilledVolume()
            self.assertTrue(np.array_equal(voxelData.voxels.shape, [128,128,128]))
            self.assertTrue(volume > 0)
        
class TestSunCgSceneLoader(unittest.TestCase):
        
    def testLoadHouseFromJson(self):
        scene = SunCgSceneLoader.loadHouseFromJson('0004d52d1aeeb8ae6de39d6bd993e992', TEST_SUNCG_DATA_DIR)
        self.assertTrue(scene.getTotalNbHouses() == 1)
        self.assertTrue(scene.getTotalNbRooms() == 4)
        self.assertTrue(scene.getTotalNbObjects() == 59)
        self.assertTrue(scene.getTotalNbAgents() == 1)
        
class TestSunCgModelLights(unittest.TestCase):
        
    def testLoadFromJson(self):
        filename = os.path.join(TEST_SUNCG_DATA_DIR, 'metadata', 'suncgModelLights.json')
        info = SunCgModelLights(filename)
        
        lights = info.getLightsForModel(modelId='s__1296')
        self.assertTrue(len(lights) == 2)
        
        lights = info.getLightsForModel(modelId='83')
        self.assertTrue(len(lights) == 0)
        
        lights = info.getLightsForModel(modelId='377')
        self.assertTrue(len(lights) == 1)
        
        lights = info.getLightsForModel(modelId='178')
        self.assertTrue(len(lights) == 1)
        
    def testRenderWithModelLights(self):
        
        filename = os.path.join(TEST_SUNCG_DATA_DIR, 'metadata', 'suncgModelLights.json')
        info = SunCgModelLights(filename)
        
        scene = Scene()
        
        modelId = 's__1296'
        modelFilename = os.path.join(TEST_SUNCG_DATA_DIR, "object", str(modelId), str(modelId) + ".egg")
        assert os.path.exists(modelFilename)
        model = loadModel(modelFilename)
        model.setName('model-' + str(modelId))
        model.show()
        
        objectsNp = scene.scene.attachNewNode('objects')
        objNp = objectsNp.attachNewNode('object-' + str(modelId))
        model.reparentTo(objNp)
        
        # Calculate the center of this object
        minBounds, maxBounds = model.getTightBounds()
        centerPos = minBounds + (maxBounds - minBounds) / 2.0
         
        # Add offset transform to make position relative to the center
        model.setTransform(TransformState.makePos(-centerPos))
        
        # Add lights to model
        for lightNp in info.getLightsForModel(modelId):
            lightNp.node().setShadowCaster(True, 512, 512)
            lightNp.reparentTo(model)
            scene.scene.setLight(lightNp)

        try:
            viewer = Viewer(scene, interactive=False, shadowing=True)
    
            viewer.cam.setTransform(TransformState.makePos(LVector3f(0.5, 0.5, 3.0)))
            viewer.cam.lookAt(lightNp)
            
            for _ in range(20):
                viewer.step()
            time.sleep(1.0)
            
        finally:
            viewer.destroy()
            viewer.graphicsEngine.removeAllWindows()
        
        
if __name__ == '__main__':
    logging.basicConfig(level=logging.WARN)
    np.seterr(all='raise')
    unittest.main()
        