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
import logging
import numpy as np
import unittest
from panda3d.core import NodePath

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data")
TEST_SUNCG_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "suncg")

from multimodalmaze.semantic import MaterialColorTable, SuncgSemantics, MaterialTable, DimensionTable
from multimodalmaze.suncg import loadModel, SunCgSceneLoader
    
class TestMaterialColorTable(unittest.TestCase):
    
    def testGetColorsFromObjectBasic(self):
        
        modelId = '317'
        modelFilename = os.path.join(TEST_SUNCG_DATA_DIR, "object", str(modelId), str(modelId) + ".egg")
        assert os.path.exists(modelFilename)
        model = loadModel(modelFilename)
        model.setName('model-' + str(modelId))
        obj = NodePath('object-' + str(modelId))
        model.reparentTo(obj)
        colorDescriptions = MaterialColorTable.getColorsFromObject(obj, mode='basic')
        self.assertTrue(len(colorDescriptions) == 1)
        self.assertTrue(colorDescriptions[0] == "silver")
        
        modelId = '83'
        modelFilename = os.path.join(TEST_SUNCG_DATA_DIR, "object", str(modelId), str(modelId) + ".egg")
        assert os.path.exists(modelFilename)
        model = loadModel(modelFilename)
        model.setName('model-' + str(modelId))
        obj = NodePath('object-' + str(modelId))
        model.reparentTo(obj)
        colorDescriptions = MaterialColorTable.getColorsFromObject(obj, mode='basic')
        self.assertTrue(len(colorDescriptions) == 1)
        self.assertTrue(colorDescriptions[0] == "white")
        
    def testGetColorsFromObjectTransparent(self):
        
        modelId = 'sphere'
        modelFilename = os.path.join(TEST_DATA_DIR, "models", "sphere.egg")
        assert os.path.exists(modelFilename)
        model = loadModel(modelFilename)
        model.setName('model-' + str(modelId))
        obj = NodePath('object-' + str(modelId))
        model.reparentTo(obj)
        colorDescriptions = MaterialColorTable.getColorsFromObject(obj, mode='basic')        
        self.assertTrue(len(colorDescriptions) == 1)
        self.assertTrue(colorDescriptions[0] == "maroon")
        
    def testGetColorsFromObjectAdvanced(self):
        
        modelId = '317'
        modelFilename = os.path.join(TEST_SUNCG_DATA_DIR, "object", str(modelId), str(modelId) + ".egg")
        assert os.path.exists(modelFilename)
        model = loadModel(modelFilename)
        model.setName('model-' + str(modelId))
        obj = NodePath('object-' + str(modelId))
        model.reparentTo(obj)
        colorDescriptions = MaterialColorTable.getColorsFromObject(obj, mode='advanced')
        self.assertTrue(len(colorDescriptions) == 1)
        self.assertTrue(colorDescriptions[0] == "navajo white")
        
        colorDescriptions = MaterialColorTable.getColorsFromObject(obj, mode='advanced', thresholdRelArea=0.0)
        self.assertTrue(len(colorDescriptions) == 2)
        self.assertTrue(colorDescriptions[0] == "navajo white")
        self.assertTrue(colorDescriptions[1] == "dark slate gray")
        
        modelId = '210'
        modelFilename = os.path.join(TEST_SUNCG_DATA_DIR, "object", str(modelId), str(modelId) + ".egg")
        assert os.path.exists(modelFilename)
        model = loadModel(modelFilename)
        model.setName('model-' + str(modelId))
        obj = NodePath('object-' + str(modelId))
        model.reparentTo(obj)
        colorDescriptions = MaterialColorTable.getColorsFromObject(obj, mode='advanced')
        self.assertTrue(len(colorDescriptions) == 2)
        self.assertTrue(colorDescriptions[0] == "dark gray")
        self.assertTrue(colorDescriptions[1] == "cadet blue")
        
    def testGetColorsFromObjectXkcd(self):
        
        modelId = '317'
        modelFilename = os.path.join(TEST_SUNCG_DATA_DIR, "object", str(modelId), str(modelId) + ".egg")
        assert os.path.exists(modelFilename)
        model = loadModel(modelFilename)
        model.setName('model-' + str(modelId))
        obj = NodePath('object-' + str(modelId))
        model.reparentTo(obj)
        colorDescriptions = MaterialColorTable.getColorsFromObject(obj, mode='xkcd')
        self.assertTrue(len(colorDescriptions) == 1)
        self.assertTrue(colorDescriptions[0] == "light peach")

class TestMaterialTable(unittest.TestCase):
        
    def testGetMaterialNameFromObject(self):
         
        modelId = '317'
        modelFilename = os.path.join(TEST_SUNCG_DATA_DIR, "object", str(modelId), str(modelId) + ".egg")
        assert os.path.exists(modelFilename)
        model = loadModel(modelFilename)
        model.setName('model-' + str(modelId))
        obj = NodePath('object-' + str(modelId))
        model.reparentTo(obj)
        materialDescriptions = MaterialTable.getMaterialNameFromObject(obj)
        self.assertTrue(len(materialDescriptions) == 1)
        self.assertTrue(materialDescriptions[0] == "wood")
         
        materialDescriptions = MaterialTable.getMaterialNameFromObject(obj, thresholdRelArea=0.0)
        self.assertTrue(len(materialDescriptions) == 1)
        self.assertTrue(materialDescriptions[0] == "wood")
        
class TestDimensionTable(unittest.TestCase):
        
    def testGetDimensionsFromObject(self):
        
        modelId = '274'
        modelFilename = os.path.join(TEST_SUNCG_DATA_DIR, "object", str(modelId), str(modelId) + ".egg")
        assert os.path.exists(modelFilename)
        model = loadModel(modelFilename)
        model.setName('model-' + str(modelId))
        obj = NodePath('object-' + str(modelId))
        model.reparentTo(obj)
         
        # XXX: should use the full metadata files if descriptors are not precomputed
        modelInfoFilename = os.path.join(TEST_SUNCG_DATA_DIR, "metadata", "models.csv")
        modelCatFilename = os.path.join(TEST_SUNCG_DATA_DIR, "metadata", "ModelCategoryMapping.csv")
        dimensionDescription = DimensionTable().getDimensionsFromModelId(modelId, modelInfoFilename, modelCatFilename)
        self.assertTrue(dimensionDescription == 'normal')

class TestSuncgSemantics(unittest.TestCase):
     
    def testDescribe(self):

        scene = SunCgSceneLoader.loadHouseFromJson("0004d52d1aeeb8ae6de39d6bd993e992", TEST_SUNCG_DATA_DIR)
        semantics = SuncgSemantics(scene, TEST_SUNCG_DATA_DIR)
         
        objNode = scene.scene.find('**/object-561*')
        desc = semantics.describeObject(objNode)
        self.assertTrue(desc == "small linen coffee table made of wood")
        
if __name__ == '__main__':
    logging.basicConfig(level=logging.WARN)
    np.seterr(all='raise')
    unittest.main()
    