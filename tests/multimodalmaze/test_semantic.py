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

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data")
TEST_SUNCG_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "suncg")

from multimodalmaze.core import Object
from multimodalmaze.semantic import MaterialColorTable, SuncgSemanticWorld,\
    MaterialTable
    
class TestMaterialColorTable(unittest.TestCase):
    
    def testBasic(self):
        
        modelId = '317'
        modelFilename = os.path.join(TEST_SUNCG_DATA_DIR, "object", str(modelId), str(modelId) + ".egg")
        assert os.path.exists(modelFilename)
        instanceId = str(modelId) + '-0'
        obj = Object(instanceId, modelId, modelFilename)
        colorDescriptions = MaterialColorTable.getBasicColorsFromObject(obj, mode='basic')
        self.assertTrue(len(colorDescriptions) == 1)
        self.assertTrue(colorDescriptions[0] == "silver")
        
        modelId = '83'
        modelFilename = os.path.join(TEST_SUNCG_DATA_DIR, "object", str(modelId), str(modelId) + ".egg")
        assert os.path.exists(modelFilename)
        instanceId = str(modelId) + '-0'
        obj = Object(instanceId, modelId, modelFilename)
        colorDescriptions = MaterialColorTable.getBasicColorsFromObject(obj, mode='basic')
        self.assertTrue(len(colorDescriptions) == 1)
        self.assertTrue(colorDescriptions[0] == "white")
        
    def testBasicTransparent(self):
        
        modelId = 'sphere'
        modelFilename = os.path.join(TEST_DATA_DIR, "models", "sphere.egg")
        assert os.path.exists(modelFilename)
        instanceId = str(modelId) + '-0'
        obj = Object(instanceId, modelId, modelFilename)
        colorDescriptions = MaterialColorTable.getBasicColorsFromObject(obj, mode='basic')
        self.assertTrue(len(colorDescriptions) == 1)
        self.assertTrue(colorDescriptions[0] == "maroon")
        
    def testAdvanced(self):
        
        modelId = '317'
        modelFilename = os.path.join(TEST_SUNCG_DATA_DIR, "object", str(modelId), str(modelId) + ".egg")
        assert os.path.exists(modelFilename)
        instanceId = str(modelId) + '-0'
        obj = Object(instanceId, modelId, modelFilename)
        colorDescriptions = MaterialColorTable.getBasicColorsFromObject(obj, mode='advanced')
        self.assertTrue(len(colorDescriptions) == 1)
        self.assertTrue(colorDescriptions[0] == "navajo white")
        
        colorDescriptions = MaterialColorTable.getBasicColorsFromObject(obj, mode='advanced', thresholdRelArea=0.0)
        self.assertTrue(len(colorDescriptions) == 2)
        self.assertTrue(colorDescriptions[0] == "navajo white")
        self.assertTrue(colorDescriptions[1] == "dark slate gray")
        
        modelId = '210'
        modelFilename = os.path.join(TEST_SUNCG_DATA_DIR, "object", str(modelId), str(modelId) + ".egg")
        assert os.path.exists(modelFilename)
        instanceId = str(modelId) + '-0'
        obj = Object(instanceId, modelId, modelFilename)
        colorDescriptions = MaterialColorTable.getBasicColorsFromObject(obj, mode='advanced')
        self.assertTrue(len(colorDescriptions) == 2)
        self.assertTrue(colorDescriptions[0] == "dark gray")
        self.assertTrue(colorDescriptions[1] == "cadet blue")
        
    def testXkcd(self):
        
        modelId = '317'
        modelFilename = os.path.join(TEST_SUNCG_DATA_DIR, "object", str(modelId), str(modelId) + ".egg")
        assert os.path.exists(modelFilename)
        instanceId = str(modelId) + '-0'
        obj = Object(instanceId, modelId, modelFilename)
        colorDescriptions = MaterialColorTable.getBasicColorsFromObject(obj, mode='xkcd')
        self.assertTrue(len(colorDescriptions) == 1)
        self.assertTrue(colorDescriptions[0] == "light peach")
       
class TestMaterialTable(unittest.TestCase):
       
    def testGetMaterialNameFromObject(self):
        
        modelId = '317'
        modelFilename = os.path.join(TEST_SUNCG_DATA_DIR, "object", str(modelId), str(modelId) + ".egg")
        assert os.path.exists(modelFilename)
        instanceId = str(modelId) + '-0'
        obj = Object(instanceId, modelId, modelFilename)
        materialDescriptions = MaterialTable.getMaterialNameFromObject(obj)
        self.assertTrue(len(materialDescriptions) == 1)
        self.assertTrue(materialDescriptions[0] == "wood")
        
        materialDescriptions = MaterialTable.getMaterialNameFromObject(obj, thresholdRelArea=0.0)
        self.assertTrue(len(materialDescriptions) == 1)
        self.assertTrue(materialDescriptions[0] == "wood")
       
class TestSuncgSemanticWorld(unittest.TestCase):
     
    def testInit(self):
        _ = SuncgSemanticWorld(TEST_SUNCG_DATA_DIR)
        
    def testDescribe(self):
        world = SuncgSemanticWorld(TEST_SUNCG_DATA_DIR)
        
        modelId = '561'
        modelFilename = os.path.join(TEST_SUNCG_DATA_DIR, "object", str(modelId), str(modelId) + ".egg")
        assert os.path.exists(modelFilename)
        instanceId = str(modelId) + '-0'
        obj = Object(instanceId, modelId, modelFilename)
        desc = world.describeObject(obj)
        self.assertTrue(desc == "linen coffee table made of wood")
        
if __name__ == '__main__':
    logging.basicConfig(level=logging.WARN)
    np.seterr(all='raise')
    unittest.main()
    