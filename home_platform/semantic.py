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

from string import digits

from home_platform.constants import MATERIAL_TABLE, MATERIAL_COLOR_TABLE
from home_platform.suncg import ModelCategoryMapping, ModelInformation,\
    ObjectVoxelData
from home_platform.rendering import getColorAttributesFromModel

logger = logging.getLogger(__name__)

class MaterialTable(object):
    @staticmethod
    def getMaterialNameFromObject(obj, thresholdRelArea=0.2):

        nodePath = obj.find('**/model*')

        # Get the list of materials
        areas, _, _, textures = getColorAttributesFromModel(nodePath)

        # Get the most dominant material based on threshold on relative surface area
        materialDescriptions = []
        for area, texture in zip(areas, textures):
            if texture is None or not area >= thresholdRelArea: continue

            # Remove any digits
            texture = texture.translate(None, digits)

            # Remove trailing underscores
            texture = texture.rstrip("_")

            # NOTE: handle many variations of textile and wood in SUNCG texture names
            if "textile" in texture:
                texture = "textile"
            if "wood" in texture:
                texture = "wood"

            if texture in MATERIAL_TABLE["materials"]:
                textureName = MATERIAL_TABLE["materials"][texture]
                materialDescriptions.append(textureName)
            else:
                logger.debug('Unsupported texture basename '
                             'for material semantics: %s' % (texture))

        # Remove duplicates (if any)
        materialDescriptions = list(set(materialDescriptions))

        return materialDescriptions


class MaterialColorTable(object):
    @staticmethod
    def getColorsFromObject(obj, mode='advanced', thresholdRelArea=0.2):

        nodePath = obj.find('**/model*')

        # Get the list of materials
        areas, colors, transparencies, _ = getColorAttributesFromModel(nodePath)

        if mode == 'basic':
            table = MATERIAL_COLOR_TABLE["BasicColorTable"]
        elif mode == 'advanced':
            table = MATERIAL_COLOR_TABLE["AdvancedColorTable"]
        elif mode == 'xkcd':
            table = MATERIAL_COLOR_TABLE["XkcdColorTable"]
        else:
            raise Exception('Unsupported color mode: %s' % (mode))

        # Get the most dominant colors based on threshold on relative surface area
        colorDescriptions = []
        for area, color, _ in zip(areas, colors, transparencies):
            if not area >= thresholdRelArea: continue

            # TODO: compare color in HSV or HSL domain instead of RGB?
            # hsvColor = colorsys.rgb_to_hsv(*color)

            # Find nearest color
            minDistance = np.Inf
            bestColorName = None
            for colorName, refColor in table.iteritems():
                dist = np.linalg.norm(np.array(refColor) /
                                      255.0 - np.array(color), ord=2)
                if dist < minDistance:
                    minDistance = dist
                    bestColorName = colorName

            colorDescriptions.append(bestColorName)

        # Remove duplicates (if any)
        colorDescriptions = list(set(colorDescriptions))

        return colorDescriptions

class DimensionTable(object):
    
    #NOTE: This table is assumed to be sorted.
    #      The values are in units of standard deviation (e.g. 2.0 x sigma)
    overallSizeTable = [ ['tiny', -2.0],
                          ['small', -1.0],
                          #['normal', 0.0],
                          ['large', 1.0],
                          ['huge', 2.0]]
    
    @staticmethod
    def getDimensionsFromModelId(modelId, modelInfoFilename, modelCatFilename):
        
        modelInfo = ModelInformation(modelInfoFilename)
        modelCat = ModelCategoryMapping(modelCatFilename)
        
        refModelDimensions = None
        otherSimilarDimensions = []
        refModelId = str(modelId)
        refCategory = modelCat.getCoarseGrainedCategoryForModelId(refModelId)
        for modelId in modelInfo.model_info.keys():
            category = modelCat.getCoarseGrainedCategoryForModelId(modelId)
            if refCategory == category:
                info = modelInfo.getModelInfo(modelId)
                
                # FIXME: handle the general case where for the front vector, do not ignore
                # NOTE: SUNCG is using the Y-up coordinate system
                frontVec = info['front']
                if np.count_nonzero(frontVec) > 1 or not np.array_equal(frontVec, [0,0,1]):
                    continue

                width, height, depth = info['aligned_dims'] / 100.0 # cm to m
                otherSimilarDimensions.append([width, height, depth])
                
                if refModelId == modelId:
                    refModelDimensions = np.array([width, height, depth])
                
        otherSimilarDimensions = np.array(otherSimilarDimensions)
        logger.debug('Number of similar objects found in dataset: %d' % (otherSimilarDimensions.shape[0]))

        # Volume statistics (assume a gaussian distribution)
        # XXX: use a more general histogram method to define the categories, rather than simply comparing the deviation to the mean
        refVolume = np.prod(refModelDimensions)
        otherVolumes = np.prod(otherSimilarDimensions, axis=-1)
        mean = np.mean(otherVolumes)
        std = np.std(otherVolumes)
        
        # Compare the deviation to the mean
        overallSizeTag = None
        diff = refVolume - mean
        for tag, threshold in DimensionTable.overallSizeTable:
            if threshold >= 0.0:
                if diff > threshold * std:
                    overallSizeTag = tag
            else:
                if diff < threshold * std:
                    overallSizeTag = tag
                    
        if overallSizeTag is None:
            overallSizeTag = 'normal'
        
        return overallSizeTag

class SuncgSemantics(object):
    
    # XXX: is not a complete list of movable objects, and the list is redundant with the one for physics
    movableObjectCategories = ['table', 'dressing_table', 'sofa', 'trash_can', 'chair', 'ottoman', 'bed']
    
    def __init__(self, scene, suncgDatasetRoot):
        self.scene = scene
        self.suncgDatasetRoot = suncgDatasetRoot
        
        if suncgDatasetRoot is not None:
            self.categoryMapping = ModelCategoryMapping(
                os.path.join(
                    self.suncgDatasetRoot,
                    'metadata',
                    'ModelCategoryMapping.csv')
            )
        else:
            self.categoryMapping = None

        self._initLayoutModels()
        self._initAgents()
        self._initObjects()
    
        self.scene.worlds['semantics'] = self

    def _initLayoutModels(self):
        
        # Load layout objects as meshes
        for _ in self.scene.scene.findAllMatches('**/layouts/object*/model*'):
            # Nothing to do
            pass
    
    def _initAgents(self):
    
        # Load agents
        for _ in self.scene.scene.findAllMatches('**/agents/agent*'):
            # Nothing to do
            pass
            
    def _initObjects(self):
    
        # Load objects
        for model in self.scene.scene.findAllMatches('**/objects/object*/model*'):
            modelId = model.getParent().getTag('model-id')
            
            objNp = model.getParent()
            
            semanticsNp = objNp.attachNewNode('semantics')
            
            # Categories
            coarseCategory = self.categoryMapping.getCoarseGrainedCategoryForModelId(modelId)
            semanticsNp.setTag('coarse-category', coarseCategory)
            
            fineCategory = self.categoryMapping.getFineGrainedCategoryForModelId(modelId)
            semanticsNp.setTag('fine-category', fineCategory)
            
            # Estimate mass of object based on volumetric data and default material density
            objVoxFilename = os.path.join(self.suncgDatasetRoot, 'object_vox', 'object_vox_data', modelId, modelId + '.binvox')
            voxelData = ObjectVoxelData.fromFile(objVoxFilename)
            volume = voxelData.getFilledVolume()
            semanticsNp.setTag('volume', str(volume))
            
            #XXX: we could get information below from physic node (if any)
            if coarseCategory in self.movableObjectCategories:
                semanticsNp.setTag('movable', 'true')
            else:
                semanticsNp.setTag('movable', 'false')
                
            #TODO: add mass information
            
            # Color information
            basicColors = MaterialColorTable.getColorsFromObject(objNp, mode='basic')
            semanticsNp.setTag('basic-colors', ",".join(basicColors))
            
            advancedColors = MaterialColorTable.getColorsFromObject(objNp, mode='advanced')
            semanticsNp.setTag('advanced-colors', ",".join(advancedColors))

            # Material information
            materials = MaterialTable.getMaterialNameFromObject(objNp)
            semanticsNp.setTag('materials', ",".join(materials))
            
            # Object size information
            modelInfoFilename = os.path.join(self.suncgDatasetRoot,
                                            'metadata',
                                            'models.csv')
            modelCatFilename = os.path.join(self.suncgDatasetRoot,
                                            'metadata',
                                            'ModelCategoryMapping.csv')
            overallSize = DimensionTable.getDimensionsFromModelId(modelId, modelInfoFilename, modelCatFilename)
            semanticsNp.setTag('overall-size', str(overallSize))
            
    def describeObject(self, obj):
        
        semanticsNp = obj.find('**/semantics')
        if not semanticsNp.isEmpty():
            
            items = []
            
            sizeDescription = semanticsNp.getTag('overall-size')
            if sizeDescription == 'normal':
                sizeDescription = ''
            items.append(sizeDescription)

            colorDescription = semanticsNp.getTag('advanced-colors')
            items.append(colorDescription)
     
            categoryDescription = semanticsNp.getTag('fine-category')
            categoryDescription = categoryDescription.replace("_", " ")
            items.append(categoryDescription)
     
            materialDescription = semanticsNp.getTag('materials')
            materialDescription = 'made of ' + materialDescription
            items.append(materialDescription)
     
            desc = " ".join(items)    
        else:
            desc = ""
        
        return desc
