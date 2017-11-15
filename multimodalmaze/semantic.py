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

from panda3d.core import Loader, LoaderOptions, NodePath, Filename

from multimodalmaze.constants import MATERIAL_TABLE, MATERIAL_COLOR_TABLE
from multimodalmaze.suncg import ModelCategoryMapping, ModelInformation
from multimodalmaze.rendering import getColorAttributesFromModel

logger = logging.getLogger(__name__)

class MaterialTable(object):
    @staticmethod
    def getMaterialNameFromObject(obj, thresholdRelArea=0.2):

        # Load the model
        loader = Loader.getGlobalPtr()
        loaderOptions = LoaderOptions()
        node = loader.loadSync(Filename(obj.modelFilename), loaderOptions)
        if node is not None:
            nodePath = NodePath(node)
        else:
            raise IOError('Could not load model file: %s' % (obj.modelFilename))

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

        # Unload model
        nodePath.removeNode()

        return materialDescriptions


class MaterialColorTable(object):
    @staticmethod
    def getBasicColorsFromObject(obj, mode='advanced', thresholdRelArea=0.2):

        # Load the model
        loader = Loader.getGlobalPtr()
        loaderOptions = LoaderOptions()
        node = loader.loadSync(Filename(obj.modelFilename), loaderOptions)
        if node is not None:
            nodePath = NodePath(node)
        else:
            raise IOError('Could not load model file: %s' % (obj.modelFilename))

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

        # Unload model
        nodePath.removeNode()

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
    def getDimensionsFromObject(obj, modelInfoFilename, modelCatFilename):
        
        modelInfo = ModelInformation(modelInfoFilename)
        modelCat = ModelCategoryMapping(modelCatFilename)
        
        refModelDimensions = None
        otherSimilarDimensions = []
        refModelId = str(obj.modelId)
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

class SemanticWorld(object):
    def __init__(self):
        pass

    def addObjectToScene(self, obj):
        pass

    def addRoomToScene(self, room):
        pass

    def addHouseToScene(self, house):
        pass


class SuncgSemanticWorld(SemanticWorld):
    def __init__(self, datasetRoot):
        self.categoryMapping = ModelCategoryMapping(
            os.path.join(
                datasetRoot,
                'metadata',
                'ModelCategoryMapping.csv')
        )

    def _describeObjectCategory(self, obj):
        category = self.categoryMapping.getFineGrainedCategoryForModelId(obj.modelId)
        desc = category.replace("_", " ")
        return desc

    def _describeObjectColor(self, obj):
        colors = MaterialColorTable.getBasicColorsFromObject(obj, mode='advanced')
        desc = ', '.join(colors)
        return desc

    def _describeObjectMaterial(self, obj):
        materials = MaterialTable.getMaterialNameFromObject(obj)
        desc = 'made of ' + ','.join(materials)
        return desc

    def describeObject(self, obj):
        items = []

        # TODO: color attribute of the main material RGB values
        colorDescription = self._describeObjectColor(obj)
        items.append(colorDescription)

        # TODO: category attribute from the SUNCG mapping
        categoryDescription = self._describeObjectCategory(obj)
        items.append(categoryDescription)

        # TODO: material attribute of the main textures
        materialDescription = self._describeObjectMaterial(obj)
        items.append(materialDescription)

        desc = " ".join(items)
        return desc
