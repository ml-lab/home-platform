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
import scipy.ndimage
import numpy as np

from string import digits

from panda3d.core import GeomVertexReader, GeomTristrips, GeomTriangles, ColorAttrib, TextureAttrib, TransparencyAttrib, \
    LVecBase4f, LVecBase3f
from panda3d.core import Loader, LoaderOptions, NodePath, Filename

from multimodalmaze.constants import MATERIAL_TABLE, MATERIAL_COLOR_TABLE
from multimodalmaze.suncg import ModelCategoryMapping

logger = logging.getLogger(__name__)


def getSurfaceAreaFromGeom(geom):
    totalArea = 0.0
    for k in range(geom.getNumPrimitives()):
        prim = geom.getPrimitive(k)
        vdata = geom.getVertexData()
        vertex = GeomVertexReader(vdata, 'vertex')
        assert isinstance(prim, (GeomTristrips, GeomTriangles))

        # Decompose into triangles
        prim = prim.decompose()
        for p in range(prim.getNumPrimitives()):
            s = prim.getPrimitiveStart(p)
            e = prim.getPrimitiveEnd(p)

            triPts = []
            for i in range(s, e):
                vi = prim.getVertex(i)
                vertex.setRow(vi)
                v = vertex.getData3f()
                triPts.append([v.x, v.y, v.z])
            triPts = np.array(triPts)

            # calculate the semi-perimeter and area
            a = np.linalg.norm(triPts[0] - triPts[1], 2)
            b = np.linalg.norm(triPts[1] - triPts[2], 2)
            c = np.linalg.norm(triPts[2] - triPts[0], 2)
            s = (a + b + c) / 2
            area = (s * (s - a) * (s - b) * (s - c)) ** 0.5
            totalArea += area

    return totalArea


def getColorAttributesFromModel(model):
    areas = []
    rgbColors = []
    textures = []
    transparencies = []
    for nodePath in model.findAllMatches('**/+GeomNode'):
        geomNode = nodePath.node()

        for n in range(geomNode.getNumGeoms()):
            state = geomNode.getGeomState(n)

            rgbColor = None
            texture = None
            isTransparent = False
            if state.hasAttrib(TextureAttrib.getClassType()):
                texAttr = state.getAttrib(TextureAttrib.getClassType())
                tex = texAttr.getTexture()

                # Load texture image from file and compute average color
                texFilename = str(tex.getFullpath())
                img = scipy.ndimage.imread(texFilename)

                texture = os.path.splitext(os.path.basename(texFilename))[0]

                # TODO: handle black-and-white and RGBA texture
                assert img.dtype == np.uint8
                assert img.ndim == 3 and img.shape[-1] == 3

                rgbColor = (np.mean(img, axis=(0, 1)) / 255.0).tolist()

            elif state.hasAttrib(ColorAttrib.getClassType()):
                colorAttr = state.getAttrib(ColorAttrib.getClassType())
                color = colorAttr.getColor()

                if isinstance(color, LVecBase4f):
                    rgbColor = [color[0], color[1], color[2]]
                    alpha = color[3]

                    if state.hasAttrib(TransparencyAttrib.getClassType()):
                        transAttr = state.getAttrib(TransparencyAttrib.getClassType())
                        if transAttr.getMode() != TransparencyAttrib.MNone and alpha < 1.0:
                            isTransparent = True
                    elif alpha < 1.0:
                        isTransparent = True

                elif isinstance(color, LVecBase3f):
                    rgbColor = [color[0], color[1], color[2]]
                else:
                    raise Exception('Unsupported color class type: '
                                    '%s' % (color.__class__.__name__))

            rgbColors.append(rgbColor)
            transparencies.append(isTransparent)

            geom = geomNode.getGeom(n)
            area = getSurfaceAreaFromGeom(geom)
            areas.append(area)
            textures.append(texture)

    areas = np.array(areas)
    areas /= np.sum(areas)

    return areas, rgbColors, transparencies, textures


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
