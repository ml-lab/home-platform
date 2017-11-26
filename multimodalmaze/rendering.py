# Copyright (c) 2017, IGLU consortium
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without 
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright 
#    notice, this list of conditions and the following disclaimer.
#   
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT 
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR 
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY 
# OF SUCH DAMAGE.

import os
import logging
import numpy as np
import scipy.ndimage

from panda3d.core import VBase4, PointLight, AmbientLight, AntialiasAttrib, \
                         GeomVertexReader, GeomTristrips, GeomTriangles, LineStream, SceneGraphAnalyzer, \
                         LVecBase3f, LVecBase4f, TransparencyAttrib, ColorAttrib, TextureAttrib, GeomEnums
                         
from panda3d.core import GraphicsEngine, GraphicsPipeSelection, Loader, RescaleNormalAttrib, \
                         Texture, GraphicsPipe, GraphicsOutput, FrameBufferProperties, WindowProperties, Camera, PerspectiveLens, ModelNode

from multimodalmaze.core import World

logger = logging.getLogger(__name__)

MODEL_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "models")

class Panda3dRenderer(World):

    def __init__(self, scene, size=(512,512), shadowing=False, mode='offscreen', zNear=0.1, zFar=1000.0, fov=40.0, depth=True, modelLightsInfo=None, cameraTransform=None):

        super(Panda3dRenderer, self).__init__()
        
        self.__dict__.update(scene=scene, size=size, mode=mode, zNear=zNear, zFar=zFar, fov=fov, 
                             depth=depth, shadowing=shadowing, modelLightsInfo=modelLightsInfo, cameraTransform=cameraTransform)
        
        self.graphicsEngine = GraphicsEngine.getGlobalPtr()
        self.loader = Loader.getGlobalPtr()
        self.graphicsEngine.setDefaultLoader(self.loader)
        
        # Change some scene attributes for rendering
        self.scene.scene.setAttrib(RescaleNormalAttrib.makeDefault())
        self.scene.scene.setTwoSided(0)
        
        self._initModels()
        
        selection = GraphicsPipeSelection.getGlobalPtr()
        self.pipe = selection.makeDefaultPipe()
        logger.debug('Using %s' % (self.pipe.getInterfaceName()))
        
        # Attach a camera to every agent in the scene
        self.cameras = []
        for agentNp in self.scene.scene.findAllMatches('**/agents/agent*'):
            camera = agentNp.attachNewNode(ModelNode('camera'))
            if self.cameraTransform is not None:
                camera.setTransform(cameraTransform)
            camera.node().setPreserveTransform(ModelNode.PTLocal)
            self.cameras.append(camera)
        
        self.rgbBuffers = dict()
        self.rgbTextures = dict()
        self.depthBuffers = dict()
        self.depthTextures = dict()
        
        self._initRgbCapture()
        if self.depth:
            self._initDepthCapture()

        self._addDefaultLighting()

        self.scene.worlds['render'] = self

    def _initModels(self):
        
        for model in self.scene.scene.findAllMatches('**/+ModelNode'):
            
            objectNp = model.getParent()
            rendererNp = objectNp.attachNewNode('render')
            model = model.copyTo(rendererNp)
            model.show()
            
            # Reparent render node below the existing physic node (if any)
            physicsNp = objectNp.find('**/physics')
            if not physicsNp.isEmpty():
                rendererNp.reparentTo(physicsNp)

    def _initRgbCapture(self):

        for camera in self.cameras:
            
            camNode = Camera('RGB camera')
            lens = PerspectiveLens()
            lens.setFov(self.fov)
            lens.setAspectRatio(float(self.size[0]) / float(self.size[1]))
            lens.setNear(self.zNear)
            lens.setFar(self.zFar)
            camNode.setLens(lens)
            camNode.setScene(self.scene.scene)
            cam = camera.attachNewNode(camNode)
            
            winprops = WindowProperties.size(self.size[0], self.size[1])
            fbprops = FrameBufferProperties.getDefault()
            fbprops = FrameBufferProperties(fbprops)
            fbprops.setRgbaBits(8, 8, 8, 0)
            
            flags = GraphicsPipe.BFFbPropsOptional
            if self.mode == 'onscreen':
                flags = flags | GraphicsPipe.BFRequireWindow
            elif self.mode == 'offscreen':
                flags = flags | GraphicsPipe.BFRefuseWindow
            else:
                raise Exception('Unsupported rendering mode: %s' % (self.mode))
            
            buf = self.graphicsEngine.makeOutput(self.pipe, 'RGB buffer', 0, fbprops,
                                                 winprops, flags)
            if buf is None:
                raise Exception('Unable to create RGB buffer')
            
            # Set to render at the end
            buf.setSort(10000)
            
            dr = buf.makeDisplayRegion()
            dr.setSort(0)
            dr.setCamera(cam)
            dr = camNode.getDisplayRegion(0)
            
            tex = Texture()
            tex.setFormat(Texture.FRgb8)
            tex.setComponentType(Texture.TUnsignedByte)
            buf.addRenderTexture(tex, GraphicsOutput.RTMCopyRam, GraphicsOutput.RTPColor)
            #XXX: should use tex.setMatchFramebufferFormat(True)?
        
            agent = camera.getParent()
            self.rgbBuffers[agent.getName()] = buf
            self.rgbTextures[agent.getName()] = tex
    
    def _initDepthCapture(self):
        
        for camera in self.cameras:
        
            camNode = Camera('Depth camera')
            lens = PerspectiveLens()
            lens.setFov(self.fov)
            lens.setAspectRatio(float(self.size[0]) / float(self.size[1]))
            lens.setNear(self.zNear)
            lens.setFar(self.zFar)
            camNode.setLens(lens)
            camNode.setScene(self.scene.scene)
            cam = camera.attachNewNode(camNode)
            
            winprops = WindowProperties.size(self.size[0], self.size[1])
            fbprops = FrameBufferProperties.getDefault()
            fbprops = FrameBufferProperties(fbprops)
            fbprops.setRgbColor(False)
            fbprops.setRgbaBits(0, 0, 0, 0)
            fbprops.setStencilBits(0)
            fbprops.setMultisamples(0)
            fbprops.setBackBuffers(0)
            fbprops.setDepthBits(16)
            
            flags = GraphicsPipe.BFFbPropsOptional
            if self.mode == 'onscreen':
                flags = flags | GraphicsPipe.BFRequireWindow
            elif self.mode == 'offscreen':
                flags = flags | GraphicsPipe.BFRefuseWindow
            else:
                raise Exception('Unsupported rendering mode: %s' % (self.mode))
            
            buf = self.graphicsEngine.makeOutput(self.pipe, 'Depth buffer', 0, fbprops,
                                                 winprops, flags)
            if buf is None:
                raise Exception('Unable to create depth buffer')
            
            # Set to render at the end
            buf.setSort(10000)
            
            dr = buf.makeDisplayRegion()
            dr.setSort(0)
            dr.setCamera(cam)
            dr = camNode.getDisplayRegion(0)
            
            tex = Texture()
            tex.setFormat(Texture.FDepthComponent)
            tex.setComponentType(Texture.TFloat)
            buf.addRenderTexture(tex, GraphicsOutput.RTMCopyRam, GraphicsOutput.RTPDepth)
            #XXX: should use tex.setMatchFramebufferFormat(True)?
            
            agent = camera.getParent()
            self.depthBuffers[agent.getName()] = buf
            self.depthTextures[agent.getName()] = tex
        
    def setWireframeOnly(self):
        self.scene.scene.setRenderModeWireframe()
        
    def showRoomLayout(self, showCeilings=True, showWalls=True, showFloors=True):
        
        for np in self.scene.scene.findAllMatches('**/layouts/**/render/*c'):
            if showCeilings:
                np.show()
            else:
                np.hide()
    
        for np in self.scene.scene.findAllMatches('**/layouts/**/render/*w'):
            if showWalls:
                np.show()
            else:
                np.hide()
            
        for np in self.scene.scene.findAllMatches('**/layouts/**/render/*f'):
            if showFloors:
                np.show()
            else:
                np.hide()
        
    def destroy(self):
        self.graphicsEngine.removeAllWindows()
        del self.pipe

    def getRgbImages(self, channelOrder="RGB"):
        images = dict()
        for name, tex in self.rgbTextures.iteritems():
            data = tex.getRamImageAs(channelOrder)
            image = np.frombuffer(data.get_data(), np.uint8) # Must match Texture.TUnsignedByte
            image.shape = (tex.getYSize(), tex.getXSize(), 3)
            image = np.flipud(image)
            images[name] = image
            
        return images
    
    def getDepthImages(self, mode='normalized'):
        
        images = dict()
        if self.depth:
        
            for name, tex in self.depthTextures.iteritems():
        
                data = tex.getRamImage().get_data()
                nbBytesComponentFromData = len(data) / (tex.getYSize() * tex.getXSize())
                if nbBytesComponentFromData == 4:
                    depthImage = np.frombuffer(data, np.float32) # Must match Texture.TFloat
                elif nbBytesComponentFromData == 2:
                    # NOTE: This can happen on some graphic hardware, where unsigned 16-bit data is stored
                    #       despite setting the texture component type to 32-bit floating point.
                    depthImage = np.frombuffer(data, np.uint16).astype()
                    depthImage = depthImage.astype(np.float32) / 65535
                    
                depthImage.shape = (tex.getYSize(), tex.getXSize())
                depthImage = np.flipud(depthImage)
                
                if mode == 'distance':
                    # NOTE: in Panda3d, the returned depth image seems to be already linearized
                    depthImage = self.zNear + depthImage / (self.zFar - self.zNear)
        
                    # Adapted from: https://stackoverflow.com/questions/6652253/getting-the-true-z-value-from-the-depth-buffer
                    #depthImage = 2.0 * depthImage - 1.0
                    #depthImage = 2.0 * self.zNear * self.zFar / (self.zFar + self.zNear - depthImage * (self.zFar - self.zNear))
                    
                elif mode == 'normalized':
                    # Nothing to do
                    pass
                else:
                    raise Exception('Unsupported output depth image mode: %s' % (mode))
                
                images[name] = depthImage
        else:
            
            for name, _ in self.depthTextures.iteritems():
                images[name] = np.zeros(self.size, dtype=np.float32)
        
        return images

    def step(self, dt):
        
        self.graphicsEngine.renderFrame()
        
        #NOTE: we need to call frame rendering twice in onscreen mode because of double-buffering
        if self.mode == 'onscreen':
            self.graphicsEngine.renderFrame()
        
    def getRenderInfo(self):
        sga = SceneGraphAnalyzer()
        sga.addNode(self.scene.scene.node())
        
        ls = LineStream()
        sga.write(ls)
        desc = []
        while ls.isTextAvailable():
            desc.append(ls.getLine())
        desc = '\n'.join(desc)
        return desc

    def _addDefaultLighting(self):
        alight = AmbientLight('alight')
        alight.setColor(VBase4(0.2, 0.2, 0.2, 1))
        alnp = self.scene.scene.attachNewNode(alight)
        self.scene.scene.setLight(alnp)
        
        for camera in self.cameras:
            
            #NOTE: Point light following the camera
            plight = PointLight('plight')
            plight.setColor(VBase4(1.0, 1.0, 1.0, 1))
            plnp = camera.attachNewNode(plight)
            self.scene.scene.setLight(plnp)
            
            if self.shadowing:
                # Use a 512x512 resolution shadow map
                plight.setShadowCaster(True, 512, 512)
    
                # Enable the shader generator for the receiving nodes
                self.scene.scene.setShaderAuto()
                self.scene.scene.setAntialias(AntialiasAttrib.MAuto)

        if self.modelLightsInfo is not None:
            
            # Add model-related lights (e.g. lamps)
            for model in self.scene.scene.findAllMatches('**/+ModelNode'):
                modelId = model.getNetTag('model-id')
                for lightNp in self.modelLightsInfo.getLightsForModel(modelId):
                    
                    if self.shadowing:
                        # Use a 512x512 resolution shadow map
                        lightNp.node().setShadowCaster(True, 512, 512)
                    
                    lightNp.reparentTo(model)
                    
                    self.scene.scene.setLight(lightNp)

def get3DPointsFromModel(model):
    geomNodes = model.findAllMatches('**/+GeomNode')
    
    pts = []
    for nodePath in geomNodes:
        nodePts = []
        geomNode = nodePath.node()
        for i in range(geomNode.getNumGeoms()):
            geom = geomNode.getGeom(i)
            vdata = geom.getVertexData()
            vertex = GeomVertexReader(vdata, 'vertex')
            while not vertex.isAtEnd():
                v = vertex.getData3f()
                nodePts.append([v.x, v.y, v.z])
        pts.append(nodePts)
    return np.array(pts)
    
def get3DTrianglesFromModel(model):
    
    # Calculate the net transformation
    transform = model.getNetTransform()
    transformMat = transform.getMat()
    
    # Get geometry data from GeomNode instances inside the model
    geomNodes = model.findAllMatches('**/+GeomNode')
    
    triangles = []
    for nodePath in geomNodes:
        geomNode = nodePath.node()
        
        for n in range(geomNode.getNumGeoms()):
            geom = geomNode.getGeom(n)
            vdata = geom.getVertexData()
            
            for k in range(geom.getNumPrimitives()):
                prim = geom.getPrimitive(k)
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
                        
                        # Apply transformation
                        v = transformMat.xformPoint(v)
                        
                        triPts.append([v.x, v.y, v.z])

                    triangles.append(triPts)
            
    triangles = np.array(triangles)
            
    return triangles

def getSurfaceAreaFromGeom(geom, transform=None):
    
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
                
                # Apply transformation
                if transform is not None:
                    v = transform.xformPoint(v)
                
                triPts.append([v.x, v.y, v.z])
            triPts = np.array(triPts)

            # calculate the semi-perimeter and area
            a = np.linalg.norm(triPts[0] - triPts[1], 2)
            b = np.linalg.norm(triPts[1] - triPts[2], 2)
            c = np.linalg.norm(triPts[2] - triPts[0], 2)
            s = (a + b + c) / 2
            area = (s*(s-a)*(s-b)*(s-c)) ** 0.5
            totalArea += area

    return totalArea

def getColorAttributesFromVertexData(geom, transform=None):
    
    colorsTotalAreas = dict()
    for k in range(geom.getNumPrimitives()):
        prim = geom.getPrimitive(k)
        vdata = geom.getVertexData()
        assert isinstance(prim, (GeomTristrips, GeomTriangles))
        
        # Check if color is defined for vertex
        isColorDefined = False        
        for i, geomVertexCol in enumerate(vdata.getFormat().getColumns()):
            if geomVertexCol.getContents() == GeomEnums.CColor:
                isColorDefined = True
                break
        assert isColorDefined
        
        vertex = GeomVertexReader(vdata, 'vertex')
        vertexColor = GeomVertexReader(vdata, 'color')
                
        # Decompose into triangles
        prim = prim.decompose()
        for p in range(prim.getNumPrimitives()):
            s = prim.getPrimitiveStart(p)
            e = prim.getPrimitiveEnd(p)
            
            color = None
            triPts = []
            for i in range(s, e):
                vi = prim.getVertex(i)
                vertex.setRow(vi)
                vertexColor.setRow(vi)
                v = vertex.getData3f()
                
                # NOTE: all vertex of the same polygon (triangles) should have the same color,
                #       so only grab it once.
                if color is None:
                    color = vertexColor.getData4f()
                    color = (color[0], color[1], color[2], color[3])
            
                triPts.append([v.x, v.y, v.z])
            triPts = np.array(triPts)
                
            # Apply transformation
            if transform is not None:
                v = transform.xformPoint(v)
            
            # calculate the semi-perimeter and area
            a = np.linalg.norm(triPts[0] - triPts[1], 2)
            b = np.linalg.norm(triPts[1] - triPts[2], 2)
            c = np.linalg.norm(triPts[2] - triPts[0], 2)
            s = (a + b + c) / 2
            area = (s*(s-a)*(s-b)*(s-c)) ** 0.5
            
            if color in colorsTotalAreas:
                colorsTotalAreas[color] += area
            else:
                colorsTotalAreas[color] = area
    
    areas = []        
    rgbColors = []
    transparencies = []
    for color, area in colorsTotalAreas.iteritems():
        areas.append(area)
        rgbColors.append(list(color[:3]))
        
        # Check transparency
        isTransparent = color[3] < 1.0
        transparencies.append(isTransparent)
            
    return areas, rgbColors, transparencies


def getColorAttributesFromModel(model):
    
    # Calculate the net transformation
    transform = model.getNetTransform()
    transformMat = transform.getMat()
    
    areas = []
    rgbColors = []
    textures = []
    transparencies = []
    for nodePath in model.findAllMatches('**/+GeomNode'):
        geomNode = nodePath.node()
        
        for n in range(geomNode.getNumGeoms()):
            state = geomNode.getGeomState(n)
        
            geom = geomNode.getGeom(n)
            area = getSurfaceAreaFromGeom(geom, transformMat)
        
            if state.hasAttrib(TextureAttrib.getClassType()):
                # Get color from texture
                texAttr = state.getAttrib(TextureAttrib.getClassType())
                tex = texAttr.getTexture()
                
                # Load texture image from file and compute average color
                texFilename = str(tex.getFullpath())
                img = scipy.ndimage.imread(texFilename)

                texture = os.path.splitext(os.path.basename(texFilename))[0]
                
                #TODO: handle black-and-white and RGBA texture
                assert img.dtype == np.uint8
                assert img.ndim == 3 and img.shape[-1] == 3
                
                rgbColor = (np.mean(img, axis=(0,1)) / 255.0).tolist()

                rgbColors.append(rgbColor)
                transparencies.append(False)
                areas.append(area)
                textures.append(texture)

            elif state.hasAttrib(ColorAttrib.getClassType()):
                colorAttr = state.getAttrib(ColorAttrib.getClassType())
                
                if (colorAttr.getColorType() == ColorAttrib.TFlat or colorAttr.getColorType() == ColorAttrib.TOff):
                    # Get flat color
                    color = colorAttr.getColor()
                    
                    isTransparent = False
                    if isinstance(color, LVecBase4f):
                        rgbColor= [color[0], color[1], color[2]]
                        alpha = color[3]
                        
                        if state.hasAttrib(TransparencyAttrib.getClassType()):
                            transAttr = state.getAttrib(TransparencyAttrib.getClassType())
                            if transAttr.getMode() != TransparencyAttrib.MNone and alpha < 1.0:
                                isTransparent = True
                        elif alpha < 1.0:
                            isTransparent = True
                            
                    elif isinstance(color, LVecBase3f):
                        rgbColor= [color[0], color[1], color[2]]
                    else:
                        raise Exception('Unsupported color class type: %s' % (color.__class__.__name__))
                
                    rgbColors.append(rgbColor)
                    transparencies.append(isTransparent)
                    areas.append(area)
                    textures.append(None)
                
                else:
                    # Get colors from vertex data
                    verAreas, verRgbColors, vertransparencies = getColorAttributesFromVertexData(geom, transformMat)
                    areas.extend(verAreas)
                    rgbColors.extend(verRgbColors)
                    transparencies.extend(vertransparencies)
                    textures.extend([None,]*len(vertransparencies))
            
    areas = np.array(areas)
    areas /= np.sum(areas)
            
    return areas, rgbColors, transparencies, textures
