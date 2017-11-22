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

from panda3d.core import VBase4, Mat4, PointLight, AmbientLight, AntialiasAttrib, \
                         GeomVertexReader, GeomTristrips, GeomTriangles, LineStream, SceneGraphAnalyzer, \
                         LVecBase3f, LVecBase4f, TransparencyAttrib, ColorAttrib, TextureAttrib, TransformState, GeomEnums
                         
from panda3d.core import GraphicsEngine, GraphicsPipeSelection, Loader, LoaderOptions, NodePath, RescaleNormalAttrib, Filename, \
                         Texture, GraphicsPipe, GraphicsOutput, FrameBufferProperties, WindowProperties, Camera, PerspectiveLens, ModelNode

logger = logging.getLogger(__name__)

MODEL_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "models")

class RenderObject(object):
   
    def getTransform(self):
        return NotImplementedError()

    def setTransform(self):
        return NotImplementedError()

class RenderWorld(object):

    def addAgentToScene(self, agent):
        return NotImplementedError()

    def addObjectToScene(self, obj):
        return NotImplementedError()
    
    def addRoomToScene(self, room):
        return NotImplementedError()
    
    def addHouseToScene(self, house):
        return NotImplementedError()

    def step(self, time):
        return NotImplementedError()
    
    def resetScene(self):
        return NotImplementedError()


class Panda3dRenderObject(RenderObject):
    
    def __init__(self, nodePath, recenterTransform=None):
        self.nodePath = nodePath
    
        if recenterTransform is None:
            recenterTransform = TransformState.makeIdentity()
        self.recenterTransform = recenterTransform
    
    def getTransform(self):
        transform = self.nodePath.node().getTransform()
        mat = transform.compose(TransformState.makePos(-self.recenterTransform.getPos())).getMat()
        return np.array([[mat[0][0], mat[0][1], mat[0][2], mat[0][3]],
                         [mat[1][0], mat[1][1], mat[1][2], mat[1][3]],
                         [mat[2][0], mat[2][1], mat[2][2], mat[2][3]],
                         [mat[3][0], mat[3][1], mat[3][2], mat[3][3]]])

    def setTransform(self, transform):
        mat = Mat4(*transform.ravel())
        self.nodePath.setTransform(TransformState.makeMat(mat).compose(self.recenterTransform))

    def getRecenterPosition(self):
        position = self.recenterTransform.getPos()
        return np.array([position.x, position.y, position.z])

class Panda3dRenderWorld(RenderWorld):

    #TODO: add a debug mode showing wireframe only?
    #      render.setRenderModeWireframe()

    def __init__(self, size=(512,512), shadowing=False, showCeiling=True, mode='offscreen', zNear=0.1, zFar=1000.0, fov=75.0, depth=True):
        
        self.size = size
        self.mode = mode
        self.zNear = zNear
        self.zFar = zFar
        self.fov = fov
        self.depth = depth
        self.graphicsEngine = GraphicsEngine.getGlobalPtr()
        self.loader = Loader.getGlobalPtr()
        self.graphicsEngine.setDefaultLoader(self.loader)
        
        self.render = NodePath('render')
        self.render.setAttrib(RescaleNormalAttrib.makeDefault())
        self.render.setTwoSided(0)
        
        selection = GraphicsPipeSelection.getGlobalPtr()
        self.pipe = selection.makeDefaultPipe()
        logger.debug('Using %s' % (self.pipe.getInterfaceName()))
        
        self.camera = self.render.attachNewNode(ModelNode('camera'))
        self.camera.node().setPreserveTransform(ModelNode.PTLocal)
        
        self.scene = self.render.attachNewNode('scene')
        
        self._initRgbCapture()
        
        if self.depth:
            self._initDepthCapture()
        
        self.__dict__.update(shadowing=shadowing, showCeiling=showCeiling)

        self.agent = None

    def _initRgbCapture(self):

        camNode = Camera('RGB camera')
        lens = PerspectiveLens()
        lens.setFov(self.fov)
        lens.setAspectRatio(float(self.size[0]) / float(self.size[1]))
        lens.setNear(self.zNear)
        lens.setFar(self.zFar)
        camNode.setLens(lens)
        camNode.setScene(self.render)
        cam = self.camera.attachNewNode(camNode)
        
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
        
        self.rgbBuffer = buf
        self.rgbTex = tex
    
    def _initDepthCapture(self):
        
        camNode = Camera('Depth camera')
        lens = PerspectiveLens()
        lens.setFov(self.fov)
        lens.setAspectRatio(float(self.size[0]) / float(self.size[1]))
        lens.setNear(self.zNear)
        lens.setFar(self.zFar)
        camNode.setLens(lens)
        camNode.setScene(self.render)
        cam = self.camera.attachNewNode(camNode)
        
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
        
        self.depthBuffer = buf
        self.depthTex = tex
        
    def destroy(self):
        self.graphicsEngine.removeAllWindows()
        del self.pipe

    def getRgbImage(self, channelOrder="RGB"):
        data = self.rgbTex.getRamImageAs(channelOrder)
        image = np.frombuffer(data.get_data(), np.uint8) # Must match Texture.TUnsignedByte
        image.shape = (self.rgbTex.getYSize(), self.rgbTex.getXSize(), 3)
        image = np.flipud(image)
        return image
    
    def getDepthImage(self, mode='normalized'):
        
        if self.depth:
        
            data = self.depthTex.getRamImage().get_data()
            nbBytesComponentFromData = len(data) / (self.depthTex.getYSize() * self.depthTex.getXSize())
            if nbBytesComponentFromData == 4:
                depthImage = np.frombuffer(data, np.float32) # Must match Texture.TFloat
            elif nbBytesComponentFromData == 2:
                # NOTE: This can happen on some graphic hardware, where unsigned 16-bit data is stored
                #       despite setting the texture component type to 32-bit floating point.
                depthImage = np.frombuffer(data, np.uint16).astype()
                depthImage = depthImage.astype(np.float32) / 65535
                
            depthImage.shape = (self.depthTex.getYSize(), self.depthTex.getXSize())
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
        else:
            depthImage = np.zeros(self.size, dtype=np.float32)
        
        return depthImage

    def step(self):
        
        # Update position of camera based on agent
        if self.agent is not None:
            transform = self.agent.transform
            mat = Mat4(*transform.ravel())
            self.camera.setMat(mat)
        
        self.graphicsEngine.renderFrame()
        
        #NOTE: we need to call frame rendering twice in onscreen mode because of double-buffering
        if self.mode == 'onscreen':
            self.graphicsEngine.renderFrame()
        
    def _renderInfo(self):
        sga = SceneGraphAnalyzer()
        sga.addNode(self.render.node())
        
        ls = LineStream()
        sga.write(ls)
        desc = []
        while ls.isTextAvailable():
            desc.append(ls.getLine())
        desc = '\n'.join(desc)
        return desc
        
    def _loadModel(self, modelPath):
        loaderOptions = LoaderOptions()
        node = self.loader.loadSync(Filename(modelPath), loaderOptions)
        if node is not None:
            nodePath = NodePath(node)
        else:
            raise IOError('Could not load model file: %s' % (modelPath))
        return nodePath

    def addDefaultLighting(self):
        alight = AmbientLight('alight')
        alight.setColor(VBase4(0.2, 0.2, 0.2, 1))
        alnp = self.scene.attachNewNode(alight)
        self.scene.setLight(alnp)
        
        #NOTE: Point light following the camera
        plight = PointLight('plight')
        plight.setColor(VBase4(1.0, 1.0, 1.0, 1))
        plnp = self.camera.attachNewNode(plight)
        self.scene.setLight(plnp)
        
        if self.shadowing:
            # Use a 512x512 resolution shadow map
            plight.setShadowCaster(True, 512, 512)

            # Enable the shader generator for the receiving nodes
            self.scene.setShaderAuto()
            self.scene.setAntialias(AntialiasAttrib.MAuto)
    
    def setCamera(self, mat):
        mat = Mat4(*mat.ravel())
        self.camera.setMat(mat)
    
    def addAgentToScene(self, agent):
        
        if not self.agent is None:
            raise NotImplementedError('Agent already present in the scene. Support for multiple agents is not yet available')
        
        nodePath = self.scene.attachNewNode('agent-' + str(agent.instanceId))
        nodePath.reparentTo(self.scene)
        
        if agent.modelFilename is not None:
            model = self._loadModel(agent.modelFilename)
            model.reparentTo(nodePath)
            
            instance = Panda3dRenderObject(nodePath)
            agent.setRenderObject(instance)
        else:
            instance = Panda3dRenderObject(nodePath)
            agent.setRenderObject(instance)
        
        agent.assertConsistency()
        
        self.agent = agent

        return nodePath # for backwards compatibility
        
    def addObjectToScene(self, obj):

        nodePath = self.scene.attachNewNode('object-' + str(obj.instanceId))
        model = self._loadModel(obj.modelFilename)
        model.reparentTo(nodePath)
        
        instance = Panda3dRenderObject(nodePath)
        obj.setRenderObject(instance)
        
        obj.assertConsistency()
        
    def addRoomToScene(self, room):

        for modelFilename in room.modelFilenames:
            
            partId = os.path.splitext(os.path.basename(modelFilename))[0]
            objNodePath = self.scene.attachNewNode('room-' + str(room.instanceId) + '-' + partId)
            model = self._loadModel(modelFilename)
            model.reparentTo(objNodePath)
            
            if not self.showCeiling and 'c' in os.path.basename(modelFilename):
                objNodePath.hide()
                
        for idx, obj in enumerate(room.objects):
            self.addObjectToScene(obj)
            room.objects[idx] = obj
    
    def addHouseToScene(self, house):

        for room in house.rooms:
            self.addRoomToScene(room)
        
        for room in house.grounds:
            self.addRoomToScene(room)
        
        for idx, obj in enumerate(house.objects):
            self.addObjectToScene(obj)
            house.objects[idx] = obj

    def resetScene(self):
        # FIXME: find out why this function outputs a lot of logs to STDERR 
        childNodes = self.scene.ls()
        if childNodes is not None:
            for node in childNodes:
                node.detachNode()
                node.removeNode()
        self.scene.clearLight()
        self.scene.clearShader()
        self.scene.detachNode()
        self.scene.removeNode()
        self.scene = self.render.attachNewNode('scene')

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
