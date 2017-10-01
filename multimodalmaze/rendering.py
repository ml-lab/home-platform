
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

from panda3d.core import Vec3, VBase4, Mat4, PointLight, AmbientLight, AntialiasAttrib, CS_yup_right, CS_zup_right, \
                         GeomVertexReader, GeomTristrips, Material
                         
from panda3d.core import GraphicsEngine, GraphicsPipeSelection, Loader, LoaderOptions, NodePath, RescaleNormalAttrib, Filename, \
                         Texture, GraphicsPipe, GraphicsOutput, FrameBufferProperties, WindowProperties, Camera, PerspectiveLens, ModelNode
from direct.showbase.ShowBase import ShowBase

logger = logging.getLogger(__name__)

MODEL_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "models")

class RenderWorld(object):

    def __init__(self):
        pass

    def addAgentToScene(self, agent):
        pass

    def addObjectToScene(self, obj):
        pass
    
    def addRoomToScene(self, room):
        pass
    
    def addHouseToScene(self, house):
        pass

    def step(self, time):
        pass
    
    def resetScene(self):
        pass

#TODO: add support for multithreading?
#      see: https://www.panda3d.org/manual/index.php/Multithreaded_Render_Pipeline

class Panda3dRenderWorld(RenderWorld):

    #TODO: add a debug mode showing wireframe only?
    #      render.setRenderModeWireframe()

    def __init__(self, size=(512,512), shadowing=True, showCeiling=True):
        
        self.size = size
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
        self._initDepthCapture()
        
        self.__dict__.update(shadowing=shadowing, showCeiling=showCeiling)

    def _initRgbCapture(self):

        camNode = Camera('RGB camera')
        lens = PerspectiveLens()
        lens.setAspectRatio(1.0)
        #lens.setNear(5.0)
        #lens.setFar(500.0)
        camNode.setLens(lens)
        camNode.setScene(self.render)
        cam = self.camera.attachNewNode(camNode)
        
        winprops = WindowProperties.getDefault()
        winprops = WindowProperties(winprops)
        winprops.setSize(self.size[0], self.size[1])
        fbprops = FrameBufferProperties.getDefault()
        fbprops = FrameBufferProperties(fbprops)
        fbprops.setRgbColor(1)
        fbprops.setColorBits(24)
        fbprops.setAlphaBits(8)
        fbprops.setDepthBits(1) 
        flags = GraphicsPipe.BFFbPropsOptional | GraphicsPipe.BFRefuseWindow
        buf = self.graphicsEngine.makeOutput(self.pipe, 'RGB buffer', 0, fbprops,
                                             winprops, flags)
        
        dr = buf.makeDisplayRegion()
        dr.setSort(0)
        dr.setCamera(cam)
        dr = camNode.getDisplayRegion(0)
        
        tex = Texture()
        tex.setFormat(Texture.FRgb)
        buf.addRenderTexture(tex, GraphicsOutput.RTMCopyRam, GraphicsOutput.RTPColor)
        
        self.rgbBuffer = buf
        self.rgbTex = tex
    
    def _initDepthCapture(self):
        
        camNode = Camera('Depth camera')
        lens = PerspectiveLens()
        lens.setAspectRatio(1.0)
        #lens.setNear(5.0)
        #lens.setFar(500.0)
        camNode.setLens(lens)
        camNode.setScene(self.render)
        cam = self.camera.attachNewNode(camNode)
        
        winprops = WindowProperties.getDefault()
        winprops = WindowProperties(winprops)
        winprops.setSize(self.size[0], self.size[1])
        fbprops = FrameBufferProperties()
        fbprops.setColorBits(0)
        fbprops.setAlphaBits(0)
        fbprops.setStencilBits(0)
        fbprops.setMultisamples(0)
        fbprops.setDepthBits(1)
        flags = GraphicsPipe.BFFbPropsOptional | GraphicsPipe.BFRefuseWindow
        buf = self.graphicsEngine.makeOutput(self.pipe, 'Depth buffer', 0, fbprops,
                                             winprops, flags)
        
        dr = buf.makeDisplayRegion()
        dr.setSort(0)
        dr.setCamera(cam)
        dr = camNode.getDisplayRegion(0)
        
        tex = Texture()
        tex.setFormat(Texture.FDepthComponent)
        buf.addRenderTexture(tex, GraphicsOutput.RTMCopyRam, GraphicsOutput.RTPDepth)
    
        self.depthBuffer = buf
        self.depthTex = tex
        
    def destroy(self):
        self.graphicsEngine.removeAllWindows()
        del self.pipe

    def getRgbImage(self, channelOrder="RGBA"):
        data = self.rgbTex.getRamImageAs(channelOrder)
        image = np.frombuffer(data.get_data(), np.uint8)
        image.shape = (self.rgbTex.getYSize(), self.rgbTex.getXSize(), self.rgbTex.getNumComponents())
        image = np.flipud(image)
        return image
    
    def getDepthImage(self):
        data = self.depthTex.getRamImage()
        depthImage = np.frombuffer(data.get_data(), np.float32)
        depthImage.shape = (self.depthTex.getYSize(), self.depthTex.getXSize(), self.depthTex.getNumComponents())
        depthImage = np.flipud(depthImage)
        return depthImage

    def step(self):
        #NOTE: we may need to call frame rendering twice because of double-buffering
        self.graphicsEngine.renderFrame()
        
    def loadModel(self, modelPath):
        loaderOptions = LoaderOptions()
        node = self.loader.loadSync(Filename(modelPath), loaderOptions)
        if node is not None:
            nodePath = NodePath(node)
        else:
            raise IOError('Could not load model file: %s' % (modelPath))
        return nodePath

    def addDefaultLighting(self):
        
        for x,y in [(0,0),(30,0),(0,30),(-30,0),(0,-30)]:
            plight = PointLight('plight')
            plight.setColor(VBase4(0.7, 0.7, 0.7, 1))
            
            node = self.scene.find("**/house*")
            if len(node.getNodes()) == 0:
                node = self.scene.find("**/room*")
                if len(node.getNodes()) == 0:
                    node = self.scene.find("**/object*")
            
            if len(node.getNodes()) == 0:
                sceneCenter = Vec3(0,0,0)
            else:
                sceneCenter = node.getBounds().getCenter()
            
            plnp = self.scene.attachNewNode(plight)
            plnp.setPos(sceneCenter+Vec3(x,y,15))
            self.scene.setLight(plnp)

            if self.shadowing:
                # Use a 512x512 resolution shadow map
                plight.setShadowCaster(True, 512, 512)
                
        alight = AmbientLight('alight')
        alight.setColor(VBase4(0.2, 0.2, 0.2, 1))
        alnp = self.scene.attachNewNode(alight)
        self.scene.setLight(alnp)
        
        if self.shadowing:
            # Enable the shader generator for the receiving nodes
            self.scene.setShaderAuto()
            self.scene.setAntialias(AntialiasAttrib.MAuto)
    
    def setCamera(self, mat):
        mat = Mat4(*mat.ravel())
        self.camera.setMat(mat)
    
    def addAgentToScene(self, agent):
        
        node = self.scene.attachNewNode('agent-' + str(agent.instanceId))
        if agent.modelFilename is not None:
            model = self.loadModel(agent.modelFilename)
            model.reparentTo(node)
        
        node.reparentTo(self.scene)
        return node
        
    def addObjectToScene(self, obj):

        node = self.scene.attachNewNode('object-' + str(obj.instanceId))
        model = self.loadModel(obj.modelFilename)
        model.reparentTo(node)
        
        if obj.transform is not None:
            # 4x4 column-major transformation matrix from object coordinates to scene coordinates
            transformMat = Mat4(*obj.transform.ravel())
            yupTransformMat = Mat4.convertMat(CS_zup_right, CS_yup_right)
            zupTransformMat = Mat4.convertMat(CS_yup_right, CS_zup_right)
            model.setMat(model.getMat() * yupTransformMat * transformMat * zupTransformMat)
        
        #node.reparentTo(self.scene)
        return model
    
    def addRoomToScene(self, room):

        node = self.scene.attachNewNode('room-' + str(room.instanceId))
        for modelFilename in room.modelFilenames:
            
            partId = os.path.splitext(os.path.basename(modelFilename))[0]
            objNode = node.attachNewNode('room-' + str(room.instanceId) + '-' + partId)
            model = self.loadModel(modelFilename)
            model.reparentTo(objNode)
            
            if not self.showCeiling and 'c' in os.path.basename(modelFilename):
                objNode.hide()
                
        for obj in room.objects:
            objNode = self.addObjectToScene(obj)
            objNode.reparentTo(node)
            
        return node
    
    def addHouseToScene(self, house):

        node = self.scene.attachNewNode('house-' + str(house.instanceId))
    
        for room in house.rooms:
            roomNode = self.addRoomToScene(room)
            roomNode.reparentTo(node)
        
        for obj in house.objects:
            objNode = self.addObjectToScene(obj)
            objNode.reparentTo(node)
        
        return node

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
    geomNodes = model.findAllMatches('**/+GeomNode')
    
    pts = []
    hvlines = []
    for nodePath in geomNodes:
        geomNode = nodePath.node()
        
        if 'WallInside' not in geomNode.name: continue
        print geomNode
        
        for n in range(geomNode.getNumGeoms()):
            geom = geomNode.getGeom(n)
    
            for k in range(geom.getNumPrimitives()):
                geom = geomNode.getGeom(k)
                prim = geom.getPrimitive(k)
                vdata = geom.getVertexData()
                vertex = GeomVertexReader(vdata, 'vertex')
                assert isinstance(prim, GeomTristrips)
                
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
                        #print "prim %s has vertex %s: %s" % (p, vi, repr(v))
                    pts.extend(triPts)
                        
                    triPts = np.array(triPts)
                    for i in range(len(triPts)-1):
                        hvline = triPts[i+1] - triPts[i]
                        if hvline[2] < 0.01:
                            hvline *= 0.0
                        hvlines.append(hvline)
            
    return np.array(pts), np.array(hvlines)

# From: https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
def angle_between(v1, v2):
    return np.arccos(np.clip(np.dot(v1/np.linalg.norm(v1), 
                                    v2/np.linalg.norm(v2)), -1.0, 1.0))
    