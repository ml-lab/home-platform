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
#  - Neither the name of the NECOTIS research group nor the names of its contributors 
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
import numpy as np

from panda3d.core import Vec3, Mat4, CS_zup_right, CS_yup_right, LoaderOptions, Filename, NodePath, Loader, ClockObject, BitMask32

from panda3d.bullet import BulletWorld, BulletTriangleMesh, BulletRigidBodyNode, BulletBoxShape, BulletTriangleMeshShape, \
                            BulletDebugNode, BulletPlaneShape, BulletCapsuleShape, BulletCharacterControllerNode, ZUp


#TODO: sweep map with agent to find navigable 2D map
#    see: https://www.panda3d.org/manual/index.php/Bullet_Queries

class PhysicWorld(object):
    
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

class Panda3dBulletPhysicWorld(PhysicWorld):

    def __init__(self, debug=False):

        self.physicWorld = BulletWorld()
        self.physicWorld.setGravity(Vec3(0, 0, -9.81))
        self.globalClock = ClockObject.getGlobalClock()
        
        self.render = NodePath('physic-render')
        
        # Plane
        shape = BulletPlaneShape(Vec3(0, 0, 1), 0)
        node = BulletRigidBodyNode('physic-ground')
        node.addShape(shape)
        self.groundNodePath = self.render.attachNewNode(node)
        self.groundNodePath.setPos(0, 0, 0)
        self.physicWorld.attach(node)
        
        if debug:
            debugNode = BulletDebugNode('physic-debug')
            debugNode.showWireframe(True)
            debugNode.showConstraints(True)
            debugNode.showBoundingBoxes(True)
            debugNode.showNormals(True)
            self.debugNodePath = self.render.attachNewNode(debugNode)
            self.debugNodePath.show()
            self.physicWorld.setDebugNode(debugNode)
        else:
            self.debugNodePath = None
    
    def _loadModel(self, modelPath):
        loader = Loader.getGlobalPtr()
        loaderOptions = LoaderOptions()
        node = loader.loadSync(Filename(modelPath), loaderOptions)
        if node is not None:
            nodePath = NodePath(node)
        else:
            raise IOError('Could not load model file: %s' % (modelPath))
        return nodePath
    
    def connectToRenderWorld(self, renderWorld):

        # Add debug node, if any
        if self.debugNodePath is not None:
            self.debugNodePath.reparentTo(renderWorld.render)
        self.groundNodePath.reparentTo(renderWorld.render)

        # Loop throught all physic-related nodepaths in graph
        for physicNodePath in self.render.getChildren():
            
            # Find matching nodepath in render-related graph
            name = physicNodePath.getName()
            renderNodePath = renderWorld.render.find('**/%s*' % (name))
            if renderNodePath.getNumNodes() == 0:
                raise Exception('Could not find matching nodepath for rendering: %s' % (name))
            
            # Reparent physic-related node to render graph
            physicNodePath.reparentTo(renderNodePath.getParent())
            renderNodePath.reparentTo(physicNodePath)
    
    def addAgentToScene(self, agent, radius=0.25, height=1.6):
        
        #NOTE: implement agent with Bullet Character Controller
        #      see: https://www.panda3d.org/manual/index.php/Bullet_Character_Controller
        shape = BulletCapsuleShape(radius, height - 2*radius, ZUp)
        node = BulletCharacterControllerNode(shape, radius, 'agent-' + str(agent.instanceId))
        self.physicWorld.attach(node)
        nodePath = self.render.attachNewNode(node)
        nodePath.setCollideMask(BitMask32.allOn())
        return nodePath
                
    def addObjectToScene(self, obj, dynamic=True, mode='bbox'):

        # Load model from file
        model = self._loadModel(obj.modelFilename)
        
        if obj.transform is not None:
            # 4x4 column-major transformation matrix from object coordinates to scene coordinates
            transformMat = Mat4(*obj.transform.ravel())
            yupTransformMat = Mat4.convertMat(CS_zup_right, CS_yup_right)
            zupTransformMat = Mat4.convertMat(CS_yup_right, CS_zup_right)
            model.setMat(model.getMat() * yupTransformMat * transformMat * zupTransformMat)
            
        if mode == 'exact':
            # Use exact triangle mesh approximation
            mesh = BulletTriangleMesh()
            geomNodes = model.findAllMatches('**/+GeomNode')
            for nodePath in geomNodes:
                geomNode = nodePath.node()
                for n in range(geomNode.getNumGeoms()):
                    geom = geomNode.getGeom(n)
                    mesh.addGeom(geom)
            #TODO: is is the shape that is dynamic?
            shape = BulletTriangleMeshShape(mesh, dynamic=dynamic)
            
            node = BulletRigidBodyNode('object-' + str(obj.instanceId))
            node.setMass(0.0)
            node.addShape(shape)
        elif mode == 'bbox':
            # Bounding box approximation
            minBounds, maxBounds = model.getTightBounds()
            dims = maxBounds - minBounds
            shape = BulletBoxShape(Vec3(dims.x/2, dims.y/2, dims.z/2))

            node = BulletRigidBodyNode('object-' + str(obj.instanceId))
            node.setMass(0.0)
            node.addShape(shape)
        else:
            raise Exception('Unknown mode type for physic object collision shape: %s' % (mode))
    
        nodePath = self.render.attachNewNode(node)
        nodePath.setMat(model.getMat())
    
        model.detachNode()
        
        self.physicWorld.attach(node)
        
        return nodePath
    
    def addRoomToScene(self, room):

        nodes = []
        for modelFilename in room.modelFilenames:
            
            partId = os.path.splitext(os.path.basename(modelFilename))[0]
            model = self._loadModel(modelFilename)
            
            mesh = BulletTriangleMesh()
            geomNodes = model.findAllMatches('**/+GeomNode')
            for nodePath in geomNodes:
                geomNode = nodePath.node()
                for n in range(geomNode.getNumGeoms()):
                    geom = geomNode.getGeom(n)
                    mesh.addGeom(geom)
            shape = BulletTriangleMeshShape(mesh, dynamic=False)       
            node = BulletRigidBodyNode('room-' + str(room.instanceId) + '-' + partId)
            node.setMass(0.0)
            node.addShape(shape)
            
            self.physicWorld.attach(node)
            np = self.render.attachNewNode(node)

            nodes.append(node)
            model.detachNode()
                
        for obj in room.objects:
            node = self.addObjectToScene(obj)
            nodes.append(node)

        return nodes

    def addHouseToScene(self, house):

        nodes = []
        for room in house.rooms:
            roomNodes = self.addRoomToScene(room)
            nodes.extend(roomNodes)
        
        for obj in house.objects:
            objNode = self.addObjectToScene(obj)
            nodes.append(objNode)
            
        return nodes
    
    def step(self):
        dt = self.globalClock.getDt()
        self.physicWorld.doPhysics(dt)
    
    def resetScene(self):
        pass

def calculate2dMapFromScene(house, actor):
    pass

    # NOTE: no need for rendering here
    
    #TODO: get the bounding box of the scene.
    
    #TODO: using the X and Y dimensions of the bounding box, discretize the 2D plan into a uniform grid. 
  
    #TODO: sweep the position of the agent across the grid, checking if collision/contacts occurs with objects or walls in the scene.
    
    #TODO: using random starting points, use the brushfire algorithm to see which cells are assessible for navigation and path-planning in the scene.