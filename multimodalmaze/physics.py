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

from panda3d.core import Vec3, Mat4, CS_zup_right, CS_yup_right

from panda3d.bullet import BulletTriangleMesh, BulletRigidBodyNode, BulletBoxShape, BulletTriangleMeshShape, \
                            BulletDebugNode, BulletPlaneShape


#TODO: sweep map with agent to find navigable 2D map
#    see: https://www.panda3d.org/manual/index.php/Bullet_Queries

# # from panda3d.core import Mat4, CS_zup_right, CS_yup_right
# #yupTransformMat = Mat4.convertMat(CS_zup_right, CS_yup_right)
# yupTransformMat = np.array([[1,0,0,0],
#                             [0,0,-1,0],
#                             [0,1,0,0],
#                             [0,0,0,1]])
# 
# #zupTransformMat = Mat4.convertMat(CS_yup_right, CS_zup_right)
# zupTransformMat = np.array([[1,0,0,0],
#                             [0,0,1,0],
#                             [0,-1,0,0],
#                             [0,0,0,1]])
# 
# #print 'yupTransformMat = ', yupTransformMat
# #print 'zupTransformMat = ', zupTransformMat
# #transform = np.dot(np.dot(transformMat, yupTransformMat), zupTransformMat)


class PhysicWorld(object):
    
    def __init__(self):
        pass

    def addObjectToScene(self, obj):
        pass
    
    def addRoomToScene(self, room):
        pass
    
    def addHouseToScene(self, house):
        pass

    def step(self, time):
        pass
    
    def run(self):
        pass
    
    def resetScene(self):
        pass

class Panda3dBulletPhysicWorld(PhysicWorld):

    def __init__(self, debug=False):

        from panda3d.bullet import BulletWorld
        self.world = BulletWorld()
        self.world.setGravity(Vec3(0, 0, -9.81))
        
        # Plane
        shape = BulletPlaneShape(Vec3(0, 0, 1), 0)
        node = BulletRigidBodyNode('Ground')
        node.addShape(shape)
        np = base.render.attachNewNode(node)
        np.setPos(0, 0, 0)
        self.world.attach(node)
        
        if debug:
            debugNode = BulletDebugNode('Debug')
            debugNode.showWireframe(True)
            debugNode.showConstraints(True)
            debugNode.showBoundingBoxes(True)
            debugNode.showNormals(True)
            debugNP = base.render.attachNewNode(debugNode)
            debugNP.show()
            
        self.world.setDebugNode(debugNP.node())
            
    def addObjectToScene(self, obj, dynamic=True, mode='bbox'):

        # Load model from file
        model = base.loader.loadModel(obj.modelFilename)
        
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
            node = BulletRigidBodyNode(obj.instanceId)
            node.setMass(0.0)
            node.addShape(shape)
        elif mode == 'bbox':
            # Bounding box approximation
            minBounds, maxBounds = model.getTightBounds()
            dims = maxBounds - minBounds
            shape = BulletBoxShape(Vec3(dims.x/2, dims.y/2, dims.z/2))

            node = BulletRigidBodyNode(obj.instanceId)
            node.setMass(0.0)
            node.addShape(shape)
        else:
            raise Exception('Unknown mode type for physic object collision shape: %s' % (mode))
    
        np = base.render.attachNewNode(node)
        np.setMat(model.getMat())
    
        model.detachNode()
        
        self.world.attach(node)
        
        return node
    
    def addRoomToScene(self, room):

        nodes = []
        for modelFilename in room.modelFilenames:
            
            partId = os.path.splitext(os.path.basename(modelFilename))[0]
            objInstanceId = 'room-' + str(room.instanceId) + '-' + partId
            model = base.loader.loadModel(modelFilename)
            
            mesh = BulletTriangleMesh()
            geomNodes = model.findAllMatches('**/+GeomNode')
            for nodePath in geomNodes:
                geomNode = nodePath.node()
                for n in range(geomNode.getNumGeoms()):
                    geom = geomNode.getGeom(n)
                    mesh.addGeom(geom)
            shape = BulletTriangleMeshShape(mesh, dynamic=False)       
            node = BulletRigidBodyNode(objInstanceId)
            node.setMass(0.0)
            node.addShape(shape)
            
            self.world.attach(node)
            np = base.render.attachNewNode(node)

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
    
    def step(self, t=0.0):
        dt = globalClock.getDt()
        self.world.doPhysics(dt)
    
    def run(self):
        while True:
            self.step()
    
    def resetScene(self):
        pass

def calculate2dMapFromScene(house, actor):
    pass

    # NOTE: no need for rendering here
    
    #TODO: get the bounding box of the scene.
    
    #TODO: using the X and Y dimensions of the bounding box, discretize the 2D plan into a uniform grid. 
  
    #TODO: sweep the position of the agent across the grid, checking if collision/contacts occurs with objects or walls in the scene.
    
    #TODO: using random starting points, use the brushfire algorithm to see which cells are assessible for navigation and path-planning in the scene.