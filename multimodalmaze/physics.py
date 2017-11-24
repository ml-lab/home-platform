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

from panda3d.core import Vec3, NodePath, BitMask32, TransformState, LVecBase3f

from panda3d.bullet import BulletWorld, BulletTriangleMesh, BulletRigidBodyNode, BulletBoxShape, BulletTriangleMeshShape, \
                            BulletDebugNode, BulletCapsuleShape, BulletConvexHullShape

from multimodalmaze.core import World
from multimodalmaze.suncg import ObjectVoxelData, ModelCategoryMapping
from multimodalmaze.utils import mat4ToNumpyArray

logger = logging.getLogger(__name__)
                
def getCollisionShapeFromModel(model, mode='box', defaultCentered=False):
    
    #NOTE: make sure the position is relative to the center of the object
    minBounds, maxBounds = model.getTightBounds()
    offset = minBounds + (maxBounds - minBounds) / 2.0
    
    transform = TransformState.makeIdentity()
    if mode == 'mesh':
        # Use exact triangle mesh approximation
        mesh = BulletTriangleMesh()
        geomNodes = model.findAllMatches('**/+GeomNode')
        for nodePath in geomNodes:
            geomNode = nodePath.node()
            for n in range(geomNode.getNumGeoms()):
                geom = geomNode.getGeom(n)
                mesh.addGeom(geom)
        shape = BulletTriangleMeshShape(mesh, dynamic=False)
        transform = model.getTransform()
        
    elif mode == "sphere":
        minBounds, maxBounds = model.getTightBounds()
        dims = maxBounds - minBounds
        radius = np.sqrt(np.square(dims[0]) + np.square(dims[1])) / 2.0
        height = dims[2]
        shape = BulletCapsuleShape(radius, 2 * radius)
        if not defaultCentered:
            transform = TransformState.makePos(offset)
        
    elif mode == "hull":
        shape = BulletConvexHullShape()
        geomNodes = model.findAllMatches('**/+GeomNode')
        for nodePath in geomNodes:
            geomNode = nodePath.node()
            for n in range(geomNode.getNumGeoms()):
                geom = geomNode.getGeom(n)
                shape.addGeom(geom)
        
    elif mode == 'box':
        # Bounding box approximation
        minBounds, maxBounds = model.getTightBounds()
        dims = maxBounds - minBounds
        shape = BulletBoxShape(Vec3(dims.x / 2, dims.y / 2, dims.z  /2))
        if not defaultCentered:
            transform = TransformState.makePos(offset)
        
    elif mode == 'capsule':
        minBounds, maxBounds = model.getTightBounds()
        dims = maxBounds - minBounds
        radius = np.sqrt(np.square(dims[0]) + np.square(dims[1])) / 2.0
        height = dims[2]
        shape = BulletCapsuleShape(radius, height - 2 * radius)
        if not defaultCentered:
            transform = TransformState.makePos(offset)
        
    else:
        raise Exception('Unknown mode type for physic object collision shape: %s' % (mode))
    
    return shape, transform
                
class Panda3dBulletPhysics(World):

    # NOTE: the model ids of objects that correspond to opened doors. They will be ignored for collisions.
    openedDoorModelIds = [
        '122', '133', '214', '246', '247', '361', '73', '756', '757', '758', '759', '760',
        '761', '762', '763', '764', '765', '768', '769', '770', '771', '778', '779', '780',
        's__1762', 's__1763', 's__1764', 's__1765', 's__1766', 's__1767', 's__1768', 's__1769',
        's__1770', 's__1771', 's__1772', 's__1773',
    ]

    # FIXME: is not a complete list of movable objects
    movableObjectCategories = ['table', 'dressing_table', 'sofa', 'trash_can', 'chair', 'ottoman', 'bed']

    # For more material, see table: http://www.ambrsoft.com/CalcPhysics/Density/Table_2.htm
    defaultDensity = 1000.0  # kg/m^3

    # For more coefficients, see table: https://www.thoughtspike.com/friction-coefficients-for-bullet-physics/
    defaultMaterialFriction = 0.7
    
    defaultMaterialRestitution = 0.5

    def __init__(self, scene, suncgDatasetRoot=None, debug=False, objectMode='box',
                       agentRadius=0.1, agentHeight=1.6, agentMass=60.0, agentMode='capsule'):

        super(Panda3dBulletPhysics, self).__init__()

        self.__dict__.update(scene=scene, suncgDatasetRoot=suncgDatasetRoot, debug=debug, objectMode=objectMode,
                             agentRadius=agentRadius, agentHeight=agentHeight, agentMass=agentMass, agentMode=agentMode)
        
        if suncgDatasetRoot is not None:
            self.modelCatMapping = ModelCategoryMapping(os.path.join(suncgDatasetRoot, "metadata", "ModelCategoryMapping.csv"))
        else:
            self.modelCatMapping = None

        self.bulletWorld = BulletWorld()
        self.bulletWorld.setGravity(Vec3(0, 0, -9.81))
        
        if debug:
            debugNode = BulletDebugNode('physic-debug')
            debugNode.showWireframe(True)
            debugNode.showConstraints(False)
            debugNode.showBoundingBoxes(True)
            debugNode.showNormals(False)
            self.debugNodePath = self.scene.scene.attachNewNode(debugNode)
            self.debugNodePath.show()
            self.bulletWorld.setDebugNode(debugNode)
        else:
            self.debugNodePath = None
            
        self._initLayoutModels()
        self._initAgents()
        self._initObjects()
    
        self.scene.worlds['physics'] = self
    
    def destroy(self):
        # Nothing to do
        pass
    
    def _initLayoutModels(self):
        
        # Load layout objects as meshes
        for model in self.scene.scene.findAllMatches('**/layouts/object*/model*'):
            
            shape, transform = getCollisionShapeFromModel(model, mode='mesh', defaultCentered=False)
            
            node = BulletRigidBodyNode('physics')
            node.setMass(0.0)
            node.setFriction(self.defaultMaterialFriction)
            node.setRestitution(self.defaultMaterialRestitution)
            node.setStatic(True)
            node.addShape(shape)
            node.setDeactivationEnabled(True)
            node.setIntoCollideMask(BitMask32.allOn())
            self.bulletWorld.attach(node)
            
            # Attach the physic-related node to the scene graph
            physicsNp = model.getParent().attachNewNode(node)
            physicsNp.setTransform(transform)
            
            if node.isStatic():
                model.getParent().setTag('physics-mode', 'static')
            else:
                model.getParent().setTag('physics-mode', 'dynamic')
            
            # Reparent render and acoustics nodes (if any) below the new physic node
            #XXX: should be less error prone to just reparent all children (except the hidden model)
            renderNp = model.getParent().find('**/render')
            if not renderNp.isEmpty():
                renderNp.reparentTo(physicsNp)
            acousticsNp = model.getParent().find('**/acoustics')
            if not acousticsNp.isEmpty():
                acousticsNp.reparentTo(physicsNp)
    
            # NOTE: we need this to update the transform state of the internal bullet node
            physicsNp.node().setTransformDirty()
    
            # Validation
            assert np.allclose(mat4ToNumpyArray(physicsNp.getNetTransform().getMat()),
                               mat4ToNumpyArray(model.getNetTransform().getMat()), atol=1e-6)
    
    def _initAgents(self):
    
        # Load agents
        for agent in self.scene.scene.findAllMatches('**/agents/agent*'):
            
            transform = TransformState.makeIdentity()
            if self.agentMode == 'capsule':
                shape = BulletCapsuleShape(self.agentRadius, self.agentHeight - 2*self.agentRadius)
            elif self.agentMode == 'sphere':
                shape = BulletCapsuleShape(self.agentRadius, 2*self.agentRadius)
                
            # XXX: use BulletCharacterControllerNode class, which already handles local transform?
            node = BulletRigidBodyNode('physics')
            node.setMass(self.agentMass)
            node.setStatic(False)
            node.setFriction(self.defaultMaterialFriction)
            node.setRestitution(self.defaultMaterialRestitution)
            node.addShape(shape)
            self.bulletWorld.attach(node)
            
            # Constrain the agent to have fixed position on the Z-axis
            node.setLinearFactor(Vec3(1.0, 1.0, 0.0))
    
            # Constrain the agent to have rotation around the Z-axis only
            node.setAngularFactor(Vec3(0.0, 0.0, 1.0))
            
            node.setIntoCollideMask(BitMask32.allOn())
            node.setDeactivationEnabled(True)
            
            if node.isStatic():
                agent.setTag('physics-mode', 'static')
            else:
                agent.setTag('physics-mode', 'dynamic')
            
            # Attach the physic-related node to the scene graph
            physicsNp = NodePath(node)
            physicsNp.setTransform(transform)
            
            # Reparent all child nodes below the new physic node
            for child in agent.getChildren():
                child.reparentTo(physicsNp)
            physicsNp.reparentTo(agent)
            
            # NOTE: we need this to update the transform state of the internal bullet node
            physicsNp.node().setTransformDirty()                
            
            # Validation
            assert np.allclose(mat4ToNumpyArray(physicsNp.getNetTransform().getMat()),
                               mat4ToNumpyArray(agent.getNetTransform().getMat()), atol=1e-6)
            
    def _initObjects(self):
    
        # Load objects
        for model in self.scene.scene.findAllMatches('**/objects/object*/model*'):
            modelId = model.getParent().getTag('model-id')
            
            # XXX: we could create BulletGhostNode instance for non-collidable objects, but we would need to filter out the collisions later on
            if not modelId in self.openedDoorModelIds:
                
                shape, transform = getCollisionShapeFromModel(model, self.objectMode, defaultCentered=True)
                
                node = BulletRigidBodyNode('physics')
                node.addShape(shape)
                node.setFriction(self.defaultMaterialFriction)
                node.setRestitution(self.defaultMaterialRestitution)
                node.setIntoCollideMask(BitMask32.allOn())
                node.setDeactivationEnabled(True)
                self.bulletWorld.attach(node)
                
                if self.suncgDatasetRoot is not None:
                    
                    # Check if it is a movable object
                    category = self.modelCatMapping.getCoarseGrainedCategoryForModelId(modelId)
                    if category in self.movableObjectCategories:
                        # Estimate mass of object based on volumetric data and default material density
                        objVoxFilename = os.path.join(self.suncgDatasetRoot, 'object_vox', 'object_vox_data', modelId, modelId + '.binvox')
                        voxelData = ObjectVoxelData.fromFile(objVoxFilename)
                        mass = Panda3dBulletPhysics.defaultDensity * voxelData.getFilledVolume()
                        node.setMass(mass)
                    else:
                        node.setMass(0.0)
                        node.setStatic(True)
                else:
                    node.setMass(0.0)
                    node.setStatic(True)
                
                if node.isStatic():
                    model.getParent().setTag('physics-mode', 'static')
                else:
                    model.getParent().setTag('physics-mode', 'dynamic')
                
                # Attach the physic-related node to the scene graph
                physicsNp = model.getParent().attachNewNode(node)
                physicsNp.setTransform(transform)
                
                # Reparent render and acoustics nodes (if any) below the new physic node
                #XXX: should be less error prone to just reparent all children (except the hidden model)
                renderNp = model.getParent().find('**/render')
                if not renderNp.isEmpty():
                    renderNp.reparentTo(physicsNp)
                acousticsNp = model.getParent().find('**/acoustics')
                if not acousticsNp.isEmpty():
                    acousticsNp.reparentTo(physicsNp)
                
                # NOTE: we need this to update the transform state of the internal bullet node
                physicsNp.node().setTransformDirty()
                
                # Validation
                assert np.allclose(mat4ToNumpyArray(physicsNp.getNetTransform().getMat()),
                                   mat4ToNumpyArray(model.getParent().getNetTransform().getMat()), atol=1e-6)
                
            else:
                logger.debug('Object %s ignored from physics' % (modelId))

    def step(self, dt):
        self.bulletWorld.doPhysics(dt)

    def isCollision(self, root):
        isCollisionDetected = False
        if isinstance(root.node(), BulletRigidBodyNode):
            result = self.bulletWorld.contactTest(root.node())
            if result.getNumContacts() > 0:
                isCollisionDetected = True
        else:
            for nodePath in root.findAllMatches('**/+BulletBodyNode'):
                result = self.bulletWorld.contactTest(nodePath.node())
                if result.getNumContacts() > 0:
                    isCollisionDetected = True
        return isCollisionDetected

    def calculate2dNavigationMap(self, agent, z=0.1, precision=0.1, yup=True):
    
        agentRbNp = agent.find('**/+BulletRigidBodyNode')
    
        # Calculate the bounding box of the scene
        bounds = []
        for nodePath in self.scene.scene.findAllMatches('**/object*/+BulletRigidBodyNode'):
            node = nodePath.node()

            #NOTE: the bounding sphere doesn't seem to take into account the transform, so apply it manually (translation only)
            bsphere = node.getShapeBounds()
            center = nodePath.getNetTransform().getPos()
            bounds.extend([center + bsphere.getMin(), center + bsphere.getMax()])
                
        minBounds, maxBounds = np.min(bounds, axis=0), np.max(bounds, axis=0)
        
        # Using the X and Y dimensions of the bounding box, discretize the 2D plan into a uniform grid with given precision
        X = np.arange(minBounds[0], maxBounds[0], step=precision)
        Y = np.arange(minBounds[1], maxBounds[1], step=precision)
        nbTotalCells = len(X) * len(Y)
        threshold10Perc = int(nbTotalCells / 10)
      
        # XXX: the simulation needs to be run a little before moving the agent, not sure why
        self.bulletWorld.doPhysics(0.1)
      
        # Sweep the position of the agent across the grid, checking if collision/contacts occurs with objects or walls in the scene.
        occupancyMap = np.zeros((len(X), len(Y)))
        occupancyMapCoord = np.zeros((len(X), len(Y), 2))
        n = 0
        for i,x in enumerate(X):
            for j,y in enumerate(Y):
                agentRbNp.setPos(LVecBase3f(x, y, z))
                
                if self.isCollision(agentRbNp):
                    occupancyMap[i,j] = 1.0
        
                occupancyMapCoord[i,j,0] = x
                occupancyMapCoord[i,j,1] = y
                
                n += 1
                if n % threshold10Perc == 0:
                    logger.debug('Collision test no.%d (out of %d total)' % (n, nbTotalCells))
        
        if yup:
            # Convert to image format (y,x)
            occupancyMap = np.flipud(occupancyMap.T)
            occupancyMapCoord = np.flipud(np.transpose(occupancyMapCoord, axes=(1, 0, 2)))

        return occupancyMap, occupancyMapCoord
