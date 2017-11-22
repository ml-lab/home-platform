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

from panda3d.core import Vec3, Mat4, LoaderOptions, Filename, NodePath, Loader, ClockObject, BitMask32, TransformState, \
    LVector3f

from panda3d.bullet import BulletWorld, BulletTriangleMesh, BulletRigidBodyNode, BulletBoxShape, \
    BulletTriangleMeshShape, \
    BulletDebugNode, BulletCapsuleShape, BulletConvexHullShape, BulletBodyNode

from multimodalmaze.suncg import ObjectVoxelData, ModelCategoryMapping

logger = logging.getLogger(__name__)


class PhysicObject(object):
    def getTransform(self):
        return NotImplementedError()

    def setTransform(self):
        return NotImplementedError()

    def setLinearVelocity(self, velocity):
        return NotImplementedError()

    def setAngularVelocity(self, velocity):
        return NotImplementedError()

    def isCollision(self):
        return NotImplementedError()


class PhysicWorld(object):
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


class Panda3dBulletPhysicObject(PhysicObject):
    def __init__(self, world, nodePath, recenterTransform=None):
        self.world = world
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

    def getRecenterPosition(self):
        position = self.recenterTransform.getPos()
        return np.array([position.x, position.y, position.z])

    def setTransform(self, transform):
        mat = Mat4(*transform.ravel())
        self.nodePath.setTransform(TransformState.makeMat(mat).compose(self.recenterTransform))

    def setLinearVelocity(self, velocity):

        # Apply the local transform to the velocity
        # XXX: use BulletCharacterControllerNode class, which already handles local transform?
        mat = self.nodePath.node().getTransform().getMat()
        rotMat = np.array([[mat[0][0], mat[0][1], mat[0][2]],
                           [mat[1][0], mat[1][1], mat[1][2]],
                           [mat[2][0], mat[2][1], mat[2][2]]])
        velocity = np.dot(velocity, rotMat)

        velocity = Vec3(velocity[0], velocity[1], velocity[2])
        self.nodePath.node().setLinearVelocity(velocity)

    def applyImpulse(self, impulse):
        proper_vector = LVector3f(impulse[0], impulse[1], impulse[2])
        self.nodePath.node().applyCentralImpulse(proper_vector)

    def setAngularVelocity(self, velocity):
        velocity = Vec3(velocity[0], velocity[1], velocity[2])
        self.nodePath.node().setAngularVelocity(velocity)

    def isCollision(self):
        isCollisionDetected = False

        node = self.nodePath.node()
        if isinstance(node, BulletBodyNode):
            result = self.world.contactTest(node)
            if result.getNumContacts() > 0:
                isCollisionDetected = True
        else:
            for nodePath in self.nodePath.findAllMatches('**/+BulletBodyNode'):
                node = nodePath.node()
                if isinstance(node, BulletBodyNode):
                    result = self.world.contactTest(node)
                    if result.getNumContacts() > 0:
                        isCollisionDetected = True

        return isCollisionDetected


def getCollisionShapeFromModel(model, mode='box'):
    # NOTE: make sure the position is relative to the center of the object
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

    elif mode == "sphere":
        minBounds, maxBounds = model.getTightBounds()
        dims = maxBounds - minBounds
        radius = np.sqrt(np.square(dims[0]) + np.square(dims[1])) / 2.0
        height = dims[2]
        shape = BulletCapsuleShape(radius, 2 * radius)
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
        shape = BulletBoxShape(Vec3(dims.x / 2, dims.y / 2, dims.z / 2))
        transform = TransformState.makePos(offset)

    elif mode == 'capsule':
        minBounds, maxBounds = model.getTightBounds()
        dims = maxBounds - minBounds
        radius = np.sqrt(np.square(dims[0]) + np.square(dims[1])) / 2.0
        height = dims[2]
        shape = BulletCapsuleShape(radius, height - 2 * radius)
        transform = TransformState.makePos(offset)

    else:
        raise Exception('Unknown mode type for physic object collision shape: %s' % (mode))

    return shape, transform


class Panda3dBulletPhysicWorld(PhysicWorld):
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

    def __init__(self, suncgDatasetRoot=None, debug=False):

        self.suncgDatasetRoot = suncgDatasetRoot
        if suncgDatasetRoot is not None:
            self.modelCatMapping = ModelCategoryMapping(
                os.path.join(suncgDatasetRoot, "metadata", "ModelCategoryMapping.csv"))
        else:
            self.modelCatMapping = None

        self.physicWorld = BulletWorld()
        self.physicWorld.setGravity(Vec3(0, 0, -9.81))
        self.globalClock = ClockObject.getGlobalClock()

        self.render = NodePath('physic-render')

        self.agent_physics_node = None

        if debug:
            debugNode = BulletDebugNode('physic-debug')
            debugNode.showWireframe(True)
            debugNode.showConstraints(False)
            debugNode.showBoundingBoxes(False)
            debugNode.showNormals(False)
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

    def addAgentToScene(self, agent, radius=0.1, height=1.6, mass=60.0, mode='capsule'):

        transform = TransformState.makeIdentity()
        if agent.modelFilename is not None:

            # Load model from file
            model = self._loadModel(agent.modelFilename)
            shape, transform = getCollisionShapeFromModel(model, mode)
        elif mode == 'capsule':
            shape = BulletCapsuleShape(radius, height - 2 * radius)
        elif mode == 'sphere':
            shape = BulletCapsuleShape(radius, 2 * radius)

        # XXX: use BulletCharacterControllerNode class, which already handles local transform?
        node = BulletRigidBodyNode('agent-' + str(agent.instanceId))
        node.setMass(mass)
        node.setStatic(False)
        node.setFriction(self.defaultMaterialFriction)
        node.setRestitution(self.defaultMaterialRestitution)
        node.addShape(shape)

        # Constrain the agent to have fixed position on the Z-axis
        node.setLinearFactor(Vec3(1.0, 1.0, 0.0))

        # Constrain the agent to have rotation around the Z-axis only
        node.setAngularFactor(Vec3(0.0, 0.0, 1.0))

        node.setIntoCollideMask(BitMask32.allOn())
        node.setDeactivationEnabled(False)

        self.agent_physics_node = node

        self.physicWorld.attach(node)
        nodePath = self.render.attachNewNode(node)

        instance = Panda3dBulletPhysicObject(self.physicWorld, nodePath)
        instance.recenterTransform = transform
        agent.setPhysicObject(instance)
        agent.assertConsistency()

    def addObjectToScene(self, obj, mode='box'):

        # Load model from file
        model = self._loadModel(obj.modelFilename)
        shape, transform = getCollisionShapeFromModel(model, mode)

        # XXX: we could create BulletGhostNode instance for non-collidable objects, but we would need to filter out the collisions later on
        if not obj.modelId in self.openedDoorModelIds:
            node = BulletRigidBodyNode('object-' + str(obj.instanceId))

            if self.suncgDatasetRoot is not None:

                # Check if it is a movable object
                category = self.modelCatMapping.getCoarseGrainedCategoryForModelId(obj.modelId)
                if category in self.movableObjectCategories:
                    # Estimate mass of object based on volumetric data and default material density
                    objVoxFilename = os.path.join(self.suncgDatasetRoot, 'object_vox', 'object_vox_data',
                                                  str(obj.modelId), str(obj.modelId) + '.binvox')
                    voxelData = ObjectVoxelData.fromFile(objVoxFilename)
                    mass = Panda3dBulletPhysicWorld.defaultDensity * voxelData.getFilledVolume()
                    node.setMass(mass)
                else:
                    node.setMass(0.0)
                    node.setStatic(True)
            else:
                node.setMass(0.0)
                node.setStatic(True)

            node.setFriction(self.defaultMaterialFriction)
            node.setRestitution(self.defaultMaterialRestitution)
            node.addShape(shape)
            node.setIntoCollideMask(BitMask32.allOn())
            node.setDeactivationEnabled(False)

            model.detachNode()
            self.physicWorld.attach(node)
            nodePath = self.render.attachNewNode(node)

            instance = Panda3dBulletPhysicObject(self.physicWorld, nodePath)
            instance.recenterTransform = transform
            obj.setPhysicObject(instance)

        else:
            logger.debug('Object %s ignored from physics' % (obj.instanceId))

        obj.assertConsistency()

    def addRoomToScene(self, room):

        roomNodePath = self.render.attachNewNode('room-' + str(room.instanceId))
        for modelFilename in room.modelFilenames:
            partId = os.path.splitext(os.path.basename(modelFilename))[0]
            model = self._loadModel(modelFilename)

            shape, _ = getCollisionShapeFromModel(model, mode='mesh')
            node = BulletRigidBodyNode('room-' + str(room.instanceId) + '-' + partId)
            node.setMass(0.0)
            node.setFriction(self.defaultMaterialFriction)
            node.setRestitution(self.defaultMaterialRestitution)
            node.setStatic(True)
            node.addShape(shape)
            node.setDeactivationEnabled(False)
            node.setIntoCollideMask(BitMask32.allOn())

            self.physicWorld.attach(node)
            roomNodePath.attachNewNode(node)
            model.detachNode()

        instance = Panda3dBulletPhysicObject(self.physicWorld, roomNodePath)
        room.setPhysicObject(instance)

        for obj in room.objects:
            self.addObjectToScene(obj)

    def addHouseToScene(self, house):

        for room in house.rooms:
            self.addRoomToScene(room)

        for room in house.grounds:
            self.addRoomToScene(room)

        for obj in house.objects:
            self.addObjectToScene(obj)

    def step(self):

        # XXX: if we only use bullet for collision detection among static objects, we may not even need to simulate physics  

        # Update physics from global clock
        dt = self.globalClock.getDt()
        self.physicWorld.doPhysics(dt)

    def calculate2dNavigationMap(self, agent, z=0.1, precision=0.1, yup=True):

        # Calculate the bounding box of the scene
        bounds = []
        for node in self.physicWorld.getRigidBodies():
            # Filter ground planes and agents
            if not 'agent' in node.getName():
                # NOTE: the bounding sphere doesn't seem to take into account the transform, so apply it manually (translation only)
                bsphere = node.getShapeBounds()
                center = node.getTransform().getPos()
                bounds.extend([center + bsphere.getMin(), center + bsphere.getMax()])
        minBounds, maxBounds = np.min(bounds, axis=0), np.max(bounds, axis=0)

        # Using the X and Y dimensions of the bounding box, discretize the 2D plan into a uniform grid with given precision
        X = np.arange(minBounds[0], maxBounds[0], step=precision)
        Y = np.arange(minBounds[1], maxBounds[1], step=precision)
        nbTotalCells = len(X) * len(Y)
        threshold10Perc = int(nbTotalCells / 10)

        # XXX: the simulation needs to be run a little before moving the agent, not sure why
        self.physicWorld.doPhysics(0.1)

        # Sweep the position of the agent across the grid, checking if collision/contacts occurs with objects or walls in the scene.
        occupancyMap = np.zeros((len(X), len(Y)))
        occupancyMapCoord = np.zeros((len(X), len(Y), 2))
        n = 0
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                agent.setPosition([x, y, z])

                if agent.isCollision():
                    occupancyMap[i, j] = 1.0

                occupancyMapCoord[i, j, 0] = x
                occupancyMapCoord[i, j, 1] = y

                n += 1
                if n % threshold10Perc == 0:
                    logger.debug('Collision test no.%d (out of %d total)' % (n, nbTotalCells))

        if yup:
            # Convert to image format (y,x)
            occupancyMap = np.flipud(occupancyMap.T)
            occupancyMapCoord = np.flipud(np.transpose(occupancyMapCoord, axes=(1, 0, 2)))

        return occupancyMap, occupancyMapCoord
