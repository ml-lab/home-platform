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
import glob
import json
import logging
import numpy as np

logger = logging.getLogger(__name__)

def eulerToMatrix(theta):
    # Adapted from: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    R_x = np.array([[1,         0,                  0                   ],
                [0,         np.cos(theta[0]), -np.sin(theta[0]) ],
                [0,         np.sin(theta[0]), np.cos(theta[0])  ]
                ])
                 
    R_y = np.array([[np.cos(theta[1]),    0,      np.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-np.sin(theta[1]),   0,      np.cos(theta[1])  ]
                    ])
                 
    R_z = np.array([[np.cos(theta[2]),    -np.sin(theta[2]),    0],
                    [np.sin(theta[2]),    np.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                     
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R

def isRotationMatrix(R):
    # Adapted from: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def matrixToEuler(R):
    # Adapted from: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    assert(isRotationMatrix(R))
    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
 
    singular = sy < 1e-6
    if  not singular :
        x = np.arctan2(R[2,1] , R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else :
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])
    

class Object(object):
    
    def __init__(self, instanceId, modelId=None, modelFilename=None, static=False):
        self.instanceId = instanceId
        self.modelId = modelId
        self.modelFilename = modelFilename
        self.attributes = {}
        self.static = static

        self.transform = np.eye(4,4)

        self.physicInstance = None
        self.renderInstance = None
        self.acousticInstance = None

    def setTransform(self, transform):
        transform = np.atleast_2d(transform)
        assert transform.ndim == 2
        assert transform.shape[0] == transform.shape[1] == 4
        
        self.transform = transform
        self._forceTransform()

    def setPosition(self, position):
        position = np.atleast_1d(position)
        assert len(position) == 3
        
        self.transform[-1,:3] = position
        self._forceTransform()
        
    def setLinearVelocity(self, velocity):
        if self.physicInstance is not None:
            self.physicInstance.setLinearVelocity(velocity)
    
    def setAngularVelocity(self, velocity):
        if self.physicInstance is not None:
            self.physicInstance.setAngularVelocity(velocity)
            
    def isCollision(self):
        isCollisionDetected = False
        if self.physicInstance is not None:
            isCollisionDetected = self.physicInstance.isCollision()
        return isCollisionDetected
            
    def getPosition(self):
        self._syncTransform()
        return np.copy(self.transform[-1,:3])
    
    def getOrientation(self):
        self._syncTransform()
        return matrixToEuler(self.transform[:3,:3])
    
    def getTransform(self):
        self._syncTransform()
        return np.copy(self.transform)
    
    def setOrientation(self, theta):
        theta = np.atleast_1d(theta)
        assert len(theta) == 3
        
        self.transform[:3,:3] = eulerToMatrix(theta)
        self._forceTransform()
    
    def clearAttributes(self):
        self.attributes.clear()

    def _forceTransform(self):
        if self.physicInstance is not None:
            self.physicInstance.setTransform(self.transform)
        if self.renderInstance is not None:
            self.renderInstance.setTransform(self.transform)
        if self.acousticInstance is not None:
            self.acousticInstance.setTransform(self.transform)

    def _syncTransform(self):
        # Get updated transform from the physic instance
        if self.physicInstance is not None:
            self.transform = self.physicInstance.getTransform()
            
        # Apply it to all other instances (rendering, acoustics)
        if self.renderInstance is not None:
            self.renderInstance.setTransform(self.transform)
        if self.acousticInstance is not None:
            self.acousticInstance.setTransform(self.transform)

    def assertConsistency(self, atol=1e-6):
        if self.physicInstance is not None:
            assert np.allclose(self.transform, self.physicInstance.getTransform(), atol=atol)
        if self.renderInstance is not None:
            assert np.allclose(self.transform, self.renderInstance.getTransform(), atol=atol)
        if self.acousticInstance is not None:
            assert np.allclose(self.transform, self.acousticInstance.getTransform(), atol=atol)
            
    def setPhysicObject(self, instance):
        self.physicInstance = instance
        self.physicInstance.setTransform(self.transform)
        
    def setAcousticObject(self, instance):
        self.acousticInstance = instance
        self.acousticInstance.setTransform(self.transform)
        
    def setRenderObject(self, instance):
        self.renderInstance = instance
        self.renderInstance.setTransform(self.transform)

class Agent(Object):
    
    def __init__(self, instanceId, modelFilename=None):
        super(Agent, self).__init__(instanceId, None, modelFilename, static=False)

    #TODO: support full transform here?
    def setCameraOrientation(self, theta):
        if self.renderInstance is not None:
            self.renderInstance.setCameraOrientation(theta)

class Room(object):

    def __init__(self, modelId, sceneId, levelId, modelFilenames, objects=None):
        self.levelId = levelId
        self.instanceId = modelId
        self.modelId = modelId
        self.sceneId = sceneId
        self.modelFilenames = modelFilenames
        if objects is None:
            objects = []
        self.objects = objects
        self.attributes = {}
        
        self.transform = np.eye(4,4)
        
        self.physicInstance = None
        self.renderInstance = None
        self.acousticInstance = None

    def clearAttributes(self):
        self.attributes.clear()
    
    def setPhysicObject(self, instance):
        self.physicInstance = instance
        #self.physicInstance.setTransform(self.transform)
        
    def setAcousticObject(self, instance):
        self.acousticInstance = instance
        #self.acousticInstance.setTransform(self.transform)
        
    def setRenderObject(self, instance):
        self.renderInstance = instance
        #self.renderInstance.setTransform(self.transform)
    
#TODO: add support for multiple levels
    
class House(object):
    
    def __init__(self, sceneId, rooms=None, objects=None, showCeiling=True):
        self.instanceId = sceneId
        self.sceneId = sceneId
        if rooms is None:
            rooms = []
        self.rooms = rooms
        if objects is None:
            objects = []
        self.objects = objects
        self.showCeiling = showCeiling
        self.attributes = {}
        
    def clearAttributes(self):
        self.attributes.clear()

    def getNbLevels(self):
        levelIds = []
        for room in self.rooms:
            levelIds.append(room.levelId)
        return len(set(levelIds))

    @staticmethod
    def loadFromJson(filename, datasetRoot):
        
        with open(filename) as f:
            data = json.load(f)
        
        objectIds = {}
        objects = []
        rooms = []
        sceneId = data['id']
        for levelId, level in enumerate(data['levels']):
            logger.debug('Loading Level %s to scene' % (str(levelId)))
            
            roomByNodeIndex = {}
            for nodeIndex, node in enumerate(level['nodes']):
                if not node['valid'] == 1: continue
                    
                modelId = node['modelId']
                    
                if node['type'] == 'Room':
                    logger.debug('Loading Room %s to scene' % (modelId))
                    
                    # Get room model filenames
                    modelFilenames = []
                    for roomObjFilename in glob.glob(os.path.join(datasetRoot, 'room', sceneId, modelId + '*.obj')):
                        
                        # Convert extension from OBJ + MTL to EGG format
                        f, _ = os.path.splitext(roomObjFilename)
                        modelFilename = f + ".egg"
                        if not os.path.exists(modelFilename):
                            raise Exception('The SUNCG dataset object models need to be convert to Panda3D EGG format!')
                        modelFilenames.append(modelFilename)
                    
                    room = Room(modelId, sceneId, levelId, modelFilenames)
                    rooms.append(room)
                    
                    for childNodeIndex in node['nodeIndices']:
                        roomByNodeIndex[childNodeIndex] = room
                    
                elif node['type'] == 'Object':
                    
                    logger.debug('Loading Object %s to scene' % (modelId))
                    
                    # Convert extension from OBJ + MTL to EGG format
                    objFilename = os.path.join(datasetRoot, 'object', node['modelId'], node['modelId'] + '.obj')
                    assert os.path.exists(objFilename)
                    f, _ = os.path.splitext(objFilename)
                    modelFilename = f + ".egg"
                    if not os.path.exists(modelFilename):
                        raise Exception('The SUNCG dataset object models need to be convert to Panda3D EGG format!')
                    
                    # 4x4 column-major transformation matrix from object coordinates to scene coordinates
                    transform = np.array(node['transform']).reshape((4,4))
                    
                    # Transform from Y-UP to Z-UP coordinate systems
                    yupTransform = np.array([[1, 0, 0, 0],
                                            [0, 0, -1, 0],
                                            [0, 1, 0, 0],
                                            [0, 0, 0, 1]])
                    
                    zupTransform = np.array([[1, 0, 0, 0],
                                            [0, 0, 1, 0],
                                            [0, -1, 0, 0],
                                            [0, 0, 0, 1]])
                    
                    transform = np.dot(np.dot(yupTransform, transform), zupTransform)
                    
                    # Instance identification
                    if modelId in objectIds:
                        objectIds[modelId] = objectIds[modelId] + 1
                    else:
                        objectIds[modelId] = 0
                    instanceId = modelId + '-' + str(objectIds[modelId])
                    
                    obj = Object(instanceId, modelId, modelFilename)
                    obj.setTransform(transform)
                    
                    room = None
                    if nodeIndex in roomByNodeIndex:
                        room = roomByNodeIndex[nodeIndex]
                        room.objects.append(obj)

                    obj.attributes['room'] = room
                    obj.attributes['level'] = levelId
                else:
                    raise Exception('Unsupported node type: %s' % (node['type']))
                
        return House(sceneId, rooms, objects)
