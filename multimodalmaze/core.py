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

class Agent(object):
    
    def __init__(self, instanceId, modelFilename=None, transform=None):
        self.instanceId = instanceId
        self.transform = transform
        self.modelFilename = modelFilename
        self.attributes = {}

    def clearAttributes(self):
        self.attributes.clear()

class Object(object):
    
    def __init__(self, instanceId, modelId, modelFilename, parentRoom=None, transform=None):
        self.instanceId = instanceId
        self.modelId = modelId
        self.parentRoom = parentRoom
        self.transform = transform
        self.modelFilename = modelFilename
        self.attributes = {}

    def clearAttributes(self):
        self.attributes.clear()

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

    def clearAttributes(self):
        self.attributes.clear()
    
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
            
            roomByNodeIndex = {}
            for nodeIndex, node in enumerate(level['nodes']):
                if not node['valid'] == 1: continue
                    
                modelId = node['modelId']
                    
                if node['type'] == 'Room':
                    logger.debug('Loading Room %s to scene' % (modelId))
                    
                    # Get room model filenames
                    modelFilenames = []
                    for roomObjFilename in glob.glob(os.path.join(datasetRoot, 'room', sceneId, modelId + '*.obj')):
                        
                        # Convert from OBJ + MTL to EGG format
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
                    
                    # Convert from OBJ + MTL to EGG format
                    objFilename = os.path.join(datasetRoot, 'object', node['modelId'], node['modelId'] + '.obj')
                    assert os.path.exists(objFilename)
                    f, _ = os.path.splitext(objFilename)
                    modelFilename = f + ".egg"
                    if not os.path.exists(modelFilename):
                        raise Exception('The SUNCG dataset object models need to be convert to Panda3D EGG format!')
                     
                    # 4x4 column-major transformation matrix from object coordinates to scene coordinates
                    transform = np.array(node['transform']).reshape((4,4))
                    
                    # Instance identification
                    if modelId in objectIds:
                        objectIds[modelId] = objectIds[modelId] + 1
                    else:
                        objectIds[modelId] = 0
                    instanceId = modelId + '-' + str(objectIds[modelId])
                    
                    if nodeIndex in roomByNodeIndex:
                        room = roomByNodeIndex[nodeIndex]
                        obj = Object(instanceId, modelId, modelFilename, room, transform)
                        room.objects.append(obj)
                    else:
                        obj = Object(instanceId, modelId, modelFilename, None, transform)
                        objects.append(obj)
                else:
                    raise Exception('Unsupported node type: %s' % (node['type']))
                
        return House(sceneId, rooms, objects)
