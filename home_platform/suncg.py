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
import re
import json
import csv
import logging
import numpy as np

from panda3d.core import NodePath, Loader, LoaderOptions, Filename, TransformState,\
    LMatrix4f, Spotlight, LVector3f, PointLight, PerspectiveLens, CS_zup_right, CS_yup_right

from home_platform.constants import MODEL_CATEGORY_MAPPING
from home_platform.core import Scene
from home_platform.utils import mat4ToNumpyArray

logger = logging.getLogger(__name__)

def loadModel(modelPath):
    loader = Loader.getGlobalPtr()
    loaderOptions = LoaderOptions()
    node = loader.loadSync(Filename(modelPath), loaderOptions)
    if node is not None:
        nodePath = NodePath(node)
        nodePath.setTag('model-filename', os.path.abspath(modelPath))
    else:
        raise IOError('Could not load model file: %s' % (modelPath))
    return nodePath

def ignoreVariant(modelId):
    suffix = "_0"
    if modelId.endswith(suffix):
        modelId = modelId[:len(modelId) - len(suffix)]
    return modelId

def data_dir():
    """ Get SUNCG data path (must be symlinked to ~/.suncg)

    :return: Path to suncg dataset
    """

    if 'SUNCG_DATA_DIR' in os.environ:
        path = os.path.abspath(os.environ['SUNCG_DATA_DIR'])
    else:
        path = os.path.join(os.path.abspath(os.path.expanduser('~')), ".suncg")
        
    rooms_exist = os.path.isdir(os.path.join(path, "room"))
    houses_exist = os.path.isdir(os.path.join(path, "house"))
    if not os.path.isdir(path) or not rooms_exist or not houses_exist:
        raise Exception("Couldn't find the SUNCG dataset in '~/.suncg' or with environment variable SUNCG_DATA_DIR. "
                        "Please symlink the dataset there, so that the folders "
                        "'~/.suncg/room', '~/.suncg/house', etc. exist.")

    return path


class ModelInformation(object):
    header = 'id,front,nmaterials,minPoint,maxPoint,aligned.dims,index,variantIds'

    def __init__(self, filename):
        self.model_info = {}

        self._parseFromCSV(filename)

    def _parseFromCSV(self, filename):
        with open(filename, 'rb') as f:
            reader = csv.reader(f, delimiter=',')
            for i, row in enumerate(reader):
                if i == 0:
                    rowStr = ','.join(row)
                    assert rowStr == ModelInformation.header
                else:
                    model_id, front, nmaterials, minPoint, \
                    maxPoint, aligned_dims, _, variantIds = row
                    if model_id in self.model_info:
                        raise Exception('Model %s already exists!' % (model_id))

                    front = np.fromstring(front, dtype=np.float64, sep=',')
                    nmaterials = int(nmaterials)
                    minPoint = np.fromstring(minPoint, dtype=np.float64, sep=',')
                    maxPoint = np.fromstring(maxPoint, dtype=np.float64, sep=',')
                    aligned_dims = np.fromstring(aligned_dims, dtype=np.float64, sep=',')
                    variantIds = variantIds.split(',')
                    self.model_info[model_id] = {'front': front,
                                                 'nmaterials': nmaterials,
                                                 'minPoint': minPoint,
                                                 'maxPoint': maxPoint,
                                                 'aligned_dims': aligned_dims,
                                                 'variantIds': variantIds}

    def getModelInfo(self, modelId):
        return self.model_info[ignoreVariant(modelId)]


class ModelCategoryMapping(object):
    def __init__(self, filename):
        self.model_id = []
        self.fine_grained_class = {}
        self.coarse_grained_class = {}
        self.nyuv2_40class = {}
        self.wnsynsetid = {}
        self.wnsynsetkey = {}

        self._parseFromCSV(filename)

    def _parseFromCSV(self, filename):
        with open(filename, 'rb') as f:
            reader = csv.reader(f, delimiter=',')
            for i, row in enumerate(reader):
                if i == 0:
                    rowStr = ','.join(row)
                    assert rowStr == MODEL_CATEGORY_MAPPING["header"]
                else:
                    _, model_id, fine_grained_class, \
                    coarse_grained_class, _, nyuv2_40class, \
                    wnsynsetid, wnsynsetkey = row
                    if model_id in self.model_id:
                        raise Exception('Model %s already exists!' % (model_id))
                    self.model_id.append(model_id)

                    self.fine_grained_class[model_id] = fine_grained_class
                    self.coarse_grained_class[model_id] = coarse_grained_class
                    self.nyuv2_40class[model_id] = nyuv2_40class
                    self.wnsynsetid[model_id] = wnsynsetid
                    self.wnsynsetkey[model_id] = wnsynsetkey

    def _printFineGrainedClassListAsDict(self):
        for c in sorted(set(self.fine_grained_class.values())):
            name = c.replace("_", " ")
            print "'%s':'%s'," % (c, name)

    def _printCoarseGrainedClassListAsDict(self):
        for c in sorted(set(self.coarse_grained_class.values())):
            name = c.replace("_", " ")
            print "'%s':'%s'," % (c, name)

    def getFineGrainedCategoryForModelId(self, modelId):
        return self.fine_grained_class[ignoreVariant(modelId)]

    def getCoarseGrainedCategoryForModelId(self, modelId):
        return self.coarse_grained_class[ignoreVariant(modelId)]

    def getFineGrainedClassList(self):
        return sorted(set(self.fine_grained_class.values()))

    def getCoarseGrainedClassList(self):
        return sorted(set(self.coarse_grained_class.values()))


class ObjectVoxelData(object):
    def __init__(self, voxels, translation, scale):
        self.voxels = voxels
        self.translation = translation
        self.scale = scale

    def getFilledVolume(self):
        nbFilledVoxels = np.count_nonzero(self.voxels)
        perVoxelVolume = self.scale / np.prod(self.voxels.shape)
        return nbFilledVoxels * perVoxelVolume

    @staticmethod
    def fromFile(filename):

        with open(filename, 'rb') as f:

            # Read header line and version
            line = f.readline().decode('ascii').strip()  # u'#binvox 1'
            header, version = line.split(" ")
            if header != '#binvox':
                raise Exception('Unable to read header from file: %s' % (filename))
            version = int(version)
            assert version == 1

            # Read dimensions and transforms
            line = f.readline().decode('ascii').strip()  # u'dim 128 128 128'
            items = line.split(" ")
            assert items[0] == 'dim'
            depth, height, width = np.fromstring(" ".join(items[1:]), sep=' ', dtype=np.int)

            # XXX: what is this translation component?
            line = f.readline().decode('ascii').strip()  # u'translate -0.176343 -0.356254 0.000702'
            items = line.split(" ")
            assert items[0] == 'translate'
            translation = np.fromstring(" ".join(items[1:]), sep=' ', dtype=np.float)

            line = f.readline().decode('ascii').strip()  # u'scale 0.863783'
            items = line.split(" ")
            assert items[0] == 'scale'
            scale = float(items[1])

            # Read voxel data
            line = f.readline().decode('ascii').strip()  # u'data'
            assert line == 'data'

            size = width * height * depth
            voxels = np.zeros((size,), dtype=np.int8)

            nrVoxels = 0
            index = 0
            endIndex = 0
            while endIndex < size:
                value = np.fromstring(f.read(1), dtype=np.uint8)[0]
                count = np.fromstring(f.read(1), dtype=np.uint8)[0]
                endIndex = index + count
                assert endIndex <= size

                voxels[index:endIndex] = value
                if value != 0:
                    nrVoxels += count
                index = endIndex

            # NOTE: we should by now have reach the end of the file
            assert f.readline() == ''

            # FIXME: not sure about the particular dimension ordering here!
            voxels = voxels.reshape((width, height, depth))

            logger.debug('Number of non-empty voxels read from file: %d' % (nrVoxels))

        return ObjectVoxelData(voxels, translation, scale)

def reglob(path, exp):
    # NOTE: adapted from https://stackoverflow.com/questions/13031989/regular-expression-using-in-glob-glob-of-python
    m = re.compile(exp)
    res = [f for f in os.listdir(path) if m.search(f)]
    res = map(lambda x: "%s/%s" % ( path, x, ), res)
    return res

class SunCgModelLights(object):
    
    def __init__(self, filename):
        
        with open(filename) as f:
            self.data = json.load(f)
        
        self.supportedModelIds = self.data.keys()
    
    def getLightsForModel(self, modelId):
        lights = []
        if modelId in self.supportedModelIds:
            
            for n, lightData in enumerate(self.data[modelId]):
                               
                attenuation = LVector3f(*lightData['attenuation'])
                
                #TODO: implement light power
                #power = float(lightData['power'])
                
                positionYup = LVector3f(*lightData['position'])
                yupTozupMat = LMatrix4f.convertMat(CS_yup_right, CS_zup_right)
                position = yupTozupMat.xformVec(positionYup)
                            
                colorHtml = lightData['color']
                color = LVector3f(*[int('0x' + colorHtml[i:i+2], 16) for i in range(1, len(colorHtml), 2)]) / 255.0
                            
                direction = None
                lightType = lightData['type']
                lightName = modelId + '-light-' + str(n)
                if lightType == 'SpotLight':
                    light = Spotlight(lightName)
                    light.setAttenuation(attenuation)
                    light.setColor(color)
                     
                    cutoffAngle = float(lightData['cutoffAngle'])
                    lens = PerspectiveLens()
                    lens.setFov(cutoffAngle / np.pi * 180.0)
                    light.setLens(lens)
                     
                    # NOTE: unused attributes
                    #dropoffRate = float(lightData['dropoffRate'])
                     
                    directionYup = LVector3f(*lightData['direction'])
                    direction = yupTozupMat.xformVec(directionYup)
                    
                elif lightType == 'PointLight':
                    light = PointLight(lightName)
                    light.setAttenuation(attenuation)
                    light.setColor(color)
                    
                elif lightType == 'LineLight':
                    #XXX: we may wish to use RectangleLight from the devel branch of Panda3D 
                    light = PointLight(lightName)
                    light.setAttenuation(attenuation)
                    light.setColor(color)
                    
                    # NOTE: unused attributes
                    #dropoffRate = float(lightData['dropoffRate'])
                    #cutoffAngle = float(lightData['cutoffAngle'])
                    
                    #position2Yup = LVector3f(*lightData['position2'])
                    #position2 = yupTozupMat.xformVec(position2Yup)
                
                    #directionYup = LVector3f(*lightData['direction'])
                    #direction = yupTozupMat.xformVec(directionYup)
                    
                else:
                    raise Exception('Unsupported light type: %s' % (lightType))
                
                lightNp = NodePath(light)
                
                # Set position and direction of light
                lightNp.setPos(position)
                if direction is not None:
                    targetPos = position + direction
                    lightNp.look_at(targetPos, LVector3f.up())
                
                lights.append(lightNp)
                
        return lights
    
    def isModelSupported(self, modelId):
        isSupported = False
        if modelId in self.supportedModelIds:
            isSupported = True
        return isSupported

class SunCgSceneLoader(object):
    
    @staticmethod
    def getHouseJsonPath(base_path, house_id):
        return os.path.join(
            base_path,
            "house",
            house_id,
            "house.json")
    
    @staticmethod
    def loadHouseFromJson(houseId, datasetRoot):
        
        filename = SunCgSceneLoader.getHouseJsonPath(datasetRoot, houseId)
        with open(filename) as f:
            data = json.load(f)
        assert houseId == data['id']
        houseId = str(data['id'])
        
        # Create new node for house instance
        houseNp = NodePath('house-' + str(houseId))
        
        objectIds = {}
        for levelId, level in enumerate(data['levels']):
            logger.debug('Loading Level %s to scene' % (str(levelId)))
            
            # Create new node for level instance
            levelNp = houseNp.attachNewNode('level-' + str(levelId))
            
            roomNpByNodeIndex = {}
            for nodeIndex, node in enumerate(level['nodes']):
                if not node['valid'] == 1: continue
                    
                modelId = str(node['modelId'])
                    
                if node['type'] == 'Room':
                    logger.debug('Loading Room %s to scene' % (modelId))
                    
                    # Create new nodes for room instance
                    roomNp = levelNp.attachNewNode('room-' + str(modelId))
                    roomLayoutsNp = roomNp.attachNewNode('layouts')
                    roomObjectsNp = roomNp.attachNewNode('objects')
                    
                    # Load models defined for this room 
                    for roomObjFilename in reglob(os.path.join(datasetRoot, 'room', houseId),
                                                  modelId + '[a-z].obj'):
                        
                        # Convert extension from OBJ + MTL to EGG format
                        f, _ = os.path.splitext(roomObjFilename)
                        modelFilename = f + ".egg"
                        if not os.path.exists(modelFilename):
                            raise Exception('The SUNCG dataset object models need to be convert to Panda3D EGG format!')
                        
                        # Create new node for object instance
                        objectNp = NodePath('object-' + str(modelId) + '-0')
                        objectNp.reparentTo(roomLayoutsNp)
                        
                        model = loadModel(modelFilename)
                        model.setName('model-' + os.path.basename(f))
                        model.reparentTo(objectNp)
                        model.hide()
                    
                    if 'nodeIndices' in node:
                        for childNodeIndex in node['nodeIndices']:
                            roomNpByNodeIndex[childNodeIndex] = roomObjectsNp
                    
                elif node['type'] == 'Object':
                    
                    logger.debug('Loading Object %s to scene' % (modelId))
                    
                    # Instance identification
                    if modelId in objectIds:
                        objectIds[modelId] = objectIds[modelId] + 1
                    else:
                        objectIds[modelId] = 0
                    
                    # Create new node for object instance
                    objectNp = NodePath('object-' + str(modelId) + '-' + str(objectIds[modelId]))
                    
                    #TODO: loading the BAM format would be much more efficient
                    # Convert extension from OBJ + MTL to EGG format
                    objFilename = os.path.join(datasetRoot, 'object', node['modelId'], node['modelId'] + '.obj')
                    assert os.path.exists(objFilename)
                    f, _ = os.path.splitext(objFilename)
                    modelFilename = f + ".egg"
                    if not os.path.exists(modelFilename):
                        raise Exception('The SUNCG dataset object models need to be convert to Panda3D EGG format!')
                    
                    model = loadModel(modelFilename)
                    model.setName('model-' + os.path.basename(f))
                    model.reparentTo(objectNp)
                    model.hide()
                    
                    # 4x4 column-major transformation matrix from object coordinates to scene coordinates
                    transform = np.array(node['transform']).reshape((4,4))
                    
                    # Transform from Y-UP to Z-UP coordinate systems
                    #TODO: use Mat4.convertMat(CS_zup_right, CS_yup_right)
                    yupTransform = np.array([[1, 0, 0, 0],
                                            [0, 0, -1, 0],
                                            [0, 1, 0, 0],
                                            [0, 0, 0, 1]])
                    
                    zupTransform = np.array([[1, 0, 0, 0],
                                            [0, 0, 1, 0],
                                            [0, -1, 0, 0],
                                            [0, 0, 0, 1]])
                    
                    transform = np.dot(np.dot(yupTransform, transform), zupTransform)
                    transform = TransformState.makeMat(LMatrix4f(*transform.ravel()))
                    
                    # Calculate the center of this object
                    minBounds, maxBounds = model.getTightBounds()
                    centerPos = minBounds + (maxBounds - minBounds) / 2.0
                    
                    # Add offset transform to make position relative to the center
                    objectNp.setTransform(transform.compose(TransformState.makePos(centerPos)))
                    model.setTransform(TransformState.makePos(-centerPos))
                    
                    # Get the parent nodepath for the object (room or level)
                    if nodeIndex in roomNpByNodeIndex:
                        objectNp.reparentTo(roomNpByNodeIndex[nodeIndex])
                    else:
                        objectNp.reparentTo(levelNp)
                        
                    # Validation
                    assert np.allclose(mat4ToNumpyArray(model.getNetTransform().getMat()),
                                       mat4ToNumpyArray(transform.getMat()), atol=1e-6)
    
                    objectNp.setTag('model-id', str(modelId))
                    objectNp.setTag('level-id', str(levelId))
                    objectNp.setTag('house-id', str(houseId))
                
                elif node['type'] == 'Ground':
                    
                    logger.debug('Loading Ground %s to scene' % (modelId))
                    
                    # Create new nodes for ground instance
                    groundNp = levelNp.attachNewNode('ground-' + str(modelId))
                    groundLayoutsNp = groundNp.attachNewNode('layouts')
                    
                    # Load model defined for this ground
                    for groundObjFilename in reglob(os.path.join(datasetRoot, 'room', houseId),
                                                  modelId + '[a-z].obj'):
                    
                        # Convert extension from OBJ + MTL to EGG format
                        f, _ = os.path.splitext(groundObjFilename)
                        modelFilename = f + ".egg"
                        if not os.path.exists(modelFilename):
                            raise Exception('The SUNCG dataset object models need to be convert to Panda3D EGG format!')
                
                        objectNp = NodePath('object-' + str(modelId) + '-0')
                        objectNp.reparentTo(groundLayoutsNp)
                
                        model = loadModel(modelFilename)
                        model.setName('model-' + os.path.basename(f))
                        model.reparentTo(objectNp)
                        model.hide()
                
                else:
                    raise Exception('Unsupported node type: %s' % (node['type']))
                
                
        scene = Scene()
        houseNp.reparentTo(scene.scene)
                
        # Recenter objects in rooms
        for room in scene.scene.findAllMatches('**/room*'):
         
            # Calculate the center of this room
            minBounds, maxBounds = room.getTightBounds()
            centerPos = minBounds + (maxBounds - minBounds) / 2.0
              
            # Add offset transform to room node
            room.setTransform(TransformState.makePos(centerPos))
              
            # Add recentering transform to all children nodes
            for childNp in room.getChildren():
                childNp.setTransform(TransformState.makePos(-centerPos))
         
        # Recenter objects in grounds
        for ground in scene.scene.findAllMatches('**/ground*'):
         
            # Calculate the center of this ground
            minBounds, maxBounds = ground.getTightBounds()
            centerPos = minBounds + (maxBounds - minBounds) / 2.0
              
            # Add offset transform to ground node
            ground.setTransform(TransformState.makePos(centerPos))
              
            # Add recentering transform to all children nodes
            for childNp in ground.getChildren():
                childNp.setTransform(TransformState.makePos(-centerPos))
                
        return scene
