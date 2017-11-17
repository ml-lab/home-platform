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
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt

from multimodalmaze.core import House, Agent, Object
from multimodalmaze.rendering import Panda3dRenderWorld
from multimodalmaze.acoustics import EvertAcousticWorld
from multimodalmaze.semantic import SuncgSemanticWorld

CDIR = os.path.dirname(os.path.realpath(__file__))

try:  
    SUNCG_DATA_DIR = os.environ["SUNCG_DATA_DIR"]
except KeyError: 
    print "Please set the environment variable SUNCG_DATA_DIR"
    sys.exit(1)

class BasicEnvironment(object):
    
    def __init__(self, size=(256, 256), enableAcoustics=True):

        self.enableAcoustics = enableAcoustics

        # Create default agent
        self.agent = Agent('agent')

        self.renderWorld = Panda3dRenderWorld(size, shadowing=False, showCeiling=False)
        self.renderWorld.addDefaultLighting()
        self.renderWorld.addAgentToScene(self.agent)

        self.acousticWorld = EvertAcousticWorld(samplingRate=16000, maximumOrder=3, materialAbsorption=False, frequencyDependent=False, showCeiling=False)
        self.acousticWorld.minRayRadius = 0.02
        self.acousticWorld.rayColors = [
                    (1.0, 0.0, 0.0, 1.0), # red
                    (1.0, 0.0, 0.0, 1.0), # red
                ]
        
        self.acousticWorld.addAgentToScene(self.agent)

        self.worlds = {
            "render": self.renderWorld,
            'acoustics': self.acousticWorld
        }

    def destroy(self):
        if self.renderWorld is not None:
            self.renderWorld.destroy()

    def loadHouse(self, house, levelIds=None, roomIds=None):

        if levelIds is not None:
            for room in house.rooms:
                if str(room.levelId) in levelIds:
                    self.worlds["render"].addRoomToScene(room)
                    if self.enableAcoustics:
                        self.worlds["acoustics"].addRoomToScene(room)
        else:
            
            if roomIds is not None:
                for room in house.rooms:
                    if str(room.modelId) in roomIds:
                        self.worlds["render"].addRoomToScene(room)
                        if self.enableAcoustics:
                            self.worlds["acoustics"].addRoomToScene(room)
            else:
                self.worlds["render"].addHouseToScene(house)
                if self.enableAcoustics:
                    self.worlds["acoustics"].addHouseToScene(house)

    def step(self):
        self.worlds["render"].step()
        if self.enableAcoustics:
            self.worlds["acoustics"].step()

def getScreenshot(env, world, outfilename):

    env.step()
    image = env.worlds[world].getRgbImage()
    
    fig = plt.figure(figsize=(16,16), facecolor='white', frameon=False, tight_layout=True)
    plt.axis("off")
    ax = plt.subplot(111)
    ax.imshow(image)
    
    fig.savefig(os.path.join(CDIR, outfilename), dpi=100)
    #plt.show()
    plt.close(fig)

def getDepthScreenshot(env, world, outfilename):

    env.step()
    image = env.worlds[world].getDepthImage()
    
    fig = plt.figure(figsize=(16,16), facecolor='white', frameon=False, tight_layout=True)
    plt.axis("off")
    ax = plt.subplot(111)
    ax.imshow(image, cmap='gray')
    
    fig.savefig(os.path.join(CDIR, outfilename), dpi=100)
    #plt.show()
    plt.close(fig)

def generateRendering():
    
    filename = House.getJsonPath(SUNCG_DATA_DIR, '0d7dde2bfdaa9474817c31167baf9187')
    house = House.loadFromJson(filename, SUNCG_DATA_DIR)
    
    env = BasicEnvironment(size=(256,256), enableAcoustics=False)
    env.loadHouse(house)
    env.renderWorld.addDefaultLighting()
    
    # Render top-down overall view
    env.agent.setPosition((41, -60.0, 15))
    env.agent.setOrientation((np.pi/4, 0.0, 0.0))
    getScreenshot(env, 'render', 'render-top-view-overall.png')
    
    # Render top-down view
    env.agent.setPosition((41, -36.0, 15))
    env.agent.setOrientation((np.pi/2, 0.0, 0.0))
    getScreenshot(env, 'render', 'render-top-view.png')
    
    # Render agent-view (human)
    env.agent.setPosition((42.43, -32.38, 3.75))
    env.agent.setOrientation((0.0, 0.0, -4*np.pi/7))
    getScreenshot(env, 'render', 'render-agent-view.png')
    getDepthScreenshot(env, 'render', 'render-agent-view-depth.png')
    
    env.destroy()

def generateAcoustics():

    filename = House.getJsonPath(SUNCG_DATA_DIR, '0d7dde2bfdaa9474817c31167baf9187')
    house = House.loadFromJson(filename, SUNCG_DATA_DIR)

    env = BasicEnvironment(size=(256,256))
    env.loadHouse(house, roomIds=['fr_1rm_2', 'fr_1rm_4', 'fr_1rm_5', 
                                  'fr_1rm_9', 'fr_1rm_10', 'fr_1rm_11'])
    env.renderWorld.addDefaultLighting()
    
    # Define agent (human)
    env.agent.setPosition((42.43, -32.38, 4.75))
    env.agent.setOrientation((0.0, 0.0, -4*np.pi/7))
    
    # Define a sound source (cat)
    instanceId = 'cat'
    modelId = '689'
    source = Object(instanceId, modelId)
    env.acousticWorld.addStaticSourceToScene(source)
    source.setPosition((36.73, -37.20, 3.25))
    
    # Acoustics top-view
    mat = np.array([[1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [41, -36.0, 15, 1]])
    env.acousticWorld.setCamera(mat)
    getScreenshot(env, 'acoustics', 'acoustics-top-view.png')
    
    env.destroy()

def generateSemantics():
    
    world = SuncgSemanticWorld(SUNCG_DATA_DIR)
    
    for modelId in ['s__790', 's__640', '625', 's__596', 's__792', '755']:
    
        modelFilename = os.path.join(SUNCG_DATA_DIR, "object", str(modelId), str(modelId) + ".egg")
        assert os.path.exists(modelFilename)
        instanceId = str(modelId) + '-0'
        obj = Object(instanceId, modelId, modelFilename)
        desc = world.describeObject(obj)
        
        print 'Model id (%s): %s' % (modelId, desc)

def main():
    
    generateRendering()
    generateAcoustics()
    generateSemantics()
    
    return 0

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARN)
    sys.exit(main())
    