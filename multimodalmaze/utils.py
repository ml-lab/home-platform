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

import sys
import numpy as np

from panda3d.core import ClockObject, AmbientLight, VBase4, PointLight, AntialiasAttrib, TextNode

from direct.showbase.ShowBase import ShowBase, WindowProperties
from direct.task.TaskManagerGlobal import taskMgr
from direct.showbase.DirectObject import DirectObject
from direct.task.Task import Task
from direct.gui.OnscreenText import OnscreenText

def mat4ToNumpyArray(mat):
    return np.array([[mat[0][0], mat[0][1], mat[0][2], mat[0][3]],
                     [mat[1][0], mat[1][1], mat[1][2], mat[1][3]],
                     [mat[2][0], mat[2][1], mat[2][2], mat[2][3]],
                     [mat[3][0], mat[3][1], mat[3][2], mat[3][3]]])

def vec3ToNumpyArray(vec):
    return np.array([vec.x, vec.y, vec.z])

class FirstPersonCamera(DirectObject):
    '''
    First person camera controller.
    Adapted from: http://www.panda3d.org/forums/viewtopic.php?t=11657
    '''
    ##   First person camera controller, "free view"/"FPS" style.
    #   
    #    Simple camera mouse look and WASD key controller
    #    shift to go faster,
    #    r and f keys move camera up/down,
    #    q and e keys rotate camera,
    #    hit enter to start/stop controls.
    #    If a refNode is specified, heading and up/down are performed wrt the
    #    reference node (usually the root node of scene, i.e. base.render)
    #    and camera behaves more similarly to an "FPS" camera.
   
    ## Constructor
    # @param gameaApp: the game application to which this controller
    # applies, that should be ShowBase derived.
    # @param camera: the camera to which this controller applies
    # @param refNode: reference node wrt heading and up/down are performed
    def __init__(self, gameApp, camera, refNode=None):
        '''
        Constructor
        '''
       
        self.gameApp = gameApp
        self.camera = camera
        if refNode != None:
            self.refNode = refNode
        else:
            self.refNode = self.camera
        self.running = False
        self.time = 0
        self.centX = self.gameApp.win.getProperties().getXSize() / 2
        self.centY = self.gameApp.win.getProperties().getYSize() / 2
       
        # key controls
        self.forward = False
        self.backward = False
        self.fast = 1.0
        self.left = False
        self.right = False
        self.up = False
        self.down = False
        self.up = False
        self.down = False
        self.rollLeft = False
        self.rollRight = False
       
        # sensitivity settings
        self.movSens = 2
        self.movSensFast = self.movSens * 5
        self.rollSens = 50
        self.sensX = self.sensY = 0.2       
       
        self.gameApp.disableMouse()
        self.camera.setP(self.refNode, 0)
        self.camera.setR(self.refNode, 0)
        
        # hide mouse cursor, comment these 3 lines to see the cursor
        props = WindowProperties()
        props.setCursorHidden(True)
        self.gameApp.win.requestProperties(props)
        
        # reset mouse to start position:
        self.gameApp.win.movePointer(0, self.centX, self.centY)             
        self.gameApp.taskMgr.add(self.cameraTask, 'HxMouseLook::cameraTask')       
        
        #Task for changing direction/position
        self.accept("w", setattr, [self, "forward", True])
        self.accept("shift-w", setattr, [self, "forward", True])
        self.accept("w-up", setattr, [self, "forward", False])
        self.accept("s", setattr, [self, "backward", True])
        self.accept("shift-s", setattr, [self, "backward", True])
        self.accept("s-up", setattr, [self, "backward", False])
        self.accept("a", setattr, [self, "left", True])
        self.accept("shift-a", setattr, [self, "left", True])
        self.accept("a-up", setattr, [self, "left", False])
        self.accept("d", setattr, [self, "right", True])
        self.accept("shift-d", setattr, [self, "right", True])
        self.accept("d-up", setattr, [self, "right", False])
        self.accept("r", setattr, [self, "up", True])
        self.accept("shift-r", setattr, [self, "up", True])
        self.accept("r-up", setattr, [self, "up", False])
        self.accept("f", setattr, [self, "down", True])
        self.accept("shift-f", setattr, [self, "down", True])
        self.accept("f-up", setattr, [self, "down", False])
        self.accept("q", setattr, [self, "rollLeft", True])
        self.accept("q-up", setattr, [self, "rollLeft", False])
        self.accept("e", setattr, [self, "rollRight", True])
        self.accept("e-up", setattr, [self, "rollRight", False])
        self.accept("shift", setattr, [self, "fast", 10.0])
        self.accept("shift-up", setattr, [self, "fast", 1.0])
       
    ## Camera rotation task
    def cameraTask(self, task):
        dt = task.time - self.time
         
        # handle mouse look
        md = self.gameApp.win.getPointer(0)       
        x = md.getX()
        y = md.getY()
         
        if self.gameApp.win.movePointer(0, self.centX, self.centY):   
            self.camera.setH(self.refNode, self.camera.getH(self.refNode)
                             - (x - self.centX) * self.sensX)
            self.camera.setP(self.camera, self.camera.getP(self.camera)
                             - (y - self.centY) * self.sensY)       
       
        # handle keys:
        if self.forward == True:
            self.camera.setY(self.camera, self.camera.getY(self.camera)
                             + self.movSens * self.fast * dt)
        if self.backward == True:
            self.camera.setY(self.camera, self.camera.getY(self.camera)
                             - self.movSens * self.fast * dt)
        if self.left == True:
            self.camera.setX(self.camera, self.camera.getX(self.camera)
                             - self.movSens * self.fast * dt)
        if self.right == True:
            self.camera.setX(self.camera, self.camera.getX(self.camera)
                             + self.movSens * self.fast * dt)
        if self.up == True:
            self.camera.setZ(self.refNode, self.camera.getZ(self.refNode)
                             + self.movSens * self.fast * dt)
        if self.down == True:
            self.camera.setZ(self.refNode, self.camera.getZ(self.refNode)
                             - self.movSens * self.fast * dt)           
        if self.rollLeft == True:
            self.camera.setR(self.camera, self.camera.getR(self.camera)
                             - self.rollSens * dt)
        if self.rollRight == True:
            self.camera.setR(self.camera, self.camera.getR(self.camera)
                             + self.rollSens * dt)
           
        self.time = task.time       
        return Task.cont

class Viewer(ShowBase):

    def __init__(self, scene, size=(800,600), interactive=True, shadowing=False):
        ShowBase.__init__(self)

        self.__dict__.update(scene=scene, size=size, interactive=interactive,
                             shadowing=shadowing)
        
        # Change window size
        wp = WindowProperties()
        wp.setSize(size[0], size[1])
        wp.setTitle("Viewer")
        self.win.requestProperties(wp)
        
        # Reparent the scene to render.
        self.scene.scene.reparentTo(self.render)

        # Task
        self.globalClock = ClockObject.getGlobalClock()
        taskMgr.add(self.update, 'update')
        
        self._addDefaultLighting()
        
        if self.interactive:
            self.setupEvents()
            self.mouseLook = FirstPersonCamera(self, self.cam, self.scene.scene)
        
    def setupEvents(self):
        
        self.escapeEventText = OnscreenText(text="ESC: Quit",
                                style=1, fg=(1,1,1,1), pos=(-1.3, 0.95),
                                align=TextNode.ALeft, scale = .05)
  
        #Set up the key input 
        self.accept('escape', sys.exit)
        
    def _addDefaultLighting(self):
        alight = AmbientLight('alight')
        alight.setColor(VBase4(0.2, 0.2, 0.2, 1))
        alnp = self.render.attachNewNode(alight)
        self.render.setLight(alnp)
        
        #NOTE: Point light following the camera
        plight = PointLight('plight')
        plight.setColor(VBase4(1.0, 1.0, 1.0, 1))
        plnp = self.cam.attachNewNode(plight)
        self.render.setLight(plnp)
        
        if self.shadowing:
            # Use a 512x512 resolution shadow map
            plight.setShadowCaster(True, 512, 512)

            # Enable the shader generator for the receiving nodes
            self.render.setShaderAuto()
            self.render.setAntialias(AntialiasAttrib.MAuto)
        
    def update(self, task):
        
        dt = self.globalClock.getDt()
        
        # Simulate physics
        if 'physics' in self.scene.worlds:
            self.scene.worlds['physics'].step(dt)
        
        # Rendering
        if 'render' in self.scene.worlds:
            self.scene.worlds['render'].step(dt)
        
        # Simulate acoustics
        if 'acoustics' in self.scene.worlds:
            self.scene.worlds['acoustics'].step(dt)
            
        return task.cont
    
    def step(self):
        taskMgr.step()
        