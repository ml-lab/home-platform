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

from panda3d.core import ClockObject, AmbientLight, VBase4, PointLight, AntialiasAttrib, TextNode, LVector3f

from direct.showbase.ShowBase import ShowBase, WindowProperties
from direct.gui.OnscreenText import OnscreenText

def mat4ToNumpyArray(mat):
    return np.array([[mat[0][0], mat[0][1], mat[0][2], mat[0][3]],
                     [mat[1][0], mat[1][1], mat[1][2], mat[1][3]],
                     [mat[2][0], mat[2][1], mat[2][2], mat[2][3]],
                     [mat[3][0], mat[3][1], mat[3][2], mat[3][3]]])

def vec3ToNumpyArray(vec):
    return np.array([vec.x, vec.y, vec.z])

class Controller(ShowBase):

    def __init__(self, scene, size=(800,600), zNear=0.1, zFar=1000.0, fov=40.0, shadowing=False, showPosition=False, cameraTransform=None):
        ShowBase.__init__(self)

        self.__dict__.update(scene=scene, size=size, fov=fov,
                             zNear=zNear, zFar=zFar, shadowing=shadowing, showPosition=showPosition, cameraTransform=cameraTransform)
        
        # Find agent and reparent camera to it
        self.agent = self.scene.scene.find('**/agents/agent*/+BulletRigidBodyNode')
        self.camera.reparentTo(self.agent)
        if self.cameraTransform is not None:
            self.camera.setTransform(cameraTransform)
        
        lens = self.cam.node().getLens()
        lens.setFov(self.fov)
        lens.setNear(self.zNear)
        lens.setFar(self.zFar)
        
        # Change window size
        wp = WindowProperties()
        wp.setSize(size[0], size[1])
        wp.setTitle("Controller")
        wp.setCursorHidden(True)
        self.win.requestProperties(wp)
        
        self.disableMouse()
        
        self.time = 0
        self.centX = self.win.getProperties().getXSize() / 2
        self.centY = self.win.getProperties().getYSize() / 2
        self.win.movePointer(0, self.centX, self.centY)
        
        # key controls
        self.forward = False
        self.backward = False
        self.fast = 1.0
        self.left = False
        self.right = False
       
        # sensitivity settings
        self.movSens = 2
        self.movSensFast = self.movSens * 5
        self.sensX = self.sensY = 0.2       
        
        # Reparent the scene to render.
        self.scene.scene.reparentTo(self.render)

        # Task
        self.globalClock = ClockObject.getGlobalClock()
        self.taskMgr.add(self.update, 'controller-update')
        
        self._addDefaultLighting()
        self._setupEvents()
        
    def _addDefaultLighting(self):
        alight = AmbientLight('alight')
        alight.setColor(VBase4(0.2, 0.2, 0.2, 1))
        alnp = self.render.attachNewNode(alight)
        self.render.setLight(alnp)
        
        #NOTE: Point light following the camera
        plight = PointLight('plight')
        plight.setColor(VBase4(0.4, 0.4, 0.4, 1))
        plnp = self.cam.attachNewNode(plight)
        self.render.setLight(plnp)
        
        if self.shadowing:
            # Use a 512x512 resolution shadow map
            plight.setShadowCaster(True, 512, 512)

            # Enable the shader generator for the receiving nodes
            self.render.setShaderAuto()
            self.render.setAntialias(AntialiasAttrib.MAuto)
        
    def _setupEvents(self):
        
        self.escapeEventText = OnscreenText(text="ESC: Quit",
                                style=1, fg=(1,1,1,1), pos=(-1.3, 0.95),
                                align=TextNode.ALeft, scale = .05)
  
        if self.showPosition:
            self.positionText = OnscreenText(text="Position: ",
                                    style=1, fg=(1,1,1,1), pos=(-1.3, 0.85),
                                    align=TextNode.ALeft, scale = .05)
  
            self.orientationText = OnscreenText(text="Orientation: ",
                                    style=1, fg=(1,1,1,1), pos=(-1.3, 0.80),
                                    align=TextNode.ALeft, scale = .05)
  
        #Set up the key input 
        self.accept('escape', sys.exit)
        self.accept("w", setattr, [self, "forward", True])
        self.accept("w-up", setattr, [self, "forward", False])
        self.accept("s", setattr, [self, "backward", True])
        self.accept("s-up", setattr, [self, "backward", False])
        self.accept("a", setattr, [self, "left", True])
        self.accept("a-up", setattr, [self, "left", False])
        self.accept("d", setattr, [self, "right", True])
        self.accept("d-up", setattr, [self, "right", False])
        self.accept("shift", setattr, [self, "fast", 10.0])
        self.accept("shift-up", setattr, [self, "fast", 1.0])
        
    def update(self, task):
        
        #dt = self.globalClock.getDt()
        dt = task.time - self.time
         
        # handle mouse look
        md = self.win.getPointer(0)
        x = md.getX()
        y = md.getY()
        
        if self.win.movePointer(0, self.centX, self.centY):
            self.agent.setH(self.agent, self.agent.getH(self.agent) - (x - self.centX) * self.sensX)
            self.agent.setP(self.agent, self.agent.getP(self.agent) - (y - self.centY) * self.sensY)
            self.agent.setR(0.0)

        linearVelocityX = 0.0
        linearVelocityY = 0.0

        if self.forward == True:
            linearVelocityY += self.movSens * self.fast
        if self.backward == True:
            linearVelocityY -= self.movSens * self.fast
        if self.left == True:
            linearVelocityX -= self.movSens * self.fast
        if self.right == True:
            linearVelocityX += self.movSens * self.fast
            
        linearVelocity = LVector3f(linearVelocityX, linearVelocityY, 0.0)
           
        # Apply the local transform to the velocity
        # XXX: use BulletCharacterControllerNode class, which already handles local transform?
        rotMat = self.agent.node().getTransform().getMat().getUpper3()
        linearVelocity = rotMat.xformVec(linearVelocity)
        linearVelocity.z = 0.0
        self.agent.node().setLinearVelocity(linearVelocity)
        
        if self.showPosition:
            position = self.agent.getNetTransform().getPos()
            hpr = self.agent.getNetTransform().getHpr()
            self.positionText.setText('Position: (x = %4.2f, y = %4.2f, z = %4.2f)' % (position.x, position.y, position.z))
            self.orientationText.setText('Orientation: (h = %4.2f, p = %4.2f, r = %4.2f)' % (hpr.x, hpr.y, hpr.z))
        
        self.time = task.time
        
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
        self.taskMgr.step()
                
    def destroy(self):
        self.taskMgr.remove('controller-update')
        ShowBase.destroy(self)
        
class Viewer(ShowBase):

    def __init__(self, scene, size=(800,600), zNear=0.1, zFar=1000.0, fov=40.0, shadowing=False, interactive=True,
                 showPosition=False):
        ShowBase.__init__(self)

        self.__dict__.update(scene=scene, size=size, fov=fov, shadowing=shadowing,
                             zNear=zNear, zFar=zFar, interactive=interactive, showPosition=showPosition)
        
        lens = self.cam.node().getLens()
        lens.setFov(self.fov)
        lens.setNear(self.zNear)
        lens.setFar(self.zFar)
        
        # Change window size
        wp = WindowProperties()
        wp.setSize(size[0], size[1])
        wp.setTitle("Viewer")
        wp.setCursorHidden(True)
        self.win.requestProperties(wp)
        
        self.disableMouse()
        
        self.time = 0
        self.centX = self.win.getProperties().getXSize() / 2
        self.centY = self.win.getProperties().getYSize() / 2
       
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
       
        # sensitivity settings
        self.movSens = 2
        self.movSensFast = self.movSens * 5
        self.sensX = self.sensY = 0.2       
       
        self.cam.setP(self.cam, 0)
        self.cam.setR(0)
        
        # reset mouse to start position:
        self.win.movePointer(0, self.centX, self.centY)     
        
        # Reparent the scene to render.
        self.scene.scene.reparentTo(self.render)

        # Task
        self.globalClock = ClockObject.getGlobalClock()
        self.taskMgr.add(self.update, 'viewer-update')
        
        self._addDefaultLighting()
        self._setupEvents()
        
    def _setupEvents(self):
        
        self.escapeEventText = OnscreenText(text="ESC: Quit",
                                style=1, fg=(1,1,1,1), pos=(-1.3, 0.95),
                                align=TextNode.ALeft, scale = .05)
  
        if self.showPosition:
            self.positionText = OnscreenText(text="Position: ",
                                    style=1, fg=(1,1,1,1), pos=(-1.3, 0.85),
                                    align=TextNode.ALeft, scale = .05)
  
            self.orientationText = OnscreenText(text="Orientation: ",
                                    style=1, fg=(1,1,1,1), pos=(-1.3, 0.80),
                                    align=TextNode.ALeft, scale = .05)
  
        #Set up the key input 
        self.accept('escape', sys.exit)
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
        self.accept("shift", setattr, [self, "fast", 10.0])
        self.accept("shift-up", setattr, [self, "fast", 1.0])
        
    def _addDefaultLighting(self):
        alight = AmbientLight('alight')
        alight.setColor(VBase4(0.2, 0.2, 0.2, 1))
        alnp = self.render.attachNewNode(alight)
        self.render.setLight(alnp)
        
        #NOTE: Point light following the camera
        plight = PointLight('plight')
        plight.setColor(VBase4(0.9, 0.9, 0.9, 1))
        plnp = self.cam.attachNewNode(plight)
        self.render.setLight(plnp)
        
        if self.shadowing:
            # Use a 512x512 resolution shadow map
            plight.setShadowCaster(True, 512, 512)

            # Enable the shader generator for the receiving nodes
            self.render.setShaderAuto()
            self.render.setAntialias(AntialiasAttrib.MAuto)
        
    def update(self, task):
        
        #dt = self.globalClock.getDt()
        dt = task.time - self.time
         
        if self.interactive:
         
            # handle mouse look
            md = self.win.getPointer(0)       
            x = md.getX()
            y = md.getY()
             
            if self.win.movePointer(0, self.centX, self.centY):   
                self.cam.setH(self.cam, self.cam.getH(self.cam)
                                 - (x - self.centX) * self.sensX)
                self.cam.setP(self.cam, self.cam.getP(self.cam)
                                 - (y - self.centY) * self.sensY)       
                self.cam.setR(0)
                
            # handle keys:
            if self.forward == True:
                self.cam.setY(self.cam, self.cam.getY(self.cam)
                                 + self.movSens * self.fast * dt)
            if self.backward == True:
                self.cam.setY(self.cam, self.cam.getY(self.cam)
                                 - self.movSens * self.fast * dt)
            if self.left == True:
                self.cam.setX(self.cam, self.cam.getX(self.cam)
                                 - self.movSens * self.fast * dt)
            if self.right == True:
                self.cam.setX(self.cam, self.cam.getX(self.cam)
                                 + self.movSens * self.fast * dt)
            if self.up == True:
                self.cam.setZ(self.cam, self.cam.getZ(self.cam)
                                 + self.movSens * self.fast * dt)
            if self.down == True:
                self.cam.setZ(self.cam, self.cam.getZ(self.cam)
                                 - self.movSens * self.fast * dt)           
        
        if self.showPosition:
            position = self.cam.getNetTransform().getPos()
            hpr = self.cam.getNetTransform().getHpr()
            self.positionText.setText('Position: (x = %4.2f, y = %4.2f, z = %4.2f)' % (position.x, position.y, position.z))
            self.orientationText.setText('Orientation: (h = %4.2f, p = %4.2f, r = %4.2f)' % (hpr.x, hpr.y, hpr.z))
           
        self.time = task.time       
        
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
        self.taskMgr.step()
        
    def destroy(self):
        self.taskMgr.remove('viewer-update')
        ShowBase.destroy(self)
        