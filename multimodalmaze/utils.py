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

import numpy as np

from panda3d.core import ClockObject, AmbientLight, VBase4, PointLight,\
    AntialiasAttrib

from direct.showbase.ShowBase import ShowBase, WindowProperties
from direct.task.TaskManagerGlobal import taskMgr

def mat4ToNumpyArray(mat):
    return np.array([[mat[0][0], mat[0][1], mat[0][2], mat[0][3]],
                     [mat[1][0], mat[1][1], mat[1][2], mat[1][3]],
                     [mat[2][0], mat[2][1], mat[2][2], mat[2][3]],
                     [mat[3][0], mat[3][1], mat[3][2], mat[3][3]]])

def vec3ToNumpyArray(vec):
    return np.array([vec.x, vec.y, vec.z])

class Viewer(ShowBase):

    def __init__(self, scene, size=(800,600), shadowing=False):
        ShowBase.__init__(self)

        self.scene = scene
        self.size = size
        self.shadowing = shadowing
        
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
        