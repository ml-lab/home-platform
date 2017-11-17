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
from timeit import default_timer as timer

# Disable v-sync
import os
os.environ['vblank_mode'] = '0'

# Load configuration file with CPU backend (TinyPanda software rendering)
from panda3d.core import loadPrcFile
loadPrcFile(os.path.join(os.path.dirname(os.path.realpath(__file__)), "ConfigCPU.prc"))

from benchmark import BenchmarkEnvironment

def getFpsAll(nbSteps):
    
    env = BenchmarkEnvironment(activeEngines=['physics', 'render', 'acoustics'])
    
    start = timer()
    env.simulate(nbSteps=nbSteps)
    end = timer()
    elapsed = (end - start)
    
    env.destroy()
    
    fps = nbSteps/elapsed
    return fps

def getFpsRenderOnly(nbSteps):
    
    env = BenchmarkEnvironment(activeEngines=['render'])
    
    start = timer()
    env.simulate(nbSteps=nbSteps)
    end = timer()
    elapsed = (end - start)
    
    env.destroy()
    
    fps = nbSteps/elapsed
    return fps

def getFpsPhysicsOnly(nbSteps):
    
    env = BenchmarkEnvironment(activeEngines=['physics'])
    
    start = timer()
    env.simulate(nbSteps=nbSteps)
    end = timer()
    elapsed = (end - start)
    
    env.destroy()
    
    fps = nbSteps/elapsed
    return fps

def getFpsAcousticsOnly(nbSteps):
    
    env = BenchmarkEnvironment(activeEngines=['acoustics'])
    
    start = timer()
    env.simulate(nbSteps=nbSteps)
    end = timer()
    elapsed = (end - start)
    
    env.destroy()
    
    fps = nbSteps/elapsed
    return fps

def main():
    
    nbSteps = 4000
    print 'FPS (all): ', getFpsAll(nbSteps)
    print 'FPS (render-only): ',  getFpsRenderOnly(nbSteps)
    print 'FPS (physics-only): ',  getFpsPhysicsOnly(nbSteps)
    print 'FPS (acoustics-only): ',  getFpsAcousticsOnly(nbSteps)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
    