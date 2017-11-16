# Copyright (c) 2017, IGLU consortium
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without 
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright 
#    notice, this list of conditions and the following disclaimer.
#   
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT 
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR 
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY 
# OF SUCH DAMAGE.

import os
import sys
import logging
import numpy as np
import Tkinter as tk
import ImageTk

from multimodalmaze.core import House
from multimodalmaze.env import BasicEnvironment

TEST_SUNCG_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "tests", "data", "suncg")

logger = logging.getLogger(__name__)

def main():
    
    root = tk.Tk()
    
    env = BasicEnvironment()
    agent = env.agent
    
    houseFilename = os.path.join(TEST_SUNCG_DATA_DIR, "house", "0004d52d1aeeb8ae6de39d6bd993e992", "house.json")
    house = House.loadFromJson(houseFilename, TEST_SUNCG_DATA_DIR)
    env.loadHouse(house)
    
    agent.setPosition((42, -39, 1))
    agent.setOrientation((0.0, 0.0, -np.pi/3))
    
    image = np.zeros(env.renderWorld.size)
    img = ImageTk.Image.fromarray(image)
    imgTk = ImageTk.PhotoImage(img)
    
    panel = tk.Label(root, image=imgTk)
    panel.pack(side="bottom", fill="y", expand="yes")
    
    linearVelocity = np.zeros(3)
    angularVelocity = np.zeros(3)

    def move(direction):
        if direction == "Up":
            linearVelocity[1] = 1.0
        elif direction == "Down":
            linearVelocity[1] = -1.0
        elif direction == "Left":
            linearVelocity[0] = -1.0
        elif direction == "Right":
            linearVelocity[0] += 1.0
    
    def look(direction):
        if direction == "Up":
            angularVelocity[0] += 2
        elif direction == "Down":
            angularVelocity[0] -= 2
        elif direction == "Left":
            angularVelocity[2] += 2
        elif direction == "Right":
            angularVelocity[2] -= 2
    
    def get_dir_for_wasd(wasd):
        if wasd == "w":
            return "Up"
        elif wasd == "s":
            return "Down"
        elif wasd == "a":
            return "Left"
        elif wasd == "d":
            return "Right"
    
    def key_pressed(event):
        if event.keysym in ["Left", "Right", "Up", "Down"]:
            look(event.keysym)
        elif event.keysym in ["w", "a", "s", "d"]:
            move(get_dir_for_wasd(event.keysym))
    
    root.bind("<Key>", key_pressed)
    
    try:
        while True:
            
            agent.setLinearVelocity(linearVelocity)
            agent.setAngularVelocity(angularVelocity)
            
            # Simulate
            env.step()
            
            # Grab some observations
            position = agent.getPosition()
            orientation = agent.getOrientation()
            image = env.renderWorld.getRgbImage()
            collision = agent.isCollision()
            
            print 'Position: %s (x,y,z)' % (str(position))
            print 'Orientation: %s (h,p,r)' % (str(orientation))
            print 'Collision detected: %s' % (str(collision))
            
            # Refresh window
            img = ImageTk.Image.fromarray(image)
            imgTk = ImageTk.PhotoImage(img)
            panel.configure(image=imgTk)
            panel.image = imgTk
            root.update()
            
    except KeyboardInterrupt:
        pass
            
    return 0

if __name__ == "__main__":
    sys.exit(main())
