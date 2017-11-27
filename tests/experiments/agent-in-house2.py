from direct.actor.Actor import Actor
from direct.showbase.ShowBase import ShowBase, Vec3, Mat4
from direct.task.TaskManagerGlobal import taskMgr
import matplotlib.pyplot as plt

from home_platform.core import House, Agent
from home_platform.physics import Panda3dBulletPhysicWorld
from home_platform.rendering import Panda3dRenderWorld
import os
import numpy as np
import time

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data")
TEST_SUNCG_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "suncg")

render = Panda3dRenderWorld(shadowing=False, showCeiling=True, mode='offscreen')

house = House.loadFromJson(
    os.path.join(TEST_SUNCG_DATA_DIR, "house", "0004d52d1aeeb8ae6de39d6bd993e992", "house.json"),
    TEST_SUNCG_DATA_DIR)

modelFilename = os.path.join(TEST_DATA_DIR, "models", "cube.egg")
agent = Agent('agent-0', modelFilename)
nodePath = render.addAgentToScene(agent)
nodePath.setColor(1, 0, 0, 1)
nodePath.setScale(1, 1, 1)

pose = np.zeros(3)
pose[0] = 45
pose[1] = -50
pose[2] = 1.5
nodePath.setPos(pose[0], pose[1], pose[2])

rotation = np.zeros(3)
nodePath.setHpr(rotation[0], rotation[1], rotation[2])

render.addHouseToScene(house)
render.addDefaultLighting()

# mat = np.array([0.999992, 0.00394238, 0, 0,
#                 -0.00295702, 0.750104, -0.661314, 0,
#                 -0.00260737, 0.661308, 0.75011, 0,
#                 43.621, -55.7499, 12, 1])
# render.setCamera(mat)

render.camera.reparentTo(nodePath)  # glue cam to the agent

# render.camera.setY(-10)
render.step()
image = render.getRgbImage()
depth = render.getDepthImage()

# fig = plt.figure(figsize=(16, 8))
# plt.axis("off")
# ax = plt.subplot(121)
# ax.imshow(image)
# ax = plt.subplot(122)
# ax.imshow(depth.squeeze(), cmap='binary')
# plt.show(block=False)
# time.sleep(1.0)
# plt.close(fig)



import Tkinter as tk
import ImageTk

root = tk.Tk()

img = ImageTk.Image.fromarray(image)
imgTk = ImageTk.PhotoImage(img)

panel = tk.Label(root, image=imgTk)
panel.pack(side="bottom", fill="y", expand="yes")


def move(dir):
    if dir == "Up":
        pose[1] += .1
    elif dir == "Down":
        pose[1] -= .1
    elif dir == "Left":
        pose[0] -= .1
    elif dir == "Right":
        pose[0] += .1


def look(dir):
    if dir == "Up":
        rotation[1] += 2
    elif dir == "Down":
        rotation[1] -= 2
    elif dir == "Left":
        rotation[0] += 2
    elif dir == "Right":
        rotation[0] -= 2


def refresh_img():
    # render.setCamera(mat)
    render.step()
    image = render.getRgbImage()

    img = ImageTk.Image.fromarray(image)
    imgTk = ImageTk.PhotoImage(img)

    panel.configure(image=imgTk)
    panel.image = imgTk


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

    print ("pose", pose, "hpr", rotation)

    nodePath.setHpr(rotation[0], rotation[1], rotation[2])
    nodePath.setPos(pose[0], pose[1], pose[2])

    refresh_img()


root.bind("<Key>", key_pressed)

root.mainloop()
