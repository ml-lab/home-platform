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

render = Panda3dRenderWorld(shadowing=False, showCeiling=False, mode='offscreen')

house = House.loadFromJson(
    os.path.join(TEST_SUNCG_DATA_DIR, "house", "0004d52d1aeeb8ae6de39d6bd993e992", "house.json"),
    TEST_SUNCG_DATA_DIR)

render.addHouseToScene(house)
render.addDefaultLighting()

mat = np.array([0.999992, 0.00394238, 0, 0,
                -0.00295702, 0.750104, -0.661314, 0,
                -0.00260737, 0.661308, 0.75011, 0,
                43.621, -55.7499, 12, 1])
render.setCamera(mat)
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
        mat[5] += .01
    elif dir == "Down":
        mat[5] -= .01
    elif dir == "Left":
        mat[4] -= .01
    elif dir == "Right":
        mat[4] += .01


def look(dir):
    if dir == "Up":
        mat[13] += .1
    elif dir == "Down":
        mat[13] -= .1
    elif dir == "Left":
        mat[12] -= .1
    elif dir == "Right":
        mat[12] += .1


def refresh_img():
    render.setCamera(mat)
    render.step()
    image = render.getRgbImage()

    img = ImageTk.Image.fromarray(image)
    imgTk = ImageTk.PhotoImage(img)

    panel.configure(image=imgTk)
    panel.image = imgTk
    print (".")


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
    print (event.keysym)
    if event.keysym in ["Left", "Right", "Up", "Down"]:
        move(event.keysym)
    elif event.keysym in ["w", "a", "s", "d"]:
        look(get_dir_for_wasd(event.keysym))

    refresh_img()


root.bind("<Key>", key_pressed)

root.mainloop()
