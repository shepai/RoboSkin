# some_file.py
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

path=""
if os.name == 'nt':
    path="C:/Users/dexte/github/RoboSkin/Assets/images/"
else:
    path="/its/home/drs25/Documents/GitHub/RoboSkin/images/"

# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, 'C:/Users/dexte/github/RoboSkin/Code/')
sys.path.insert(1, '/its/home/drs25/Documents/GitHub/RoboSkin/Code')

import RoboSkin as sk

#download example pic
img = cv2.imread(path+'flat.png')
shrink=(np.array(img.shape[0:2])//3).astype(int)
img=cv2.resize(img,(shrink[1],shrink[0]),interpolation=cv2.INTER_AREA)[60:220,75:220]

skin=sk.Skin(imageFile=img) #create the image

h,w=img.shape[0:2]
env=sk.environment(w*4,w*4) #create environment

tip=sk.digiTip(img) #create tactip

for i in range(0,tip.h,10):
    tip.h-=10
    im=tip.getImage(env.get())
    e=env.get().copy()
    #get coorinares on bigger image
    y=tip.pos[0]+tip.grid.shape[0]
    x=tip.pos[1]+tip.grid.shape[1]
    im=tip.maskPush(im)
    e[max(min(tip.pos[0],e.shape[0]),0):max(min(y,e.shape[0]),0),max(min(tip.pos[1],e.shape[1]),0):max(min(x,e.shape[1]),0)]=im
    plt.imshow(e)
    plt.title(str(tip.h))
    plt.pause(0.5)

tip.h=25
for i in range(0,30,2):
    tip.moveX(10)
    im=tip.getImage(env.get())
    e=env.get().copy()
    y=tip.pos[0]+tip.grid.shape[0]
    x=tip.pos[1]+tip.grid.shape[1]
    im=tip.maskPush(im)
    e[max(min(tip.pos[0],e.shape[0]),0):max(min(y,e.shape[0]),0),max(min(tip.pos[1],e.shape[1]),0):max(min(x,e.shape[1]),0)]=im
    plt.imshow(e)
    plt.title(str(i))
    plt.pause(0.25)

#plt.show()
