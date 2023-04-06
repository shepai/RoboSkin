import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

path=""
if os.name == 'nt':
    path="C:/Users/dexte/github/RoboSkin/Assets/images/"
else:
    path="/its/home/drs25/Documents/GitHub/RoboSkin/images/"

# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, 'C:/Users/dexte/github/RoboSkin/Code/')
sys.path.insert(1, '/its/home/drs25/Documents/GitHub/RoboSkin/Code')

import RoboSkin as sk

def genEnv(w):
    env=np.zeros((w*10,w*10)) #create environment
    grad=np.random.randint(-2,5)
    c=0
    if grad<=0: c=600
    for i in range(w*10):
        for j in range(w*10):
            y=grad*j +c
            if y>i:
                env[i][j]=22
    return env,grad,c

def clearNoise(tip,skin,image,past_Frame,SPLIT):
    for i in range(100):
        im=tip.getImage(env)
        im=tip.maskPush(im)
        im_g=skin.getBinary(min_size = SIZE) #get image from skin
        image=skin.getForce(im_g,past_Frame,SPLIT,image=image,threshold=20,degrade=10) #get the force push

img = cv2.imread(path+'flat.png')
shrink=(np.array(img.shape[0:2])//3).astype(int)
img=cv2.resize(img,(shrink[1],shrink[0]),interpolation=cv2.INTER_AREA)[60:220,75:220]
img=cv2.resize(img,(np.array(img.shape[0:2])/1.5).astype(int),interpolation=cv2.INTER_AREA)
skin=sk.Skin(imageFile=img) #create the image

frame=skin.getFrame()
old_T=skin.origin
new=np.zeros_like(frame)
SPLIT=15
past_Frame=skin.getBinary()
image=np.zeros_like(past_Frame)
h,w=img.shape[0:2]
env,gr,c=genEnv(w)
startY=(w*5) + img.shape[0]//2#start in middle
if gr==0: gr=0.1
startX=(startY- c)//gr

SIZE=250
tip=sk.digiTip(img) #create tactip
tip.setPos(startX,startY) #start somewhere on line
tip.h=20

clearNoise(tip,skin,image,past_Frame,SPLIT)


for i in range(0,40,1):
    if gr>0:
        tip.moveX(5)
    else:
        tip.moveX(-5)
    im=tip.getImage(env)
    im=tip.maskPush(im)
    skin.imF=im.copy() #set it in skin
    im_g=skin.getBinary(min_size = SIZE) #get image from skin
    image=skin.getForce(im_g,past_Frame,SPLIT,image=image,threshold=20,degrade=10) #get the force push
    past_Frame=im_g.copy() #get last frame
    tactile=np.zeros_like(new)
    tactile[:,:,2]=image #show push in red
    e=env.copy()
    y=tip.pos[0]+tip.grid.shape[0]
    x=tip.pos[1]+tip.grid.shape[1]
    terrain=env[max(min(tip.pos[0],e.shape[0]),0):max(min(y,e.shape[0]),0),max(min(tip.pos[1],e.shape[1]),0):max(min(x,e.shape[1]),0)].copy()*5
    #e[max(min(tip.pos[0],e.shape[0]),0):max(min(y,e.shape[0]),0),max(min(tip.pos[1],e.shape[1]),0):max(min(x,e.shape[1]),0)]=im
    #plt.imshow(e)
    #plt.title(str(i))
    #plt.pause(0.25)
    f_=np.concatenate((past_Frame,image,terrain),axis=1).astype(np.uint8)
    f_=cv2.resize(f_,(1000,400),interpolation=cv2.INTER_AREA)
    f_=cv2.cvtColor(f_,cv2.COLOR_GRAY2RGB)
    #assert f.shape==f_.shape,"Error f.shape"+str(f.shape)+str(f_.shape)
    #out.write(f_)
    cv2.imshow('data', f_)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break