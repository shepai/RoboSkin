# some_file.py
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

#download example pic
img = cv2.imread(path+'flat.png')
shrink=(np.array(img.shape[0:2])//3).astype(int)
img=cv2.resize(img,(shrink[1],shrink[0]),interpolation=cv2.INTER_AREA)[60:220,75:220]

skin=sk.Skin(imageFile=img) #create the image

frame=skin.getFrame()
old_T=skin.origin
new=np.zeros_like(frame)
SPLIT=15
past_Frame=skin.getBinary()
image=np.zeros_like(past_Frame)

h,w=img.shape[0:2]
env=sk.environment(w*4,w*4) #create environment
SIZE=250
tip=sk.digiTip(img) #create tactip

def getDir(skin,old_T,env,tip):
    im=tip.getImage(env.get())
    im=tip.maskPush(im)
    skin.imF=im.copy() #set it in skin
    im=skin.getBinary()
    new=np.zeros_like(im)
    t_=skin.getDots(im)
    t=skin.movement(t_)
    if t.shape[0]>2:
        for i,coord in enumerate(t): #show vectors of every point
            x1=coord[1]
            y1=coord[0]
            x2=old_T[i][1]
            y2=old_T[i][0]
            cv2.arrowedLine(new, (x2, y2), (x1, y1), 255, thickness=1)#
    return im,new

"""f=np.concatenate((past_Frame,image,image),axis=1).astype(np.uint8)
f=cv2.resize(f,(1000,400),interpolation=cv2.INTER_AREA)
f=cv2.cvtColor(f,cv2.COLOR_GRAY2RGB)
h, w = f.shape[:2]
out = cv2.VideoWriter('digiTipFlow.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, (w,h))#"""

#move down
tip.h=100
for i in range(0,tip.h,10):
    tip.h-=10
    img_g,dire=getDir(skin,old_T,env,tip)
    #show in bigger picture
    e=env.get().copy()
    y=tip.pos[0]+tip.grid.shape[0]
    x=tip.pos[1]+tip.grid.shape[1]
    terrain=env.get()[max(min(tip.pos[0],e.shape[0]),0):max(min(y,e.shape[0]),0),max(min(tip.pos[1],e.shape[1]),0):max(min(x,e.shape[1]),0)].copy()*5
    #display all images
    #plt.imshow(e)
    #plt.title(str(tip.h))
    #plt.pause(0.25)
    f_=np.concatenate((img_g,dire,terrain),axis=1).astype(np.uint8)
    f_=cv2.resize(f_,(1000,400),interpolation=cv2.INTER_AREA)
    f_=cv2.cvtColor(f_,cv2.COLOR_GRAY2RGB)
    #assert f.shape==f_.shape,"Error f.shape"+str(f.shape)+str(f_.shape)
    #out.write(f_)
    cv2.imshow('data', f_)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    


#move right
tip.h=20
for i in range(0,30,2):
    tip.moveX(10)
    img_g,dire=getDir(skin,old_T,env,tip)
    #show in bigger picture
    e=env.get().copy()
    y=tip.pos[0]+tip.grid.shape[0]
    x=tip.pos[1]+tip.grid.shape[1]
    terrain=env.get()[max(min(tip.pos[0],e.shape[0]),0):max(min(y,e.shape[0]),0),max(min(tip.pos[1],e.shape[1]),0):max(min(x,e.shape[1]),0)].copy()*5
    #display all images
    #plt.imshow(e)
    #plt.title(str(tip.h))
    #plt.pause(0.25)
    f_=np.concatenate((img_g,dire,terrain),axis=1).astype(np.uint8)
    f_=cv2.resize(f_,(1000,400),interpolation=cv2.INTER_AREA)
    f_=cv2.cvtColor(f_,cv2.COLOR_GRAY2RGB)
    #assert f.shape==f_.shape,"Error f.shape"+str(f.shape)+str(f_.shape)
    out.write(f_)
    cv2.imshow('data', f_)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


#move down
for i in range(0,30,2):
    tip.moveY(10)
    img_g,dire=getDir(skin,old_T,env,tip)
    #show in bigger picture
    e=env.get().copy()
    y=tip.pos[0]+tip.grid.shape[0]
    x=tip.pos[1]+tip.grid.shape[1]
    terrain=env.get()[max(min(tip.pos[0],e.shape[0]),0):max(min(y,e.shape[0]),0),max(min(tip.pos[1],e.shape[1]),0):max(min(x,e.shape[1]),0)].copy()*5
    #display all images
    #plt.imshow(e)
    #plt.title(str(tip.h))
    #plt.pause(0.25)
    f_=np.concatenate((img_g,dire,terrain),axis=1).astype(np.uint8)
    f_=cv2.resize(f_,(1000,400),interpolation=cv2.INTER_AREA)
    f_=cv2.cvtColor(f_,cv2.COLOR_GRAY2RGB)
    #assert f.shape==f_.shape,"Error f.shape"+str(f.shape)+str(f_.shape)
    #out.write(f_)
    cv2.imshow('data', f_)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#move left
tip.h=20
for i in range(0,30,2):
    tip.moveX(-10)
    img_g,dire=getDir(skin,old_T,env,tip)
    #show in bigger picture
    e=env.get().copy()
    y=tip.pos[0]+tip.grid.shape[0]
    x=tip.pos[1]+tip.grid.shape[1]
    terrain=env.get()[max(min(tip.pos[0],e.shape[0]),0):max(min(y,e.shape[0]),0),max(min(tip.pos[1],e.shape[1]),0):max(min(x,e.shape[1]),0)].copy()*5
    #display all images
    #plt.imshow(e)
    #plt.title(str(tip.h))
    #plt.pause(0.25)
    f_=np.concatenate((img_g,dire,terrain),axis=1).astype(np.uint8)
    f_=cv2.resize(f_,(1000,400),interpolation=cv2.INTER_AREA)
    f_=cv2.cvtColor(f_,cv2.COLOR_GRAY2RGB)
    #assert f.shape==f_.shape,"Error f.shape"+str(f.shape)+str(f_.shape)
    #out.write(f_)
    cv2.imshow('data', f_)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#move up
for i in range(0,tip.h,10):
    tip.moveY(-10)
    img_g,dire=getDir(skin,old_T,env,tip)
    #show in bigger picture
    e=env.get().copy()
    y=tip.pos[0]+tip.grid.shape[0]
    x=tip.pos[1]+tip.grid.shape[1]
    terrain=env.get()[max(min(tip.pos[0],e.shape[0]),0):max(min(y,e.shape[0]),0),max(min(tip.pos[1],e.shape[1]),0):max(min(x,e.shape[1]),0)].copy()*5
    #display all images
    #plt.imshow(e)
    #plt.title(str(tip.h))
    #plt.pause(0.25)
    f_=np.concatenate((img_g,dire,terrain),axis=1).astype(np.uint8)
    f_=cv2.resize(f_,(1000,400),interpolation=cv2.INTER_AREA)
    f_=cv2.cvtColor(f_,cv2.COLOR_GRAY2RGB)
    #assert f.shape==f_.shape,"Error f.shape"+str(f.shape)+str(f_.shape)
    #out.write(f_)
    cv2.imshow('data', f_)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
#out.release() #uncomment to record video
exit()