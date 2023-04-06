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
    if grad<=0: c=1000
    for i in range(w*10):
        for j in range(w*10):
            y=grad*j +c
            if y>i:
                env[i][j]=22
    env+=np.random.normal(0,3,env.shape) #add bit of noise to simulate vibration
    env[env<0]=0
    return env,grad,c

def clearNoise(tip,skin,env,image,past_Frame,SPLIT,SIZE):
    for i in range(100):
        im,image=GetIm(tip,skin,env,image,past_Frame,SPLIT,SIZE)
        past_Frame=im.copy()
    return image
def GetIm(tip,skin,env,image,past_Frame,SPLIT,SIZE):
    im=tip.getImage(env)
    im=tip.maskPush(im)
    skin.imF=im.copy() #set it in skin
    im_g=skin.getBinary(min_size = SIZE) #get image from skin
    image=skin.getForce(im_g,past_Frame,SPLIT,image=image,threshold=20,degrade=20) #get the force push
    return im_g,image
def tap(tip,skin,env,image,past_Frame,SPLIT,SIZE):
    tip.h=25
    img_g,image = GetIm(tip,skin,env,image,past_Frame,SPLIT,SIZE)
    tip.h=20
    img_g,image = GetIm(tip,skin,env,image,past_Frame,SPLIT,SIZE)
    return img_g,image

img = cv2.imread(path+'flat.png')
shrink=(np.array(img.shape[0:2])//3).astype(int)
img=cv2.resize(img,(shrink[1],shrink[0]),interpolation=cv2.INTER_AREA)[60:220,75:220]
img=cv2.resize(img,(np.array(img.shape[0:2])).astype(int),interpolation=cv2.INTER_AREA)
skin=sk.Skin(imageFile=img) #create the image



def runTrial(img,skin,T):
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

    image=clearNoise(tip,skin,env,image,past_Frame,SPLIT,SIZE)
    dt=0.01
    t=0
    im_g,image=tap(tip,skin,env,image,past_Frame,SPLIT,SIZE) #get tap to know area
    trial_over=False
    while t<T and not trial_over:
        y=tip.pos[0]+tip.grid.shape[0]
        x=tip.pos[1]+tip.grid.shape[1]
        terrain=env[max(min(tip.pos[0],env.shape[0]),0):max(min(y,env.shape[0]),0),max(min(tip.pos[1],env.shape[1]),0):max(min(x,env.shape[1]),0)].copy()*5
        if terrain.shape!=image.shape: trial_over=True
        f_=np.concatenate((im_g,image,terrain),axis=1).astype(np.uint8)
        f_=cv2.resize(f_,(1000,400),interpolation=cv2.INTER_AREA)
        f_=cv2.cvtColor(f_,cv2.COLOR_GRAY2RGB)
        
        #assert f.shape==f_.shape,"Error f.shape"+str(f.shape)+str(f_.shape)
        #out.write(f_)
        cv2.imshow('data', f_)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        #move
        if gr>0:
            tip.moveX(5)
        else:
            tip.moveX(-5)
        #gather next image
        im_g,image = GetIm(tip,skin,env,image,past_Frame,SPLIT,SIZE)
        past_Frame=im_g.copy() #get last frame
        t+=dt



def getDir(skin,old_T,env,tip):
    im=tip.getImage(env)
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
    return new
def runTrialD(img,skin,T):
    SPLIT=15
    past_Frame=skin.getBinary()
    image=np.zeros_like(past_Frame)
    h,w=img.shape[0:2]
    env,gr,c=genEnv(w)
    startY=(w*5) + img.shape[0]//2#start in middle
    if gr==0: gr=0.1
    startX=(startY- c)//gr
    old_T=skin.origin
    SIZE=250
    tip=sk.digiTip(img) #create tactip
    tip.setPos(startX,startY) #start somewhere on line
    tip.h=22
    im_g,im=GetIm(tip,skin,env,image,past_Frame,SPLIT,SIZE)
    dt=0.01
    t=0
    dire=getDir(skin,old_T,env,tip)
    while t<T:
        y=tip.pos[0]+tip.grid.shape[0]
        x=tip.pos[1]+tip.grid.shape[1]
        terrain=env[max(min(tip.pos[0],env.shape[0]),0):max(min(y,env.shape[0]),0),max(min(tip.pos[1],env.shape[1]),0):max(min(x,env.shape[1]),0)].copy()*5     	
        #gray = cv2.cvtColor(dire, cv2.COLOR_BGR2GRAY)

        f_=np.concatenate((im_g,dire,terrain),axis=1).astype(np.uint8)
        f_=cv2.resize(f_,(1000,400),interpolation=cv2.INTER_AREA)
        f_=cv2.cvtColor(f_,cv2.COLOR_GRAY2RGB)
        
        #assert f.shape==f_.shape,"Error f.shape"+str(f.shape)+str(f_.shape)
        #out.write(f_)
        cv2.imshow('data', f_)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        #move
        if gr>0:
            tip.moveX(5)
        else:
            tip.moveX(-5)
        #gather next image
        img_g,image=GetIm(tip,skin,env,image,past_Frame,SPLIT,SIZE)
        past_Frame=img_g.copy()
        dire=getDir(skin,old_T,env,tip)
        t+=dt

runTrial(img,skin,5)