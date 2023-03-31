from arm import Leg
import time
import cv2
import numpy as np
#################################################################
"""
If the library is not in the direct path add it
"""
import sys
import os
path=""
if os.name == 'nt':
    path="C:/Users/dexte/github/RoboSkin/Assets/Video demos/"
else:
    path="/its/home/drs25/Documents/GitHub/RoboSkin/Assets/Video demos/"

# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, 'C:/Users/dexte/github/RoboSkin/Code/')
sys.path.insert(1, '/its/home/drs25/Documents/GitHub/RoboSkin/Code')
#################################################################

import RoboSkin as sk

#setup coms with skin and arm
skin=sk.Skin(device=0) 
l=Leg()

l.startPos()
time.sleep(0.5)

"""for i in range(15):
    time.sleep(0.10)
    l.moveX(-i/10)
for i in range(18):
    time.sleep(0.10)
    l.moveX(i/10)"""

def getImage(image,past_Frame):
    frame=skin.getFrame()
    im=skin.getBinary()
    if type(past_Frame)==type(None):
        past_Frame=im.copy()
    image=skin.getForce(im,past_Frame,SPLIT,image=image,degrade=200) #get the force push
    past_Frame=im.copy() #get last frame
    tactile=np.zeros_like(new)
    tactile[:,:,2]=image #show push in red
    return tactile,past_Frame,image

def touchDown(m,past_Frame):
    time.sleep(1)
    image=np.zeros_like(past_Frame)
    tactileO,past_Frame,image=getImage(image,None)
    l.moveX((-10)/10)
    time.sleep(0.2)
    tactileN,past_Frame,image=getImage(image,past_Frame)
    l.moveX((10)/10) #return to normal
    return tactileN,past_Frame,image

frame=skin.getFrame()
old_T=skin.origin
new=np.zeros_like(frame)
SPLIT=25
past_Frame=skin.getBinary()
image=np.zeros_like(past_Frame)

i=0
UP=True
first=False

while(True):
    tactile,past_Frame,image=getImage(image,past_Frame)
    
    bigTouch=np.sum(tactile)/(255*SPLIT*SPLIT)
    if bigTouch<25: #if the force is not too much
        if i==40: 
            UP=True
        elif i==0: 
            l.startPos()
            UP = False
        m=10
        if UP: 
            i-=2
            m=-10
        else: i+=2
        l.moveX(m/10)
        if not first and i==20:
            first=True
            l.setStart()
        time.sleep(0.02)
    elif bigTouch>25 and bigTouch<50:
        m=10
        if UP: 
            m=-10
        change,past_Frame,image=touchDown(m,past_Frame)
        cv2.imshow('tactile', change)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        l.setStart()
    else:
        print("Too much",bigTouch)
        UP=True
        #i=0
        #UP=True
skin.close()
cv2.destroyAllWindows()
l.close()