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




frame=skin.getFrame()
old_T=skin.origin
new=np.zeros_like(frame)
SPLIT=25
past_Frame=skin.getBinary()
image=np.zeros_like(past_Frame)
first=False
l.moveX(3)
i=2
UP=True
while(True):
    frame=skin.getFrame()
    im=skin.getBinary()
    image=skin.getForce(im,past_Frame,SPLIT,image=image,degrade=70) #get the force push
    past_Frame=im.copy() #get last frame
    tactile=np.zeros_like(new)
    tactile[:,:,2]=image #show push in red
    #tactile[:,:,0]=NEW
    cv2.imshow('tactile', tactile)
    cv2.imshow('our binary',im)
    cv2.imshow('unprocessed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    bigTouch=np.sum(tactile)/(255*SPLIT*SPLIT)
    if bigTouch<30: #if the force is not too much
        if i==30: 
            UP=not UP
        elif i==0: 
            l.startPos()
            UP = not UP
        m=10
        if UP: 
            i-=2
            m=-10
        else: i+=2
        l.moveX(m/10)
        if not first:
            first=True
            l.setStart()
        time.sleep(0.10)
    else:
        print("Too much",bigTouch)
        UP=True
        #i=0
        #UP=True
skin.close()
cv2.destroyAllWindows()
l.close()
