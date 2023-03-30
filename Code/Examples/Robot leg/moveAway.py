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


frame=skin.getFrame()
old_T=skin.origin
new=np.zeros_like(frame)
SPLIT=25
past_Frame=skin.getBinary()
image=np.zeros_like(past_Frame)

i=0
UP=True
while(True):
    frame=skin.getFrame()
    im=skin.getBinary()
    image=skin.getForce(im,past_Frame,SPLIT,image=image,degrade=50) #get the force push
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
    #break up into quaters
    q1=tactile[0:tactile.shape[0]//2,0:tactile.shape[1]//2]
    q2=tactile[0:tactile.shape[0]//2,tactile.shape[1]//2:]
    q3=tactile[tactile.shape[0]//2:,0:tactile.shape[1]//2]
    q4=tactile[tactile.shape[0]//2:,tactile.shape[1]//2:]
    quaters=np.array([np.sum(q1),np.sum(q2),np.sum(q3),np.sum(q4)]) /(255*SPLIT*SPLIT)
    if np.max(quaters)>5 and bigTouch<35: #if the force is not too much:
        #check what has the most pressure and move there
        if np.argmax(quaters)==0: 
            print("q1")
            ang=l.angle2
            l.B.move(2,ang-5)
            l.angle2=ang-5
            ang=l.angle1
            l.B.move(1,ang-5)
            l.angle1=ang-5
        elif np.argmax(quaters)==1: 
            print("q2")
            ang=l.angle2
            l.B.move(2,ang+5)
            l.angle2=ang+5
            ang=l.angle1
            l.B.move(1,ang-5)
            l.angle1=ang-5
        elif np.argmax(quaters)==2: 
            print("q3")
            ang=l.angle2
            l.B.move(2,ang+5)
            l.angle2=ang+5
            ang=l.angle1
            l.B.move(1,ang+5)
            l.angle1=ang+5
        elif np.argmax(quaters)==3: 
            print("q4")
            ang=l.angle2
            l.B.move(2,ang-5)
            l.angle2=ang-5
            ang=l.angle1
            l.B.move(1,ang+5)
            l.angle1=ang+5
    elif bigTouch>30: #if too much pressure return to start position
        l.startPos()
skin.close()
cv2.destroyAllWindows()
l.close()
