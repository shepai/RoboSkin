from arm import Leg
import time
import cv2
import numpy as np
from agent import *
import os
#################################################################
"""
If the library is not in the direct path add it
"""
import sys
import os
path=""
clear=None
if os.name == 'nt':
    path="C:/Users/dexte/github/RoboSkin/Assets/Video demos/"
    clear = lambda: os.system('cls')
else:
    path="/its/home/drs25/Documents/GitHub/RoboSkin/Assets/Video demos/"
    clear = lambda: os.system('clear')

# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, 'C:/Users/dexte/github/RoboSkin/Code/')
sys.path.insert(1, '/its/home/drs25/Documents/GitHub/RoboSkin/Code')

#################################################################

import RoboSkin as sk

#setup coms with skin and arm
skin=sk.Skin(device=0) 
l=Leg()
l.startPos()

def getImage(image,past_Frame,new,SPLIT):
    tactile=np.zeros_like(new)
    frame=skin.getFrame()
    im=skin.getBinary()
    t_=skin.getDots(im)
    t=skin.movement(t_)
    vectors=[]
    if t.shape[0]>2:
        new[t[:,0],t[:,1]]=(0,255,0)
        new[old_T[:,0],old_T[:,1]]=(0,0,255)
        for i,coord in enumerate(t): #show vectors of every point
            x1=coord[1]
            y1=coord[0]
            x2=old_T[i][1]
            y2=old_T[i][0]
            #cv2.putText(new,str(i),(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,0.2,(0,255,0))
            #d=skin.euclid(np.array([x1, y1]), np.array([x2, y2]))
            cv2.arrowedLine(tactile, (x2, y2), (x1, y1), (0, 255, 0), thickness=1)#
            vectors.append([x2-x1,y2-y1])
    if type(past_Frame)==type(None):
        past_Frame=im.copy()
    image=skin.getForce(im,past_Frame,SPLIT,image=image,degrade=200) #get the force push
    past_Frame=im.copy() #get last frame
    tactile[:,:,2]=image #show push in red
    past_Frame=im.copy()
    return tactile,past_Frame,image,vectors

def touchDown(past_Frame,new,SPLIT):
    time.sleep(1)
    image=np.zeros_like(past_Frame)
    tactileO,past_Frame,image,v=getImage(image,None,new,SPLIT)
    l.moveX((10)/10)
    time.sleep(1)
    tactileN,past_Frame,image,v=getImage(image,past_Frame,new,SPLIT)
    l.moveX((-10)/10) #return to normal
    return tactileN,past_Frame,image


old_T=skin.origin
#create an agent
arm=Agent(len(old_T.flatten()),[40,40],2)
genes=np.random.normal(0,3,(arm.num_genes,))
arm.set_genes(genes)

i=0
UP=True
first=False
LIMIT=50
dt=0.01
T=4

def runTrial(agent,T,dt):
    #create sensor and generate initial variables
    frame=skin.getFrame()
    old_T=skin.origin
    new=np.zeros_like(frame)
    SPLIT=25
    past_Frame=skin.getBinary()
    image=np.zeros_like(past_Frame)
    old_T=skin.origin #get old direction
    t=0
    fit=0
    while t<T:
        #gather signal
        tactile,past_Frame,image,vec=getImage(image,past_Frame,new,SPLIT)
        bigTouch=np.sum(tactile)/(255*SPLIT*SPLIT)
        #run through network
        v=torch.tensor(vec).flatten()
        a=agent.forward(v)
        move=np.argmax(a.detach().numpy())

        #trial
        if bigTouch>25 or l.angle2>100:
            print("Too much",bigTouch)
            l.startPos()
            time.sleep(1)
            fit=0
        else:
            if move==0:
                l.moveX(1)
            else:
                l.moveX(-1)
            fit+=dt
        cv2.imshow('tactile', tactile)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
        t+=dt
    return fit/T


#genetic agorithm
SIZE=100
pop=np.random.normal(0,3,(SIZE,arm.num_genes))



generations=500
for gen in range(generations):
    clear()
    print("Generation:",gen,"Fitness",runTrial(arm,2,dt))
    
skin.close()
cv2.destroyAllWindows()
l.close()