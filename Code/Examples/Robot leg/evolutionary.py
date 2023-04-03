from arm import Leg
import time
import cv2
import numpy as np
from agent import *
import os
from copy import deepcopy as DC
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
    vectors=old_T-t
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


def runTrial(agent,T,dt):
    #create sensor and generate initial variables
    frame=skin.getFrame()
    old_T=skin.origin
    new=np.zeros_like(frame)
    SPLIT=25
    past_Frame=skin.getBinary()
    image=np.zeros_like(past_Frame)
    old_T=skin.origin #get old direction
    for i in range(10): #cycle out noise
        tactile,past_Frame,image,vec=getImage(image,past_Frame,new,SPLIT)
    t=0
    fit=0
    l.startPos()
    while t<T:
        #gather signal
        tactile,past_Frame,image,vec=getImage(image,past_Frame,new,SPLIT)
        bigTouch=np.sum(tactile)/(255*SPLIT*SPLIT)
        #run through network
        v=torch.tensor(vec).flatten()
        a=agent.forward(v)
        move=np.argmax(a.detach().numpy())
        cv2.imshow('tactile', tactile)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        #trial
        if bigTouch>35 or l.angle2>110:
            print("Too much",bigTouch,l.angle2)
            fit=0
            break
        else:
            if move==0:
                l.moveX(1)
                fit+=dt
            else:
                l.moveX(-1)
            time.sleep(0.1)
        t+=dt
    return fit/T

def mutate(genotype):
    return genotype+np.random.normal(0,2,genotype.shape)

def gen_genotype(shape):
    return np.random.normal(0,3,(shape,))

def run(agent,population,generations=500):
    pop_size=len(population)
    shape=population.shape[1]
    fitness_matrix=np.zeros((pop_size))
    overTime=np.zeros((generations,))

    for gen in range(generations):
        clear()
        print("Generation:",gen,"Fitness",overTime[max(gen-1,0)])
        #get mask to select genotypes for battle
        mask=np.random.choice([0, 1], size=pop_size)
        inds=(mask==1).nonzero()[0]
        while len(inds)%2!=0:
            mask=np.random.choice([0, 1], size=pop_size)
            inds=(mask==1).nonzero()[0]
        #get indicies and tournament modes
        inds=inds.reshape(len(inds)//2,2).astype(int)
        fitnesses=np.zeros((len(inds)//2)).astype(int)
        new_inds=np.zeros((len(inds)//2,2)).astype(int)
        #run each trial
        for i in range(len(inds)//2):
            #select genotypes
            g1=population[inds[i][0]]
            g2=population[inds[i][1]]
            #tournament
            agent.set_genes(g1)
            f1=runTrial(agent,T,dt)
            agent.set_genes(g1)
            f2=runTrial(agent,T,dt)
            fitness_matrix[inds[i][0]]=f1
            fitness_matrix[inds[i][1]]=f2
            if f1>f2:
                fitnesses[i]=f1
                new_inds[i]=[0,i]
            elif f2>f1:
                fitnesses[i]=f2
                new_inds[i]=[1,i]
            else:
                fitnesses[i]=0
                new_inds[i]=[0,i]
        #get top values and redistribute into the array
        winners=int(len(inds)//2 *0.4)
        mutants=int(len(inds)//2 *0.4)
        other=len(inds)//2 -winners - mutants
        order=np.argsort(fitnesses)
        for i in reversed(range(len(inds)//2)): #loop through backwards leaving the winners in place
            genoWin=new_inds[i][0]
            old_index=new_inds[i][1]
            if i<(len(inds)//2)-winners and i>(len(inds)//2)-winners-mutants: #pick mutants
                population[inds[old_index][1-genoWin]]=mutate(DC(population[inds[old_index][genoWin]])) #mutate copy
            elif i<(len(inds)//2)-winners-mutants: #the other
                population[inds[old_index][1-genoWin]]=gen_genotype(shape=shape) #create new genotype
        overTime[gen]=np.max(fitness_matrix)
    return overTime

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
T=2
#genetic agorithm
SIZE=100
pop=np.random.normal(0,3,(SIZE,arm.num_genes))

generations=100
print(run(arm,pop,generations))
    
skin.close()
cv2.destroyAllWindows()
l.close()