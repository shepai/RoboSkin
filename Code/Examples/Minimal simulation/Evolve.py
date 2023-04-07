import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from copy import deepcopy as DC
import torch

path=""
clear=None
if os.name == 'nt':
    path="C:/Users/dexte/github/RoboSkin/Assets/images/"
    clear = lambda: os.system('cls')
else:
    path="/its/home/drs25/Documents/GitHub/RoboSkin/Assets/images/"
    clear = lambda: os.system('clear')

# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, 'C:/Users/dexte/github/RoboSkin/Code/')
sys.path.insert(1, '/its/home/drs25/Documents/GitHub/RoboSkin/Code')
out=None
import RoboSkin as sk
import Models.agent as agent

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
    #env+=np.random.normal(0,5,env.shape) #add bit of noise to simulate vibration
    env[env<0]=0
    return env,grad,c

def clearNoise(tip,skin,env,image,past_Frame,SPLIT,SIZE):
    for i in range(100):
        im,image,g=GetIm(tip,skin,env,image,past_Frame,SPLIT,SIZE)
        past_Frame=im.copy()
    return image
def GetIm(tip,skin,env,image,past_Frame,SPLIT,SIZE):
    im=tip.getImage(env)
    im=tip.maskPush(im)
    skin.imF=im.copy() #set it in skin
    im_g=skin.getBinary(min_size = SIZE) #get image from skin
    image,grid=skin.getForceGrid(im_g,SPLIT,image=image,threshold=20,degrade=20) #get the force push
    return im_g,image,grid
def tap(tip,skin,env,image,past_Frame,SPLIT,SIZE):
    tip.h=30
    img_g,image,g = GetIm(tip,skin,env,image,past_Frame,SPLIT,SIZE)
    tip.h=20
    img_g,image,g = GetIm(tip,skin,env,image,past_Frame,SPLIT,SIZE)
    return img_g,image,g



def runTrial(agent,img,skin,T):
    SPLIT=5
    past_Frame=skin.getBinary()
    image=np.zeros_like(past_Frame)
    h,w=img.shape[0:2]
    env,gr,c=genEnv(w)
    startY=(w*5) + img.shape[0]//2#start in middle
    if gr==0: gr=0.1
    startX=(startY- c)//gr
    if gr>0:startX-=50
    else:startX+=50
    SIZE=250
    tip=sk.digiTip(img) #create tactip
    tip.setPos(startX,startY) #start somewhere on line

    image=clearNoise(tip,skin,env,image,past_Frame,SPLIT,SIZE)
    dt=0.02
    t=0
    im_g,image,grid=tap(tip,skin,env,image,past_Frame,SPLIT,SIZE) #get tap to know area
    trial_over=False
    while t<T and not trial_over:
        y=tip.pos[0]+tip.grid.shape[0]
        x=tip.pos[1]+tip.grid.shape[1]
        terrain=env[max(min(tip.pos[0],env.shape[0]),0):max(min(y,env.shape[0]),0),max(min(tip.pos[1],env.shape[1]),0):max(min(x,env.shape[1]),0)].copy()*5
        if terrain.shape!=image.shape: trial_over=True
        if im_g.shape==terrain.shape:
            f_=np.concatenate((im_g,image,terrain),axis=1).astype(np.uint8)
            f_=cv2.resize(f_,(1000,400),interpolation=cv2.INTER_AREA)
            f_=cv2.cvtColor(f_,cv2.COLOR_GRAY2RGB)
            out.write(f_)
        #cv2.imshow('data', f_)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
        #move
        move=np.argmax(agent.forward(torch.tensor(grid.flatten())))
        if move==0: tip.moveX(5)
        elif move==1: tip.moveX(-5)
        elif move==2: tip.moveY(5)
        elif move==3: tip.moveY(-5)
        #gather next image
        im_g,image,grid = GetIm(tip,skin,env,image,past_Frame,SPLIT,SIZE)
        past_Frame=im_g.copy() #get last frame
        t+=dt
    return np.sum(terrain)/(img.shape[0]*img.shape[1]*22)/5


def mutate(genotype):
    return genotype+np.random.normal(0,2,genotype.shape)

def gen_genotype(shape):
    return np.random.normal(0,3,(shape,))

def run(agent,population,generations=500,T=2):
    pop_size=len(population)
    shape=population.shape[1]
    fitness_matrix=np.zeros((pop_size))
    overTime=np.zeros((generations,))
    gen=0
    while gen < (generations) and overTime[max(gen-1,0)]<1:
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
            f1=0
            for i in range(3):
                f1+=runTrial(agent,img,skin,T)
            f1/=3
            agent.set_genes(g1)
            f2=0
            for i in range(3):
                f2+=runTrial(agent,img,skin,T)
            f2/=3
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
        gen+=1
    return overTime

img = cv2.imread(path+'flat.png')
shrink=(np.array(img.shape[0:2])//3).astype(int)
img=cv2.resize(img,(shrink[1],shrink[0]),interpolation=cv2.INTER_AREA)[60:220,75:220]
#img=cv2.resize(img,(np.array(img.shape[0:2])).astype(int),interpolation=cv2.INTER_AREA)
skin=sk.Skin(imageFile=img) #create the image

f=np.concatenate((img,img,img),axis=1).astype(np.uint8)
f=cv2.resize(f,(1000,400),interpolation=cv2.INTER_AREA)
#f=cv2.cvtColor(f,cv2.COLOR_GRAY2RGB)
h, w = f.shape[:2]
out = cv2.VideoWriter('digiTip_evolve.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, (w,h))

#create an agent
sensor=agent.Agent(5*5,[30,30],4)
genes=np.random.normal(0,3,(sensor.num_genes,))
sensor.set_genes(genes)

dt=0.01
T=2
#genetic agorithm
SIZE=100
pop=np.random.normal(0,3,(SIZE,sensor.num_genes))

generations=1
print(run(sensor,pop,generations))
out.release()

cv2.destroyAllWindows()
