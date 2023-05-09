import cv2
import letRun #This library can be deleted, it is used for debugging
import RoboSkin as sk
import numpy as np
import matplotlib.pyplot as plt
import time
import random

path= letRun.path


#uncomment to record video
"""p=np.concatenate((new,new),axis=1)
h, w = p.shape[:2]
out = cv2.VideoWriter('skinDIrection.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, (w,h))"""
def gatherT(file,string,n=5): #number of cm
    skin=sk.Skin(videoFile=path+file) #videoFile=path+"Movement4.avi"
    old_T=skin.origin
    for i in range(random.randint(20,50)):
        im=skin.getBinary()
    DATA=[]
    past_Frame=skin.getBinary()
    for i in range(n):
        im=skin.getBinary()
        t_=skin.getDots(im)
        t=skin.movement(t_)

        DATA.append(t.copy())
    np.array(DATA)
    av=np.zeros((len(DATA),))
    for i,dat in enumerate(DATA):
        av[i]=np.linalg.norm(old_T-dat)
    dist=np.sum(av)#/len(av)
    skin.close()
    return dist

exps=10
av=np.zeros((4))
for j in range(10):
    sample=4
    modes=["speed1.avi","speed2.avi","speed3.avi","speed4.avi"]
    print("Obstacle ",j)
    speeds=[]

    for i in range(sample):
        print(modes[i])
        d=gatherT(modes[i],"speed"+str(i))
        speeds.append(d)
    av+=np.array(speeds)
    plt.plot([(i+1)/2 for i in range(sample)],speeds,label="Exp"+str(j))

av=av/exps
plt.plot([(i+1)/2 for i in range(sample)],av,'--',label="Average")

"""print("Obstacle 2")
for i in range(sample):
    d=gatherT("..")
    plt.plot([(i+1)/2 for i in range(2*2)],d,c="r")"""

plt.xlabel("Speed setting")
plt.ylabel("Average magnitude size (px)")
plt.title("How speed affects the time of arrival to the magnitude of vectors")
plt.legend(loc="lower left")
plt.show()

