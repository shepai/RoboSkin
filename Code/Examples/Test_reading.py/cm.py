import cv2
import letRun #This library can be deleted, it is used for debugging
import RoboSkin as sk
import numpy as np
import matplotlib.pyplot as plt
import time

path= letRun.path
skin=sk.Skin(device=1) #videoFile=path+"Movement4.avi"
frame=skin.getFrame()
old_T=skin.origin
new=np.zeros_like(frame)
SPLIT=5


#uncomment to record video
"""p=np.concatenate((new,new),axis=1)
h, w = p.shape[:2]
out = cv2.VideoWriter('skinDIrection.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, (w,h))"""
def gatherT(string,n=2): #number of cm
    DATA=[]
    past_Frame=skin.getBinary()
    image=np.zeros_like(past_Frame)
    time.sleep(1)
    for i in range(n*2):
        print(string,(i+1)*0.5)
        input("record> ")
        grid=None
        for i in range(10): #must loop through so image is not weird
            im=skin.getBinary()
            #image=skin.getForce(im,SPLIT,image=image,threshold=80,degrade=20,) #get the force push
            image,grid=skin.getForceGrid(im,SPLIT,image=image,threshold=10,degrade=20)
        DATA.append(np.sum(grid))
        past_Frame=image.copy
    return np.array(DATA)

sample=3
print("Obstacle 1")
for i in range(sample):
    d=gatherT("..")
    plt.plot([(i+1)/2 for i in range(2*2)],d,c="b")
print("Obstacle 2")
for i in range(sample):
    d=gatherT("..")
    plt.plot([(i+1)/2 for i in range(2*2)],d,c="r")

plt.xlabel("Pressure in cm")
plt.ylabel("Pressure in summed pixels")
plt.title("Pressure vs cm")
plt.show()
