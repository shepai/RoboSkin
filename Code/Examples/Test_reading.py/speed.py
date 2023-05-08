import cv2
import letRun #This library can be deleted, it is used for debugging
import RoboSkin as sk
import numpy as np
import matplotlib.pyplot as plt
import time

path= letRun.path


#uncomment to record video
"""p=np.concatenate((new,new),axis=1)
h, w = p.shape[:2]
out = cv2.VideoWriter('skinDIrection.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, (w,h))"""
def gatherT(file,string,n=3): #number of cm
    skin=sk.Skin(videoFile=path+file) #videoFile=path+"Movement4.avi"
    old_T=skin.origin
    for i in range(5):
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
    dist=np.sum(av)/len(av)
    skin.close()
    return dist

sample=4
modes=["Movement4.avi","Movement3.avi","Movement2.avi","Movement1.mp4"]
print("Obstacle 1")
speeds=[]
for i in range(sample):
    print(modes[i])
    d=gatherT(modes[i],"speed"+str(i))
    speeds.append(d)

plt.plot([(i+1)/2 for i in range(sample)],speeds,c="b")


"""print("Obstacle 2")
for i in range(sample):
    d=gatherT("..")
    plt.plot([(i+1)/2 for i in range(2*2)],d,c="r")"""

plt.xlabel("Pressure setting")
plt.ylabel("Average magnitude size (px)")
plt.title("How speed affects the magnitude of vectors")
plt.show()

