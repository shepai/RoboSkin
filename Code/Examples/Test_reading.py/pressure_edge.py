import cv2
import letRun #This library can be deleted, it is used for debugging
import RoboSkin as sk
import numpy as np
import matplotlib.pyplot as plt

path= letRun.path
skin=sk.Skin(device=0) #videoFile=path+"Movement4.avi"
frame=skin.getFrame()
old_T=skin.origin
new=np.zeros_like(frame)
SPLIT=25
past_Frame=skin.getBinary()
image=np.zeros_like(past_Frame)


#uncomment to record video
"""p=np.concatenate((new,new),axis=1)
h, w = p.shape[:2]
out = cv2.VideoWriter('skinDIrection.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, (w,h))"""
def gatherT(string,n=5):
    DATA=[]
    for i in range(n):
        print(string,i+1)
        input("record> ")
        frame=skin.getFrame()
        im=skin.getBinary()
        #image=skin.getForce(im,SPLIT,image=image,threshold=80,degrade=20,) #get the force push
        im,grid=skin.getForceGrid(im,SPLIT,image=image,threshold=80,degrade=20,)
        DATA.append(grid.copy())
    return DATA
NUM=2
ar=np.zeros((3,NUM,5,2))
for j in range(NUM):
    DATA=gatherT("Flat")
    av=np.zeros((len(DATA),2))
    for i,dat in enumerate(DATA):
        av[i]=np.unravel_index(dat.argmax(), dat.shape)
    weights = np.arange(1, len(DATA)+1)
    plt.scatter(av[:,0],av[:,1],c=weights, cmap='Purples',label="Central")
    ar[0][j]=av.copy()
    input("get off>")
    skin.origin=skin.zero()
    DATA=gatherT("EdgeL")
    av=np.zeros((len(DATA),2))
    for i,dat in enumerate(DATA):
        av[i]=np.unravel_index(dat.argmax(), dat.shape)
    weights = np.arange(1, len(DATA)+1)
    plt.scatter(av[:,0],av[:,1],c=weights, cmap='Greens',label="Top")
    ar[1][j]=av.copy()
    input("get off>")
    skin.origin=skin.zero()

    DATA=gatherT("EdgeR")
    av=np.zeros((len(DATA),2))
    for i,dat in enumerate(DATA):
        av[i]=np.unravel_index(dat.argmax(), dat.shape)
    weights = np.arange(1, len(DATA)+1)
    plt.scatter(av[:,0],av[:,1],c=weights, cmap='Reds',label="Bottom")
    ar[2][j]=av.copy()
    input("get off>")
    skin.origin=skin.zero()

np.save("Visualisation_samepressure",ar)

plt.colorbar()
plt.title("Scatter of average vector of different sensations for vectors")
plt.show()


skin.close()