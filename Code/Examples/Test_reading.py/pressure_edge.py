import cv2
import letRun #This library can be deleted, it is used for debugging
import RoboSkin as sk
import numpy as np
import matplotlib.pyplot as plt
import time

path= letRun.path
skin=sk.Skin(device=0) #videoFile=path+"Movement4.avi"
frame=skin.getFrame()
old_T=skin.origin
new=np.zeros_like(frame)
SPLIT=5


#uncomment to record video
"""p=np.concatenate((new,new),axis=1)
h, w = p.shape[:2]
out = cv2.VideoWriter('skinDIrection.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, (w,h))"""
def gatherT(string,n=5):
    DATA=[]
    past_Frame=skin.getBinary()
    image=np.zeros_like(past_Frame)
    time.sleep(1)
    for i in range(n):
        print(string,i+1)
        input("record> ")
        image=None
        for i in range(10): #must loop through so image is not weird
            im=skin.getBinary()
            #image=skin.getForce(im,SPLIT,image=image,threshold=80,degrade=20,) #get the force push
            image,grid=skin.getForceGrid(im,SPLIT,image=image,threshold=10,degrade=20)
        DATA.append(image/(image.shape[0]*image.shape[1]))
        past_Frame=image.copy
    return np.array(DATA)


a=None
for i in range(5):
    DATA=gatherT("Flat")
    DATA=DATA.reshape((len(DATA),DATA[0].flatten().shape[0]))
    if type(a)==type(None): a=np.zeros((5,5,DATA[0].shape[0]))
    print(np.sum(DATA,axis=1))
    plt.plot([i+1 for i in range(len(DATA))],np.sum(DATA,axis=1),label="Trial "+str(i+1))

np.save("/its/home/drs25/Documents/GitHub/RoboSkin/Code/Models/saved/pressureSaved/pressures",a)
plt.legend(loc="upper left")
plt.title("Pressure applied to the TacTip over multiple trials")
plt.xlabel("Pressure setting")
plt.ylabel("Summed value of pressure image")
plt.show()

"""NUM=2
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


skin.close()"""