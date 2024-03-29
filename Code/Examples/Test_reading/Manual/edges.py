import cv2
import letRun #This library can be deleted, it is used for debugging
import RoboSkin as sk
import numpy as np
import matplotlib.pyplot as plt

path= letRun.path
#videoFile=path+"Movement4.avi"
skin=sk.Skin(device=0) #load skin object using demo video
frame=skin.getFrame()
old_T=skin.origin
new=np.zeros_like(frame) 

#uncomment to record video
"""p=np.concatenate((new,new),axis=1)
h, w = p.shape[:2]
out = cv2.VideoWriter('skinDIrection.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, (w,h))"""
def gatherT(skin,string,n=5):
    DATA=[]
    for i in range(n):
        print(string,i+1)
        input("record> ")
        im=skin.getBinary()
        new=np.zeros_like(frame)
        t_=skin.getDots(im)
        t=skin.movement(t_)
        #show user the imagesS
        DATA.append(t.copy())
        print(t.shape)
        #out.write(np.concatenate((sk,new),axis=1)) #uncomment to record video
    DATA=(np.array(DATA))
    return DATA
NUM=2
ar=np.zeros((3,NUM,5,2))
for j in range(NUM):
    DATA=gatherT(skin,"Flat")
    av=np.zeros((len(DATA),2))
    for i,dat in enumerate(DATA):
        av[i]=np.sum(old_T-dat,axis=0)/len(old_T-dat)
    weights = np.arange(1, len(DATA)+1)
    plt.scatter(av[:,0],av[:,1],c=weights, cmap='Purples',label="Central")
    ar[0][j]=av.copy()
    input("get off>")
    skin.reset()
    old_T=skin.origin
    print("ORIGIN>",old_T.shape)
    DATA=gatherT(skin,"EdgeL")
    av=np.zeros((len(DATA),2))
    for i,dat in enumerate(DATA):
        av[i]=np.sum(old_T-dat,axis=0)/len(old_T-dat)
    weights = np.arange(1, len(DATA)+1)
    plt.scatter(av[:,0],av[:,1],c=weights, cmap='Greens',label="Top")
    ar[1][j]=av.copy()
    input("get off>")
    skin.reset()
    old_T=skin.origin
    print("ORIGIN>",old_T.shape)
    DATA=gatherT(skin,"EdgeR")
    av=np.zeros((len(DATA),2))
    for i,dat in enumerate(DATA):
        av[i]=np.sum(old_T-dat,axis=0)/len(old_T-dat)
    weights = np.arange(1, len(DATA)+1)
    plt.scatter(av[:,0],av[:,1],c=weights, cmap='Reds',label="Bottom")
    ar[2][j]=av.copy()
    input("get off>")
    skin.reset()
    old_T=skin.origin
    print("ORIGIN>",old_T.shape)
    
np.save("/its/home/drs25/Documents/GitHub/RoboSkin/Code/Models/saved/Visualisation8",ar)

plt.colorbar()
plt.title("Scatter of average vector of different sensations for vectors")
plt.show()


skin.close()