import cv2
import letRun #This library can be deleted, it is used for debugging
import RoboSkin as sk
import numpy as np
import matplotlib.pyplot as plt

path= letRun.path
#videoFile=path+"Movement4.avi"
skin=sk.Skin(device=1) #load skin object using demo video
frame=skin.getFrame()
old_T=skin.origin
new=np.zeros_like(frame) 
print(old_T.shape)
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
        t_=skin.getDots(im)
        print(">>>",t_.shape)
        t=skin.movement(t_)
        #show user the imagesS
        DATA.append(t.copy())
        print(t.shape)
        #out.write(np.concatenate((sk,new),axis=1)) #uncomment to record video
    DATA=(np.array(DATA))
    return DATA

Objects=3
colours=["r","g","b","m"]
for j in range(Objects):
    if (j+1)%2==0: print("Slippery")
    input("start>>>")
    skin.reset()
    old_T=skin.origin
    print(old_T.shape)
    d=gatherT(skin,"test")
    av=np.zeros((len(d),2))
    for i,dat in enumerate(d):
        av[i]=np.sum(old_T-dat,axis=0)/len(old_T-dat)
    if j!=Objects-1:
        lab="Non Slippery"
        if (j+1)%2==0: lab="Slippery"
        plt.scatter(av[:,0],av[:,1],c=colours[j],label=lab)#"Meterial "+str(j+1)"
    else: plt.scatter(av[:,0],av[:,1],c=colours[j],label="Control ")
    
plt.title("Directions of sensor over slippery vs non slippery")
plt.xlabel("X position")
plt.ylabel("Y position")
plt.legend(loc="upper right")
plt.show()


skin.close()