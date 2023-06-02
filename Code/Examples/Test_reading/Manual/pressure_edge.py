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
        grid=None
        for i in range(10): #must loop through so image is not weird
            im=skin.getBinary()
            #image=skin.getForce(im,SPLIT,image=image,threshold=80,degrade=20,) #get the force push
            image,grid=skin.getForceGrid(im,SPLIT,image=image,threshold=10,degrade=20)
        DATA.append(grid)
        past_Frame=image.copy
    return np.array(DATA)

p={}
for i in range(SPLIT):
    for j in range(SPLIT):
        p["field "+str(i+1)+","+str(j+1)]=[]

NUM=3
a=np.zeros((5,SPLIT**2))
for i in range(NUM):
    DATA=gatherT("Flat")
    DATA=DATA.reshape((len(DATA),DATA[0].flatten().shape[0]))
    a+=DATA

a=a/NUM
for j in range(len(a)):
    for i,key in enumerate(p):
        ar=p[key]
        ar.append(a[j][i])

#purge receptive fields of blank
p_=p.copy()
for key in p_:
    if sum(p_[key])==0: 
        del p[key]
    
classes= ("Pressure 1", "Pressure 2", "Pressure 3","Pressure 4","Pressure 5")


x = np.arange(len(classes))  # the label locations
width = 0.1  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in p.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    #ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Summed value of pressure')
ax.set_title('Pressure applied to the TacTip on receptive fields')
ax.set_xticks(x + width, classes)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, 100)

plt.show()

"""
np.save("/its/home/drs25/Documents/GitHub/RoboSkin/Code/Models/saved/pressureSaved/pressures1",a)
plt.legend(loc="upper left")
plt.title("Pressure applied to the TacTip over multiple trials")
plt.xlabel("Pressure setting")
plt.ylabel("Summed value of pressure image")
plt.show()
"""

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