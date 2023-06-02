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

def gatherT(string,n=10):
    DATA=[]
    past_Frame=skin.getBinary()
    image=np.zeros_like(past_Frame)
    #time.sleep(1)
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
NUM=3
n=10
classes=[]
for i in range(NUM):
    p["Meterial"+str(i+1)]=[]
classes= ["Trial "+str(i+1) for i in range(n)]

BIG=[]
for i in range(NUM):
    DATA=gatherT("Flat",n=n)
    DATA=DATA.reshape((len(DATA),DATA[0].flatten().shape[0]))
    BIG.append(np.sum(DATA,axis=1))

class_data=(np.array(BIG)).astype(np.int32)



trials = 3
#class_data = np.random.randint(1, 10, size=(trials, len(classes)))
#print(np.random.randint(1, 10, size=(trials, len(classes))),"\n",class_data)
# Set the bar width and positions
bar_width = 0.1
x = np.arange(trials)
width = 0.1  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')
bar_colors = ['tab:red', 'tab:blue', 'tab:orange']
# Plot the stacked bars
bottom = np.zeros(trials)
for i, cls in enumerate(classes):
    offset = width * multiplier
    ax.bar(x + offset,class_data[:, i], width, color=bar_colors,label=cls)
    bottom += class_data[:, i]
    multiplier += 1


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Summed value of pressure')
ax.set_title('Pressure applied from different surfaces')
ax.set_xticks(np.arange(3) + width, ["Meterial 1","Meterial 2","Meterial 3"])
#ax.legend(loc='upper left', ncols=3)
#ax.set_ylim(0, 100)

plt.show()
