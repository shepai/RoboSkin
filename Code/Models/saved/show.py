import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

f1=np.load("/its/home/drs25/Documents/GitHub/RoboSkin/Code/Models/saved/Visualisation1.npy")
f2=np.load("/its/home/drs25/Documents/GitHub/RoboSkin/Code/Models/saved/Visualisation2.npy")
f3=np.load("/its/home/drs25/Documents/GitHub/RoboSkin/Code/Models/saved/Visualisation3.npy")
f4=np.load("/its/home/drs25/Documents/GitHub/RoboSkin/Code/Models/saved/Visualisation4.npy")
f5=np.load("/its/home/drs25/Documents/GitHub/RoboSkin/Code/Models/saved/Visualisation5.npy")
f6=np.load("/its/home/drs25/Documents/GitHub/RoboSkin/Code/Models/saved/Visualisation6.npy")
f7=np.load("/its/home/drs25/Documents/GitHub/RoboSkin/Code/Models/saved/Visualisation7.npy")
f8=np.load("/its/home/drs25/Documents/GitHub/RoboSkin/Code/Models/saved/Visualisation8.npy")

fs=[f1,f2,f3,f4,f5,f6,f7,f8]
#ar=np.zeros((3,NUM,5,2))

#plot flat
colours=["Purples","Greens","Reds"]
for j in range(len(f1)):
    for k in range(len(fs)):
        for i in range(len(fs[k][j])):
            if j!=0:
                weights = np.arange(1, len(fs[k][j][i][:,1])+1)
                plt.scatter(fs[k][j][i][:,0],fs[k][j][i][:,1],c=weights, cmap=colours[j])
    
#plt.scatter(fs[0][0][0][:,0][0],fs[0][0][0][:,1][0],c=1,cmap="Purples",label="Top")
#plt.colorbar()
#plt.clim([-1,1])
plt.scatter(fs[0][1][0][:,0][0],fs[0][1][0][:,1][0],c=1,cmap="Greens",label="Left")
plt.colorbar()
plt.clim([-1,1])
plt.scatter(fs[0][2][0][:,0][0],fs[0][2][0][:,1][0],c=1,cmap="Reds",label="Right")
plt.colorbar()
plt.clim([-1,1])
plt.legend(loc="upper right")
plt.xlabel("vector x position")
plt.ylabel("vector y position")
plt.title("Linearly seperable left and right sensations")
plt.show()