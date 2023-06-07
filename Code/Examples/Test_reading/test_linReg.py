import letRun #This library can be deleted, it is used for debugging
import RoboSkin as sk
import numpy as np
import cv2
from sklearn.linear_model import LinearRegression

#################################################
#Get data
#################################################
path="C:/Users/dexte/github/RoboSkin/Code/Models/saved/"
vecs=np.load(path+"vectors2.npy")
data=np.load(path+"pressures2.npy")

classes=data[:,1][0]/200
def find_nearest(array, value): #return class
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

SIZE=-1
X_data=vecs[0:SIZE].reshape((len(vecs[0:SIZE])*vecs.shape[1],vecs.shape[2]*2))
y_data=data[:,1][0:SIZE].flatten()/200

reg = LinearRegression().fit(X_data, y_data)
print("Score on training",reg.score(X_data, y_data))

#################################################
#Skin reading and prediction
#################################################

skin=sk.Skin(device=1)

path= letRun.path
#videoFile=path+"Movement4.avi"
skin=sk.Skin(device=1) #load skin object using demo video
frame=skin.getFrame()
old_T=skin.origin

while(True):
    im=skin.getBinary()
    t_=skin.getDots(im)
    t=skin.movement(t_)
    v=np.zeros(t.shape)
    if t.shape[0]>2:
        for i,coord in enumerate(t): #show vectors of every point
            x1=coord[1]
            y1=coord[0]
            x2=old_T[i][1]
            y2=old_T[i][0]
            v[i] =np.array([x1-x2,y1-y2])

    p=reg.predict(np.array([v]))
    print("PREDICTION:",p)
    