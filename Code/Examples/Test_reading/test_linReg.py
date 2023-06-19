import letRun #This library can be deleted, it is used for debugging
import RoboSkin as sk
import numpy as np
import cv2
from sklearn.linear_model import LinearRegression
import pickle

#################################################
#Get data
#################################################
path1="C:/Users/dexte/github/RoboSkin/Code/Models/saved/"
vecs=np.load(path1+"vectors2.npy")
data=np.load(path1+"pressures2.npy")

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

name=path1+"pickle_weight_model.pkl"
with open(name,'rb') as file:
    weight=pickle.load(file)
#################################################
#Skin reading and prediction
#################################################

path= letRun.path
#videoFile=path+"Movement4.avi"
skin=sk.Skin(videoFile=path+"Movement4.avi") #load skin object using demo video
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
    #print(v.shape,v.reshape(1,v.shape[0]).shape)
    p=reg.predict(v.reshape(1,v.shape[0]))
    print(p)
    print("PREDICTION:",weight.predict(np.array([p/10]))[0],"g")
    