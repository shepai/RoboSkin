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
vecs=np.load(path1+"vectors.npy")
data=np.load(path1+"pressures.npy")

classes=data[:,1][0]/200
def find_nearest(array, value): #return class
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

SIZE=-1
X_data=vecs[0:SIZE].reshape((len(vecs[0:SIZE])*vecs.shape[1],vecs.shape[2]*2))
y_data=data[:,1][0:SIZE].flatten()/200

reg = LinearRegression(positive=True).fit(X_data, y_data)
print("Score on training",reg.score(X_data, y_data))

name=path1+"pickle_weight_model.pkl"
with open(name,'rb') as file:
    weight=pickle.load(file)
#################################################
#Skin reading and prediction
#################################################
name="C:/Users/dexte/OneDrive/Documents/AI/Data_Labeller/pickle_imputer.pkl"
reg1=None
with open(name,'rb') as file:
    reg1=pickle.load(file)
SIZE=0.3
def predict(reg1,dat):
    p=reg1.predict(dat)
    p=(p.reshape((p.shape[0],p.shape[1]//2,2))*255/SIZE)
    return p


path= letRun.path
skin=sk.Skin(videoFile=path+"push.avi")#videoFile=path+"push.avi") #load skin object using demo video
frame=skin.getFrame()
h=frame.shape[1]*SIZE
w=frame.shape[0]*SIZE

frame=cv2.resize(frame,(int(h),int(w)),interpolation=cv2.INTER_AREA)
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).flatten()/255
past=predict(reg1,np.array([frame]))[0]
initial=past.copy()

while(True):
    frame_=skin.getFrame()
    frame=cv2.resize(frame_,(int(h),int(w)),interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).flatten()/255
    new=np.zeros_like(frame_)+255
    points=predict(reg1,np.array([frame]))[0]
    av=[0,0]
    for j,_ in enumerate(points):
        cv2.circle(frame_,(int(_[0]),int(_[1])),2,(250,0,0),4)
    #print(v.shape,v.reshape(1,v.shape[0]).shape)
    vectors=(initial-points)
    p=reg.predict(vectors.reshape((1,vectors.flatten().shape[0])))*10
    val=weight.predict(np.array([p/100]))[0]
    if val<0: val=0
    print("PREDICTION:",val,"g")
    cv2.imshow('Image', frame_)
    q=cv2.waitKey(1) 
    if q & 0xFF == ord('q'):
        break