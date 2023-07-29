import letRun #This library can be deleted, it is used for debugging
import RoboSkin as sk
import numpy as np
import cv2
import pickle

NAME="edge_detection.avi"
path= letRun.path
skin=sk.Skin(videoFile=path+NAME)#videoFile=path+"Movement4.avi") #load skin object using demo video

cap = cv2.VideoCapture(path+NAME)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

SIZE=0.3
name="C:/Users/dexte/OneDrive/Documents/AI/Data_Labeller/pickle_imputer_small.pkl" #use standard imputer or one for small
reg=None
with open(name,'rb') as file:
    reg=pickle.load(file)

def predict(reg,dat):
    p=reg.predict(dat)
    p=(p.reshape((p.shape[0],p.shape[1]//2,2))*255/SIZE)
    return p

path= letRun.path
skin=sk.Skin(videoFile=path+"small1.avi")#videoFile=path+"push.avi" videoFile=path+"small1.avi" #load skin object using demo video
skin.sharpen=False #falsify the sharpness if recorded with sharpness
frame=skin.getFrame()
h=frame.shape[1]*SIZE
w=frame.shape[0]*SIZE
frame=cv2.resize(frame,(int(h),int(w)),interpolation=cv2.INTER_AREA)
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).flatten()/255
past=predict(reg,np.array([frame]))[0]
initial=past.copy()
initial_frame=frame.copy()

X=[]
y=[] #label as [edge surface soft hard slippery]
lastFrames=[]
STORE=5
for i in range(length): #lop through all
    frame_=skin.getFrame()
    frame=cv2.resize(frame_,(int(h),int(w)),interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).flatten()/255
    new=np.zeros_like(frame_)+255
    points=predict(reg,np.array([frame]))[0]
    #get pressure map
    diff=np.sum(np.abs(initial_frame-frame))/(frame.shape[0]*3)
    vecs=initial-points
    lastFrames.append(vecs)
    if len(lastFrames)>STORE: lastFrames.pop(0)
    if diff>0.01 and len(lastFrames)==STORE: #significant contact
        X.append(np.array(lastFrames)) #store temporal element
        y.append([0,1,0,1,0])
    
    
y=np.array(y)
X=np.array(X)
print(len(X),len(y))

np.save("C:/Users/dexte/github/RoboSkin/Code/Models/labeller/X_data_edge",X)
np.save("C:/Users/dexte/github/RoboSkin/Code/Models/labeller/y_data_edge",y)

skin.close()