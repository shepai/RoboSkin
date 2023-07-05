import cv2
import pickle
import letRun #This library can be deleted, it is used for debugging
import RoboSkin as sk
#from sklearn.linear_model import LinearRegression
import numpy as np

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
skin=sk.Skin(device=1)#videoFile=path+"push.avi" #load skin object using demo video
skin.sharpen=False #falsify the sharpness if recorded with sharpness
frame=skin.getFrame()
h=frame.shape[1]*SIZE
w=frame.shape[0]*SIZE
frame=cv2.resize(frame,(int(h),int(w)),interpolation=cv2.INTER_AREA)
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).flatten()/255
past=predict(reg,np.array([frame]))[0]
initial=past.copy()

#record
"""p=np.concatenate((skin.getFrame(),skin.getFrame()),axis=1)
h1, w1 = p.shape[:2]
out = cv2.VideoWriter('skinDIirectionReg.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, (w1,h1))"""

while True:
    frame_=skin.getFrame()
    frame=cv2.resize(frame_,(int(h),int(w)),interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).flatten()/255
    new=np.zeros_like(frame_)+255
    points=predict(reg,np.array([frame]))[0]
    av=[0,0]
    for j,_ in enumerate(points):
        cv2.putText(frame_,str(j),(int(_[0]),int(_[1])),cv2.FONT_HERSHEY_SIMPLEX,0.2,(255,0,0))
        #cv2.circle(frame_,(int(initial[j][0]),int(initial[j][1])),2,(0,0,255),3)
        #cv2.circle(frame_,(int(_[0]),int(_[1])),2,(250,0,0),4)
        #print((int(initial[j][0]),int(initial[j][1])),(int(_[0]),int(_[1])))
        cv2.arrowedLine(new,(int(initial[j][0]),int(initial[j][1])),(int(_[0]),int(_[1])), (0, 0, 0), thickness=2)
        av[0]+=_[0]
        av[1]+=_[1]
    #av[0]/=len(points)
    #av[1]/=len(points)
    #cv2.circle(new,(int(av[0]),int(av[1])),4,(0,255,0),4)
    cv2.imshow('Image', frame_)
    cv2.imshow('arrows', new)
    #cv2.imshow('sharp',sharp_image)
    q=cv2.waitKey(1) 
    if q & 0xFF == ord('q'):
        break
    #out.write(np.concatenate((frame_,new),axis=1)) #uncomment to record video
#out.release() #uncomment to record video