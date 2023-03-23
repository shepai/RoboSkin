import cv2
import letRun #This library can be deleted, it is used for debugging
import RoboSkin as sk
import numpy as np

path= letRun.path
#videoFile=path+"Movement2.avi"
skin=sk.Skin()
frame=skin.getFrame()
old_T=skin.origin
new=np.zeros_like(frame)
past= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

 
while True:
    frame=skin.getFrame()
    im=skin.getBinary()
    #direction=np.zeros_like(frame)
    new=np.zeros_like(frame)
    t_=skin.getDots(im)
    old=skin.last.copy()
    t=skin.movement(t_)
    if t.shape[0]>2:
        #new[t[:,0],t[:,1]]=(255,0,0)
        #new[t_[:,0],t_[:,1]]=(0,0,255)
        #new[old_T[:,0],old_T[:,1]]=(255,0,255)
        for i,coord in enumerate(t):
            x1=coord[1]
            y1=coord[0]
            x2=old_T[i][1]
            y2=old_T[i][0]
            #cv2.putText(new,str(i),(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,0.2,(0,255,0))
            d=skin.euclid(np.array([x1, y1]), np.array([x2, y2]))
            if d<25:
                cv2.arrowedLine(new, (x2, y2), (x1, y1), (0, 255, 0), thickness=1)#
    frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask=skin.opticFlow(frame,past)
    past=frame.copy()
    cv2.imshow('flow',mask)
    cv2.imshow('flow vector',new)
    cv2.imshow('unprocessed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break