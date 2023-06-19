import letRun #This library can be deleted, it is used for debugging
import RoboSkin as sk
import numpy as np
import cv2

path= letRun.path
skin=sk.Skin(device=1)#videoFile=path+"Movement4.avi") #load skin object using demo video

p=skin.getFrame()
h1, w1 = p.shape[:2]
out = cv2.VideoWriter('flatSkin.avi',cv2.VideoWriter_fourcc(*'DIVX'), 60, (w1,h1))

while True:
    frame=skin.getFrame()
    cv2.imshow('frame', frame)
    
    q=cv2.waitKey(1) 
    if q & 0xFF == ord('q'):
        break
    out.write(frame) #uncomment to record video

out.release()