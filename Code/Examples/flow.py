import cv2
import letRun #This library can be deleted, it is used for debugging
import RoboSkin as sk
import numpy as np

path="C:/Users/dexte/github/RoboSkin/Assets/Video demos/"
#videoFile=path+"Movement2.avi"
skin=sk.Skin(videoFile=path+"Movement4.avi")
frame=skin.getFrame()
old_T=skin.origin
new=np.zeros_like(frame)

while(True):
    im=skin.getBinary()
    #direction=np.zeros_like(frame)
    new=np.zeros_like(frame)
    t_=skin.getDots(im)
    t=skin.movement(t_)
    if t.shape[0]>2:
        new[t[:,0],t[:,1]]=(0,255,0)
        new[old_T[:,0],old_T[:,1]]=(0,0,255)
        for i,coord in enumerate(t):
            x1=coord[1]
            y1=coord[0]
            x2=old_T[i][1]
            y2=old_T[i][0]
            #cv2.putText(new,str(i),(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,0.2,(0,255,0))
            d=skin.euclid(np.array([x1, y1]), np.array([x2, y2]))
            cv2.arrowedLine(new, (x2, y2), (x1, y1), (0, 255, 0), thickness=1)#
            pass
    #show user the imagesS
    cv2.imshow('spots', new)
    cv2.imshow('our binary',im)
    #cv2.imshow('direcion',direction)
    cv2.imshow('unprocessed', skin.getFrame())
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
skin.close()
cv2.destroyAllWindows()