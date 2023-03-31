import cv2
import letRun #This library can be deleted, it is used for debugging
import RoboSkin as sk
import numpy as np

path= letRun.path
#videoFile=path+"Movement2.avi"
skin=sk.Skin(videoFile=path+"Movement4.avi") #load skin object using demo video
frame=skin.getFrame()
old_T=skin.origin
new=np.zeros_like(frame) 

#uncomment to record video
"""p=np.concatenate((new,new),axis=1)
h, w = p.shape[:2]
out = cv2.VideoWriter('skinDIrection.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, (w,h))"""

while(True):
    im=skin.getBinary()
    new=np.zeros_like(frame)
    t_=skin.getDots(im)
    t=skin.movement(t_)
    if t.shape[0]>2:
        new[t[:,0],t[:,1]]=(0,255,0)
        new[old_T[:,0],old_T[:,1]]=(0,0,255)
        for i,coord in enumerate(t): #show vectors of every point
            x1=coord[1]
            y1=coord[0]
            x2=old_T[i][1]
            y2=old_T[i][0]
            #cv2.putText(new,str(i),(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,0.2,(0,255,0))
            #d=skin.euclid(np.array([x1, y1]), np.array([x2, y2]))
            cv2.arrowedLine(new, (x2, y2), (x1, y1), (0, 255, 0), thickness=1)#
    #show user the imagesS
    sk=skin.getFrame()
    cv2.imshow('spots', new)
    cv2.imshow('our binary',im)
    cv2.imshow('unprocessed', sk)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #out.write(np.concatenate((sk,new),axis=1)) #uncomment to record video
skin.close()
cv2.destroyAllWindows()
#out.release() #uncomment to record video