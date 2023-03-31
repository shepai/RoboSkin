import cv2
import letRun #This library can be deleted, it is used for debugging
import RoboSkin as sk
import numpy as np

#path="C:/Users/dexte/github/RoboSkin/Assets/Video demos/"
path= letRun.path
skin=sk.Skin(videoFile=path+"Movement4.avi") #videoFile=path+"Movement4.avi"
frame=skin.getFrame()
old_T=skin.origin
new=np.zeros_like(frame)
SPLIT=25
past_Frame=skin.getBinary()
image=np.zeros_like(past_Frame)

#uncomment to record video
"""p=np.concatenate((new,new),axis=1)
h, w = p.shape[:2]
out = cv2.VideoWriter('skinPressure.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, (w,h))"""

while(True):
    frame=skin.getFrame()
    im=skin.getBinary()
    image=skin.getForce(im,past_Frame,SPLIT,image=image,degrade=20) #get the force push
    past_Frame=im.copy() #get last frame
    tactile=np.zeros_like(new)
    tactile[:,:,2]=image #show push in red
    #tactile[:,:,0]=NEW
    cv2.imshow('tactile', tactile)
    cv2.imshow('our binary',im)
    cv2.imshow('unprocessed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    o#ut.write(np.concatenate((frame,tactile),axis=1)) #uncomment to record video
skin.close()
cv2.destroyAllWindows()
#out.release() #uncomment to record video