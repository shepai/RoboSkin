import cv2
import letRun #This library can be deleted, it is used for debugging
import RoboSkin as sk
import numpy as np

path="C:/Users/dexte/github/RoboSkin/Assets/Video demos/"

skin=sk.Skin(videoFile=path+"Movement3.avi") #videoFile=path+"Movement2.avi"
frame=skin.getFrame()
print(frame.shape)
old_T=skin.origin
new=np.zeros_like(frame)

SPLIT=20

past_Frame=skin.getBinary()

while(True):
    frame=skin.getFrame()
    im=skin.getBinary()
    push=skin.getForce(im,past_Frame,SPLIT) #get the force push
    past_Frame=im.copy() #get last frame
    tactile=np.zeros_like(new)
    tactile[:,:,2]=push #show push in red
    #tactile[:,:,0]=NEW
    cv2.imshow('spots', new)
    cv2.imshow('tactile', tactile)
    cv2.imshow('our binary',im)
    cv2.imshow('unprocessed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
skin.close()
cv2.destroyAllWindows()
