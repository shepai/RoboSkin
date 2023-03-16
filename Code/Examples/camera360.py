import cv2
import letRun #This library can be deleted, it is used for debugging
import RoboSkin as sk
import numpy as np

skin=sk.Skin(device=sk.camera360()) #videoFile=path+"Movement2.avi"
while(True):
    frame=skin.getFrame()
    cv2.imshow('unprocessed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
skin.close()
cv2.destroyAllWindows()