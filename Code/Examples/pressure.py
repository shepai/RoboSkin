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

SPLIT=10


while(True):
    im=skin.getBinary()
    #show user the imagesS
    grid,NEW=skin.getForce(im,SPLIT)
    grid=grid.reshape(SPLIT**2,2)
    new=np.zeros_like(frame)
    new[grid[:,0],grid[:,1]]=(0,0,255)
    """for i in range(0,len(grid),3):
        for j in range(0,len(grid),3):
            d=skin.euclid(grid[i], grid[j])
            if i!=j and d<150:
                cv2.line(new, (grid[i][1],grid[i][0]), (grid[j][1],grid[j][0]), (0, 255, 0), thickness=1)"""
    push=skin.getSizeForce(im,SPLIT)
    tactile=np.zeros_like(new)
    tactile[:,:,2]=push
    tactile[:,:,0]=NEW
    cv2.imshow('spots', new)
    cv2.imshow('tactile', tactile)
    cv2.imshow('binary',im)
    cv2.imshow('unprocessed', skin.getFrame())
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
skin.close()
cv2.destroyAllWindows()
