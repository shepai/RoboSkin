import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

class Skin: #skin object for detecting movement
    def __init__(self,device=1,videoFile=""):
        if videoFile=="": self.cap = cv2.VideoCapture(self.vid)
        else: self.cap = cv2.VideoCapture(videoFile)
        self.vid=videoFile
        self.centre=np.array(self.getFrame().shape[0:-1])//2
        self.origin=self.zero()
        self.last=self.origin.copy()
        self.thetas=np.zeros_like(self.last) #store distances
    def getFrame(self): #get a frame from the camera
        ret, frame = self.cap.read()
        if not ret: #reopen
            self.cap.release()
            if self.vid=="": self.cap = cv2.VideoCapture(self.vid)
            else: self.cap = cv2.VideoCapture(self.vid)
            ret, frame = self.cap.read()
        return frame
    def adaptive(self,img,threshold=150): #get a threshold of pixels that maximizes the blobs
        frame=np.copy(img)
        sum=0
        while sum<21000000: #loop till large picel value found
            frame[frame>threshold]=255
            frame[frame<=threshold]=0
            sum=np.sum(frame)
            if sum>22000000:threshold+=1
            elif sum<22000000:threshold-=1
            frame=np.copy(img)
        frame[frame>threshold]=255
        frame[frame<=threshold]=0
        return frame
    def removeBlob(self,im,min_size = 270):
        nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(im)
        sizes = stats[:, -1]
        sizes = sizes[1:]
        nb_blobs -= 1
        # output image with only the kept components
        im_result = np.zeros_like(im_with_separated_blobs)
        im_result_with = np.zeros_like(im_with_separated_blobs)
        # for every component in the image, keep it only if it's above min_size
        for blob in range(nb_blobs):
            if sizes[blob] >= min_size:
                # see description of im_with_separated_blobs above
                im_result_with[im_with_separated_blobs == blob + 1] = 255#
            if sizes[blob] <= min_size:
                # see description of im_with_separated_blobs above
                im_result[im_with_separated_blobs == blob + 1] = 255#
        return im_result_with,im_result

    def get_processed(self,frame):
        #make gray and preprocess the binary threshold
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray=self.adaptive(frame_gray)
        #remove the blobs
        to_Show,spots=self.removeBlob(frame_gray)
        to_Show=to_Show.astype(np.uint8)
        spots=spots.astype(np.uint8)
        #replace parts
        inds=np.argwhere(to_Show == 255)
        frame[inds[:,0],inds[:,1]]=[80,80,80]
        inds=np.argwhere(spots == 255)
        frame[inds[:,0],inds[:,1]]=[255,255,255]
        return frame
    def getDots(self,gray):
        labels, nlabels = ndimage.label(gray)
        t = ndimage.center_of_mass(gray, labels, np.arange(nlabels) + 1 )
        t=np.array(t)
        t=np.rint(t).astype(int)
        temp=[]
        for i in range(len(t)):
            d=np.sqrt(np.sum((self.centre-t[i])**2))
            if d<150:
                temp.append(t[i])
        t=np.array(temp)
        return t
    def getBinary(self):
        image=self.getFrame()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary=self.adaptive(gray)
        to_Show,spots=self.removeBlob(binary)
        return spots.astype(np.uint8)
    def zero(self,iter=100): #get the best origional image 
        im=self.getBinary()
        old_T=self.getDots(im)
        max_t=[]
        for i in range(100):
            im=self.getBinary()
            t=self.getDots(im)
            if len(t)>len(max_t):
                max_t=t.copy()
        return max_t
    def euclid(self,a,b):
        return np.sqrt(np.sum((a-b)**2))
    def movement(self,new,MAXD=5):
        arrayA=new.copy()
        arrayB=self.last.copy()
        looped=np.zeros_like(arrayB)
        if len(new)>2: #check coords exiat
            used=[]
            for i, eachPoint in enumerate(arrayB): #loop through distances and pair off
                distances=np.sqrt(np.sum(eachPoint-arrayA,axis=1)**2)
                min_dist=np.argmin(distances)
                ind=np.argmin(distances)
                if distances[min_dist]<MAXD and ind not in used: #make sure within parameters
                    looped[i]=arrayA[ind]
                    used.append(ind)
            #TODO experiment with adding the unpicked lowest distances instead of the orginal point
            for i, eachPoint in enumerate(arrayB): #fill in gaps
                if np.sum(looped[i])==0:
                    looped[i]=eachPoint
            self.last=looped.copy()
            return looped
        else:
            return arrayB
    def close(self):
        self.vid.release()


skin=Skin(videoFile="C:/Users/dexte/github/Chaos-Robotics/movement.avi")
frame=skin.getFrame()
old_T=skin.origin
new=np.zeros_like(frame)

while(True):
    im=skin.getBinary()
    t_=skin.getDots(im)
    t=skin.movement(t_)
    if t_.shape[0]>2:
        new=np.zeros_like(frame)
        new[t[:,0],t[:,1]]=(0,255,0)
        #new[t_[:,0],t_[:,1]]=(0,0,255)
        new[old_T[:,0],old_T[:,1]]=(255,0,255)
        for i,coord in enumerate(t):
            x1=coord[1]
            y1=coord[0]
            x2=old_T[i][1]
            y2=old_T[i][0]
            #cv2.putText(new,str(i),(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,0.2,(0,255,0))
            d=skin.euclid(np.array([x1, y1]), np.array([x2, y2]))
            if d<25:
                cv2.arrowedLine(new, (x2, y2), (x1, y1), (0, 255, 0), thickness=1)
        #show user the imagesS
        cv2.imshow('spots', new)
        cv2.imshow('binary', im)
        cv2.imshow('unprocessed', skin.getFrame())
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
skin.close()
cv2.destroyAllWindows()