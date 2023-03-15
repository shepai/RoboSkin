import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

class Skin: #skin object for detecting movement
    def __init__(self,device=1,videoFile=""):
        if videoFile=="": self.cap = cv2.VideoCapture(device)
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
        if frame.shape[0]>500: #do not allow too large
            SF=480/frame.shape[0]
            frame = cv2.resize(frame, (int(frame.shape[1]*SF),480), interpolation = cv2.INTER_AREA)
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
    def loop_through(self,stored,used,looped,arrayA,arrayB,count,maxL=10):
        if count==maxL:
            return looped
        for i, eachPoint in enumerate(arrayB): #loop through distances and pair off
            distances=np.sqrt(np.sum(eachPoint-arrayA,axis=1)**2)
            min_dist=distances[np.argmin(distances)]
            ind=np.argmin(distances)
            if ind not in used: #make sure within parameters
                looped[i]=arrayA[ind]
                used.append(ind)
                stored[ind]=[min_dist,i]
            else: #if the index is already used
                if stored[ind][0]>min_dist: #something found better
                    j=stored[ind][1] #get old index
                    looped[i]=arrayA[ind] #set pointer
                    stored[ind]=[min_dist,i]
                    looped=self.loop_through(stored,used,looped,arrayA,arrayB,count+1,maxL=maxL)
        return looped
    def movement(self,new,MAXD=10):
        arrayA=new.copy()
        arrayB=self.last.copy()
        if len(new)>2: #check coords exist
            stored=np.zeros_like(arrayA+(2,))
            looped=np.zeros_like(arrayB)
            used=[]
            looped=self.loop_through(stored,used,looped,arrayA,arrayB,1)
            #TODO experiment with adding the unpicked lowest distances instead of the orginal point
            for i, eachPoint in enumerate(arrayB): #fill in gaps
                if np.sum(looped[i])==0:
                    looped[i]=eachPoint.copy()
            self.last=looped.copy()
            return looped
        else:
            return arrayB
    def getForce(self,im,gridSize,threshold=30):
        #get dots and cut up averages of squares to get overall force
        t=self.getDots(im)
        image=np.zeros_like(im)
        x=im.shape[1]
        y=im.shape[0]
        x_div=x//gridSize
        y_div=y//gridSize
        GRID=np.zeros((gridSize,gridSize,2))
        if len(t)>2:
            for c,i in enumerate(range(0,x,x_div)): #loop through grid space
                for d,j in enumerate(range(0,y,y_div)):
                    found=t[np.where(np.logical_and(t[:,0]>j,t[:,0]<j+y_div))]
                    found=found[np.where(np.logical_and(found[:,1]>i,found[:,1]<i+x_div))]  
                    o_found=self.origin[np.where(np.logical_and(self.origin[:,0]>j,self.origin[:,0]<j+y_div))]
                    o_found=o_found[np.where(np.logical_and(o_found[:,1]>i,o_found[:,1]<i+x_div))]  
                    if len(found)!=0:
                        mag=np.sum(found,axis=0)//len(found) #get magnitude of point
                        o_mag=np.sum(o_found,axis=0)//len(o_found) #get magnitude of origin
                        GRID[c][d]=mag
                        val=min(self.euclid(mag,o_mag)*2,255)
                        if val>threshold: image[j:j+y_div,i:i+x_div]=val #get intensity of movement
        return GRID.astype(int),image
        #t=self.movement(t_)
    def getSizeForce(self,im,gridSize,threshold=30):
        image=np.zeros_like(im)
        x=im.shape[1]
        y=im.shape[0]
        x_div=x//gridSize
        y_div=y//gridSize
        average=np.average(im)
        for c,i in enumerate(range(0,x,x_div)): #loop through grid space
            for d,j in enumerate(range(0,y,y_div)):
                    val=max(np.average(im[j:j+y_div,i:i+x_div])-average,0) 
                    if val>threshold:
                        image[j:j+y_div,i:i+x_div]=val#get intensity of movement
        return image
    def close(self):
        self.vid.release()

#C:/Users/dexte/github/Chaos-Robotics/Bio-inspired sensors/Tactip/Vid/Movement.mp4
#C:/Users/dexte/github/Chaos-Robotics/movement.avi
"""
path="C:/Users/dexte/github/RoboSkin/Assets/Video demos/"

skin=Skin(videoFile=path+"Movement3.avi") #videoFile=path+"Movement2.avi"
frame=skin.getFrame()
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
    
    for i in range(0,len(grid),3):
        for j in range(0,len(grid),3):
            d=skin.euclid(grid[i], grid[j])
            if i!=j and d<150:
                cv2.line(new, (grid[i][1],grid[i][0]), (grid[j][1],grid[j][0]), (0, 255, 0), thickness=1)
    push=skin.getSizeForce(im,SPLIT)
    tactile=np.zeros_like(new)
    tactile[:,:,2]=push
    tactile[:,:,0]=NEW
    cv2.imshow('spots', new)
    #cv2.imshow('pressure', NEW)
    #cv2.imshow('push',push)
    cv2.imshow('tactile',tactile)
    cv2.imshow('binary',im)
    cv2.imshow('unprocessed', skin.getFrame())
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
skin.close()
cv2.destroyAllWindows()
"""