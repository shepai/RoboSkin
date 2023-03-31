"""
This library is for interfacing with tactile sensors such as the TacTip. 
You can use the dataset in Assets/Video demos or you can use a physical tactile sensor with a usb webcam. 

Code by Dexter R. Shepherd
PhD student at the University of Sussex
https://profiles.sussex.ac.uk/p493975-dexter-shepherd

"""

import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import copy

class camera360: #class for interfacing with a homemade 360 camera
    def __init__(self,device1=1,device2=2,ignore=False):
        """
        @param device1 id for camera 1
        @param device2 id for camera 2
        @param ignore stops attempt of connection when checking if the camera is real or not
        """
        if not ignore:
            self.cap1 = cv2.VideoCapture(device1)
            self.cap2 = cv2.VideoCapture(device2)
    def read(self):
        """
        read the devices and stitch them together
        """
        reta, frame1 = self.cap1.read()
        retb, frame2 = self.cap2.read()
        if reta and retb:
            im=np.concatenate((frame1,frame2),axis=1)
            return True,im
        return False, None
    def release(self):
        """
        release the devices
        """
        self.cap1.release()
        self.cap2.release()
        
class Skin: #skin object for detecting movement
    def __init__(self,device=1,videoFile=""):
        """
        set up correct camera/ video component
        set up all variables for optic flow and establish a baseline
        @param device selects which camera to take from
        @param videoFile points to a file to use instead of the camera
        """
        if type(device)==type(camera360(ignore=True)): self.cap=device
        elif videoFile=="": self.cap = cv2.VideoCapture(device)
        else: self.cap = cv2.VideoCapture(videoFile)
        
        self.vid=videoFile #save viceo file
        self.centre=np.array(self.getFrame().shape[0:-1])//2 #get centre of frame
        self.startIm=None
        self.origin=self.zero() #establish baseline
        self.last=self.origin.copy()
        self.thetas=np.zeros_like(self.last) #store distances
        #optical flow parameters:
        feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.1,
                       minDistance = 7,
                       blockSize = 7 )
        # Parameters for lucas kanade optical flow
        self.lk_params = dict( winSize  = (15, 15),
                  maxLevel = 10,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.1))
        past=cv2.cvtColor(self.getFrame(), cv2.COLOR_BGR2GRAY)
        self.begin=[]
        self.good_new=[]
        self.good_old=[]
        self.p0 = cv2.goodFeaturesToTrack(past, mask = None, **feature_params)
    def getFrame(self):
        """
        get a frame from the camera
        """
        ret, frame = self.cap.read()
        if not ret: #reopen if closed
            self.cap.release()
            if self.vid=="": self.cap = cv2.VideoCapture(self.vid)
            else: self.cap = cv2.VideoCapture(self.vid)
            ret, frame = self.cap.read()
            assert ret==True,"Error, camera or video file not existant"
        if frame.shape[0]>500: #do not allow too large
            SF=480/frame.shape[0]
            frame = cv2.resize(frame, (int(frame.shape[1]*SF),480), interpolation = cv2.INTER_AREA)
        return frame
    def adaptive(self,img,threshold=150):
        """
        get a threshold of pixels that maximizes the blobs
        @param img
        @param threshold starts the binary threshold at this value and increases or decreases
        """
        frame=np.copy(img)
        sum=0
        while sum<22000000: #loop till large picel value found
            frame[frame>threshold]=255
            frame[frame<=threshold]=0
            sum=np.sum(frame)
            if sum>22000000:threshold+=1
            elif sum<22000000:threshold-=1
            frame=np.copy(img)
        frame[frame>threshold]=255
        frame[frame<=threshold]=0
        return frame
    def removeBlob(self,im,min_size = 300):
        """
        @param im
        @param min_size defins the minimum size of the blobs otherwise delete them
        """
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
        """
        make gray and preprocess the binary threshold
        @param frame
        """
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
    def getDots(self,gray,size=150,centre=None):
        """
        return the centroud positions of all the points within the binary image
        @param gray
        """
        if centre==None:centre=self.centre
        labels, nlabels = ndimage.label(gray)
        t = ndimage.center_of_mass(gray, labels, np.arange(nlabels) + 1 )
        t=np.array(t)
        t=np.rint(t).astype(int)
        temp=[]
        for i in range(len(t)):
            if type(centre)!=type(0):
                d=np.sqrt(np.sum((self.centre-t[i])**2))
                if d<size:
                    temp.append(t[i])
            else:
                temp.append(t[i])
        t=np.array(temp)
        return t
    def getBinary(self):
        """
        get the binary image from the sensor and return only the white dots (filter out glare)
        """
        image=self.getFrame()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary=self.adaptive(gray)
        to_Show,spots=self.removeBlob(binary)
        return spots.astype(np.uint8)
    def zero(self,iter=50):
        """
        get the best original image by maximizing the number of pixels
        @param iter is how many iterations it will make an average on
        """
        im=self.getBinary()
        old_T=self.getDots(im)
        max_t=[]
        for i in range(iter):
            im=self.getBinary()
            t=self.getDots(im)
            if len(t)>len(max_t):
                max_t=t.copy()
                self.startIm=im.copy()
        return max_t
    def euclid(self,a,b,axis=None):
        """
        calculate the euclid distance
        @param a is vectors a
        @param b is vectors b
        @axis is the axis you wish to calculate the distance on
        """
        return np.linalg.norm(a-b,axis=axis)
    def loop_through(self,stored,used,looped,arrayA,arrayB,count,maxL=10):
        """
        Recursive method to find the closest points
        @param stored is the items that have been visited but can be again
        @param used is the already used indicies
        @param looped is the big array of point positions
        @param arrayA is the incoming points
        @param arrayB is the og points
        @param count is a counter to preent infinite recusion 
        @param maxL is the maximum level it will search
        """
        if count==maxL:
            return looped
        for i, eachPoint in enumerate(arrayB): #loop through distances and pair off
            distances=self.euclid(eachPoint,arrayA,axis=1)#np.sqrt(np.sum(eachPoint-arrayA,axis=1)**2)
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
    def movement(self,new,referenceArray=None,MAXD=10):
        """
        detect the movement of points using the new image
        @param new
        @param maxD is the maxdepth to search
        """
        if type(referenceArray)==type(None): referenceArray=self.last.copy()
        arrayA=new.copy()
        arrayB=referenceArray.copy()
        if len(new)>2: #check coords exist
            stored=np.zeros_like(arrayA+(2,))
            looped=np.zeros_like(arrayB)
            used=[]
            looped=self.loop_through(stored,used,looped,arrayA,arrayB,1,maxL=MAXD)
            #TODO experiment with adding the unpicked lowest distances instead of the orginal point
            for i, eachPoint in enumerate(arrayB): #fill in gaps
                if np.sum(looped[i])==0:
                    looped[i]=eachPoint.copy()
            if self.noiseRed(arrayB,looped)<0.028: #if insignificant movement
                return arrayB
                #self.last=self.origin.copy()
            #self.last=looped.copy()
            return looped
        else:
            return arrayB
    def opticFlow(self,past,next):
        """
        Calculate the optic flow image
        @param past image
        @param next image
        """
        color = np.random.randint(0, 255, (100, 3))
        p1, st, err = cv2.calcOpticalFlowPyrLK(past, next, self.p0, None, **self.lk_params)
        started=False
        if p1 is not None:
            self.good_new = p1[st==1]
            self.good_old = self.p0[st==1]
        if len(self.begin)==0: #has not been chosen
            started=True
        if len(self.begin)!=len(self.good_new):
            self.begin=[]
        started=True
        mask=np.zeros_like(past)
        for i, (new, old) in enumerate(zip(self.good_new, self.good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            if i<len(color):
                if started:
                    self.begin.append([c, d])
        for i in range(len(self.begin)):
            if i<len(self.begin) and i<len(self.good_new):
                vy=self.begin[i][1]-self.good_new[i][1]
                vx=self.begin[i][0]-self.good_new[i][0]
                #vectors.append([vx,vy])
                mask = cv2.arrowedLine(mask, (int(self.begin[i][0]), int(self.begin[i][1])), (int(self.good_new[i][0]), int(self.good_new[i][1])), color[i].tolist(), 2)
        self.p0 = self.good_new.reshape(-1, 1, 2)
        return mask
    def noiseRed(self,t1,t2):
        """
        Check whether there is noise between points by ratio of average to std
        @param t1 points
        @param t2 points
        """
        average=self.euclid(t1,t2)/len(t1)
        std=np.sqrt((self.euclid(t1**2,t2**2)/len(t1))-(average**2))
        return average/max(std,0.1)
    def getForce(self,im,past,gridSize,threshold=50,image=None,degrade=1):
        """
        Get the total force acting on different areas
        @param im image
        @param past image
        @param gridSize is how many segments to device the image (gridSizexgridSize)
        @param threshold ignores anything below
        """
        im=cv2.absdiff(im,past) #get the difference between images
        if type(image)==type(None): image=np.zeros_like(im)
        else: 
            image=image-degrade
            image[image<0]=0
        x=im.shape[1]
        y=im.shape[0]
        x_div=x//gridSize
        y_div=y//gridSize
        average=0
        num=0
        for c,i in enumerate(range(0,x,x_div)): #loop through grid space
            for d,j in enumerate(range(0,y,y_div)):
                    val=np.average(im[j:j+y_div,i:i+x_div])
                    if val>threshold:
                        image[j:j+y_div,i:i+x_div]=val+150#get intensity of movement
                    if val!=0: #calculate average of filled in points
                        average+=val
                        num+=1
        if num!=0:
            image=image-(average//num) #subtract the light average
        image[image<0]=0
        return image
    def close(self):
        """
        close the camera/s
        """
        self.cap.release()

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

    def getForceA(self,im,gridSize,threshold=30):
        #get dots and cut up averages of squares to get overall force
        image=np.zeros_like(im)
        x=im.shape[1]
        y=im.shape[0]
        x_div=x//gridSize
        y_div=y//gridSize
        GRID=np.zeros((gridSize,gridSize,1))
        av=0
        co=0
        for c,i in enumerate(range(0,x,x_div)): #loop through grid space
            for d,j in enumerate(range(0,y,y_div)):
                croppedN=im[j:j+y_div,i:i+x_div]
                croppedO=self.startIm[j:j+y_div,i:i+x_div]
                t1=self.getDots(croppedO,centre=0)
                t2=self.getDots(croppedN,centre=0)
                t2=self.movement(t2,referenceArray=t1)
                if len(t1)>2:
                    dist=np.sum(self.euclid(t1,t2,axis=0))*10
                    GRID[c][d]=dist
                    #image[j:j+y_div,i:i+x_div]=dist
                    av+=dist
                    if dist>0: co+=1
        sub=copy.deepcopy(GRID.flatten())
        inds=np.zeros_like(sub)
        shapesotre=GRID.shape
        GRID=GRID.flatten()
        GRID=GRID*0
        n=255//len([inds>0])
        for i in range(len(inds)//2):
            val=np.argmax(sub)
            if sub[val]>0: GRID[val]=max(255-(n*i),30)
            sub=np.delete(sub,val)
        GRID=GRID.reshape(shapesotre)
        for c,i in enumerate(range(0,x,x_div)): #loop through grid space
            for d,j in enumerate(range(0,y,y_div)):
                image[j:j+y_div,i:i+x_div]=GRID[c][d]
        #image=image-(av//co)
        #image[image<0]=0
        return image
"""