import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt


class camera360: #class for interfacing with a homemade 360 camera
    def __init__(self,device1=1,device2=2,ignore=False):
        if not ignore:
            self.cap1 = cv2.VideoCapture(device1)
            self.cap2 = cv2.VideoCapture(device2)
    def read(self):
        reta, frame1 = self.cap1.read()
        retb, frame2 = self.cap2.read()
        if reta and retb:
            im=np.concatenate((frame1,frame2),axis=1)
            return True,im
        return False, None
    def release(self):
        self.cap1.release()
        self.cap2.release()
        
class Skin: #skin object for detecting movement
    def __init__(self,device=1,videoFile=""):
        #set up correct camera/ video component
        if type(device)==type(camera360(ignore=True)): self.cap=device
        elif videoFile=="": self.cap = cv2.VideoCapture(device)
        else: self.cap = cv2.VideoCapture(videoFile)
        
        self.vid=videoFile
        self.centre=np.array(self.getFrame().shape[0:-1])//2
        self.origin=self.zero()
        self.last=self.origin.copy()
        self.thetas=np.zeros_like(self.last) #store distances
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
    def zero(self,iter=50): #get the best origional image 
        im=self.getBinary()
        old_T=self.getDots(im)
        max_t=[]
        for i in range(100):
            im=self.getBinary()
            t=self.getDots(im)
            if len(t)>len(max_t):
                max_t=t.copy()
        return max_t
    def euclid(self,a,b,axis=None):
        return np.linalg.norm(a-b,axis=axis)
    def loop_through(self,stored,used,looped,arrayA,arrayB,count,maxL=10):
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
    def getVectors(self,current,old):
        #t_=self.getDots(current)
        kernel = np.ones((5, 5), np.uint8)
        im = cv2.erode(cv2.absdiff(current,old), kernel) 
        t_=self.getDots(im)
        return self.movement(t_)
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
            if self.noiseRed(arrayB,looped)<0.028: #if insignificant movement
                return arrayB
                #self.last=self.origin.copy()
            #self.last=looped.copy()
            return looped
        else:
            return arrayB
    def opticFlow(self,past,next):
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
            #mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            #mask = cv.arrowedLine(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            if i<len(color):
                #frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
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
        average=self.euclid(t1,t2)/len(t1)
        std=np.sqrt((self.euclid(t1**2,t2**2)/len(t1))-(average**2))
        return average/max(std,0.1)
    def getForce(self,im,past,gridSize,threshold=40):
        im=cv2.absdiff(im,past)
        image=np.zeros_like(im)
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
                        image[j:j+y_div,i:i+x_div]=val#get intensity of movement
                    if val!=0: #calculate average of filled in points
                        average+=val
                        num+=1
        image=image-(average//num) #subtract the lisght average
        image[image<0]=0
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
"""