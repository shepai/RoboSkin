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
    def __init__(self,device=1,videoFile="",imageFile=""):
        """
        set up correct camera/ video component
        set up all variables for optic flow and establish a baseline
        @param device selects which camera to take from
        @param videoFile points to a file to use instead of the camera
        @param imageFile is if we are using the digitip from the simulation
        """
        self.MAX=22000000
        if type(device)==type(camera360(ignore=True)): self.cap=device
        elif type(imageFile)!=type(""): 
            self.cap=imageFile
            self.MAX=1000
        elif videoFile=="": self.cap = cv2.VideoCapture(device)
        else: self.cap = cv2.VideoCapture(videoFile)
        self.sharpen=False
        self.imF=copy.deepcopy(imageFile)
        self.vid=videoFile #save viceo file
        self.centre=np.array(self.getFrame().shape[0:-1])//2 #get centre of frame
        self.startIm=None
        self.init_im=self.getBinary()
        #self.init_im=self.init_im//3
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
    def setImage(self,image):
        """
        set the image for the digi tip
        @param image
        """
        
        self.imF=copy.deepcopy(image)
    def getFrame(self):
        """
        get a frame from the camera
        """
        if type(self.imF)!=type(""): return self.imF#if we are using the image frame we will set this outside the loop
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
        lab= cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        # Applying CLAHE to L-channel
        # feel free to try different values for the limit and grid size:
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        cl = clahe.apply(l_channel)

        # merge the CLAHE enhanced L-channel with the a and b channel
        limg = cv2.merge((cl,a,b))

        # Converting image from LAB Color model to BGR color spcae
        frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        if self.sharpen:
            
            sharpen_filter=np.array([[-1,-1,-1],
                    [-1,9,-1],
                    [-1,-1,-1]])
            # applying kernels to the input image to get the sharpened image
            frame=cv2.filter2D(frame,-1,sharpen_filter)
        return frame
    def adaptive(self,img,threshold=100):
        """
        get a threshold of pixels that maximizes the blobs
        @param img
        @param threshold starts the binary threshold at this value and increases or decreases
        """
        frame=np.copy(img)
        sum=0
        MAXITER=1000
        i=0
        while sum<self.MAX and i<MAXITER: #loop till large picel value found
            frame[frame>threshold]=255
            frame[frame<=threshold]=0
            sum=np.sum(frame)
            if sum>self.MAX:threshold+=1
            elif sum<self.MAX:threshold-=1
            frame=np.copy(img)
            i+=1
        frame[frame>threshold]=255
        frame[frame<=threshold]=0
        return frame
    def removeBlob(self,im,min_size = 200):
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
    def getBinary(self,min_size = 300,adaptive=False):
        """
        get the binary image from the sensor and return only the white dots (filter out glare)
        """
        image=self.getFrame()
        gray=image
        if len(image.shape)==3: gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255,
	cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51,1)
        kernel = np.ones((2,2),np.uint8)
        binary = cv2.erode(binary,kernel,iterations = 2)
        if adaptive: binary=self.adaptive(gray)
        if type(self.imF)!=type(""): return binary
        to_Show,spots=self.removeBlob(binary,min_size = min_size)
        return spots.astype(np.uint8)
    def zero(self,iter=50):
        """
        get the best original image by maximizing the number of pixels
        @param iter is how many iterations it will make an average on
        """
        im=self.getBinary()
        #old_T=self.getDots(im)
        moving_av=[]
        max_t=[]#[[0,0] for i in range(300)]
        for i in range(iter):
            im=self.getBinary()
            t=self.getDots(im)
            moving_av.append(len(t))
            cur_av=sum(moving_av)/len(moving_av)
            if cur_av-len(t)<cur_av-len(max_t): #get closest to average
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
    def loop_through(self,stored,used,looped,arrayA,arrayB,count,maxL=20,dist=12):
        """
        Recursive method to find the closest points
        @param stored is the items that have been visited but can be again
        @param used is the already used indicies
        @param looped is the big array of point positions
        @param arrayA is the incoming points
        @param arrayB is the og points
        @param count is a counter to preent infinite recursion 
        @param maxL is the maximum level it will search
        @param dist is the parameter that removes shapes over a crtain distance
        """
        if count==maxL:
            return looped
        for i, eachPoint in enumerate(arrayB): #loop through distances and pair off
            distances=self.euclid(eachPoint,arrayA,axis=1)#np.sqrt(np.sum(eachPoint-arrayA,axis=1)**2)
            min_dist=distances[np.argmin(distances)]
            ind=np.argmin(distances)
            if ind not in used and distances[ind]<dist: #make sure within parameters
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
    def movement(self,new,referenceArray=None,MAXD=8):
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
            if self.noiseRed(self.origin,looped): #if insignificant movement
                self.last=self.origin.copy()
                return arrayB
                #self.last=self.origin.copy()
            self.last=looped.copy()
            return looped
        else:
            return arrayB
    def reset(self):
        self.origin=self.zero()
        self.last=self.origin.copy()
        self.init_im=self.getBinary()

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
        all=self.euclid(t1,t2,axis=1)
        average=self.euclid(t1,t2)/len(t1)
       
        std=np.sqrt((self.euclid(t1**2,t2**2)/len(t1))-(average**2))

        a=all[all>average+std]
        return False if len(a)<1 else True
    def getForce(self,im,gridSize,threshold=50,image=None,degrade=1,past=None):
        """
        Get the total force acting on different areas
        @param im image
        @param past image
        @param gridSize is how many segments to device the image (gridSizexgridSize)
        @param threshold ignores anything below
        @param past either provides the past image or a new one
        """
        if type(past)==type(None):
            past=self.init_im
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
                        image[j:j+y_div,i:i+x_div]=val*2#get intensity of movement
                    if val!=0: #calculate average of filled in points
                        average+=val
                        num+=1
        if num!=0:
            image=image-(average//num) #subtract the light average
        image[image<0]=0
        image[image>255]=255
        return image
    def getForceGrid(self,im,gridSize,threshold=50,image=None,degrade=1,past=None):
        """
        Get the total force acting on different areas return a nxn grid
        @param im image
        @param past image
        @param gridSize is how many segments to device the image (gridSizexgridSize)
        @param threshold ignores anything below
        @param past either provides the past image or a new one
        """
        if type(past)==type(None):
            past=self.init_im
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
        grid=np.zeros((gridSize,gridSize))
        for c,i in enumerate(range(0,x,x_div)): #loop through grid space
            for d,j in enumerate(range(0,y,y_div)):
                    val=np.average(im[j:j+y_div,i:i+x_div])
                    if val>threshold:
                        image[j:j+y_div,i:i+x_div]=val+50#get intensity of movement
                        grid[c][d]=val
                    if val!=0: #calculate average of filled in points
                        average+=val
                        num+=1
        if num!=0:
            image=image-(average//num) #subtract the light average
        image[image<0]=0
        image[image>255]=255
        return image,grid
    def close(self):
        """
        close the camera/s
        """
        self.cap.release()

class environment:
    def __init__(self,x,y):
        #self.grid=(np.random.random((y,x))*10).astype(int)
        p = np.zeros((y,x))
        for i in range(4):
            freq = 2**i
            lin = np.linspace(0, freq, x, endpoint=False)
            x_, y_ = np.meshgrid(lin, lin)  # FIX3: I thought I had to invert x and y here but it was a mistake
            p = self.perlin(x_, y_, seed=87) / freq + p
        
        while np.max(p)<10: #adjust for size
            p=p*10
        self.grid=p.astype(int)*-1
        self.grid[self.grid<0]=0
        print(self.grid)
    def get(self):
        return self.grid
    def perlin(self,x, y, seed=0):
        # permutation table
        np.random.seed(seed)
        p = np.arange(256, dtype=int)
        np.random.shuffle(p)
        p = np.stack([p, p]).flatten()
        # coordinates of the top-left
        xi, yi = x.astype(int), y.astype(int)
        # internal coordinates
        xf, yf = x - xi, y - yi
        # fade factors
        u, v = self.fade(xf), self.fade(yf)
        # noise components
        n00 = self.gradient(p[p[xi] + yi], xf, yf)
        n01 = self.gradient(p[p[xi] + yi + 1], xf, yf - 1)
        n11 = self.gradient(p[p[xi + 1] + yi + 1], xf - 1, yf - 1)
        n10 = self.gradient(p[p[xi + 1] + yi], xf - 1, yf)
        # combine noises
        x1 = self.lerp(n00, n10, u)
        x2 = self.lerp(n01, n11, u)  # FIX1: I was using n10 instead of n01
        return self.lerp(x1, x2, v)  # FIX2: I also had to reverse x1 and x2 here

    def lerp(self,a, b, x):
        "linear interpolation"
        return a + x * (b - a)

    def fade(self,t):
        "6t^5 - 15t^4 + 10t^3"
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    def gradient(self,h, x, y):
        "grad converts h to the right gradient vector and return the dot product with (x,y)"
        vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
        g = vectors[h % 4]
        return g[:, :, 0] * x + g[:, :, 1] * y


class digiTip:
    def __init__(self,img,startPos=[25,25]):
        self.grid=np.zeros(img.shape[0:2])
        self.h=100
        self.pos=startPos
        kernel = np.ones((1, 1), np.uint8)
        img = cv2.erode(img, kernel, iterations=1)
        self.img=img
        
    def lower(self,amount):
        self.h-=amount
    def getImage(self,env):
        y=self.pos[0]+self.grid.shape[0]
        x=self.pos[1]+self.grid.shape[1]
        subArray=env[max(min(self.pos[0],env.shape[0]),0):max(min(y,env.shape[0]),0),max(min(self.pos[1],env.shape[1]),0):max(min(x,env.shape[1]),0)]
        touch=subArray-self.h
        touch[touch<0]=0
        for i in range(len(touch)-1):
            for j in range(len(touch[0])-1):
                if self.pos[0]+i>0 and self.pos[0]+i<env.shape[0] and self.pos[1]+j>0 and self.pos[1]+j<env.shape[1]:
                    touch=self.expand(touch,j,i)
        touch[touch<0]=0  
        touch=touch*10
        touch[touch>80]=80
        return touch
    def expand(self,arr,x,y):
        item=arr[y][x]
        if item<=0: return arr
        else: #if has number decrease the outside
            if y-1>=0:
                if arr[y-1][x]<2: arr[y-1][x]+=item-1
                if arr[y-1][x]==0: 
                    arr=self.expand(arr,x,y-1)
            if y+1<len(arr):
                if arr[y+1][x]<2: arr[y+1][x]+=item-1
                if arr[y+1][x]==0:
                    arr=self.expand(arr,x,y+1)
            if x+1<len(arr[0]):
                if arr[y][x+1]<2: arr[y][x+1]+=item-1
                if arr[y][x+1]==0:
                    arr=self.expand(arr,x+1,y)
            if x-1>=0:
                if arr[y][x-1]<2: arr[y][x-1]+=item-1
                if arr[y][x-1]==0:
                    arr=self.expand(arr,x-1,y)
        return arr
    def maskPush(self,arr,DIV=2):
        image=self.img.copy()[:,:,0]
        for i in range(0,arr.shape[0]-DIV,DIV):
            for j in range(0,arr.shape[1]-DIV,DIV):
                k=np.sum(arr[i:i+DIV,j:j+DIV])/(DIV*DIV*10)//2
                if k>0:
                    #k=int(k/10 +1) #make larger than self
                    image=self.dilate(image,[DIV,DIV],[i,j],k=min(int(k),3))#self.blowUp(image,[DIV,DIV],[i,j],k=k)
        return image
    def moveX(self,units):
        self.pos[1]=units+self.pos[1]
    def moveY(self,units):
        self.pos[0]=units+self.pos[0]
    def dilate(self,image,dims,pos,k):
        area=image[pos[0]:pos[0]+dims[0],pos[1]:pos[1]+dims[1]]
        kernel = np.ones((2,2), np.uint8)
        area = cv2.dilate(area, kernel, iterations=k)
        a=int((area.shape[0]-dims[0])/2)
        b=int((area.shape[1]-dims[1])/2)
        image[pos[0]:pos[0]+dims[0],pos[1]:pos[1]+dims[1]]=area[a:dims[0]+a,b:dims[1]+b]
        return image
    def blowUp(self,image,dims,pos,k=1.2):
        area=image[pos[0]:pos[0]+dims[0],pos[1]:pos[1]+dims[1]]
        area=cv2.resize(area,(np.array(dims)*k).astype(int),interpolation=cv2.INTER_AREA)
        a=int((area.shape[0]-dims[0])/2)
        b=int((area.shape[1]-dims[1])/2)
        image[pos[0]:pos[0]+dims[0],pos[1]:pos[1]+dims[1]]=area[a:dims[0]+a,b:dims[1]+b]
        return image
    def setPos(self,x,y):
        self.pos=[y,x]
    
class tacLeg(Skin):
    def getBinary(self,min_size = 400,adaptive=False):
        """
        get the binary image from the sensor and return only the white dots (filter out glare)
        """
        image=self.getFrame()
        gray=image
        if len(image.shape)==3: gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255,
	cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51,1)
        kernel = np.ones((2,2),np.uint8)
        binary = cv2.erode(binary,kernel,iterations = 2)
        if adaptive: binary=self.adaptive(gray)
        if type(self.imF)!=type(""): return binary
        #return binary
        to_Show,spots=self.removeBlob(binary,min_size = min_size)
        return spots.astype(np.uint8)
    def getDots(self,gray,size=1,centre=None):
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
                temp.append(np.array(t[i]))
        t=np.array(temp)
        return t
    def movement(self,new,referenceArray=None,MAXD=100):
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
            #print(looped.shape)
            #TODO experiment with adding the unpicked lowest distances instead of the orginal point
            for i, eachPoint in enumerate(arrayB): #fill in gaps
                if np.sum(looped[i])==0:
                    looped[i]=eachPoint.copy()
            
                #self.last=self.origin.copy()
            self.last=looped.copy()
            return looped
        else:
            return arrayB
    def loop_through(self,stored,used,looped,arrayA,arrayB,count,maxL=20,dist=20):
        """
        Recursive method to find the closest points
        @param stored is the items that have been visited but can be again
        @param used is the already used indicies
        @param looped is the big array of point positions
        @param arrayA is the incoming points
        @param arrayB is the og points
        @param count is a counter to preent infinite recursion 
        @param maxL is the maximum level it will search
        @param dist is the parameter that removes shapes over a crtain distance
        """
        if count==maxL:
            return looped
        for i, eachPoint in enumerate(arrayB): #loop through distances and pair off
            distances=self.euclid(eachPoint,arrayA,axis=1)#np.sqrt(np.sum(eachPoint-arrayA,axis=1)**2)
            min_dist=distances[np.argmin(distances)]
            ind=np.argmin(distances)
            if ind not in used and distances[ind]<dist: #make sure within parameters
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