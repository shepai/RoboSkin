import numpy as np
import cv2
from datapath import datapath
from sys import getsizeof

class loaded:
    def __init__(self,t=20,from_=0,filename="X_data_15.npz"):
        data = np.load(datapath+filename) #load data
        for array_name in data:
            self.X=(data[array_name].astype(np.uint8))
        data = np.load(datapath+filename.replace("X","y")) #load data
        for array_name in data:
            self.y=(data[array_name].astype(np.uint8))
        self.keys=['Leather', 'Cork', 'wool', 'LacedMatt', 'Gfoam', 'Plastic', 'Carpet', 'bubble', 'Efoam', 'cotton', 'LongCarpet', 'Flat', 'felt', 'Jeans', 'Ffoam']

        print("Dataset size:",self.X.shape[0],"\nWindow size:",self.X.shape[1],"\nImage:",self.X.shape[2:])
        print("Memory needed:",round(getsizeof(self.X)/ 1024 / 1024/ 1024,2),"GB")
        assert self.X.shape[0]==self.y.shape[0],"Incorrect data size match y="+str(self.y.shape[0])+" x="+str(self.X.shape[0])
        self.X=self.X[:,from_:t]
        #randomize order
        n_samples = self.X.shape[0]
        indices = np.random.permutation(n_samples)
        shuffled_data = self.X[indices]
        shuffled_labels = self.y[indices]
        self.X=shuffled_data
        self.y=shuffled_labels
    def shuffle(self):
        n_samples = self.X.shape[0]
        indices = np.random.permutation(n_samples)
        shuffled_data = self.X[indices]
        shuffled_labels = self.y[indices]
        self.X=shuffled_data
        self.y=shuffled_labels
    def augment(self,t=4,zoom=[10,20,30,40]):
        self.orientation_augment()
        self.zoom_augment(zoom)
        #self.speed_augment(t)
    def orientation_augment(self):
        #create rotations
        self.AugmentedX=np.zeros((len(self.X)*3,*self.X.shape[1:]),dtype=np.uint8)
        self.Augmentedy=np.zeros_like(np.concatenate((self.y,self.y,self.y)))
        for k,i in enumerate(range(0,len(self.AugmentedX),3)): #loop through the normal data and new data
            for j in range(len(self.X[0])):
                self.AugmentedX[i][j]=np.copy(self.X[k][j])
                self.AugmentedX[i+1][j]=cv2.resize(cv2.rotate(self.X[k][j].copy(), cv2.ROTATE_90_CLOCKWISE),(self.X[k][j].shape[1],self.X[k][j].shape[0]),interpolation=cv2.INTER_AREA)
                self.AugmentedX[i+2][j]=cv2.resize(cv2.rotate(self.X[k][j].copy(), cv2.ROTATE_180),(self.X[k][j].shape[1],self.X[k][j].shape[0]),interpolation=cv2.INTER_AREA)
                self.Augmentedy[i+1]=self.y[k]
                self.Augmentedy[i+2]=self.y[k]
                self.Augmentedy[i]=self.y[k]
                #self.AugmentedX[i+3][j]=cv2.rotate(self.X[k][j], cv2.ROTATE_90_COUNTERCLOCKWISE)
        print("Dataset size:",self.AugmentedX.shape[0],"\nWindow size:",self.X.shape[1],"\nImage:",self.X.shape[2:])
        print("Memory needed:",round(getsizeof(self.AugmentedX)/ 1024 / 1024/ 1024,2),"GB")
        self.X = self.AugmentedX
        self.y = self.Augmentedy
        n_samples = self.X.shape[0]
        indices = np.random.permutation(n_samples)
        shuffled_data = self.X[indices]
        shuffled_labels = self.y[indices]
        self.X=shuffled_data
        self.y=shuffled_labels
        del self.AugmentedX
        del self.Augmentedy
    def applySobel(self):
        for i in range(len(self.X)): #crop all images individually
            for j in range(len(self.X[0])):
                image=self.X[i][j]
                # Apply Sobel filter in x-direction
                sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # ksize=3 for a 3x3 Sobel kernel

                # Apply Sobel filter in y-direction
                sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

                # Convert the results back to uint8
                sobel_x = np.uint8(np.absolute(sobel_x))
                sobel_y = np.uint8(np.absolute(sobel_y))

                # Combine the results to get the final edge-detected image
                sobel_combined = cv2.bitwise_or(sobel_x, sobel_y)
                self.X[i][j]=sobel_combined
    def zoom_augment(self, zoom_factors):
        dataX=self.X
        dataY=self.y
        n, t, h, w = dataX.shape
        augmented_dataX = []
        augmented_dataY = []
        
        for zoom in zoom_factors:
            crop_margin = int((zoom / 100) * min(h, w) / 2)
            cropped_and_resized = np.zeros_like(dataX)
            
            for i in range(n):
                for j in range(t):
                    # Crop the central region
                    cropped = dataX[i, j, crop_margin:h-crop_margin, crop_margin:w-crop_margin]
                    # Resize back to original dimensions
                    cropped_and_resized[i, j] = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
            
            augmented_dataX.append(cropped_and_resized)
            augmented_dataY.append(dataY.copy())  # Labels remain the same
        augmented_dataX=np.array(augmented_dataX)
        augmented_dataY=np.array(augmented_dataY)
        self.X=np.concatenate([self.X,augmented_dataX.reshape((len(zoom_factors)*augmented_dataX.shape[1],*augmented_dataX.shape[2:]))])
        self.y=np.concatenate([self.y,augmented_dataY.reshape((len(zoom_factors)*augmented_dataY.shape[1],))])
        
    def resize(self,percentage):
        h=int(self.X.shape[2]*percentage)
        w=int(self.X.shape[3]*percentage)
        new_array=np.zeros((*self.X.shape[0:2],h,w))

        for i in range(len(self.X)): #crop all images individually
            for j in range(len(self.X[0])):
                image=self.X[i][j]
                iamge = cv2.resize(image,(w,h),interpolation=cv2.INTER_AREA)
                new_array[i][j]=iamge
        self.X=new_array.copy()
    def speed_augment(self,t,speeds=5):
        L=self.X.shape[1]
        assert t*t<=L, "Cannot pick this many frames"+str(L)
        keys=np.array(list(range(1,t)))
        frames=np.zeros((self.X.shape[0]*len(keys),t,*self.X.shape[2:]),dtype=np.uint8)
        y_labels=np.zeros(self.X.shape[0]*len(keys))
        for i in range(len(keys)):
            frames_=self.X[:,::keys[i],:,:]
            frames[len(self.X)*i:len(self.X)*(i+1)]=frames_[:,:t,:,:]
            y_labels[len(self.X)*i:len(self.X)*(i+1)]=self.y
        self.X=frames
        self.y=y_labels
    def different_starts(self,t):
        multiplier=self.X.shape[1]//t
        new_array=np.zeros((len(self.X)*multiplier,t,*self.X.shape[2:]))
        new_labels=np.zeros((len(self.y)*multiplier))
        for i in range(len(self.X)):
            for j in range(0,self.X.shape[1]-t,t):
                new_array[i]=self.X[i][j:j+t]
                new_labels[i]=self.y[i]
        self.X=new_array
        self.y=new_labels
