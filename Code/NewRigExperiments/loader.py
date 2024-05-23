import os
import numpy as np
import sys
import cv2
sys.path.insert(1, '/its/home/drs25/Documents/GitHub/Rig-controller/Code/DataLogger')
def list_files(directory):
    files = []
    for item in os.listdir(directory):
        full_path = os.path.join(directory, item)
        if os.path.isfile(full_path):
            files.append(item)
    return files

def list_folders(directory):
    folder_names = []
    for item in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, item)):
            folder_names.append(item)
    return folder_names

#load the file version
class dataset:
    def __init__(self,pathways,temporal_window,delay=0,compression_dim=1,crop=None):
        self.pathways=pathways
        self.T=temporal_window
        self.X=[]
        self.y=[]
        new_dim=(int(640*compression_dim),int(480*compression_dim))
        self.keys={}
        if type(crop)!=type(None):
            crop_x=abs(crop[0]-crop[1])
            crop_y=abs(crop[2]-crop[3])
        for i,path in enumerate(self.pathways): #loop through paths
            self.keys[i]=path.split("/")[-2]
            files=list_files(path)
            for k in range(0,len(files),1): #loop through files
                file=files[k]
                if ".npz" in file: #is numpy
                    data = np.load(path+"/"+file) #load data
                    for array_name in data:
                        data=data[array_name].reshape(100,480,640,3)
                        window=data[::4][delay:delay+temporal_window]
                        a=np.zeros((len(window),new_dim[1],new_dim[0],))
                        if type(crop)!=type(None):
                            a=np.zeros((len(window),crop_y,crop_x,))
                        if compression_dim<1:
                            for j,frame in enumerate(window):
                                image=cv2.resize(frame,new_dim,interpolation=cv2.INTER_AREA) #resize to new dimentions
                                if type(crop)!=type(None): a[j]=cv2.cvtColor(image[crop[2]:crop[3],crop[0]:crop[1]],cv2.COLOR_BGR2GRAY) #crop out centre
                                else: a[j]=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                                #a[j] = cv2.cvtColor(a[j], cv2.COLOR_BGR2GRAY)
                        window=a.copy()
                        self.X.append(window)
                        self.y.append(i)
                
        self.X=np.array(self.X)
        self.y=np.array(self.y)
        print("Dataset size:",self.X.shape[0],"\nWindow size:",self.X.shape[1],"\nImage:",self.X.shape[2:])
        print("Memory needed:",round(getsizeof(self.X)/ 1024 / 1024/ 1024,2),"GB")
        #randomize order
        n_samples = self.X.shape[0]
        indices = np.random.permutation(n_samples)
        shuffled_data = self.X[indices]
        shuffled_labels = self.y[indices]
        self.X=shuffled_data
        self.y=shuffled_labels
    def crop(self,x,x1,y,y1):
        crop_x=abs(x-x1)
        crop_y=abs(y-y1)
        temp=np.zeros((self.X.shape[0],self.X.shape[1],crop_y,crop_x,3))
        for i in range(len(self.X)): #crop all images individually
            for j in range(len(self.X[0])):
                temp[i][j]=self.X[i][j][y:y1,x:x1]
        self.X=temp
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

# Replace 'path_to_directory' with the path to the directory you want to search
directory = '/its/home/drs25/Documents/data/Tactile Dataset/TacTip_Gfoam_P100 (2)/'
folders = list_folders(directory)
f=list_files
Files=[]
Folders=[]
for folder in folders:
    Files.append(directory+folder+"/"+folder+".xml")
    Folders.append(directory+folder+"/")
