from loader import list_files, Files, Folders  #precalculate what we are loading in
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sys import getsizeof
import os
import time
clear = lambda: os.system('clear')

class dataset:
    def __init__(self,pathways,temporal_window,delay=0,compression_dim=1,crop=None,start=0,end=-1):
        self.pathways=pathways
        self.T=temporal_window
        self.X=[]
        self.y=[]
        new_dim=(int(640*compression_dim),int(480*compression_dim))
        self.keys={}
        if type(crop)!=type(None):
            crop_x=abs(crop[0]-crop[1])
            crop_y=abs(crop[2]-crop[3])
        t1=time.time()
        names=[folder.split("/")[8].split("_")[1] for folder in self.pathways]
        time.sleep(2)
        for i,name in enumerate(names):
            self.keys[name]=i
        for i,path in enumerate(self.pathways[start:end]): #loop through paths
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
                        self.X.append(window.astype(np.int8))
                        self.y.append(self.keys[names[i+start]])
                clear()
                if len(np.array(self.X).shape)>=2:
                    print("Dataset size:",(i*k)+k,"\nWindow size:",self.X[0].shape[0],"\nImage:",self.X[0].shape[1:])
                    print("Approximate percentage:",round((((i*(k+1))+k)/(len(self.pathways[start:end])*len(files)))*100,2),"%")
                    print("Memory needed:",round(self.getSize("/its/home/drs25/Documents/data/Tactile Dataset/texture/textures_X_data_"+names[i+start]+".npz")/ 1024 / 1024/ 1024,2),"GB")
                    print("Time lapsed:",(time.time()-t1)/60,"minutes")
                if k%20==0:
                    self.append_to_npz("/its/home/drs25/Documents/data/Tactile Dataset/texture/textures_X_data_"+names[i+start]+".npz",np.array(self.X))
                    self.append_to_npz("/its/home/drs25/Documents/data/Tactile Dataset/texture/textures_y_data_"+names[i+start]+".npz",np.array(self.y))
                    self.X=[]
                    self.y=[]
                    self.y2=[]
            if len(self.X)>0:
                self.append_to_npz("/its/home/drs25/Documents/data/Tactile Dataset/texture/textures_X_data_"+names[i+start]+".npz",np.array(self.X))
                self.append_to_npz("/its/home/drs25/Documents/data/Tactile Dataset/texture/textures_y_data_"+names[i+start]+".npz",np.array(self.y))
                self.X=[]
                self.y=[]
                self.y2=[]
        self.X=np.array(self.X)
        self.y=np.array(self.y)
        #self.y2=np.array(self.y2)
        #print("Dataset size:",self.X.shape[0],"\nWindow size:",self.X.shape[1],"\nImage:",self.X.shape[2:])
        #print("Memory needed:",round(getsizeof(self.X)/ 1024 / 1024/ 1024,2),"GB")
    def append_to_npz(self,npz_file, new_data, array_name=None):
        if os.path.exists(npz_file):
            # Load existing data
            try:
                with np.load(npz_file, allow_pickle=True) as data:
                    if array_name is None:
                        # Use the first array name if none is provided
                        array_name = list(data.keys())[0]
                    
                    if array_name in data:
                        existing_data = data[array_name]
                        # Concatenate new data to the existing array
                        updated_data = np.concatenate((existing_data, new_data))
                    else:
                        updated_data = new_data
            except Exception as e:
                # If loading fails (e.g., empty file), just use new data
                print(f"Error loading data from '{npz_file}': {e}")
                updated_data = new_data
        else:
            # If file does not exist, use new data as the initial data
            updated_data = new_data

        # Save the updated array back to the .npz file
        np.savez_compressed(npz_file, updated_data)
    def getSize(self,npz_file):
        try:
            file_size = os.path.getsize(npz_file)
            #print(f"Size of '{npz_file}': {file_size} bytes")
            return file_size
        except FileNotFoundError:
            print(f"File '{npz_file}' not found.")
            return 0
        
names=[folder.split("/")[8].split("_")[1] for folder in Folders]
print(names)
for i in range(0,26,2):
    print(i,i+2,names[i:i+2])
    data=dataset(Folders,20,compression_dim=0.4,delay=0,crop=[60,180,40,150],start=i,end=i+2) 
print(names[27:28])
data=dataset(Folders,20,compression_dim=0.4,delay=0,crop=[60,180,40,150],start=27,end=28) 
#data=dataset(Folders,20,compression_dim=0.4,delay=0,crop=[60,180,40,150],start=i,end=i+2) 
"""data=dataset(Folders,20,compression_dim=0.4,delay=0,crop=[60,180,40,150],start=2,end=4) #20 steps approximatly 4 seconds
data=dataset(Folders,20,compression_dim=0.4,delay=0,crop=[60,180,40,150],start=4,end=6) #20 steps approximatly 4 seconds
data=dataset(Folders,20,compression_dim=0.4,delay=0,crop=[60,180,40,150],start=6,end=8) #20 steps approximatly 4 seconds
data=dataset(Folders,20,compression_dim=0.4,delay=0,crop=[60,180,40,150],start=8,end=10) #20 steps approximatly 4 seconds"""