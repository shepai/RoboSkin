from loader import list_files, Files, Folders  #precalculate what we are loading in
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sys import getsizeof
from data_xml import Loader
import os
import time
clear = lambda: os.system('clear')

class dataset:
    def __init__(self,files,temporal_window,delay=0,compression_dim=1,crop=None,start=0,end=-1):
        self.X=[]
        self.y=[]
        self.y2=[]
        new_dim=(int(640*compression_dim),int(480*compression_dim))
        if type(crop)!=type(None):
            crop_x=abs(crop[0]-crop[1])
            crop_y=abs(crop[2]-crop[3])
        t1=time.time()
        print(print(len(files)))
        print(files)
        time.sleep(2)
        names=[folder.split("/")[8].split("_")[1] for folder in files]
        for i in range(len(files[start:end])): #loop through all files 
            loader_=Loader(files[i])
            frame=loader_.getByExperiment() #get frame
            del loader_
            for index, row in frame.iterrows():
                folder=files[i].replace(".xml","_"+row['Readings']+".npz")
                data = np.load(folder) #load data
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
                    self.y.append(row['Positions'])
                    self.y2.append(files[i])
                    self.X.append(window)
                clear()
                if len(np.array(self.X).shape)>=2:
                    print("Dataset size:",(i*index)+index,"\nWindow size:",self.X[0].shape[0],"\nImage:",self.X[0].shape[1:])
                    print("Approximate percentage:",round((((i*(index+1))+index)/(len(frame)*len(files[start:end])))*100,2),"%")
                    print("Memory needed:",round(self.getSize("/its/home/drs25/Documents/data/Tactile Dataset/Directions/directions_X_data_"+names[i+start]+".npz")/ 1024 / 1024/ 1024,2),"GB")
                    print("Time lapsed:",(time.time()-t1)/60,"minutes")
                if index%20==0:
                    self.append_to_npz("/its/home/drs25/Documents/data/Tactile Dataset/Directions/directions_X_data_"+names[i+start]+".npz",np.array(self.X))
                    self.append_to_npz("/its/home/drs25/Documents/data/Tactile Dataset/Directions/directions_y_data_"+names[i+start]+".npz",np.array(self.y))
                    self.X=[]
                    self.y=[]
                    self.y2=[]
            if len(self.X)>0:
                self.append_to_npz("/its/home/drs25/Documents/data/Tactile Dataset/Directions/directions_X_data_"+str(start)+"-"+str(end)+".npz",np.array(self.X))
                self.append_to_npz("/its/home/drs25/Documents/data/Tactile Dataset/Directions/directions_y_data_"+str(start)+"-"+str(end)+".npz",np.array(self.y))
                self.X=[]
                self.y=[]
                self.y2=[]
        self.X=np.array(self.X)
        self.y=np.array(self.y)
        #self.y2=np.array(self.y2)
        print("Dataset size:",self.X.shape[0],"\nWindow size:",self.X.shape[1],"\nImage:",self.X.shape[2:])
        print("Memory needed:",round(getsizeof(self.X)/ 1024 / 1024/ 1024,2),"GB")
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

names=[folder.split("/")[8].split("_")[1] for folder in Files]
print(len(names))
for i in range(0,26,2):
    print(i,i+2,names[i:i+2])
    data=dataset(Files,20,compression_dim=0.4,delay=0,crop=[60,180,40,150],start=i,end=i+2) 
print(names[27:28])
data=dataset(Files,20,compression_dim=0.4,delay=0,crop=[60,180,40,150],start=27,end=28) 