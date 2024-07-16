import numpy as np
from sys import getsizeof
from os import listdir
from os.path import isfile, join
import cv2
class dataset:
    def __init__(self,pathway,tempdir,temporal_window,delay=0,compression_dim=1,crop=None,label=0):
        self.pathway=pathway
        self.T=temporal_window
        self.X=[]
        new_dim=(int(640*compression_dim),int(480*compression_dim))
        self.keys={}
        if type(crop)!=type(None):
            crop_x=abs(crop[0]-crop[1])
            crop_y=abs(crop[2]-crop[3])
        files = [f for f in listdir(pathway) if isfile(join(pathway, f))]
        for k in range(0,len(files),1): #loop through files
            file=files[k]
            if ".npz" in file: #is numpy
                data = np.load(self.pathway+"/"+file) #load data
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
            
        self.X=np.array(self.X)
        np.savez_compressed(tempdir.name+"/X_"+str(label)+".npz", self.X)
