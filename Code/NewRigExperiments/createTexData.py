import numpy as np
import cv2
import matplotlib.pyplot as plt
from sys import getsizeof
from os import listdir
from os.path import isfile, join
from os import walk
import os
import time
import tempfile
from dataset_unloader import dataset
t1=time.time()
clear = lambda: os.system('clear')

path_to_files="C:/Users/dexte/Documents/Datasets/raw_tactile/"
path_to_output="C:/Users/dexte/Documents/dataset/tactile/"
#########################################################################
#Gather all folders and process them into classes
#########################################################################
f = []
for (dirpath, dirnames, filenames) in walk(path_to_files):
    f.extend(dirnames)
    break
#gather all names of classes
className=[]
list_of_labels=[]
for i in range(len(f)):
    f[i]=dirpath+f[i]
    folder=f[i].split("/")[6].split("_")[1]
    if folder in className:
        list_of_labels.append(className.index(folder))
    else:
        list_of_labels.append(len(className))
        className.append(folder)

def output_message(string):
    clear()
    print("\tFolders:",f)
    print("\tLabels:",list_of_labels)
    print("\tClasses:",className)
    print(string)

#########################################################################
#concat into one folder per class
#########################################################################
temp_dir = tempfile.TemporaryDirectory()
for i,file in enumerate(f):
    output_message("Processing... "+str(i)+"/"+str(len(f)))
    data=dataset(file,temp_dir,20,compression_dim=0.4,delay=0,crop=[60,180,40,150],label=list_of_labels[i]) #read as dataset and convert
    del data
print("Creating Temp:",temp_dir.name)
print("\nDataset sorted...\nCompressing...")
#########################################################################
#Compress all data together into one big file
########################################################################
new_classes=[]
for (dirpath, dirnames, filenames) in walk(temp_dir.name+"/"):
    new_classes.extend(filenames)
    break
print("\tClasses found:",new_classes)
#load in all x names and evaluate the size of the array
size_x=0
shape_x=None
for i,class_ in enumerate(new_classes):
    filename=temp_dir.name+"/"+class_
    with np.load(filename) as data:  #load data
        for array_name in data:
            size_x+=len(data[array_name])
            shape_x=data[array_name][0].shape
            break
#create X
X=np.zeros((size_x,*shape_x))+1 #load into memory
marker=0
for i,class_ in enumerate(new_classes):
    output_message("Compressing X data... "+str(i)+"/"+str(len(new_classes)))
    filename=temp_dir.name+"/"+class_
    with np.load(filename) as data:  #load data
        for array_name in data:
            X[marker:marker+len(data[array_name])]=data[array_name] #copy over contents
            marker+=len(data[array_name])
            break
        
np.savez_compressed(path_to_output+"X_data.npz",X)
del X
#create_y
y=np.zeros((size_x,))+1 #load into memory
marker=0
for i,class_ in enumerate(new_classes):
    output_message("Compressing y data... "+str(i)+"/"+str(len(new_classes)))
    label=int(class_.split("_")[1].split(".")[0])
    filename=temp_dir.name+"/"+class_
    with np.load(filename) as data:  #load data
        for array_name in data:
            y[marker:marker+len(data[array_name])]=label #copy over contents
            marker+=len(data[array_name])
            break
np.savez_compressed(path_to_output+"y_data.npz",y)
del y

print("Data compressed!",round((time.time()-t1)/(60*60),3),"Hours")
# use temp_dir, and when done:
temp_dir.cleanup()
del temp_dir
