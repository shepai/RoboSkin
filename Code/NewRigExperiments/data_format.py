import numpy as np
from sys import getsizeof
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
from sklearn.preprocessing import LabelEncoder
import os
import time
import random
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from models import *
from datapath import datapath
from data_loader import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device=torch.device("cpu")
print(f"Using device: {device}")
csfont = {'fontname':'Times New Roman'}
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def genData(frm,to,percentage=1,n=-1):
    torch.cuda.empty_cache()
    data=loaded(to,from_=frm,filename="X_data_newMorph.npz") #X_data_15.npz
    data.applySobel()
    data.resize(percentage)
    data.augment()
    data.X=data.X[:n]
    data.y=data.y[:n]
    return org_data(data, (len(data.X),1,abs(frm-to)*data.X.shape[2],data.X.shape[3]),n=n)

def genDataANN(frm,to,percentage=1,n=-1):
    torch.cuda.empty_cache()
    data=loaded(to,from_=frm,filename="X_data_15.npz")
    data.applySobel()
    data.resize(percentage)
    data.augment()
    data.X=data.X[:n]
    data.y=data.y[:n]
    return org_data(data, (len(data.X),abs(frm-to)*data.X.shape[2]*data.X.shape[3]),n=n)
def gen3DData(frm,to,percentage=1,n=-1):
    torch.cuda.empty_cache()
    data=loaded(to,from_=frm,filename="X_data_15.npz")
    data.applySobel()
    data.augment()
    data.resize(percentage)
    data.X=data.X[:n]
    data.y=data.y[:n]
    return org_data(data, (len(data.X),1,abs(frm-to),data.X.shape[2],data.X.shape[3]),n=n)
    
def genLSTMData(frm,to,percentage=1,n=-1):
    torch.cuda.empty_cache()
    data=loaded(to,from_=frm,filename="X_data_15.npz")
    data.applySobel()
    data.augment()
    data.resize(percentage)
    data.X=data.X[:n]
    data.y=data.y[:n]
    #print(data.X.shape,(len(data.X),abs(frm-to),data.X.shape[2]*data.X.shape[3]))
    return org_data(data, (len(data.X),abs(frm-to),data.X.shape[2]*data.X.shape[3]),n=n)
def genCNNLSTMData(frm,to,percentage=1,n=-1):
    torch.cuda.empty_cache()
    data=loaded(to,from_=frm,filename="X_data_15.npz")
    data.applySobel()
    data.augment()
    data.resize(percentage)
    data.X=data.X[:n]
    data.y=data.y[:n]
    return org_data(data, (len(data.X),1,abs(frm-to),data.X.shape[2],data.X.shape[3]),n=n)


def org_data(data,shape,n=-1):
    torch.cuda.empty_cache()
    data.shuffle()
    print("LOADED DATASET...")
    # Example: if train_labels are strings, use LabelEncoder to convert them to integers
    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(data.y)
    one_hot_labels = torch.nn.functional.one_hot(torch.tensor(train_labels_encoded), num_classes=len(np.unique(train_labels_encoded)))
    print("Memory left",round(torch.cuda.mem_get_info()[1]/ 1024 / 1024/ 1024,2),"GB")
    print(data.X.shape,shape)
    x_data=data.X.reshape(shape)
    del data
    x_data=(x_data-np.mean(x_data))/(np.max(x_data)-np.min(x_data)) #preprocessing
    
    train_images_tensor = torch.tensor(x_data, dtype=torch.float32).to(device)
    print("Using",round(getsizeof(x_data)/ 1024 / 1024/ 1024,2),"GB")
    del x_data
    train_labels_tensor = torch.tensor(one_hot_labels, dtype=torch.float32).to(device)

    # Create a TensorDataset
    dataset = TensorDataset(train_images_tensor, train_labels_tensor)
    # Split the dataset into training and testing sets (e.g., 80% training, 20% testing)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoader for training and testing sets
    train_loader = DataLoader(dataset=train_dataset, batch_size=40, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=40,shuffle=False)

    print(train_images_tensor.shape)
    print(train_labels_tensor.shape)
    
    return train_loader,test_loader