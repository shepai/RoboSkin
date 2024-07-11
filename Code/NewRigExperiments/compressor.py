import numpy as np
from loader import list_files, Files, Folders  #precalculate what we are loading in

files=list_files("/its/home/drs25/Documents/data/Tactile Dataset/texture_gel/")
def compress():
    X1=[]
    y1=[]
    files=[]#'X_data_wool', 'y_data_wool', 'X_data_Jeans', 'X_data_LacedMatt', 'y_data_LacedMatt', 'X_data_Gfoam', 'y_data_Gfoam', 'X_data_bubble', 'y_data_bubble', 'X_data_Efoam', 'y_data_Efoam', 'X_data_cotton', 'y_data_cotton', 'X_data_Flat', 'y_data_Flat', 'X_data_felt', 'y_data_felt', 'X_data_Ffoam', 'y_data_Ffoam']
    #keys={'Cork': 38, 'wool': 19, 'LacedMatt': 28, 'Gfoam': 30, 'Carpet': 31, 'bubble': 37, 'Efoam': 21, 'cotton': 29, 'LongCarpet': 25, 'Flat': 16, 'felt': 34, 'Jeans': 39, 'Ffoam': 36}
    #keys={'Carpet': 24, 'LacedMatt': 29, 'Cork': 31, 'Felt': 30, 'LongCarpet': 42, 'cotton': 41, 'Plastic': 25, 'Flat': 32, 'Ffoam': 44, 'Gfoam': 28, 'bubble': 27, 'Efoam': 43, 'jeans': 35, 'Leather': 38}
    #keys={'Plasticd2': 4, 'Plasticd1': 7, 'Plasticd3': 11}
    keys={'Leather': 34, 'Cork': 41, 'wool': 21, 'LacedMatt': 31, 'Gfoam': 33, 'Plastic': 22, 'Carpet': 35, 'bubble': 40, 'Efoam': 24, 'cotton': 32, 'LongCarpet': 28, 'Flat': 17, 'felt': 37, 'Jeans': 42, 'Ffoam': 39}
    l=list(keys.keys())
    #l=["felt","Flat"]
    for name in l:
         files.append("X_data_"+name)
         files.append("y_data_"+name)
    data = np.load("/its/home/drs25/Documents/data/Tactile Dataset/texture/"+"textures_"+files[0]+".npz") #load data
    X_shape=0
    other=0
    for file in files:
         if "X_"in file:
            for array_name in data:
                X_shape+=len(data[array_name])
                other=data[array_name][0].shape
    X=np.zeros((X_shape,*other),dtype=np.uint8)
    print("x created")
    xi=0
    for file in files:
        data = np.load("/its/home/drs25/Documents/data/Tactile Dataset/texture/"+"textures_"+file+".npz") #load data
        if "X_" in file:
            for array_name in data:
                if len(data[array_name].shape)>1:
                    print(data[array_name].dtype,xi,xi+len(data[array_name]))
                    array=data[array_name].astype(np.uint8)
                    X[xi:xi+len(array)]=array.copy()
                    xi+=len(array)
                    del array
    print("Converting x")
    #X=np.concatenate(X1,dtype=np.uint8)
    np.savez_compressed("/its/home/drs25/Documents/data/Tactile Dataset/datasets/X_textures_all15",X)
    del X
    y=np.zeros((X_shape,),dtype=np.uint8)
    yi=0
    print("y created")
    for file in files:
        data = np.load("/its/home/drs25/Documents/data/Tactile Dataset/texture/"+"textures_"+file+".npz") #load data
        if "y_" in file:
            for array_name in data:
                    print(data[array_name].dtype)
                    array=data[array_name].astype(np.uint8)
                    y[yi:yi+len(array)]=array
                    yi+=len(array)
    
    del data
    print(len(y1),y1[0].shape,len(X1),X1[0].shape)
    print("Converting y")
    #y=np.concatenate(y1,dtype=np.uint8)
    np.savez_compressed("/its/home/drs25/Documents/data/Tactile Dataset/datasets/y_textures_all15",y)
    del y
    
compress()
