import numpy as np
from loader import list_files, Files, Folders  #precalculate what we are loading in

files=list_files("/its/home/drs25/Documents/data/Tactile Dataset/texture")
def compress():
    X1=[]
    y1=[]
    files=['X_data_wool', 'y_data_wool', 'X_data_jeans', 'y_data_jeans', 'X_data_LacedMatt', 'y_data_LacedMatt', 'X_data_Gfoam', 'y_data_Gfoam', 'X_data_bubble', 'y_data_bubble', 'X_data_Efoam', 'y_data_Efoam', 'X_data_cotton', 'y_data_cotton', 'X_data_Flat', 'y_data_Flat', 'X_data_felt', 'y_data_felt', 'X_data_Ffoam', 'y_data_Ffoam']

    for file in files:
        data = np.load("/its/home/drs25/Documents/data/Tactile Dataset/texture/"+"textures_"+file+".npz") #load data
        if "jeans" not in file:
            if "X_" in file:
                for array_name in data:
                    if len(data[array_name].shape)>1:
                        X1.append(data[array_name].astype(np.int8))
                        print(file,len(X1[-1]))
            elif "y_" in file:
                for array_name in data:
                        y1.append(data[array_name].astype(np.int8))
                        print(file,len(y1[-1]))
        
    X=np.concatenate(X1)
    y=np.concatenate(y1)
    np.savez_compressed("/its/home/drs25/Documents/data/Tactile Dataset/X_texture",X)
    np.savez_compressed("/its/home/drs25/Documents/data/Tactile Dataset/y_texture",y)

compress()
