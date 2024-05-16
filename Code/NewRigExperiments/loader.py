import os
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


# Replace 'path_to_directory' with the path to the directory you want to search
directory = '/its/home/drs25/Documents/data/Tactile Dataset/TacTip_Gfoam_P100 (2)/'
folders = list_folders(directory)
f=list_files
Files=[]
Folders=[]
for folder in folders:
    Files.append(directory+folder+"/"+folder+".xml")
    Folders.append(directory+folder+"/")
