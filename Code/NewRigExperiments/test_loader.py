import numpy as np
import os

def load_files(directory, type_="circle"):
    files = []
    data_list = []
    label_list = []
    keys = {}

    for item in os.listdir(directory):
        full_path = os.path.join(directory, item)
        filename = full_path.split("/")[-1]
        print("Loading:", filename)

        if not os.path.isfile(full_path):
            continue

        files.append(item)
        data = np.load(full_path).astype(np.uint8)
        newlabel = filename.split("_")[2]

        if newlabel not in keys:
            keys[newlabel] = len(keys)

        num = keys[newlabel]

        if type_ == "circle" or type_ == "pressure":
            data = data[:, :, :len(np.arange(10, 100, 10))]

        data = data.reshape((1 * 2 * len(np.arange(10, 100, 10)), 50, *data.shape[4:]))
        data_list.append(data)
        label_list.append(np.ones((data.shape[0],)) * num)

    numpy = np.concatenate(data_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    return numpy.astype(np.uint8), label.astype(np.uint8)

path="/its/home/drs25/Documents/data/Tactile Dataset/nonlinear/pressure"
savepath="/its/home/drs25/Documents/data/Tactile Dataset/datasets/"

x,y=load_files(path,"pressure")
np.savez_compressed(savepath+"X_nonlinear_TT",x)
np.savez_compressed(savepath+"y_nonlinear_TT",y)