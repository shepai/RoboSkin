import numpy as np
import os
from tempfile import mkdtemp
import os.path as path
import numpy as np
import os
from tempfile import mkdtemp

def load_files_memory_efficient(directory, type_="circle", temp_dir=None):
    if temp_dir is None:
        temp_dir = mkdtemp()
    
    # First pass: collect labels and create mapping
    materials = ['Carpet', 'LacedMatt', 'wool', 'Cork', 'Felt', 'LongCarpet', 'cotton', 'Plastic', 'Flat', 'foamf', 'foamg', 'bubble', 'foame', 'jeans', 'Leather']
    keys = {material.lower(): index for index, material in enumerate(materials)}
    file_paths = []
    
    for item in os.listdir(directory):
        full_path = os.path.join(directory, item)
        if not os.path.isfile(full_path):
            continue
            
        filename = os.path.basename(full_path)
        newlabel = filename.split("_")[2].lower()
        
        if newlabel not in keys:
            keys[newlabel] = len(keys)
        file_paths.append(full_path)
    
    # Initialize memory-mapped arrays for final output
    sample_file = np.load(file_paths[0])
    if type_ == "circle" or type_ == "pressure":
        sample_file = sample_file[:, :, :len(np.arange(10, 100, 10))]
    
    sample_shape = (1 * 2 * len(np.arange(10, 100, 10)), 50, *sample_file.shape[4:])
    
    # Create memory-mapped arrays for data and labels
    data_memmap_path = os.path.join(temp_dir, 'data_memmap.dat')
    label_memmap_path = os.path.join(temp_dir, 'label_memmap.dat')
    
    # Calculate total size needed
    total_samples = 0
    for file_path in file_paths:
        data = np.load(file_path)
        if type_ == "circle" or type_ == "pressure":
            data = data[:, :, :len(np.arange(10, 100, 10))]
        total_samples += data.shape[0] * 2 * len(np.arange(10, 100, 10))
    
    # Initialize memmap files
    data_memmap = np.memmap(data_memmap_path, dtype=np.uint8, mode='w+', 
                           shape=(total_samples, *sample_shape[1:]))
    label_memmap = np.memmap(label_memmap_path, dtype=np.uint8, mode='w+',
                            shape=(total_samples,))
    
    # Second pass: process files one at a time
    current_idx = 0
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        print("Processing:", filename)
        
        data = np.load(file_path).astype(np.uint8)
        newlabel = filename.split("_")[2].lower()
        num = keys[newlabel]
        
        if type_ == "circle" or type_ == "pressure":
            data = data[:, :, :len(np.arange(10, 100, 10))]
        
        data = data.reshape((1 * 2 * len(np.arange(10, 100, 10)), 50, *data.shape[4:]))
        num_samples = data.shape[0]
        
        # Write to memmap
        data_memmap[current_idx:current_idx + num_samples] = data
        label_memmap[current_idx:current_idx + num_samples] = np.ones((num_samples,)) * num
        
        current_idx += num_samples
    
    # Flush changes to disk
    data_memmap.flush()
    label_memmap.flush()
    
    # Reload as regular arrays (or keep as memmap if you prefer)
    final_data = np.array(data_memmap)
    final_labels = np.array(label_memmap)
    
    # Clean up temporary files
    del data_memmap
    del label_memmap
    try:
        os.remove(data_memmap_path)
        os.remove(label_memmap_path)
    except:
        pass
    
    return final_data, final_labels

# Usage
path = "/its/home/drs25/Documents/data/Tactile Dataset/nonlinear/pressure"
savepath = "/its/home/drs25/Documents/data/Tactile Dataset/datasets/"

x, y = load_files_memory_efficient(path, "pressure")
np.savez_compressed(savepath + "X_nonlinear_TT", x)
np.savez_compressed(savepath + "y_nonlinear_TT", y)