import numpy as np 
import pandas as pd 
import os 
import cv2
#create dataframe for labels 
df_empty = pd.DataFrame({
    "Index":[],
    "Texture_class":[],
    "Non_linear":[]
})
labels=['carpet', 'lacedmatt', 'wool', 'cork', 'Felt', 'longcarpet', 'cotton', 'plastic', 'flat', 'ffoam', 'gfoam', 'bubble', 'efoam', 'jeans', 'leather']
keys = {labels[i].lower():i for i in range(len(labels))}
keys['foame']=keys['efoam']
keys['foamg']=keys['gfoam']
keys['foamf']=keys['ffoam']
def downscale_videos(images, scale=0.25):
    N, T, H, W = images.shape
    new_H, new_W = int(H * scale), int(W * scale)

    scaled = np.empty((N, T, new_H, new_W), dtype=images.dtype)
    for n in range(N):
        for t in range(T):
            scaled[n, t] = cv2.resize(images[n, t], (new_W, new_H), interpolation=cv2.INTER_AREA)
    return scaled
#loop through first folder, open each file, perform cropping ect... 
def get_temp(directory_in_str,temp_name,frame,T=5,start=0):
    c=start
    X=[]
    directory = os.fsencode(directory_in_str)
    for per,file in enumerate(os.listdir(directory)):
        filename = os.fsdecode(file)
        if filename.endswith(".npy"): 
            # print(os.path.join(directory, filename))
            try:
                images=np.load(directory_in_str+filename)
                words=filename.split("_")
                if len(images.shape)>6: 
                    texture=words[1]
                    nl=0
                    images=images.reshape(images.shape[0]*images.shape[1]*images.shape[2]*images.shape[3],images.shape[4],images.shape[5],images.shape[6])
                else:
                    texture=words[2]
                    nl=1
                    images=images.reshape(images.shape[0]*images.shape[1]*images.shape[2],images.shape[3],images.shape[4],images.shape[5])
                for i in range(0,images.shape[1]-T,T):
                    imagesub=images[:,i:i+T,:].astype(np.uint8)
                    scaled = downscale_videos(imagesub,0.3)
                    X.append(scaled)
                    #make them cropped, greyscale and 
                    for j in range(len(imagesub)):
                        new_item={"Index":c,"Texture_class":keys[texture.lower()],"Non_linear":nl}
                        frame=pd.concat([frame, pd.DataFrame([new_item])], ignore_index=True)
                        c+=1
            except ValueError:
                pass
        print(len(frame),per,"/",len(os.listdir(directory)))
    X=np.array(X).astype(np.uint8)
    X=X.reshape((X.shape[0]*X.shape[1],T,X.shape[3],X.shape[4]))
    np.save(temp_name,X)
    return frame
# save to temp file
df_empty=get_temp("/mnt/data0/drs25/data/1/","/mnt/data0/drs25/data/temp1",df_empty)
# loop through second folder 
df_empty=get_temp("/mnt/data0/drs25/data/1/1/","/mnt/data0/drs25/data/temp2",df_empty,start=len(df_empty))

#combine temps
ar1=np.load("/mnt/data0/drs25/data/temp1.npy")
ar2=np.load("/mnt/data0/drs25/data/temp2.npy")
X=np.concatenate([ar1,ar2])
np.save("/mnt/data0/drs25/data/optical-tactile-dataset-for-textures/X_MorphB_all",X)
df_empty.to_csv("/mnt/data0/drs25/data/optical-tactile-dataset-for-textures/X_MorphB_all.csv")
#save labels 