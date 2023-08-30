import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split

import letRun #This library can be deleted, it is used for debugging
import RoboSkin as sk
import cv2 
import pickle
from sklearn import preprocessing
import os

#if linux

torch.cuda.empty_cache() 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.version.cuda)
print("GPU:",torch.cuda.is_available())

path="C:/Users/dexte/github/RoboSkin/Code/Models/labeller/"
if os.name == 'nt':
    path="C:/Users/dexte/github/RoboSkin/Code/Models/labeller/" #use standard imputer or one for small
else:
    path="/its/home/drs25/RoboSkin/Code/Models/labeller/" #for linux
    
def sigmoid(x):                                        
   return 1 / (1 + np.exp(-x))


##DATA SET CREATOR
class dataset:
    def __init__(self,names=["flat_detection.avi","edge_detection.avi","flat_slip_detection.avi","soft_detection.avi","flat_slip_detection_paper.avi","nothing.avi"]):
        self.path=letRun.path
        self.names=names
        self.SIZE=0.3
        name=""
        if os.name == 'nt':
            name="C:/Users/dexte/OneDrive/Documents/AI/Data_Labeller/pickle_imputer_small.pkl" #use standard imputer or one for small
        else:
            name="/its/home/drs25/RoboSkin/Code/Models/TacTip reader/pickle_imputer_small.pkl" #for linux
        self.reg=None
        with open(name,'rb') as file:
            self.reg=pickle.load(file)
    def gen_image_data(self,STORE=5,y_labels=[],scale=False):
        BIG_DATA_X=None
        BIG_DATA_y=None
        assert len(y_labels)>=len(self.names),"Incorrect length of labels"
        for j,name in enumerate(self.names):
            skin=sk.Skin(videoFile=self.path+name)#videoFile=path+"Movement4.avi") #load skin object using demo video
            cap = cv2.VideoCapture(self.path+name)
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            skin.sharpen=False #falsify the sharpness if recorded with sharpness
            frame=skin.getFrame()
            h=frame.shape[1]*self.SIZE
            w=frame.shape[0]*self.SIZE
            frame=cv2.resize(frame,(int(h),int(w)),interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).flatten()/255
            
            initial_frame=frame.copy()
            counter=0
            X=[]
            y=[] #label as [edge surface soft hard slippery]
            lastFrames=[]
            gyro=np.zeros((length,6))
            try:
                gyro=np.load(self.path+name.replace(".avi","")+"_gyro.npy")
            except:
                pass
            for i in range(length): #lop through all
                frame_=skin.getFrame()
                frame=cv2.resize(frame_,(int(h),int(w)),interpolation=cv2.INTER_AREA)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                #get pressure map
                diff=np.sum(np.abs(initial_frame-frame.flatten()))/(frame.flatten().shape[0]*3)
                frame = np.uint8(frame)
                #current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #
                frame = cv2.adaptiveThreshold(
                        frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 1
                    )
                kernel = np.ones((2, 2), np.uint8)
                frame = cv2.erode(frame, kernel, iterations=1)
                lastFrames.append(frame/255)
                if len(lastFrames)>STORE: lastFrames.pop(0)
                if diff>0.01 and len(lastFrames)==STORE: #significant contact
                    #all_frames=np.zeros(())
                    X.append(np.array(lastFrames).reshape((int(w)*STORE,int(h)))) #store temporal element
                    y.append(y_labels[j])
                    counter+=1

            if type(BIG_DATA_y)==type(None):
                BIG_DATA_y=np.array(y.copy())
                BIG_DATA_X=np.array(X.copy())
            else:
                BIG_DATA_y= np.concatenate((np.array(y.copy()),BIG_DATA_y))
                BIG_DATA_X= np.concatenate((np.array(X.copy()),BIG_DATA_X))
            print(name,"Length:",counter,"/",str(length),"X-size:",BIG_DATA_X.shape)
        print("Completed creation",np.average(BIG_DATA_X),np.std(BIG_DATA_X))
        if scale:
            """scaler1 = preprocessing.StandardScaler().fit(a)
            a = scaler1.transform(a)"""
            BIG_DATA_X=(BIG_DATA_X-np.average(BIG_DATA_X))/np.std(BIG_DATA_X)
        return BIG_DATA_X,BIG_DATA_y
    


#d=dataset()


class SimpleConv2DNeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size,layers=[1000,500,200],drop_out_prob=0.2):
        super(SimpleConv2DNeuralNetwork, self).__init__()
        input_channels=1 #greyscale
        self.conv_layer = nn.Conv2d(in_channels=input_channels, out_channels=8, kernel_size=3)
        # Add 2D convolutional layer
        self.conv_layer = nn.Conv2d(in_channels=input_channels, out_channels=8, kernel_size=3)
        
        # Calculate the size of the tensor after convolution and pooling
        conv_output_size = self._get_conv_output_size(input_channels, input_size[0], input_size[1])  # Adjust the last two dimensions
        # Add hidden layers
        self.fc=[nn.Linear(int(conv_output_size), layers[0])]
        self.fc.append(nn.Sigmoid())
        self.fc.append(nn.Dropout(p=drop_out_prob))
        for i in range(1,len(layers)): #create layers 
                self.fc.append(nn.Linear(layers[i-1], layers[i]))
                self.fc.append(nn.Sigmoid())
                self.fc.append(nn.Dropout(p=drop_out_prob))
        self.fc.append(nn.Linear(layers[-1], output_size))
        self.fc_layers = nn.Sequential(*self.fc)
    def forward(self, x):
        x=self.conv_layer(x)
        x = torch.relu(x)
        # Reshape the tensor to match the input size of the first fully connected layer
        x = x.view(x.size(0), -1)
        x=self.fc_layers(x)
        return x
    def _get_conv_output_size(self, input_channels, height, width):
        dummy_input = torch.zeros(1, input_channels, height, width)
        dummy_output = self.conv_layer(dummy_input)
        return dummy_output.view(-1).size(0)

# After training, you can use the trained model for predictions on new data.
# For example, if you have new input data 'X_new', you can do:


def get_acc(predictions,should_be=[]):
    predictions*=10 #normalize
    pred_array=np.zeros_like(predictions)

    if len(predictions[0])>4:
        inds=np.argmax(predictions[:,0:2],axis=1) #convert at threshold
        inds2=np.argmax(predictions[:,2:-1],axis=1) #convert at threshold
        for i in range(len(pred_array)):
            pred_array[i][inds[i]]=1
            pred_array[i][2+inds2[i]]=1
    else:
        inds=np.argmax(predictions,axis=1) #convert at threshold
        for i in range(len(pred_array)):
            pred_array[i][inds[i]]=1
    correct=0
    for j,pred in enumerate(pred_array): #loopthrough predictions
        inds=np.where(pred==1) #
        if len(should_be)>0: #validation task
            sb=should_be[j]*10
            sb=np.where(sb==1)
            if len(sb[0])==len(inds[0]):
                c=0
                for i in range(len(sb[0])):
                    if sb[0][i]==inds[0][i]: c+=1
                if c==len(sb[0]): correct+=1
    return correct/len(should_be)

def create_dataset(array,t):
    d=dataset(names=array)
    d.SIZE=0.2
    #X,ya=d.gen_image_data(STORE=5,y_labels=[[1,0,0,0],[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],scale=True)
    X,ya=d.gen_image_data(STORE=t,y_labels=[[1,0],[0,1]],scale=True)

    #x,y =d.generate(STORE=5,y_labels=[[1,0,0],[1,0,0],[0,0,1]],scale=True)
    print(np.sum(X))
    #Xa=X/10
    ya=ya/10
    #y=y/10
    X, data_test, Y, labels_test = train_test_split(X, ya, test_size=0.20, random_state=42)
    #X, data_test, Y, labels_test = train_test_split(x, y , test_size=0.20, random_state=42)

    print(X.shape,Y.shape)
    print(data_test.shape,labels_test.shape)
    # Define the size of the input (n) and output (m) layers
    
    return X, data_test, Y, labels_test


def train(model,num_epochs,output=True):
    loss_ar=[]
    accuracy=[]
    for epoch in range(num_epochs):
        # Forward pass
        y_pred = model(X_tensor.float())

        # Compute the loss
        loss = criterion(y_pred, y_tensor)

        # Zero gradients, backward pass, and update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_ar.append(loss.item())
        #predict
        with torch.no_grad():
            model.eval()
            predictions = model(X_tensor)
        a=get_acc(predictions.cpu(),should_be=Y)
        accuracy.append(a)
        # Print the current loss to monitor training progress
        if epoch%100==0 and output:
            with torch.no_grad():
                 model.eval()
                 predictions = model(torch.tensor(data_test, dtype=torch.float32).view(data_test.shape[0],1,data_test.shape[1],data_test.shape[2]).to(device))

            a=get_acc(predictions.cpu(),should_be=labels_test)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}","Accuracy Train:",accuracy[-1],"Accuracy test:",a)
    return np.array(loss_ar),np.array(accuracy)

trials=20
a2=np.zeros((len(range(0,6)),trials,500))
for i in range(0,6):
    torch.cuda.empty_cache() 
    d=dataset(names=["lego.avi","smooth.avi"])
    size=0.3-(0.05*i)
    d.SIZE=size
    X,ya=d.gen_image_data(STORE=5,y_labels=[[1,0],[0,1]],scale=True)
    print(np.sum(X))
    ya=ya/10
    X, data_test, Y, labels_test = train_test_split(X, ya, test_size=0.20, random_state=42)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)
    n_inputs = X.shape[1]
    m_outputs = Y.shape[1]
    for j in range(trials):
        # Create the neural network
        #Create the neural network
        model = SimpleConv2DNeuralNetwork([X.shape[1], X.shape[2]], m_outputs,layers=[1000],drop_out_prob=0.1).to(device)
        #model=Network().to(device)

        # Define the loss function and optimizer
        criterion = nn.MSELoss()#nn.CrossEntropyLoss() #nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001)
        loss,accuracy=train(model,500)
        a2[i][j]=accuracy
        

np.save("/its/home/drs25/RoboSkin/Code/Models/surfaceModel/accuracyRes",a2)


print("*************************************")
print("Lego classification: ",np.max(a2))
print("*************************************")