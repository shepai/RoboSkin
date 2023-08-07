import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import letRun #This library can be deleted, it is used for debugging
import RoboSkin as sk
import cv2 
import pickle
from sklearn import preprocessing
import os

from mpremote import pyboard
import serial, time,sys,glob

class Board:
    def __init__(self):
        self.COM=None
        self.VALID_CHANNELS=[i for i in range(10)]
    def connect(self,com):
        """
        Connect to a com given
        @param com of serial port
        @param fileToRun is the file that executes on board to allow hardware interfacing
        """
        self.COM=pyboard.Pyboard(com) #connect to board
        self.COM.enter_raw_repl() #open for commands
        print("Successfully connected")
    def runFile(self,fileToRun='mOTRO CONTROL.py'):
        """
        runFile allows the user to send a local file to the device and run it
        @param fileToRun is the file that will be locally installed
        """
        if self.COM==None:
            raise OSError("Connect to device before running file")
        self.COM.execfile(fileToRun) #run the file controlling the sensors
    def serial_ports(self):
        """
        Read the all ports that are open for serial communication
        @returns array of COMS avaliable
        """
        if sys.platform.startswith('win'):
            ports = ['COM%s' % (i + 1) for i in range(256)]
        elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
            # this excludes your current terminal "/dev/tty"
            ports = glob.glob('/dev/tty[A-Za-z]*')
        elif sys.platform.startswith('darwin'):
            ports = glob.glob('/dev/tty.*')
        else:
            raise EnvironmentError('Unsupported platform') #if the OS is not one of the main

        result = []
        for port in ports:
            try:
                s = serial.Serial(port)
                s.close()
                result.append(port)
            except (OSError, serial.SerialException):
                pass
        return result
    def getData(self):
        return np.array(self.COM.exec('get_data()').decode("utf-8").replace("\r\n","").split()).astype(float)
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.version.cuda)
print("GPU:",torch.cuda.is_available())

class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size,layers=[500,100,50],drop_out_prob=0.2):
        super(SimpleNeuralNetwork, self).__init__()
        self.fc=[nn.Linear(input_size, layers[0])]
        self.fc.append(nn.Sigmoid())
        self.fc.append(nn.Dropout(p=drop_out_prob))
        for i in range(1,len(layers)): #create layers 
                self.fc.append(nn.Linear(layers[i-1], layers[i]))
                self.fc.append(nn.Sigmoid())
                self.fc.append(nn.Dropout(p=drop_out_prob))
        self.fc.append(nn.Linear(layers[-1], output_size))
        self.fc_layers = nn.Sequential(*self.fc)
    def forward(self, x):
        x=self.fc_layers(x)
        return x


T=5
# Create the neural network
model = SimpleNeuralNetwork(T*((133*2)+6), 4).to(device)

model.load_state_dict(torch.load("/its/home/drs25/Documents/GitHub/RoboSkin/Code/Models/surfaceModel/state/model.pth"))
model.eval()

B=Board()
#get serial boards and connect to first one
COM=""
while COM=="":
    try:
        res=B.serial_ports()
        print("ports:",res)
        B.connect(res[0])
        B.runFile("/its/home/drs25/Documents/GitHub/RoboSkin/Code/Examples/gyro/code.py") #C:/Users/dexte/github/RoboSkin/Code/Examples/gyro/
        COM=res[0]
    except IndexError:
        time.sleep(1)


SIZE=0.3
name=letRun.path.replace("Assets/Video demos/","Code/Models/TacTip reader/")+"pickle_imputer.pkl" #use standard imputer or one for small
reg=None
with open(name,'rb') as file:
    reg=pickle.load(file)

def predict(reg,dat):
    p=reg.predict(dat)
    p=(p.reshape((p.shape[0],p.shape[1]//2,2))*255/SIZE)
    return p

path= letRun.path
skin=sk.Skin(device=0)#videoFile=path+"push.avi" #load skin object using demo video
skin.sharpen=False #falsify the sharpness if recorded with sharpness
frame=skin.getFrame()
h=frame.shape[1]*SIZE
w=frame.shape[0]*SIZE
frame=cv2.resize(frame,(int(h),int(w)),interpolation=cv2.INTER_AREA)
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).flatten()/255
past=predict(reg,np.array([frame]))[0]
initial=past.copy()

ar=[]
vectors=[]

options=["hard","soft","slippery","no touch"]

while True:
    frame_=skin.getFrame()
    frame=cv2.resize(frame_,(int(h),int(w)),interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).flatten()/255
    points=predict(reg,np.array([frame]))[0]
    for j,_ in enumerate(points):
        cv2.arrowedLine(frame_,(int(initial[j][0]),int(initial[j][1])),(int(_[0]),int(_[1])), (0, 0, 0), thickness=2)
    ar.append(B.getData())
    
    vecs=initial-points
    vectors.append(vecs)
    if len(vectors)>T: #only save within temporal size of T
        vectors.pop(0)
        ar.pop(0)
        #make array
        a=[]
        for i in range(T):
            a.append(np.concatenate((vectors[i].flatten(),ar[i].flatten())))
        a=np.array(a).flatten()
        a=(a-np.average(a))/np.std(a)
        predictions = model(torch.tensor(a.reshape(1,a.shape[0]), dtype=torch.float32).to(device)).cpu().detach().numpy()
        #print(predictions)
        ind=np.argmax(predictions[0])
        cv2.putText(frame_,"Prediction: "+options[ind],(int(50),int(50)),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0))

    cv2.imshow('Image', frame_)
    #cv2.imshow('sharp',sharp_image)
    q=cv2.waitKey(1) 
    if q & 0xFF == ord('q'):
        break


    
