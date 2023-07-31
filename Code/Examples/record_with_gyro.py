import letRun #This library can be deleted, it is used for debugging
import RoboSkin as sk
import numpy as np
import cv2

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


B=Board()
#get serial boards and connect to first one
COM=""
while COM=="":
    try:
        res=B.serial_ports()
        print("ports:",res)
        B.connect(res[0])
        B.runFile("C:/Users/dexte/github/RoboSkin/Code/Examples/gyro/code.py")
        COM=res[0]
    except IndexError:
        time.sleep(1)


path= letRun.path
skin=sk.Skin(device=1)#videoFile=path+"Movement4.avi") #load skin object using demo video
skin.sharpen=False #falsify the sharpness if recorded with sharpness
p=skin.getFrame()
h1, w1 = p.shape[:2]
NAME="edge_detection"
out = cv2.VideoWriter(path+NAME+".avi",cv2.VideoWriter_fourcc(*'DIVX'), 120, (w1,h1))
ar=[]
while True:
    frame=skin.getFrame()
    cv2.imshow('frame', frame)
    ar.append(B.getData())
    q=cv2.waitKey(1) 
    if q & 0xFF == ord('q'):
        break
    out.write(frame) #uncomment to record video

np.save(path+NAME+'_gyro',np.array(ar))
out.release()