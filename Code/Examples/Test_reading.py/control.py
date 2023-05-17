
from mpremote import pyboard
import serial, time
import sys
import glob
import letRun #This library can be deleted, it is used for debugging
import RoboSkin as sk
import numpy as np

"""
Setup control with micropython device
"""

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
    def moveX(self,num):
        self.COM.exec_raw_no_follow('b.moveX('+str(num)+')')#.decode("utf-8").replace("/r/n","")
    def moveZ(self,num):
        self.COM.exec_raw_no_follow('b.moveZ('+str(num)+')')#.decode("utf-8").replace("/r/n","")
    def close(self):
        self.COM.close()

class Experiment:
    #17 steps = 1cm on z axis
    def __init__(self,board,device=1,split=5):
        self.b=board
        self.path = letRun.path
        self.skin=sk.Skin(device=device) #load skin object using demo video
        self.frame=self.skin.getFrame()
        self.old_T=self.skin.origin
        self.SPLIT=split
        self.dist=-1
        self.image=np.zeros_like(self.skin.getBinary())
        #zero
        for i in range(50):
            im=self.skin.getBinary()
            self.image=self.skin.getForce(im,self.SPLIT,image=self.image,threshold=20,degrade=20,) #get the force push
    def move_till_touch(self):
        not_touched=True
        while not_touched:
            im=self.skin.getBinary()
            self.image=self.skin.getForce(im,self.SPLIT,image=self.image,threshold=20,degrade=20,) #get the force push
            tactile=np.zeros_like(self.frame)
            tactile[:,:,2]=self.image #show push in red
            self.b.moveZ(-1) #move down
            #print(np.sum(tactile)/(self.SPLIT*self.SPLIT*255))
            if np.sum(tactile)/(self.SPLIT*self.SPLIT*255) > 3: #if touched
                not_touched=False
    def run_edge(self):
        pass
    def moveZ(self,cm,dir): #dir must be 1 or -1
        assert dir==1 or dir==-1, "Incorrect direction, must be 1 or -1"
        cm=cm*17 #17 steps per cm
        for i in range(0,round(cm)):
            ex.b.moveZ(1*dir) #move up
    def run_pressure(self,cm_samples=2,step=0.5):
        a=[]
        self.move_till_touch() #be touching the platform
        for i in np.arange(0, cm_samples, step):
            im=self.skin.getBinary()
            self.image=self.skin.getForce(im,self.SPLIT,image=self.image,threshold=20,degrade=20,) #get the force push
            time.sleep(1)
            self.moveZ(i,-1) #move down
            time.sleep(1)
            im=self.skin.getBinary()
            self.image=self.skin.getForce(im,self.SPLIT,image=self.image,threshold=20,degrade=20,) #get the force push
            a.append(np.sum(self.image)/(self.SPLIT*self.SPLIT*255))
            self.moveZ(i,1) #move back
        return a


B=Board()
#get serial boards and connect to first one
COM=""
while COM=="":
    try:
        res=B.serial_ports()
        print("ports:",res)
        B.connect(res[0])
        B.runFile("C:/Users/dexte/github/RoboSkin/Code/Examples/Test_reading.py/mOTRO CONTROL.py")
        COM=res[0]
    except IndexError:
        time.sleep(1)

ex=Experiment(B)
ex.moveZ(1,1)
a=ex.run_pressure()
print(a)