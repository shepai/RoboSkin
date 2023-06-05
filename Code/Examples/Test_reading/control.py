
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
    def setSpeed(self,speed):
        self.COM.exec_raw_no_follow('b.speed='+str(speed))#.decode("utf-8").replace("/r/n","")
    def close(self):
        self.COM.close()

class Experiment:
    #17 steps = 1cm on z axis
    def __init__(self,board,device=1,split=5,th=2):
        self.b=board
        self.path = letRun.path
        self.skin=sk.Skin(device=device) #load skin object using demo video
        self.frame=self.skin.getFrame()
        self.old_T=self.skin.origin
        self.SPLIT=split
        self.dist=-1
        self.th=th
        self.IIMM=None
    def move_till_touch(self,im__=None):
        not_touched=True
        #zero
        self.image=np.zeros_like(self.skin.getBinary())
        for i in range(50):
            im=self.skin.getBinary()
            self.image,grid=self.skin.getForceGrid(im,self.SPLIT,image=self.image,threshold=20,degrade=20,past=im__) #get the force push
        while not_touched: #loop till touching surface
            im=self.skin.getBinary()
            self.image,grid=self.skin.getForceGrid(im,self.SPLIT,image=self.image,threshold=20,degrade=20,past=im__) #get the force push
            tactile=np.zeros_like(self.frame)
            tactile[:,:,2]=self.image #show push in red
            self.b.moveZ(-1) #move down
            print(np.sum(grid),">",np.sum(grid)/(self.SPLIT*self.SPLIT))
            if np.sum(grid)/(self.SPLIT*self.SPLIT) > self.th: #if touched
                not_touched=False
    def read_vectors(self,old_T):
        im=self.skin.getBinary()
        t_=self.skin.getDots(im)
        t=self.skin.movement(t_)
        return old_T-t
    def run_edge(self,num_samples,flat=True,left=True,right=True):
        self.skin.reset()
        old_T=self.skin.origin
        Image=self.skin.getBinary() #get initial image
        self.move_till_touch(Image) #be touching the platform
        fl_dt=np.zeros((num_samples,2))
        l_dt=np.zeros((num_samples,2))
        r_dt=np.zeros((num_samples,2))
        if flat:
            for i in range(num_samples):
                self.moveZ(0.5,-1) #move down
                vectors=self.read_vectors(old_T)
                fl_dt[i]=np.sum(vectors,axis=0)/len(vectors)
                time.sleep(1)
                self.moveZ(0.5,1) #move up
                time.sleep(1)
        if left:
            for i in range(num_samples):
                self.moveZ(0.5,-1) #move down
                self.moveX(1,-1)
                vectors=self.read_vectors(old_T)
                l_dt[i]=np.sum(vectors,axis=0)/len(vectors)
                time.sleep(1)
                self.moveZ(0.5,1) #move up
                self.moveX(1,1)
                time.sleep(1)
        if right:
            for i in range(num_samples):
                self.moveZ(0.5,-1) #move down
                self.moveX(1,1)
                vectors=self.read_vectors(old_T)
                r_dt[i]=np.sum(vectors,axis=0)/len(vectors)
                time.sleep(1)
                self.moveZ(0.5,1) #move up
                self.moveX(1,-1)
                time.sleep(1)
        return fl_dt,l_dt,r_dt
    def test_speed(self,speeds=[]):
        self.skin.reset()
        #self.moveX(1,-1)
        old_T=self.skin.origin
        Image=self.skin.getBinary() #get initial image
        self.move_till_touch(Image) #be touching the platform
        r_dt=np.zeros((len(speeds)))
        for j,sp in enumerate(speeds):
            self.b.setSpeed(sp)
            self.moveZ(0.5,-1) #move down
            self.moveX(0.5-(j/10),1)
            vectors=self.read_vectors(old_T)
            r_dt[j]=np.sum(np.linalg.norm(vectors))
            time.sleep(1)
            self.moveZ(0.5,1) #move up
            self.moveX(0.5-(j/10),-1)
            time.sleep(1)
        self.b.setSpeed(20)
        return r_dt
    def moveZ(self,cm,dir): #dir must be 1 or -1
        assert dir==1 or dir==-1, "Incorrect direction, must be 1 or -1"
        cm=cm*17 #17 steps per cm
        for i in range(0,round(cm)):
            im=self.skin.getBinary()
            if type(self.IIMM)!=type(None): self.image,grid=self.skin.getForceGrid(im,self.SPLIT,image=self.image,threshold=20,degrade=20,past=self.IIMM) #get the force push
            self.b.moveZ(1*dir) #move up
    def moveX(self,cm,dir): #dir must be 1 or -1
        assert dir==1 or dir==-1, "Incorrect direction, must be 1 or -1"
        cm=cm*26 #26 steps per cm
        for i in range(0,round(cm)):
            im=self.skin.getBinary()
            if type(self.IIMM)!=type(None): self.image,grid=self.skin.getForceGrid(im,self.SPLIT,image=self.image,threshold=20,degrade=20,past=self.IIMM) #get the force push
            self.b.moveX(1*dir) #move up
    def run_pressure(self,cm_samples=2,step=0.5):
        a=[]
        Image=self.skin.getBinary() #get initial image
        self.IIMM=Image.copy()
        self.move_till_touch(Image) #be touching the platform
        for i in np.arange(0, cm_samples, step):
            #print("depth:",i)
            im=self.skin.getBinary()
            self.image,grid=self.skin.getForceGrid(im,self.SPLIT,image=self.image,threshold=20,degrade=20,past=Image) #get the force push
            time.sleep(1)
            self.moveZ(i,-1) #move down
            time.sleep(1)
            im=self.skin.getBinary()
            self.image=self.skin.getForce(im,self.SPLIT,image=self.image,threshold=20,degrade=20,past=Image) #get the force push
            a.append(np.sum(grid)/(self.SPLIT*self.SPLIT))
            self.moveZ(i,1) #move back
        self.moveZ(1,1) #move back
        return a



