
from mpremote import pyboard
import serial, time
import sys
import glob
import letRun #This library can be deleted, it is used for debugging
import RoboSkin as sk
import numpy as np
import pickle 

SIZE=0.3
name="C:/Users/dexte/OneDrive/Documents/AI/Data_Labeller/pickle_imputer.pkl"
reg=None
with open(name,'rb') as file:
    reg=pickle.load(file)

def predict(reg1,dat):
    p=reg1.predict(dat)
    p=(p.reshape((p.shape[0],p.shape[1]//2,2))*255/SIZE)
    return p
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
    def getWeight(self):
        return float(self.COM.exec('get_pressure()').decode("utf-8").replace("\r\n",""))
import cv2
class Experiment:
    #17 steps = 1cm on z axis
    def __init__(self,board,device=1,split=10,th=2):
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
            
            #print(np.sum(grid),">",np.sum(grid)/(self.SPLIT*self.SPLIT))
            if np.sum(grid)/(self.SPLIT*self.SPLIT) > self.th: #if touched
                not_touched=False
    def read_vectors(self,old_T):
        im=self.skin.getBinary()
        t_=self.skin.getDots(im)
        t=self.skin.movement(t_)
        return old_T-t
    def run_edge(self,num_samples,mode=1,flat=True,left=True,right=True):
        self.skin.reset()
        old_T=self.skin.origin
        Image=self.skin.getBinary() #get initial image
        self.move_till_touch(Image) #be touching the platform
        fl_dt=np.zeros((num_samples,2))
        l_dt=np.zeros((num_samples,2))
        r_dt=np.zeros((num_samples,2))
        frame=self.skin.getFrame()
        h=frame.shape[1]*SIZE
        w=frame.shape[0]*SIZE
        frame=cv2.resize(frame,(int(h),int(w)),interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).flatten()/255
        past=predict(reg,np.array([frame]))[0]
        initial=past.copy()
        vectors=None
        if flat:
            for i in range(num_samples):
                self.moveZ(0.5,-1) #move down
                if mode==1:vectors=self.read_vectors(old_T)
                elif mode==2:
                    frame_=self.skin.getFrame()
                    frame=cv2.resize(frame_,(int(h),int(w)),interpolation=cv2.INTER_AREA)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).flatten()/255
                    points=predict(reg,np.array([frame]))[0]
                    vectors=initial-points
                fl_dt[i]=np.sum(vectors,axis=0)/len(vectors)
                time.sleep(1)
                self.moveZ(0.5,1) #move up
                time.sleep(1)
        if left:
            for i in range(num_samples):
                self.moveZ(0.5,-1) #move down
                self.moveX(1,-1)
                if mode==1:vectors=self.read_vectors(old_T)
                elif mode==2:
                    frame_=self.skin.getFrame()
                    frame=cv2.resize(frame_,(int(h),int(w)),interpolation=cv2.INTER_AREA)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).flatten()/255
                    points=predict(reg,np.array([frame]))[0]
                    vectors=initial-points
                l_dt[i]=np.sum(vectors,axis=0)/len(vectors)
                time.sleep(1)
                self.moveZ(0.5,1) #move up
                self.moveX(1,1)
                time.sleep(1)
        if right:
            for i in range(num_samples):
                self.moveZ(0.5,-1) #move down
                self.moveX(1,1)
                if mode==1:vectors=self.read_vectors(old_T)
                elif mode==2:
                    frame_=self.skin.getFrame()
                    frame=cv2.resize(frame_,(int(h),int(w)),interpolation=cv2.INTER_AREA)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).flatten()/255
                    points=predict(reg,np.array([frame]))[0]
                    vectors=initial-points
                r_dt[i]=np.sum(vectors,axis=0)/len(vectors)
                time.sleep(1)
                self.moveZ(0.5,1) #move up
                self.moveX(1,-1)
                time.sleep(1)
        return fl_dt,l_dt,r_dt
    def test_speed(self,speeds=[],mode=1):
        #self.skin.reset()
        #self.moveX(1,-1)
        frame=self.skin.getFrame()
        h=frame.shape[1]*SIZE
        w=frame.shape[0]*SIZE
        frame=cv2.resize(frame,(int(h),int(w)),interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).flatten()/255
        past=predict(reg,np.array([frame]))[0]
        initial=past.copy()
        old_T=self.skin.origin
        Image=self.skin.getBinary() #get initial image
        self.move_till_touch(Image) #be touching the platform
        r_dt=np.zeros((len(speeds)))
        v=[]
        for j,sp in enumerate(speeds):
            self.b.setSpeed(sp)
            self.moveZ(0.5,-1) #move down
            self.moveX(0.5-(j/10),1)
            vectors=self.read_vectors(old_T)
            r_dt[j]=np.sum(np.linalg.norm(vectors))
            if mode==1:v.append(self.getVectors())
            elif mode==2:
                frame_=self.skin.getFrame()
                frame=cv2.resize(frame_,(int(h),int(w)),interpolation=cv2.INTER_AREA)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).flatten()/255
                points=predict(reg,np.array([frame]))[0]
                v.append(initial-points)
            
            time.sleep(1)
            self.moveZ(0.5,1) #move up
            self.moveX(0.5-(j/10),-1)
            time.sleep(1)
        self.b.setSpeed(20)
        return r_dt,v 
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
        x=[]
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
            a.append((np.max(grid)+np.average(grid))/(self.SPLIT*self.SPLIT))
            x.append(self.b.getWeight())
            self.moveZ(i,1) #move back
        self.moveZ(1,1) #move back
        return a,x
    def getMagnitude(self):
        im=self.skin.getBinary()
        t_=self.skin.getDots(im)
        t=self.skin.movement(t_)
        v=np.zeros(t.shape)
        if t.shape[0]>2:
            for i,coord in enumerate(t): #show vectors of every point
                x1=coord[1]
                y1=coord[0]
                x2=self.old_T[i][1]
                y2=self.old_T[i][0]
                v[i] =np.array([x1-x2,y1-y2])
        return np.sqrt(np.sum(np.square(np.abs(v))))
    def getVectors(self):
        im=self.skin.getBinary()
        t_=self.skin.getDots(im)
        t=self.skin.movement(t_)
        v=np.zeros(t.shape)
        if t.shape[0]>2:
            for i,coord in enumerate(t): #show vectors of every point
                x1=coord[1]
                y1=coord[0]
                x2=self.old_T[i][1]
                y2=self.old_T[i][0]
                v[i] =np.array([x1-x2,y1-y2])
        return v
    def run_pressure_2(self,cm_samples=2,step=0.5):
        a=[]
        Image=self.skin.getBinary() #get initial image
        #self.skin.reset()
        #self.old_T=self.skin.origin
        self.move_till_touch(Image) #be touching the platform
        x=[]
        v=[]
        for i in np.arange(0, cm_samples, step):
            #print("depth:",i)
            mag=self.getMagnitude()
            time.sleep(1)
            self.moveZ(i,-1) #move down
            time.sleep(1)
            mag=self.getMagnitude()
            a.append(mag)
            x.append(self.b.getWeight())
            v.append(self.getVectors())
            self.moveZ(i,1) #move back
        self.moveZ(1,1) #move back
        return a,x,v
    def run_pressure_3(self,cm_samples=2,step=0.5):
        a=[]
        Image=self.skin.getBinary() #get initial image
        #self.skin.reset()
        #self.old_T=self.skin.origin
        self.move_till_touch(Image) #be touching the platform
        x=[]
        v=[]
        frame=self.skin.getFrame()
        h=frame.shape[1]*SIZE
        w=frame.shape[0]*SIZE
        frame=cv2.resize(frame,(int(h),int(w)),interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).flatten()/255
        past=predict(reg,np.array([frame]))[0]
        initial=past.copy()
        for i in np.arange(0, cm_samples, step):
            #print("depth:",i)
            mag=self.getMagnitude()
            time.sleep(1)
            self.moveZ(i,-1) #move down
            time.sleep(1)
            mag=self.getMagnitude()
            frame_=self.skin.getFrame()
            frame=cv2.resize(frame_,(int(h),int(w)),interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).flatten()/255
            points=predict(reg,np.array([frame]))[0]

            a.append(mag)
            x.append(self.b.getWeight())
            v.append(initial-points)
            self.moveZ(i,1) #move back
        self.moveZ(1,1) #move back
        return a,x,v



