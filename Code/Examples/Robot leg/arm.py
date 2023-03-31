import numpy as np
import math as maths
from mpremote import pyboard
import serial, time
import sys
import glob


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
    def runFile(self,fileToRun='main.py'):
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
    def move(self,motor,angle):
        if angle<180 and angle>0:
            self.COM.exec_raw_no_follow('moveMotor('+str(motor)+','+str(angle)+')')#.decode("utf-8").replace("\r\n","")
    def close(self):
        self.COM.close()


class Leg:
    def __init__(self):
        self.Board=Board()
        #get serial boards and connect to first one
        COM=""
        while COM=="":
            try:
                res=self.Board.serial_ports()
                print("ports:",res)
                self.Board.connect(res[0])
                COM=res[0]
            except IndexError:
                time.sleep(1)
        self.Board.runFile("/its/home/drs25/Documents/GitHub/RoboSkin/Code/Examples/Robot leg/main_program.py")
        
        self.angle1=160
        self.angle2=10
        self.angle3=130
        self.x=10
        self.x1=0
        self.d=20
        self.A=160
        self.B=maths.degrees(maths.acos((self.x)/self.d))
        self.C=130
    def moveSpeed(self,num,angleStart,angleEnd,t=0.05):
        angleEnd=int(angleEnd)
        angleStart=int(angleStart)
        if angleEnd!=angleStart:
            rang=None
            if angleStart>angleEnd: rang=reversed(range(angleEnd,angleStart,5))
            else: rang=range(angleStart,angleEnd,5)
            for i in rang:
                self.Board.move(num,i)
                time.sleep(t)
    def setStart(self):
        self.A=self.angle1
        self.B=self.angle2
        self.C=self.angle3
    def startPos(self):
        #self.B.move(1,170)
        #self.B.move(2,maths.degrees(maths.acos((self.x)/self.d)))
        #self.B.move(3,140)
        self.moveSpeed(1,self.angle1,self.A)
        self.moveSpeed(2,self.angle2,self.B)
        self.moveSpeed(3,self.angle3,self.C)
        self.angle1=self.A
        self.angle2=self.B
        self.angle3=self.C
    def moveX(self,x):
        angle1=self.angle2
        angle2=self.angle3
        mov=maths.asin((1)/self.d) * x #one unit of distance times distance
        #mov2=maths.acos((x)/self.d)
        angle1=angle1-maths.degrees(mov)
        angle2=angle2+maths.degrees(mov)
        self.Board.move(2,angle1)
        self.Board.move(3,angle2)
        self.angle2=angle1
        self.angle3=angle2
    def move(self,num,Bydegrees):
        deg=[self.angle1,self.angle2,self.angle3]
        d=deg[num-1]
        self.Board.move(num,d+Bydegrees)
        if d+Bydegrees>0 and d+Bydegrees<180:
            if num==1: self.angle1=d+Bydegrees
            elif num==2: self.angle2=d+Bydegrees
            elif num==3: self.angle3=d+Bydegrees
        #print(maths.degrees(mov),self.angle2,self.angle3)
    def moveY(self,y):
        pass
    def close(self):
        self.Board.close()

