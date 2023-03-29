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
        self.COM.exec_raw_no_follow('moveMotor('+str(motor)+','+str(angle)+')')#.decode("utf-8").replace("\r\n","")
    def close(self):
        self.COM.close()


class Leg:
    def __init__(self):
        self.B=Board()
        #get serial boards and connect to first one
        COM=""
        while COM=="":
            try:
                res=self.B.serial_ports()
                print("ports:",res)
                self.B.connect(res[0])
                COM=res[0]
            except IndexError:
                time.sleep(1)
        self.B.runFile("/its/home/drs25/Documents/GitHub/RoboSkin/Code/Examples/Robot leg/main_program.py")
        self.angle1=0
        self.angle2=0
        self.angle3=0
        self.x=10
        self.x1=0
        self.d=20
    def startPos(self):
        self.B.move(1,170)
        self.B.move(2,maths.degrees(maths.acos((self.x)/self.d)))
        self.B.move(3,140)
        self.angle1=170
        self.angle2=20
        self.angle3=140
    def moveX(self,x):
        angle1=self.angle2
        angle2=self.angle3
        mov=maths.asin((x)/self.d)
        #mov2=maths.acos((x)/self.d)
        angle1=angle1-maths.degrees(mov)
        angle2=angle2+maths.degrees(mov)
        self.B.move(2,angle1)
        self.B.move(3,angle2)
        self.angle2=angle1
        self.angle3=angle2

        print(maths.degrees(mov),self.angle2,self.angle3)
    def moveY(self,y):
        pass
    def close(self):
        self.B.close()

