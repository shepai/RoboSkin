import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import math as maths
import random
from mpremote import pyboard
import serial, time
import sys
import glob
from multiprocessing import Process

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


class arm:
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
        self.angle1=0
        self.angle2=0
        self.angle3=0
    def startPos(self):
        self.B.move(1,170)
        self.B.move(2,20)
        self.B.move(3,140)
        self.angle1=170
        self.angle2=20
        self.angle3=140
    def moveX(self,num):
        angle1=self.angle1
        angle2=self.angle2
        


