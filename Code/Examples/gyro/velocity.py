from mpremote import pyboard
import serial, time,sys,glob
import numpy as np
import matplotlib.pyplot as plt
import time

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


# Constants
time_interval = 0.1  # Time interval between consecutive accelerometer readings (seconds)

# Initialize variables
velocity = np.zeros(3)  # Initial velocity in x, y, and z directions
time_interval=time.time()
past=B.getData()[0:3]
try:
    while True:
        acceleration = B.getData()[0:3]  # Get acceleration data from the accelerometer
        t=time.time()-time_interval
        time_interval=time.time()
        # Convert acceleration data to SI units (if necessary)
        # For example, if the data is in g, you can convert it to m/s² by multiplying with 9.81 m/s² (standard gravity)
        # acceleration *= 9.81

        # Integrate acceleration to get velocity
        velocity = acceleration * t

        # Do something with the velocity data, such as displaying it or using it for further calculations
        print("Velocity:", velocity)

except KeyboardInterrupt:
    # Stop the loop when Ctrl+C is pressed
    pass
