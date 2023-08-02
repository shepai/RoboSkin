
from mpremote import pyboard
import serial, time,sys,glob
import numpy as np
import matplotlib.pyplot as plt

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


# Initialize an empty 2x2 grid of subplots for the sensor readings
fig, axs = plt.subplots(3, figsize=(12, 10))
plt.ion()  # Turn on interactive mode for real-time plotting
arr=[[0],[0],[0]]
vel=[[0],[0],[0]]
MAX_LEN=1000
dt=0.1
try:
    while True:
        data = B.getData()  # Get data from the sensor
        arr[0].append(data[0])
        arr[1].append(data[1])
        arr[2].append(data[2])
        vel[0].append(vel[0][-1]+arr[0][-1]*dt)
        vel[1].append(vel[1][-1]+arr[1][-1]*dt)
        vel[2].append(vel[2][-1]+arr[2][-1]*dt)
        if len(arr[0])>MAX_LEN: #prevent getting too large
            arr[0].pop(0)
            arr[1].pop(1)
            arr[2].pop(2)
            vel[0].pop(0)
            vel[1].pop(1)
            vel[2].pop(2)
        for i in range(3):
            axs[i].cla()  # Clear the previous plot
            axs[i].plot(arr[i],c="b")  # Plot the data for the corresponding sensor reading
            axs[i].plot(vel[i],c="r")  # Plot the data for the corresponding sensor reading
            axs[i].set_title(f'Sensor {i} Reading')  # Set title for the subplot

        plt.pause(0.1)  # Pause for a short duration to update the plot
except KeyboardInterrupt:
    # Stop the loop when Ctrl+C is pressed
    pass

plt.ioff()  # Turn off interactive mode after the loop
plt.show()  # Display the final plot