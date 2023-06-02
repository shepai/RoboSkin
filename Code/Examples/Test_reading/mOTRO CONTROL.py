import PicoRobotics
import utime
from machine import freq, Pin

class Move:
    def __init__(self,speed=20):
        self.board=PicoRobotics.KitronikPicoRobotics()
        self.speed=speed
        self.board.motorOff(1)
        self.pushX=Pin(4 , Pin.IN, Pin.PULL_UP)
    def moveZ(self,num,speed=20):
        direction="f"
        if num<0:
            direction="r"
        for step in range(abs(num)):
            self.board.step(2,direction,speed)
    def moveX(self,num,stopFunc=None):
        direction="f"
        if num<0:
            direction="r"
        step=0
        stop=False
        if self.pushX.value() or direction=="f": #not touching edge
            self.board.motorOn(1, direction, self.speed)
            while step < abs(num) and not stop:
                utime.sleep(0.3)
                if stopFunc!=None: stop=stopFunc
                if not self.pushX.value():
                    break
                step+=1
            
        self.board.motorOff(1)



b=Move()
"""b.moveZ(-5)
utime.sleep(3)
b.moveZ(5)

b.moveX(-5)
utime.sleep(3)
b.moveX(5)"""