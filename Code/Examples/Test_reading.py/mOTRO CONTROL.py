import PicoRobotics
import utime

class Move:
    def __init__(self,speed=20):
        self.board=PicoRobotics.KitronikPicoRobotics()
        self.speed=speed
        self.board.motorOff(1)
    def moveZ(self,num):
        direction="f"
        if num<0:
            direction="r"
        for step in range(abs(num)):
            self.board.step(2,direction,self.speed)
    def moveX(self,num,stopFunc=None):
        direction="f"
        if num<0:
            direction="r"
        step=0
        stop=False
        self.board.motorOn(1, direction, self.speed)
        while step < abs(num) and not stop:
            utime.sleep(0.3)
            if stopFunc!=None: stop=stopFunc
            step+=1
            
        self.board.motorOff(1)
b=Move()
"""b.moveZ(-5)
utime.sleep(3)
b.moveZ(5)

b.moveX(-5)
utime.sleep(3)
b.moveX(5)"""