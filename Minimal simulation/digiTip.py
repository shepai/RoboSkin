import numpy as np
import matplotlib.pyplot as plt

class environment:
    def __init__(self,x,y):
        self.grid=(np.random.random((y,x))*10).astype(int)
    def get(self):
        return self.grid

class digiTip:
    def __init__(self,x,y,startPos=(25,25)):
        self.grid=np.zeros((y,x))
        self.h=10
        self.pos=startPos
    def lower(self,amount):
        self.h-=amount
    def getImage(self,env):
        y=self.pos[0]+self.grid.shape[0]
        x=self.pos[1]+self.grid.shape[1]
        subArray=env[self.pos[0]:y,self.pos[1]:x]
        touch=subArray-self.h
        touch[touch<0]=0
        for i in range(len(touch)):
            for j in range(len(touch[0])):
                touch=self.expand(touch,i,j)
        touch=touch*50
        touch[touch>80]=80
        return touch
    def expand(self,arr,x,y):
        item=arr[y][x]
        if item<=0: return arr
        else: #if has number decrease the outside
            if y-1>=0:
                if arr[y-1][x]==0: 
                    arr[y-1][x]+=item-1
                    arr=self.expand(arr,x,y-1)
            if y+1<len(arr):
                if arr[y+1][x]==0:
                    arr[y+1][x]+=item-1
                    arr=self.expand(arr,x,y+1)
            if x+1<len(arr[0]):
                if arr[y][x+1]==0:
                    arr[y][x+1]+=item-1
                    arr=self.expand(arr,x+1,y)
            if x-1>=0:
                if arr[y][x-1]==0:
                    arr[y][x-1]+=item-1
                    arr=self.expand(arr,x-1,y)
        return arr
            


env=environment(50,50) #create environment
tip=digiTip(10,10) #create tactip


for i in range(10):
    tip.h-=1
    im=tip.getImage(env.get())
    plt.imshow(im)
    plt.show()

