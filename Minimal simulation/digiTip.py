import numpy as np
import matplotlib.pyplot as plt
import cv2

class environment:
    def __init__(self,x,y):
        #self.grid=(np.random.random((y,x))*10).astype(int)
        p = np.zeros((y,x))
        for i in range(4):
            freq = 2**i
            lin = np.linspace(0, freq, x, endpoint=False)
            x_, y_ = np.meshgrid(lin, lin)  # FIX3: I thought I had to invert x and y here but it was a mistake
            p = self.perlin(x_, y_, seed=87) / freq + p
        
        while np.max(p)<10: #adjust for size
            p=p*10
        self.grid=p.astype(int)*-1
        self.grid[self.grid<0]=0
        print(self.grid)
    def get(self):
        return self.grid
    def perlin(self,x, y, seed=0):
        # permutation table
        np.random.seed(seed)
        p = np.arange(256, dtype=int)
        np.random.shuffle(p)
        p = np.stack([p, p]).flatten()
        # coordinates of the top-left
        xi, yi = x.astype(int), y.astype(int)
        # internal coordinates
        xf, yf = x - xi, y - yi
        # fade factors
        u, v = self.fade(xf), self.fade(yf)
        # noise components
        n00 = self.gradient(p[p[xi] + yi], xf, yf)
        n01 = self.gradient(p[p[xi] + yi + 1], xf, yf - 1)
        n11 = self.gradient(p[p[xi + 1] + yi + 1], xf - 1, yf - 1)
        n10 = self.gradient(p[p[xi + 1] + yi], xf - 1, yf)
        # combine noises
        x1 = self.lerp(n00, n10, u)
        x2 = self.lerp(n01, n11, u)  # FIX1: I was using n10 instead of n01
        return self.lerp(x1, x2, v)  # FIX2: I also had to reverse x1 and x2 here

    def lerp(self,a, b, x):
        "linear interpolation"
        return a + x * (b - a)

    def fade(self,t):
        "6t^5 - 15t^4 + 10t^3"
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    def gradient(self,h, x, y):
        "grad converts h to the right gradient vector and return the dot product with (x,y)"
        vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
        g = vectors[h % 4]
        return g[:, :, 0] * x + g[:, :, 1] * y


class digiTip:
    def __init__(self,img,startPos=[25,25]):
        self.grid=np.zeros(img.shape[0:2])
        self.h=100
        self.pos=startPos
        self.img=img
    def lower(self,amount):
        self.h-=amount
    def getImage(self,env):
        y=self.pos[0]+self.grid.shape[0]
        x=self.pos[1]+self.grid.shape[1]
        subArray=env[max(min(self.pos[0],env.shape[0]),0):max(min(y,env.shape[0]),0),max(min(self.pos[1],env.shape[1]),0):max(min(x,env.shape[1]),0)]
        touch=subArray-self.h
        touch[touch<0]=0
        for i in range(len(touch)-1):
            for j in range(len(touch[0])-1):
                if self.pos[0]+i>0 and self.pos[0]+i<env.shape[0] and self.pos[1]+j>0 and self.pos[1]+j<env.shape[1]:
                    touch=self.expand(touch,j,i)
        touch[touch<0]=0  
        touch=touch*10
        touch[touch>80]=80
        return touch
    def expand(self,arr,x,y):
        item=arr[y][x]
        if item<=0: return arr
        else: #if has number decrease the outside
            if y-1>=0:
                if arr[y-1][x]<5: arr[y-1][x]+=item-1
                if arr[y-1][x]==0: 
                    arr=self.expand(arr,x,y-1)
            if y+1<len(arr):
                if arr[y+1][x]<5: arr[y+1][x]+=item-1
                if arr[y+1][x]==0:
                    arr=self.expand(arr,x,y+1)
            if x+1<len(arr[0]):
                if arr[y][x+1]<5: arr[y][x+1]+=item-1
                if arr[y][x+1]==0:
                    arr=self.expand(arr,x+1,y)
            if x-1>=0:
                if arr[y][x-1]<5: arr[y][x-1]+=item-1
                if arr[y][x-1]==0:
                    arr=self.expand(arr,x-1,y)
        return arr
    def maskPush(self,arr,DIV=20):
        image=self.img.copy()[:,:,0]
        for i in range(0,arr.shape[0]-DIV,DIV):
            for j in range(0,arr.shape[1]-DIV,DIV):
                k=np.sum(arr[i:i+DIV,j:j+DIV])/(DIV*DIV*10)
                if k>0:
                    k=(k/50)+1 #make larger than self
                    image=self.blowUp(image,[DIV,DIV],[i,j],k=k)
        return image
    def moveX(self,units):
        self.pos[1]=units+self.pos[1]
    def moveY(self,units):
        self.pos[0]=units+self.pos[0]
    def blowUp(self,image,dims,pos,k=1.2):
        area=image[pos[0]:pos[0]+dims[0],pos[1]:pos[1]+dims[1]]
        area=cv2.resize(area,(np.array(dims)*k).astype(int),interpolation=cv2.INTER_AREA)
        a=int((area.shape[0]-dims[0])/2)
        b=int((area.shape[1]-dims[1])/2)
        image[pos[0]:pos[0]+dims[0],pos[1]:pos[1]+dims[1]]=area[a:dims[0]+a,b:dims[1]+b]
        return image
    
img = cv2.imread('C:/Users/dexte/github/RoboSkin/Assets/images/flat.png')
shrink=(np.array(img.shape[0:2])//3).astype(int)
img=cv2.resize(img,(shrink[1],shrink[0]),interpolation=cv2.INTER_AREA)[60:220,75:220]

h,w=img.shape[0:2]
env=environment(w*4,w*4) #create environment

tip=digiTip(img) #create tactip

for i in range(0,tip.h,10):
    tip.h-=10
    im=tip.getImage(env.get())
    e=env.get().copy()
    y=tip.pos[0]+tip.grid.shape[0]
    x=tip.pos[1]+tip.grid.shape[1]
    im=tip.maskPush(im)
    e[max(min(tip.pos[0],e.shape[0]),0):max(min(y,e.shape[0]),0),max(min(tip.pos[1],e.shape[1]),0):max(min(x,e.shape[1]),0)]=im
    plt.imshow(e)
    plt.title(str(tip.h))
    plt.pause(0.5)

tip.h=25
for i in range(0,30,1):
    tip.moveX(10)
    im=tip.getImage(env.get())
    e=env.get().copy()
    y=tip.pos[0]+tip.grid.shape[0]
    x=tip.pos[1]+tip.grid.shape[1]
    im=tip.maskPush(im)
    e[max(min(tip.pos[0],e.shape[0]),0):max(min(y,e.shape[0]),0),max(min(tip.pos[1],e.shape[1]),0):max(min(x,e.shape[1]),0)]=im
    plt.imshow(e)
    plt.title(str(i))
    plt.pause(0.25)

#plt.show()
