"""
n this implementation, n is the number of neurons in the network. The train method takes a 2D array X of shape (m, n), where m is the number of patterns to train on, and each row of X is a pattern. The predict method takes a 1D array x of length n, which is the input pattern to the network. The asynchronous parameter controls whether the network should update the neurons synchronously or asynchronously during prediction.

The _predict_sync and _predict_async methods are private methods that implement the synchronous and asynchronous update rules, respectively. The synchronous update rule updates all neurons in parallel, whereas the asynchronous update rule updates each neuron one at a time in a random order. The asynchronous update rule is generally slower but can be more robust to noise and can converge to a wider range of attractor states.


"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TKAgg')

class HNN:
    def __init__(self,n):
        self.n=n
        self.W=np.random.normal(0,2,(n,n))
        np.fill_diagonal(self.W, 0)
    def forward(self,y):
        y=np.dot(y,self.W) #forward pass
        y[y>0] = 1 #activation
        y[y<0] = -1
        return y #return the state
    def train(self,label,num=50):
        states=self.run(label,num)
        n=states.shape[1]
        for i in range(len(self.W)):
            for j in range(len(self.W[i])):
                if i!=j:
                    self.W[i][j]=(1/n)*np.sum(states[:,i]*states[:,j]) #hebbian learning
    def run(self,y,t):
        states=np.zeros((t,self.n))
        states[0]=y
        for i in range(1,t):
            states[i]=self.forward(states[i-1])
        return states

Image = np.array([-1,1,-1 ,1,-1,1, -1,1,-1])
Image2 = np.array([1,1,1 ,1,1,1, -1,1,-1])
net=HNN(9)
#train oon images
net.train(Image)
#net.train(Image2)


states=net.run(Image,2)

Image=Image.reshape((3,3))
Image[Image<0]=0

plt.title("Origin")
plt.imshow(Image)
plt.show()

for state in states[1:]:
    im=state.reshape(3,3)
    im[im<0]=0
    plt.title("other")
    plt.imshow(im)
    plt.show()

states=net.run(Image2,2)

Image2=Image2.reshape((3,3))
Image2[Image2<0]=0

plt.title("Origin")
plt.imshow(Image2)
plt.show()

for state in states[1:]:
    im=state.reshape(3,3)
    im[im<0]=0
    plt.title("other")
    plt.imshow(im)
    plt.show()

