import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TKAgg')

class CTRNN:
    def __init__(self, size):
        self.size = size
        self.time_constant = np.ones(size)
        self.bias = np.zeros(size)
        self.activation = np.zeros(size)
        self.output = np.zeros(size)
        self.weight = np.zeros((size, size))
        
    def set_parameters(self, time_constant, bias, weight):
        self.time_constant = time_constant
        self.bias = bias
        self.weight = weight
        
    def step(self, input, step_size=0.01):
        net_input = np.dot(self.weight, self.activation) + self.bias + input
        dx_dt = (1/self.time_constant) * (-self.activation + net_input)
        self.activation += step_size * dx_dt
        self.output = np.tanh(self.activation)

Image = np.array([-1,1,-1 ,1,-1,1, -1,1,-1])
Image2 = np.array([1,1,1 ,1,1,1, -1,1,-1])
net=CTRNN(9)

net.step(Image)
print(net.output)
net.step(Image)
print(net.output)
net.step(Image)
print(net.output)
net.step(Image)
print(net.output)
#train oon images
"""net.train(Image)
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

"""