from control import *
import matplotlib.pyplot as plt

B=Board()
#get serial boards and connect to first one
COM=""
while COM=="":
    try:
        res=B.serial_ports()
        print("ports:",res)
        B.connect(res[0])
        B.runFile("C:/Users/dexte/github/RoboSkin/Code/Examples/Test_reading.py/mOTRO CONTROL.py")
        COM=res[0]
    except IndexError:
        time.sleep(1)
#B.moveZ(10)
c=0
ex=Experiment(B)
#ex.moveZ(1,1)
samples=1
samples_2=2
data=np.zeros((samples,3,samples_2,2))

for i in range(samples):
    print("Trial:",i+1)
    f=ex.test_speed(samples_2,[10,20,30,40])
    #save to main data
    data[i][0]=f
    weights = np.arange(1, samples_2+1)
    plt.scatter(f[:,0],f[:,1],c=weights, cmap='Purples')


#plot the labels
"""plt.scatter(f[:,0],f[:,1],c=weights, cmap='Purples',label="Central")
plt.scatter(l[:,0],l[:,1],c=weights, cmap='Greens',label="Top")
plt.scatter(r[:,0],r[:,1],c=weights, cmap='Reds',label="Bottom")
plt.legend(loc="upper right")"""
plt.colorbar()
plt.title("Scatter of average vector of different sensations for vectors")
plt.show()

np.save("C:/Users/dexte/github/RoboSkin/code/Models/saved/dirs",data)

