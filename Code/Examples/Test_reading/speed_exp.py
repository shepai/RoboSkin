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
        B.runFile("C:/Users/dexte/github/RoboSkin/Code/Examples/Test_reading/mOTRO CONTROL.py")
        COM=res[0]
    except IndexError:
        time.sleep(1)
#B.moveZ(10)
c=0
ex=Experiment(B,split=10,th=1.5)
#ex.moveZ(1,1)
ex.moveX(2.5,1)
samples=15
speeds=[10,20,30,40]
data=np.zeros((samples,len(speeds)))
vecs=np.zeros((samples,len(speeds),133,2))
try:
    for i in range(samples):
        print("Trial:",i+1)
        f,v=ex.test_speed(speeds,mode=2)
        ex.moveZ(2,1)
        #save to main data
        data[i]=f
        vecs[i]=v.copy()
        weights = np.arange(1, len(speeds)+1)
        plt.plot(speeds,f,label="Exp"+str(i))
        np.save("C:/Users/dexte/github/RoboSkin/code/Models/saved/speeds2",data)
        np.save("C:/Users/dexte/github/RoboSkin/code/Models/saved/speed_vectors2",vecs)
except KeyboardInterrupt:
    print("Interrupt")
    ex.moveZ(1.5,1)
    exit()       
#plot the labels
"""plt.scatter(f[:,0],f[:,1],c=weights, cmap='Purples',label="Central")
plt.scatter(l[:,0],l[:,1],c=weights, cmap='Greens',label="Top")
plt.scatter(r[:,0],r[:,1],c=weights, cmap='Reds',label="Bottom")
plt.legend(loc="upper right")"""
plt.xlabel("Speed setting")
plt.ylabel("Average magnitude size (px)")
plt.title("How speed affects the time of arrival to the magnitude of vectors")
plt.legend(loc="lower left")
plt.show()



