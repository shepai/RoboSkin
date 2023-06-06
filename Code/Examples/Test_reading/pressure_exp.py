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


ex=Experiment(B,split=5,th=0.7)
#ex.moveZ(1,1)
samples=30
CM=1
ST=0.2
data=np.zeros((samples,2,len(np.arange(0, CM, ST))))
log=np.zeros((samples,2))
print("begin",ex.b.getWeight())
try:
    for i in range(samples):
        a,x=ex.run_pressure_2(cm_samples=CM,step=ST)
        #ex.moveZ(1,1)
        print("Trial:",i+1,"Range:",a[-1]-a[0])
        print(x)
        data[i][0]=np.array(a)
        data[i][1]=np.array(x)
        log[i][0]=ex.skin.origin.shape[0]
        if data[i][0][0]>data[i][0][1] or data[i][0][1]>data[i][0][2] or data[i][0][2]>data[i][0][3] or data[i][0][3]>data[i][0][4]: #error detected
            log[i][1]=1
        plt.plot(x,a)
        np.save("C:/Users/dexte/github/RoboSkin/code/Models/saved/pressures",data)
        np.save("C:/Users/dexte/github/RoboSkin/code/Models/saved/errorlog",log)
except KeyboardInterrupt:
    print("Interrupt")
    ex.moveZ(1.5,1)
    exit()

#np.save("C:/Users/dexte/github/RoboSkin/code/Models/saved/pressures1",data)
plt.xlabel("Pressure in cm")
plt.ylabel("Pressure in summed pixels")
plt.title("Pressure vs cm")
plt.show()

