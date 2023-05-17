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
B.moveZ(10)
ex=Experiment(B)
ex.moveZ(1,1)
samples=10
CM=2
ST=0.5
for i in range(samples):
    a=ex.run_pressure(cm_samples=CM,step=ST)
    print("Trial:",i)
    plt.plot([i for i in np.arange(0, CM, ST)],a)

plt.xlabel("Pressure in cm")
plt.ylabel("Pressure in summed pixels")
plt.title("Pressure vs cm")
plt.show()

