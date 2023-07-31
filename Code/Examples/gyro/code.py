from machine import Pin, I2C
import utime
from mpu6050 import init_mpu6050, get_mpu6050_data
 
i2c = I2C(1, scl=Pin(15), sda=Pin(14), freq=400000)
print(i2c. scan())
init_mpu6050(i2c)
 
def get_data():
    data = get_mpu6050_data(i2c)
    print(data['accel']['x'], data['accel']['y'], data['accel']['z'],data['gyro']['x'],data['gyro']['y'], data['gyro']['z'])
    
#while True:
#    
#    get_data()
#    utime.sleep(0.5)