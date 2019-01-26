import time
from mpu6050 import mpu6050
import ujson

i2c_address = 0x68
sleep_reg = 0x6B
data_reg_start = 0x3B
data_reg_end = 0x48

sensor = mpu6050(0x68)

recording = []

while True:
    last_time = time.time()
    while len(recording) < 6000:
        data = sensor.get_all_data() # [accel, gyro, temp]
        data = [
            time.time(),
            data[0]['x'],
            data[0]['y'],
            data[0]['z'],
            data[1]['x'],
            data[1]['y'],
            data[1]['z'],
            data[2]
        ]
        recording.append(data)
    ujson.dump(recording, open('recordings2.json','w'))
    recording = []
    print(last_time)



