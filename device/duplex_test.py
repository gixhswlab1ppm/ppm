import serial
import time
import struct
import threading
import ujson
import rpi_rt_pipeline
dev = serial.Serial('/dev/ttyUSB0', 9600)# 15200

n_wf = 0
n_all = 0
entry_list = []
last_update_time = time.time()

#print(ser.in_waiting)

def process_rt(file_name):
    sensor_data = ujson.load(open(file_name,'r'))
    rpi_rt_pipeline.rt_process_file(sensor_data)


def daemon():
    global last_update_time
    while True:
        print(time.time() - last_update_time)
        if time.time() - last_update_time > 3:
            ujson.dump(entry_list, open(str(last_update_time)+'.json', 'w'))
            print('file emitted!')
            threading.Thread(target=process_rt, args=(str(last_update_time)+'.json',)).start()
            last_update_time = time.time()
        time.sleep(2)

def rt_process(entry):
    global last_update_time
    last_update_time = time.time()
    # print(entry)
    entry = [e for e in entry]
    entry.append(last_update_time)
    entry_list.append(entry)


threading.Thread(target=daemon).start()

while True:
    dev.read_until('!'.encode())
    res = dev.read_until('@'.encode())
    n_all += 1
    if not len(res) == 35:  # 34+1
        continue
    n_wf += 1
    rt_process(struct.unpack_from('<fffffffhhh', res))
    n_wf += 1
    # print(time.time(), data)
    #ser.write(bytes([1]))
    # ser.flush()
    #time.sleep(0.5)



