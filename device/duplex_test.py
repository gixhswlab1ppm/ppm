import serial
import time
import struct
ser = serial.Serial('/dev/ttyACM0', 9600)
n_wf = 0
n_all = 0
t = time.time()
while True:
    #print(ser.in_waiting)
    ser.read_until('!'.encode())
    #print('found start')
    res = ser.read_until('@'.encode())
    #print('found end')
    n_all +=1
    if not len(res) == 35: # 34+1
        print('DROP', len(res), res)
    else:
        n_wf +=1
        #print(struct.unpack_from('<fffffffhhh', res))
    # print(time.time(), data)
    #ser.write(bytes([1]))
    # ser.flush()
    #time.sleep(0.5)
    if (n_all % 100 == 0):
        print('DROP RATE', 1-n_wf/n_all, time.time() - t);
        t = time.time()
