import serial
import time
import struct
import threading
import ujson
import collections
import numpy as np
import math
import os
import circular_buffer
from algorithm import swing_count_svc, hit_detection_svc

# global settings
dev = serial.Serial('/dev/ttyUSB0', 9600)  # 15200
packet_dt = np.dtype([
    ('ts', np.uint32),
    ('accx', np.float32),
    ('accy', np.float32),
    ('accz', np.float32),
    ('gyrox', np.float32),
    ('gyroy', np.float32),
    ('gyroz', np.float32),
    ('a0', np.int16),
    ('a1', np.int16),
    ('a2', np.int16)])  # must be wrapped as a tuple
packet_fmt = '<Lffffffhhh'
proc_buff_size = 1 << 10 # approx >20sec
packet_ver = 1
n_readers = 2

# assumption must hold true: data arrives *continuously*
proc_buf = circular_buffer.circular_buffer(1 << 9, n_readers, packet_dt)

# TODO adaptive sleep policy (upon None | hasData)


def swing_counter():
    time.sleep(5)
    while True:
        # based on observation of 15 packet/sec
        chunk = proc_buf.try_read(0, 150)
        if chunk == None:
            time.sleep(.5)  # long sleep until data ready
            continue
        result = swing_count_svc(chunk, 0, [1, 2, 3, 4, 5, 6])
        # TODO display result


def hit_detector():
    time.sleep(5)
    while True:
        chunk = proc_buf.try_read(1, 15)  # approx. per sec
        if chunk == None:
            time.sleep(.1)
            continue
        result = hit_detection_svc(chunk, 0, [7, 8, 9])


# def lt_precise_insights(file_name):
#     # load all data since program start
#     sensor_data = ujson.load(open(file_name, 'r'))
#     rpi_rt_pipeline.rt_process_file(sensor_data)


threading.Thread(target=swing_counter).start()
threading.Thread(target=hit_detector).start()


while True:
    packet = None
    if packet_ver < 1:
        dev.read_until('!'.encode())
        res = dev.read_until('@'.encode())
        if not len(res) == (1 + packet_dt.itemsize):
            continue
        packet = tuple(struct.unpack_from(packet_fmt, res))
    else:
        res = dev.read_until()
        packet = tuple(ujson.decode(res))

    # data validation (no guarantee even if passed)
    if not len(packet) == len(packet_dt):
        continue

    proc_buf.write_one(packet)
