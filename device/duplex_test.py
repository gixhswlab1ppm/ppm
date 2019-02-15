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
from algorithm import swing_count_svc, hit_detection_svc, fft_svc

debug = True # true if output more debug info to console

dev_arduino = False # false if data is collected from arduino in real time
dev_button = False # true if the start/end physical button is available
dev_display = True # true if e-ink display is available

if dev_button:
    from gpiozero import Button
    button = Button(7)

if dev_display:
    from display import update_field

if dev_arduino:
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
proc_buff_size = 1 << 11  # 50 dp/sec -> 40 sec
packet_ver = 1
n_readers = 3

is_alive = False # arduino data handling switch

# assumption must hold true: data arrives *continuously*
proc_buf = circular_buffer.circular_buffer(
    proc_buff_size, n_readers, packet_dt)

# TODO adaptive sleep policy (upon None | hasData)


def swing_counter():
    time.sleep(5)
    if debug:
        swing_sum = 0
    while True:
        # based on observation of 15 packet/sec
        chunk = proc_buf.try_read(0, 150)
        if chunk is None:
            time.sleep(.5 if not is_alive else 5)  # long sleep until data ready
            continue
        result = swing_count_svc(np.array(chunk.tolist()).reshape(len(chunk), len(packet_dt)), 0, [1, 2, 3, 4, 5, 6])
        if debug:
            print('swing counter result available')
        if dev_display and debug:
            swing_sum += math.trunc(np.median(result))
            update_field(0, swing_sum)
        # TODO display result


def hit_detector():
    time.sleep(5)
    if debug:
        hit_sum = 0
    while True:
        chunk = proc_buf.try_read(1, 15)  # approx. per sec
        if chunk is None:
            time.sleep(.1 if not is_alive else 5)
            continue
        result = hit_detection_svc(np.array(chunk.tolist()).reshape(len(chunk), len(packet_dt)), 0, [7, 8, 9])
        if debug:
            print('hit detector result available')
        if dev_display and debug:
            hit_sum += len([pair for pair in result if pair[1] + pair[2]>0.01])
            update_field(1, hit_sum)


def fft_near_realtime():
    time.sleep(5)
    while True:
        chunk = proc_buf.try_read(2, 1 << 9)  # ~10sec
        if chunk is None:
            time.sleep(5 if not is_alive else 5)
            continue
        result = fft_svc(np.array(chunk.tolist()).reshape(len(chunk), len(packet_dt)), 0, [1, 2, 3, 4, 5, 6], win_len=0)
        if debug:
            print('fft_nrt result available')



# def lt_precise_insights(file_name):
#     # load all data since program start
#     sensor_data = ujson.load(open(file_name, 'r'))
#     rpi_rt_pipeline.rt_process_file(sensor_data)

def user_input_handler(): # when not alive, worker thread will still run till data processed
    worker_threads = [
        threading.Thread(target=swing_counter),
        threading.Thread(target=hit_detector),
        threading.Thread(target=fft_near_realtime)
    ]
    for th in worker_threads:
        th.start()
    if debug:
        print('worker threads started')
    def end():
        if debug:
            print('activity ends')
        global is_alive
        is_alive = False
        if dev_button:
            button.when_pressed = start
    def start():
        if debug:
            print('activity starts')
        global is_alive
        is_alive = True
        if dev_button:
            button.when_pressed = end
    if debug:
        print('boot seq complete')
    print('waiting for start')
    if dev_button:
        button.when_pressed = start


threading.Thread(target=user_input_handler).start()

    
if __name__ == "__main__":
    if dev_arduino:
        while True:
            if not is_alive:
                if debug:
                    print('sleeping for 2 sec')
                time.sleep(2)
                continue

            packet = None
            if packet_ver < 1:
                dev.read_until('!'.encode())
                res = dev.read_until('@'.encode())
                if not len(res) == (1 + packet_dt.itemsize):
                    continue
                packet = tuple(struct.unpack_from(packet_fmt, res))
            else:
                # print('reading sensor data')
                res = str(dev.read_until())
                res = res[res.find('['):res.find(']')+1]
                res = ujson.decode(res)
                packet = tuple(res)

            # data validation (no guarantee even if passed)
            if not len(packet) == len(packet_dt):
                continue

            proc_buf.write_one(packet)
    else:
        if debug:
            print('simulation starts')
        def run_simulate(data):
            time.sleep(1.5)
            global is_alive
            is_alive = True
            delays = .05 + np.random.rand(len(data))/10
            for i,d in enumerate(delays):
                proc_buf.write_one(data[i])
                time.sleep(d)
            if debug:
                print('simulation ends')
        import os
        import json
        dir_path = os.path.dirname(os.path.realpath(__file__))
        file_path = dir_path + '/1550209189_0.json'
        data = json.load(open(file_path, 'r'))
        data = np.array([tuple(s) for s in data], dtype=packet_dt)
        threading.Thread(target=run_simulate, args=(data,)).start()
        while True:
            time.sleep(10)


