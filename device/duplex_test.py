debug = True  # true if output more debug info to console
simulate = False

storage_remote = False

dev_arduino = True  # false if data is collected from arduino in real time
dev_button = True # true if the start/end physical button is available
dev_display = True  # true if e-ink display is available
dev_pi = True  # true if computing is on pi w/ flat folder structure

# 0x68
from mpu6050 import mpu6050
mpu = mpu6050(0x68)
from Adafruit_ADS1x15 import ADS1115
adc = ADS1115()


import threading
if dev_display:
    import display
    threading.Thread(target=display.render_welcome_screen).start()

if dev_button:
    from gpiozero import Button
    button = Button(21)

if dev_arduino:
    # 15200, ttyACM0, ttyUSB0, serial0
    # import serial
    # dev = serial.Serial('/dev/serial0', 9600)
    dev = None

import time
import numpy as np
import circular_buffer
from algorithm import swing_count_svc, hit_report_svc, fft_svc, get_swing_count_from_ts_clusters_by_feature

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
proc_buff_size = 1 << 9  # 50 dp/sec -> 40 sec
packet_ver = 2
n_readers = 4

is_paused = False  # arduino data handling switch
ui_state = 0  # 0: Keep Flat, 1: User Profile, 2: Keep Swinging, 3: Running, 4: Paused

# assumption must hold true: data arrives *continuously*
proc_buf = circular_buffer.circular_buffer(
    proc_buff_size, n_readers, packet_dt, storage_remote)

import json


import profile_manager 
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
if dev_pi:
    file_path = dir_path + '/'
else:
    file_path = dir_path[:-7] + '/data/'
profile = profile_manager.profile_manager(file_path + 'profile.json')


def swing_counter():
    time.sleep(5)
    if debug:
        swing_sum = 0
    while True:
        # based on observation of 15 packet/sec
        chunk = proc_buf.try_read(0, 20)
        if chunk is None:
            time.sleep(5 if is_paused else 1)
            continue
        cluster_ts_per_feature, _ = swing_count_svc(
            np.array(chunk.tolist()).reshape(len(chunk), len(packet_dt)),
            0,
            [1, 2, 3, 4, 5, 6],
            profile.get_field("th"))
        swing_count = get_swing_count_from_ts_clusters_by_feature(
            cluster_ts_per_feature)
        swing_sum += swing_count
        if swing_count > 0:
            if debug:
                print('swing counter result available', swing_count, proc_buf.get_len(0))
            if dev_display:
                display.async_update_display_partial(1, swing_sum)  # TODO async?
        time.sleep(0) # yield

def hit_reporter():
    time.sleep(5)
    if debug:
        count_sum = 0
        dt_avg = 0
        ds_avg = 0
        st_avg = 0
    while True:
        chunk = proc_buf.try_read(1, 1 << 4)  # approx. per 2 sec
        if chunk is None:
            time.sleep(5 if is_paused else 1)
            continue
        h_dt, h_st, h_events = hit_report_svc(
            np.array(chunk.tolist()).reshape(len(chunk), len(packet_dt)), 0, [7, 8, 9])
        count_delta = len(h_dt)
        if count_delta > 0:
            if debug:
                print('hit detector result available', len(h_dt), proc_buf.get_len(1))
            dt_avg = 0.8*dt_avg + 0.2*np.average(h_dt)
            st_avg = 0.8*st_avg + 0.2*np.average(h_st)
            count_sum += count_delta
            if dev_display:
                display.async_update_display_partial(0, count_sum)
                display.async_update_display_partial(2, int(100 * dt_avg))
                display.async_update_display_partial(3, int(100 * st_avg))
        time.sleep(0)


def fft_near_realtime():
    time.sleep(5)
    while True:
        # rolling basis: 128 + (new) 32 data ~ 13 + (new) 3 sec
        if proc_buf.get_len(2) < 1 << 7:
            time.sleep(5)
        chunk = proc_buf.try_read(2, 1 << 5)
        if chunk is None:
            time.sleep(5 if is_paused else 5)
            continue
        result = fft_svc(np.array(chunk.tolist()).reshape(
            len(chunk), len(packet_dt)), 0, [1, 2, 3, 4, 5, 6], win_len=0)
        if debug:
            print('fft_nrt result available')


def on_navigate_to_profile_screen():
    if debug:
        print('on_navigate_to_profile_screen')

    def profile_screen_button_handler():
        profile.update_id((profile.get_id() + 1) % profile.get_len())
        if debug:
            print('profile switched to ', profile.get_id())
        threading.Thread(target=display.render_profile_screen,
                         args=(profile.get_id(),)).start()

    global ui_state
    ui_state = 1
    if dev_display:
        threading.Thread(target=display.render_profile_screen, args=(profile.get_id(), )).start()
    profile_start_time = time.time()
    if dev_button:
        button.when_pressed = profile_screen_button_handler
    while time.time() - profile_start_time < 5:
        time.sleep(.1)
    if dev_button:
        button.when_pressed = None


def on_navigate_to_train_screen():
    if debug:
        print('on_navigate_to_train_screen')
    global ui_state
    ui_state = 2
    if dev_display:
        threading.Thread(target=display.render_train_screen).start()
    time.sleep(4) # skip some in case user starts late
    chunk = None
    while True:
        chunk = proc_buf.try_read(0, 80)
        if chunk is None:
            time.sleep(2)
        else:
            break

    _, curr_thresholds = swing_count_svc(
        np.array(chunk.tolist()).reshape(len(chunk), len(packet_dt)),
        0,
        [1, 2, 3, 4, 5, 6])
    profile.update_field("th", curr_thresholds.tolist())
    if debug:
        print('learned thresholds', curr_thresholds)


def on_navigate_to_run_screen():
    if debug:
        print('on_navigate_to_run_screen')

    def run_screen_button_handler():
        global is_paused
        is_paused = not is_paused
        if debug:
            print('activity ', 'paused' if is_paused else 'resumed')
        if dev_display:
            if is_paused:
                threading.Thread(target=display.render_pause_screen).start()
            else:
                threading.Thread(target=display.render_run_screen).start()

    global ui_state
    ui_state = 3
    worker_threads = [swing_counter, hit_reporter]
    for th in worker_threads:
        thr = threading.Thread(target=th)
        thr.daemon = True
        thr.start()
    if dev_display:
        threading.Thread(target=display.render_run_screen).start()
    if dev_button:
        button.when_pressed = run_screen_button_handler

def communication_start():
    if simulate:
        if debug:
            print('simulation starts')

        def run_simulate(data):
            delays = .05 + np.random.rand(len(data))/10
            delays /= 10.
            for i, d in enumerate(delays):
                proc_buf.write_one(data[i])
                time.sleep(d)
            if debug:
                print('simulation ends')

        data = json.load(open(file_path + '1551925483_0.json', 'r'))
        data = np.array([tuple(s) for s in data], dtype=packet_dt)
        threading.Thread(target=run_simulate, args=(data,)).start()
    elif dev_arduino:
        if debug:
            n_packet = 0
            n_packet_success = 0
        addr_adc = 0x48
        addr_mpu = 0x68
        import ujson
        while True:
            if debug:
                n_packet += 1
            packet = None
            if packet_ver < 1:
                import struct
                dev.read_until('!'.encode())
                res = dev.read_until('@'.encode())
                if not len(res) == (1 + packet_dt.itemsize):
                    continue
                packet = tuple(struct.unpack_from(packet_fmt, res))
            elif packet_ver == 1:
                try:
                    res = str(dev.read_until())
                    res = res[res.find('['):res.find(']')+1]
                    res = ujson.decode(res)
                    packet = tuple(res)
                except:
                    if debug:
                        print('packet abandoned due to decode error')
            elif packet_ver == 2:
                mpu_raw = mpu.get_all_data()
                packet = (time.time(), mpu_raw[0]['x'],mpu_raw[0]['y'],mpu_raw[0]['z'],mpu_raw[1]['x'],mpu_raw[1]['y'],mpu_raw[1]['z'], adc.read_adc(0), adc.read_adc(1), adc.read_adc(2))
            # data validation (no guarantee even if passed)
            if (packet is None):
            #or (not len(packet) == len(packet_dt)):
                continue

            if debug:
                n_packet_success += 1
                if n_packet % 50 == 0:
                    print('packet success rate: {0:.2f}'.format(
                        n_packet_success/n_packet))

            proc_buf.write_one(packet)

if __name__ == "__main__":
    on_navigate_to_profile_screen()
    commu_daemon = threading.Thread(target=communication_start)
    commu_daemon.daemon = True
    commu_daemon.start()
    th = profile.get_field("th")
    if th is None:
        on_navigate_to_train_screen()
    on_navigate_to_run_screen()
    commu_daemon.join() # prevent from quiting

