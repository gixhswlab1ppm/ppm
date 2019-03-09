debug = True  # true if output more debug info to console
storage_remote = False

dev_arduino = True  # false if data is collected from arduino in real time
dev_button = True  # true if the start/end physical button is available
dev_display = True  # true if e-ink display is available
dev_pi = True  # true if computing is on pi w/ flat folder structure

ui_state = 0  # 0: Keep Flat, 1: User Profile, 2: Keep Swinging, 3: Running, 4: Paused

import threading
if dev_display:
    import display
    threading.Thread(target=display.render_welcome_screen).start()

if dev_button:
    from gpiozero import Button
    button = Button(7)

if dev_arduino:
    # 15200, ttyACM0, ttyUSB0, serial0
    import serial
    dev = serial.Serial('/dev/serial0', 9600)

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
proc_buff_size = 1 << 8  # 50 dp/sec -> 40 sec
packet_ver = 1
n_readers = 4

is_alive = False  # arduino data handling switch

# assumption must hold true: data arrives *continuously*
proc_buf = circular_buffer.circular_buffer(
    proc_buff_size, n_readers, packet_dt, storage_remote)

import json

class profile_manager:
    fields = ["id", "th"]
    def __init__(self):
        self.profiles = json.load(open('profile.json', 'r'))
    def _get_index(self, _id):
        return [i for i,p in enumerate(self.profiles) if p["id"] == _id][0]
    def update_id(self, _id):
        self.profile_id = _id
    def update(self, profile):
        self.profiles[self._get_index(self.profile_id)] = profile
        json.dump(self.profiles, open('profile.json', 'w'))
    def update_field(self, field, val):
        profile = self.profiles[self._get_index(self.profile_id)]
        profile[field] = val
        self.update(profile)
    def get_field(self, field):
        if field in self.profiles[self._get_index(self.profile_id)]:
            return self.profiles[self._get_index(self.profile_id)][field]
        else:
            return None
    def get_id(self):
        return self.profile_id
    def get_len(self):
        return 5

profile = profile_manager()

def swing_counter():
    time.sleep(5)
    if debug:
        swing_sum = 0
    while True:
        # based on observation of 15 packet/sec
        chunk = proc_buf.try_read(0, 20)
        if chunk is None:
            # long sleep until data ready
            time.sleep(5 if not is_alive else .5)
            continue
        cluster_ts_per_feature, _ = swing_count_svc(
            np.array(chunk.tolist()).reshape(len(chunk), len(packet_dt)),
            0,
            [1, 2, 3, 4, 5, 6],
            profile.get_field("th"))
        swing_count = get_swing_count_from_ts_clusters_by_feature(
            cluster_ts_per_feature)
        if debug:
            print('swing counter result available')
        if dev_display:
            swing_sum += swing_count
            display.update_display_partial(0, swing_sum)  # TODO async?


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
            time.sleep(5 if not is_alive else .1)
            continue
        h_dt, h_st, h_events = hit_report_svc(
            np.array(chunk.tolist()).reshape(len(chunk), len(packet_dt)), 0, [7, 8, 9])
        if debug:
            print('hit detector result available')
        if dev_display:
            dt_avg = 0.8*dt_avg + 0.2*h_dt
            st_avg = 0.8*st_avg + 0.2*h_st
            count_delta = len(h_dt)
            if count_delta > 0:
                count_sum += count_delta
                display.update_display_partial(1, count_sum)
                display.update_display_partial(2, int(100 * dt_avg))
                display.update_display_partial(3, int(100 * st_avg))


def fft_near_realtime():
    time.sleep(5)
    while True:
        # rolling basis: 128 + (new) 32 data ~ 13 + (new) 3 sec
        if proc_buf.get_len(2) < 1 << 7:
            time.sleep(5)
        chunk = proc_buf.try_read(2, 1 << 5)
        if chunk is None:
            time.sleep(5 if not is_alive else 5)
            continue
        result = fft_svc(np.array(chunk.tolist()).reshape(
            len(chunk), len(packet_dt)), 0, [1, 2, 3, 4, 5, 6], win_len=0)
        if debug:
            print('fft_nrt result available')


# def lt_precise_insights(file_name):
#     # load all data since program start
#     sensor_data = ujson.load(open(file_name, 'r'))
#     rpi_rt_pipeline.rt_process_file(sensor_data)

def user_input_handler():  # when not alive, worker thread will still run till data processed
    worker_threads = [
        threading.Thread(target=swing_counter),
        threading.Thread(target=hit_reporter),
        # threading.Thread(target=fft_near_realtime)
    ]
    for th in worker_threads:
        th.start()
    if debug:
        print('worker threads started')
    # 0: Keep Flat, 1: User Profile, 2: Keep Swinging, 3: Running, 4: Paused

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


def ui_1_button_pressed_handler():
    profile.update_id((profile.get_id() + 1)%profile.get_len())
    threading.Thread(target=display.render_profile_screen,
                     args=(profile.get_id(),)).start()


def on_navigate_to_profile_screen():
    global ui_state
    ui_state = 1
    threading.Thread(target=display.render_profile_screen).start()
    profile_start_time = time.time()
    if dev_button:
        button.when_pressed = ui_1_button_pressed_handler
    while time.time() - profile_start_time < 5:
        time.sleep(.1)
    if dev_button:
        button.when_pressed = None

def on_navigate_to_train_screen():
    global ui_state
    ui_state = 2
    threading.Thread(target=display.render_train_screen).start()
    chunk = None
    while True:
        chunk = proc_buf.try_read(0, 80)
        if chunk is None:
            time.sleep(2)

    _, curr_thresholds = swing_count_svc(
        np.array(chunk.tolist()).reshape(len(chunk), len(packet_dt)),
        0,
        [1, 2, 3, 4, 5, 6])
    profile.update_field("th", curr_thresholds)

def on_navigate_to_run_screen():
    global ui_state
    ui_state = 3
    threading.Thread(target=display.render_run_screen).start()

if __name__ == "__main__":
    on_navigate_to_profile_screen()
    th = profile.get_field("th")
    if th is None:
        on_navigate_to_train_screen()
    on_navigate_to_run_screen()

    if dev_arduino:
        import ujson
        while True:
            if not is_alive:
                if debug:
                    print('sleeping for 2 sec')
                time.sleep(2)
                continue

            packet = None
            if packet_ver < 1:
                import struct
                dev.read_until('!'.encode())
                res = dev.read_until('@'.encode())
                if not len(res) == (1 + packet_dt.itemsize):
                    continue
                packet = tuple(struct.unpack_from(packet_fmt, res))
            else:
                # print('reading sensor data')
                try:
                    res = str(dev.read_until())
                    res = res[res.find('['):res.find(']')+1]
                    res = ujson.decode(res)
                    packet = tuple(res)
                except:
                    if debug:
                        print('packet abandoned due to decode error')

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
            for i, d in enumerate(delays):
                proc_buf.write_one(data[i])
                time.sleep(d)
            if debug:
                print('simulation ends')

        if dev_pi:
            file_path = dir_path + '/1550355620_0.json'
        else:
            file_path = dir_path[:-7] + '/data/1550355620_0.json'
        data = json.load(open(file_path, 'r'))
        data = np.array([tuple(s) for s in data], dtype=packet_dt)
        threading.Thread(target=run_simulate, args=(data,)).start()
        while True:
            time.sleep(10)
