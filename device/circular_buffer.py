import numpy as np
import threading
import time
import math
import json

debug = True
stroage = True


storage_svc = None

def init_storage():
    global storage_svc
# firebase storage code
    import pyrebase

    config = {
      "apiKey": "AIzaSyDnDjyTHvySbS6gHN9kA_xsRhj_cqjGdac",
      "authDomain": "hwsw-lab.firebaseapp.com",
      "databaseURL": "https://hwsw-lab.firebaseio.com",
      "projectID": "hwsw-lab",
      "storageBucket": "hwsw-lab.appspot.com",
      "messagingSenderId": "709816656037"
    }

    firebase = pyrebase.initialize_app(config)
    auth = firebase.auth()
    user = auth.sign_in_with_email_and_password('opsec@google.com', 'opsec1')
    storage_svc = firebase.storage()


class circular_buffer():
    '''
    An amazing fixed-array based circular buffer supporting:
    - Single writer, multi reader independent read
    - Data invalidated upon successful read for each reader
    - Bulk write/read
    - Thread safety
    - Per-epoch dumping
    - Custom datatype
    '''
    def __init__(self, max_len, n_readers, dtype): 
        self._lock = threading.Lock()
        self._maxlen = max_len
        self._buf = np.empty((self._maxlen, ), dtype=dtype)
        self._dtype_has_fields = self._buf.dtype.fields is None
        self._valid = False*np.ones((n_readers, self._maxlen, ), dtype=bool)
        # pos to read; no check for writers
        self._readers = [self._maxlen-1]*n_readers
        self._writer = self._maxlen-1  # position to write; no check for readers
        self._epoch = 0
        self._ts = math.floor(time.time())

    def write_one(self, data):
        with self._lock:
            self._buf[self._writer] = data
            self._readers = [(r - 1) % self._maxlen if r ==
                             self._writer and self._valid[i, r] == True else r for i, r in enumerate(self._readers)]
            self._valid[:, self._writer] = True
            if self._writer == 0:  # reached 1 epoch, now dumping to file; synchronous for now
                file_name = str(self._ts) + "_" + str(self._epoch) + ".json"
                json.dump(np.flip(self._buf, axis=0).tolist(), open(file_name, 'w')) # only supports serializable dtype or serializable fields of dtype
                
                if not (storage_svc is None):
                # as admin - TRY NOT TO USE
                # storage.child("images/example.jpg").put("example2.jpg")
                # as user - THIS SHOULD WORK
                    try:
                        if debug:
                            print("sending file {0} to firebase".format(file_name))
                        storage_svc.put(file_name, user['idToken'])
                    except:
                        if debug:
                            print("file transmission to firebase failed")
                # end firebase storage code

                if debug:
                    print('file dumped locally @ epoch {0}'.format(self._epoch))
                self._epoch += 1
            self._writer = (self._writer - 1) % self._maxlen

    def write_iterable(self, iterable):  # only 1 thread allowed to write
        if debug:
            print('before write_iterable: {0} {1} {2}'.format(
                self._writer, self._readers, self._buf))
        for d in iterable:
            self.write_one(d)  # TODO perf improvement
        if debug:
            print('after write_iterable: {0} {1} {2}'.format(
                self._writer, self._readers, self._buf))

    # will return None if existing content not bigger than window
    # data "invisible" for current reader upon successful read
    def try_read(self, reader_idx, window):
        with self._lock:
            if reader_idx >= len(self._readers) or window > self._maxlen:
                raise Exception()
            if debug:
                print('valid item count for read {0} is {1}'.format(reader_idx, np.sum(self._valid[reader_idx, :])))
            reader = self._readers[reader_idx]
            window_ori = window
            result = np.empty(0, dtype=self._buf.dtype)
            if reader + 1 < window:
                if not np.alltrue(self._valid[reader_idx, :reader+1]) or not np.alltrue(self._valid[reader_idx, -(window-reader-1):]):
                    # print('None')
                    return None
                self._valid[reader_idx, :reader+1] = False
                self._valid[reader_idx, -(window-reader-1):] = False
                result = np.hstack(
                    [result, self._buf[:reader+1][::-1], self._buf[-(window-reader-1):][::-1]])
            else:
                if not np.alltrue(self._valid[reader_idx, reader-window+1:reader+1]):
                    # print('None')
                    return None
                self._valid[reader_idx, reader-window+1:reader+1] = False
                result = np.hstack(
                    [result, self._buf[reader-window+1:reader+1][::-1]])
            if len(result) == 0:
                # print('None')
                return None
            self._readers[reader_idx] = (reader - window_ori) % self._maxlen
            # if debug:
            #     print('reader {0} result {1}'.format(reader_idx, result))
            return np.array(result)

    def get_len(self, reader_idx):
        return np.sum(self._valid[reader_idx])

if storage:
    init_storage()

if __name__ == "__main__":
    cb = circular_buffer(5, 3, np.dtype(float))
    cb.write_iterable([4, 3, 2, 1, 0, -1, -2])
    cb.write_iterable([4, 3, 2, 1, 0, -1, -2])
    cb.write_iterable([4, 3, 2, 1, 0, -1, -2])
    print(cb.get_len(0))
    cb.try_read(0, 2)
    print(cb.get_len(0))
    cb.try_read(0, 1)
    print(cb.get_len(0))
    cb.try_read(0, 2)
    print(cb.get_len(0))
