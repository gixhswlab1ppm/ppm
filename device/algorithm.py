import numpy as np
from scipy import signal
import math
from collections import Counter
from scipy import stats
from scipy.signal import argrelextrema
import ujson
import os
import math
dir_path = os.path.dirname(os.path.realpath(__file__))

visualize = False
debug = False

if visualize == True:
    import matplotlib.pyplot as plt

#  stateless algorithms

def triangulate_centroid(readings, circles=[[-1, 0], [1, 0], [0, -math.sqrt(3)]]):
	"""
	Given a 1x3 array of readings, and 3x2 array of circles
	Returns the weighted centroid (1x2)
	"""
	return np.divide(np.dot(readings, circles), .00001+np.sum(readings))


def shift_mean(data, ts_col, feature_cols):
    return np.hstack([
        data[:, [ts_col]],
        data[:, feature_cols] - np.mean(data[:, feature_cols], axis=0)
    ])


def swing_count_svc(data, ts_col, feature_cols):
    def clustering_dedup(maximas):  # maximas: timestamps
        # merge within window until no change
        # 2-level clustering (dedup then detect, respectively)
        idx_min = maximas.min()
        idx_max = maximas.max()

        seg_span = 1500  # millisecond # TODO change to millisec if using new ts type!

        segs = np.array(maximas).reshape(-1, 1)
        if debug == True:
            print(len(segs))
        while True:
            curr_seg_idx = 0
            epoch_updated = False
            while curr_seg_idx < len(segs) - 1:
                curr_seg = segs[curr_seg_idx]
                next_seg = segs[curr_seg_idx+1]
                curr_mean = np.mean(curr_seg)
                next_mean = np.mean(next_seg)
                if next_mean - curr_mean < seg_span:
                    segs = [*segs[:curr_seg_idx],
                            [*curr_seg, *next_seg], *segs[curr_seg_idx+2:]]
                    epoch_updated = True
                curr_seg_idx += 1
            if debug == True:
                print(len(segs))
            if epoch_updated == False:
                break
        if debug == True:
            print("!!!")
        return segs

    data = shift_mean(data, ts_col, feature_cols)
    polls = []
    if debug:
        groups = []
    for i in range(0, len(feature_cols)):
        maxima_idx = np.array(argrelextrema(data[:, i+1], np.greater))[0]
        threshold = data[:, i+1].std()*0.6
        # threshold = threshold if threshold > 1 else 1
        maxima_idx_filtered = [
            m_i for m_i in maxima_idx if data[m_i, i+1] > threshold]
        if len(maxima_idx_filtered) > 0:
            ts_dedup = clustering_dedup(data[maxima_idx_filtered, 0])
            polls.append(len(ts_dedup))  # silly polling for now
            if visualize:
                plt.scatter(data[maxima_idx_filtered, 0],
                            data[maxima_idx_filtered, i+1], s=20, alpha=.5)
                plt.scatter([np.mean(tsd) for tsd in ts_dedup], data[:,
                                                                     i+1].std() * np.ones(len(ts_dedup)), s=40, alpha=.5)
                plt.plot(data[:, 0], data[:, i+1])
                plt.show()
            if debug:
                groups.append(ts_dedup)
        else:
            polls.append(0)
    if debug:
        return polls, groups
    return polls


def hit_detection_svc(data, ts_col, feature_cols):
    data = data[:, [ts_col, *feature_cols]]
    result = np.empty((len(data), 2), dtype=float)
    for i, entry in enumerate(data):
        if np.sum(entry[1:]) > 0:
            result[i] = triangulate_centroid(entry[1:]) # add "confidence": sum of values, to discriminate zero val result
        else:
            result[i] = [0, 0]
    return np.hstack([data[:, [0]], result]).reshape(-1, 3)


# win_len=0 for unwindowed fft, unit: millisec
def fft_svc(data, ts_col, feature_cols, win_len=15000):
    def fft(y_temp, topk=.1, sample_rate=0.005):  # 5e-3s, 5ms):
        """
        Given a feature vector (1-D) sorted in time domain, this function performs a Fast DFT (real part only) and returns:
        x_freq: frequencies in frequency domain
        y_freq_abs | y_freq_abs_filtered: amplitude in frequency domain 

        Arguments:
        y_temp: feature vector in time domain
        topk: \in (0,1) | [1, len(y_temp)]; returns the top frequencies & amplitudes by top percentage or count
        """
        # y_temp -= np.mean(y_temp)
        y_freq = np.fft.rfft(y_temp)
        y_freq_abs = np.abs(y_freq)
        x_freq = np.fft.rfftfreq(len(y_temp), d=sample_rate)
        if topk != None:
            topk = int(topk*len(y_freq)) if topk < 1 else topk
            y_freq_idx_by_amp = np.argsort(y_freq_abs)[::-1][:topk]
            y_freq_mask = np.ones(len(y_freq), dtype=bool)
            y_freq_mask[y_freq_idx_by_amp] = False
            y_freq_filtered = y_freq.copy()
            y_freq_filtered[y_freq_mask] = 0
            y_freq_abs_filtered = np.abs(y_freq_filtered)
            # return x_freq, y_freq_abs_filtered/y_freq_abs_filtered.max()
            return x_freq, y_freq_abs_filtered
        else:
            return x_freq, y_freq_abs

    # swing_period = (1.5, 3)

    # must treat timestamps as evenly distributed
    # aka. actual value of ts is ignored!
    data = shift_mean(data, ts_col, feature_cols)
    ts_col = data[:, 0].copy()
    mpu_time_min = data[:, 0].min()
    mpu_time_max = data[:, 0].max()
    data[:, 0] -= mpu_time_min
    data[:, 0] = np.linspace(0, mpu_time_max - mpu_time_min, len(data))

    sample_rate = 0.001 * (mpu_time_max - mpu_time_min)/len(data)

    if win_len > 0:
        n_window = math.floor((mpu_time_max - mpu_time_min)/win_len)
        n_winlen = math.floor(data.shape[0]/n_window)
        if n_winlen == 0:
            return None, None
    else:
        n_window = 1
        n_winlen = data.shape[0]

    n_feature = len(feature_cols)

    fft_freqs = np.fft.rfftfreq(n_winlen, d=sample_rate)
    fft_topk = 20
    fft_result = np.ndarray((n_feature, n_window, len(fft_freqs)))
    swing_frequency = np.ndarray((n_feature, n_window))
    for i in range(0, n_feature):
        if debug:
            print(i, "-th feature stats", stats.describe(data[:, i+1]))

        if visualize:
            plt.figure(str(i) + '-th data')
            plt.plot(data[:, 0], data[:, i+1])
            plt.figure(str(i) + '-th spectrum (A-T)')

        for j in range(0, n_window):
            x, y = fft(data[n_winlen*j:n_winlen*(j+1), i+1],
                       topk=fft_topk, sample_rate=sample_rate)
            fft_result[i, j, :] = y.copy()

            if debug:
                print('Swing period for feature {0} window {1}: {2}'.format(
                    i, j, 1/x[np.argmax(y)]))

            if visualize:
                plt.scatter(1/x, y, label=n_winlen*j +
                            i, alpha=.5, s=100*y/max(y))

            swing_frequency[i, j] = 1/x[np.argmax(y)]
    if visualize:
        plt.show()

    if win_len > 0:
        return fft_result, swing_frequency
    else:
        return fft_result[:, 0, :], swing_frequency[:, 0]


if __name__ == "__main__":
    import os
    import json
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = dir_path + '\\1550349479_0.json'
    data = np.array(json.load(open(file_path, 'r')))
    # ts_col = 10
    # mpu_cols = [1,2,3,4,5,6]
    # hit_cols = [7,8,9]
    ts_col = 0
    mpu_cols = [1, 2, 3, 4, 5, 6]
    hit_cols = [7, 8, 9]

    # check if data points are ordered & have consistent segments
    plt.scatter(range(len(data)), data[:, ts_col],s=10,alpha=.5)
    plt.show()

    swing_polls, swing_groups = list(swing_count_svc(data, ts_col, mpu_cols))
    print(swing_polls)

    hit_result = hit_detection_svc(data, ts_col, hit_cols)
    indices = np.argwhere(np.abs(np.dot(hit_result, [[0], [1], [1]])) > 0)
    import matplotlib.pyplot as plt
    plt.scatter(hit_result[indices[:, 0], 1],
                hit_result[indices[:, 0], 2], alpha=.5)
    plt.plot([-1, 1, 0, -1], [0, 0, -math.sqrt(3), 0])
    plt.show()
    # print(hit_result)

    i = mpu_cols[1]
    plt.plot(data[:, ts_col], 130 + 10*(data[:, i+1] -
                               data[:, i+1].mean())/data[:, i+1].std())
    for h in hit_cols:
        plt.scatter(data[:, ts_col], data[:, h], alpha=.5, label=h)
    for idx, ts_dedup in enumerate(swing_groups):
        plt.scatter([np.mean(tsd) for tsd in ts_dedup], (10*idx + 100) * np.ones(len(ts_dedup)), s=50, alpha=.2, label=mpu_cols[idx])
    plt.legend(loc='upper right')
    plt.show()

    for h in hit_cols:
        x = np.zeros(len(data), dtype=bool)
        x[np.argwhere(data[:,h]>0)]=True
        plt.scatter(data[:, ts_col], x, alpha=.5, label=h, s=data[:,h])
    plt.legend(loc='upper right')
    plt.show()

    fft_result, swing_freq = list(fft_svc(data, ts_col, mpu_cols))
    print(swing_freq)

    pass
