import numpy as np
from scipy import signal
import math
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import Counter
from scipy import stats
from scipy.signal import argrelextrema

sample_rate = 0.005  # 5e-3s, 5ms

def fft(y_temp, topk=.1):
    """
    Given a feature vector (1-D) sorted in time domain, this function performs a Fast DFT (real part only) and returns:
    x_freq: frequencies in frequency domain
    y_freq_abs | y_freq_abs_filtered: amplitude in frequency domain 

    Arguments:
    y_temp: feature vector in time domain
    topk: \in (0,1) | [1, len(y_temp)]; returns the top frequencies & amplitudes by top percentage or count
    """
    y_temp -= np.mean(y_temp)
    y_freq = np.fft.rfft(y_temp)
    y_freq_abs = np.abs(y_freq)
    x_freq = np.fft.rfftfreq(len(y_temp), d=sample_rate)
    if topk != None:
        topk = int(topk*len(y_freq)) if topk<1 else topk
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

def rt_process_file(mpu_obj):
    print("#######################################")
    swing_period = (1.5, 3)

    mpu = mpu_obj
    mpu = np.array(mpu)
    mpu_time_col = mpu[:, -1].copy()
    mpu[:, -1] = mpu[:, 0].copy()
    mpu[:, 0] = mpu_time_col.copy()
    mpu_time_min = mpu[:,0].min()
    mpu_time_max = mpu[:,0].max()
    mpu[:, 0] -= mpu_time_min
    mpu[:, 0] = np.linspace(0, mpu_time_max - mpu_time_min, len(mpu))

    global sample_rate
    sample_rate = (mpu_time_max - mpu_time_min)/len(mpu)

    data = mpu[:, [0,2,3,4,5,6,7]]
    n_feature = 6

    period_poll = []
    t_window = 10
    n_window = math.floor(data[-1, 0]/t_window)
    n_winlen = math.floor(data.shape[0]/n_window)
    fft_freqs = np.fft.rfftfreq(n_winlen, d=sample_rate)
    fft_topk = 20
    fft_amps = np.ndarray((n_feature, n_window, len(fft_freqs))) # 2D (feature, window) array of (list of) amps
    maxima_store = []

    for i in range(0, n_feature):
        print("@@@@@@@@@@@@@@")
        print(i, "-th feature stats", stats.describe(data[:, i+1]))
        
        maxima_idx = np.array(argrelextrema(data[:, i+1], np.greater))[0]
        threshold  = data[:, i+1].std()/2
        maxima_idx_filtered = [m_i for m_i in maxima_idx if data[m_i, i+1] > threshold]
        maxima_store.append(maxima_idx_filtered)
        print(i, "-th feature ", 'detected ', len(maxima_idx_filtered), ' peaks')

        for j in range(0, n_window):
            x, y = fft(data[n_winlen*j:n_winlen*(j+1), i+1], topk=fft_topk)
            fft_amps[i,j,:] = y.copy()
            print('Swing period for window ', j, ' is ', 1/x[np.argmax(y)])

            main_signal_idx = np.argmax(y)
            main_signal_period = 1/x[main_signal_idx]
            main_signal_amp = y[main_signal_idx]

            if main_signal_period >= swing_period[0] and main_signal_period <= swing_period[1]:
                period_poll.append((main_signal_period, main_signal_amp))

    period_pred = np.max([val for val, count in Counter([p for p,a in period_poll]).most_common(1)])
    print('Predicted swing period: ', period_pred)
    print('Swing period polls: ', period_poll)