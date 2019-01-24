# acc file name
acc_features = [
    "Linear Acceleration x (m/s^2)", "Linear Acceleration y (m/s^2)", "Linear Acceleration z (m/s^2)"]
gyro_features = [
    "Gyroscope x (rad/s)", "Gyroscope y (rad/s)", "Gyroscope z (rad/s)"]

sample_rate = 0.005  # 5e-3s, 5ms
# WARNING must keep sampling consistency!

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import math
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
    if topk <= 0:
        y_freq_idx_by_amp = np.argsort(y_freq_abs)[::-1]
        # return x_freq, y_freq_abs/y_freq_abs.max()
        return x_freq, y_freq_abs
    else:
        if topk < 1:
            topk = int(topk*len(y_freq))
        y_freq_idx_by_amp = np.argsort(y_freq_abs)[::-1][:topk]
        y_freq_mask = np.ones(len(y_freq), dtype=bool)
        y_freq_mask[y_freq_idx_by_amp] = False
        y_freq_filtered = y_freq.copy()
        y_freq_filtered[y_freq_mask] = 0
        y_freq_abs_filtered = np.abs(y_freq_filtered)
        # return x_freq, y_freq_abs_filtered/y_freq_abs_filtered.max()
        return x_freq, y_freq_abs_filtered

    #plt.figure()
    #plt.plot(freq_frequencies, freq_map_abs/freq_map_abs.max(), alpha=.2)
    #plt.show()

    # sort by abs, but take original spectrum (w/ imaginary part)

    # print("Window size: ", len(y_temp)*1000*acc_sample_rate, "ms")
    # print("Main freqs (Hz): ", x_freq[y_freq_idx_by_amp])
    # print("Main amps  (rel to max): ",
    #       y_freq_abs[y_freq_idx_by_amp]/y_freq_abs[y_freq_idx_by_amp].max())
    # print()
    # # print(freq_reserve_idx)

    # freq_mask = np.ones(len(y_freq), dtype=bool)
    # freq_mask[y_freq_idx_by_amp] = False

    # #plt.scatter(freq_reserve_idx, freq_map_abs[freq_reserve_idx], color='red')

    # freq_map_filtered = y_freq.copy()
    # freq_map_filtered[freq_mask] = 0
    # y_filtered = np.fft.irfft(freq_map_filtered)

    # plt.scatter(x_freq, np.abs(freq_map_filtered)/y_freq_abs.max(), alpha=.5)

    # #plt.figure()
    # #plt.plot(y_filtered)
    # #plt.plot(y)
    # return (x_freq, y_freq_abs) # x: frequency, y: amplititude



# align to nearest sampling time
acc = pd.read_csv('accelerometer_cutuy_swing_walk.csv', header=0).values
gyro = pd.read_csv('gyroscope_cutuy_swing_walk.csv', header=0).values
tf = np.arange(0, min(np.max(acc[:, 0]), np.max(gyro[:, 0])), sample_rate)
acc = acc[[np.argmin(abs(acc[:, 0]-t)) for t in tf]]
acc[:, 0] = np.array(tf)
gyro = gyro[[np.argmin(abs(gyro[:, 0]-t)) for t in tf]]
gyro[:, 0] = np.array(tf)

# [t, accx, accy, accz, gyrox, gyro y, gyro z]
data = np.hstack((acc[:, 0:4], gyro[:, 1:4]))

# TODO: spectrum comparison function
n_feature = data.shape[1]-1



## if VIZ
for i in range(0, n_feature):
    x, y = fft(data[:, i+1])
    print("All Data")
    ind_y = np.argsort(y)[::-1]
    print("Main periods: ", 1/x[ind_y[0]], 1/x[ind_y[1]], 1/x[ind_y[2]], 1/x[ind_y[3]], 1/x[ind_y[4]], 1/x[ind_y[5]])
    # plt.scatter(x, y, label=i, alpha=.5, s=100*y)
    plt.scatter(x, y, label=i, alpha=.5)
plt.legend(loc='upper left')
plt.show()
## endif

## if VIZ
# validation for hypothesis: spectrum among different time range should be similar
# result: yeah!
# TODO: trim outlier time ranges
# TODO: find partial-data main period versus overal period variance in different n_window settings (window_length in fact)
for i in range(0, n_feature):
    plt.figure(str(i) + '-th spectrum;')
    t_window = 10
    n_window = math.floor(data[-1,0]/t_window)
    n_winlen = math.floor(data.shape[0]/n_window)
    
    print('Feature {0}'.format(i))
    for j in range(0, n_window):
        x, y = fft(data[n_winlen*j:n_winlen*(j+1), i+1], topk=.05)
        # plt.scatter(x+j*20, y, label=n_winlen*j+i, alpha=.5)
        plt.scatter(x, y, label=n_winlen*j+i, alpha=.5)
        # print('Main freq:', x[np.argmax(y)], np.argmax(y))
        print('Main period:', 1/x[np.argmax(y)])
        # plt.legend(loc='upper left')
    # overall fft
    plt.xlim(0, 10) # max freq should be at the 2nd column
    plt.legend(loc='upper right')
    plt.figure(str(i) + '-th data')
    plt.plot(data[:, 0], data[:, i+1])

plt.show()
## endif




# WARNING Scaler is in question...
# scaler = StandardScaler()
# data = scaler.fit_transform(data[:,1:])
# pca = PCA().fit(data[:,1:])
# print(pca.explained_variance_ratio_)


pass
# df1 = pd.read_csv('gyroscope.csv', header=0)
# y1 = df1[acc_feature].values
# fft(y1)

# y2 = y1[2000:6000]
# fft(y2)

# #df.plot()
# #plt.show()


# TODO: CANCELLED adjacent frequencies combination
# This is not mathematically correct.
