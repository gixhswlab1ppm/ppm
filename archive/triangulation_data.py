# Sensor data collection on Raspberry Pi in real-time
sample_rate = 0.005  # 5e-3s, 5ms
# WARNING must keep sampling consistency!

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import math
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import Counter
import ujson
from scipy import stats
from scipy.signal import argrelextrema

# swing_period = (1.5, 3)
# 
# def fft(y_temp, topk=.1):
#     """
#     Given a feature vector (1-D) sorted in time domain, this function performs a Fast DFT (real part only) and returns:
#     x_freq: frequencies in frequency domain
#     y_freq_abs | y_freq_abs_filtered: amplitude in frequency domain 

#     Arguments:
#     y_temp: feature vector in time domain
#     topk: \in (0,1) | [1, len(y_temp)]; returns the top frequencies & amplitudes by top percentage or count
#     """
#     y_temp -= np.mean(y_temp)
#     y_freq = np.fft.rfft(y_temp)
#     y_freq_abs = np.abs(y_freq)
#     x_freq = np.fft.rfftfreq(len(y_temp), d=sample_rate)
#     if topk != None:
#         topk = int(topk*len(y_freq)) if topk<1 else topk
#         y_freq_idx_by_amp = np.argsort(y_freq_abs)[::-1][:topk]
#         y_freq_mask = np.ones(len(y_freq), dtype=bool)
#         y_freq_mask[y_freq_idx_by_amp] = False
#         y_freq_filtered = y_freq.copy()
#         y_freq_filtered[y_freq_mask] = 0
#         y_freq_abs_filtered = np.abs(y_freq_filtered)
#         # return x_freq, y_freq_abs_filtered/y_freq_abs_filtered.max()
#         return x_freq, y_freq_abs_filtered
#     else: 
#         return x_freq, y_freq_abs

mpu = ujson.load(open('C:\\Users\\James\\ppm2\\device\\1548719339.7204287.json','r'))
mpu = np.array(mpu)
mpu_time_col = mpu[:, -1].copy()
mpu[:, -1] = mpu[:, 0].copy()
mpu[:, 0] = mpu_time_col.copy()
mpu_time_min = mpu[:,0].min()
mpu_time_max = mpu[:,0].max()
mpu[:, 0] -= mpu_time_min
mpu[:, 0] = np.linspace(0, mpu_time_max - mpu_time_min, len(mpu))

sample_rate = (mpu_time_max - mpu_time_min)/len(mpu)
# mpu[:,0] -= mpu[0,0]
# tf = np.arange(0, np.max(mpu[:, 0]), sample_rate)
# mpu = mpu[[np.argmin(abs(mpu[:, 0]-t)) for t in tf]]
# mpu[:, 0] = np.array(tf)

data = mpu[:, [7, 8, 9]]
locations = {
    7 : (0, 0),
    8 : (10, 0),
    9 : (5, 10),
}
n_feature = 3

"""takes in time-series piezometer data nested array form:
[[piezo1, piezo2, piezo3],...,[piezo1, piezo2, piezo3]]
with the assumption that piezo1 is in the top left corner
piezo2 is in the top right corner equidistant from the handle
piezo3 is near the handle"""
def triangulate(data):
    coords = [[], [], []]
    i = 0
    for datum in data:
        # find r for trilateration via pythagorean theorem
        DistA = datum[0]
        DistB = datum[1]
        DistC = datum[2]

        # find p1, p2 (beginning coords, set in locations dict above)
        LatA = locations[7][0]
        LonA = locations[7][1]
        LatB = locations[8][0]
        LonB = locations[8][1]
        LatC = locations[9][0]
        LonC = locations[9][1]

        # if piezometers all sense data
        # change ands to ors if you want judgement
        if (DistA != 0) and (DistB != 0) and (DistC != 0):
            # find ratios of distance
            dist_1 = DistA / (DistA + DistB)

            dist_2 = DistA / (DistA + DistC)

            dist_3 = DistC / (DistC + DistB)

            # print(dist_1, dist_2, dist_3)

            """ 
            calculate  edges of triangle of location based on ratios
            of distance between each node
            """

            # TODO/WARNING: Right now, this might work the wrong way! It might say that it is closer if it is louder.

            loc_1x = (dist_1 * (LatB - LatA)) + LatA
            loc_1y = (dist_1 * (LonB - LonA)) + LonA

            loc_2x = (dist_2 * (LatC - LatA)) + LatA
            loc_2y = (dist_2 * (LonC - LonA)) + LonA

            loc_3x = LatB - (dist_3 * (LatB - LatC))
            loc_3y = LatC - (dist_3 * (LonB - LonC))

            # print("guess dist a/c", [loc_1x, loc_1y])
            # print("guess dist a/b", [loc_2x, loc_2y])
            # print("guess dist b/c", [loc_3x, loc_3y])

            # average the edges to find the center of the formed triangle
            # print("this is the estimate x location", loc_1x, loc_2x, loc_3x)
            x = (loc_1x + loc_2x + loc_3x)/3
            y = (loc_1y + loc_2y + loc_3y)/3

            if np.isnan(x) or np.isnan(y):
                pass
            else:
                coords[0].append(x)
                coords[1].append(10-y)
                coords[2].append(i)
                # coords.append([x, y, i])

                if (DistC < DistA):
                    if (DistC < DistB):
                        guess = ", Ball is between 1 and 2"
                if (DistB < DistA):
                    if (DistB < DistC):
                        guess = ", Ball is between 1 and 3"
                if (DistA < DistC):
                    if (DistA < DistB):
                        guess = ", Ball is between 2 and 3"
            print("Piezometer 1: ", DistA, ", 2: ", DistB, ", 3: ", DistC, ", period: ", i, guess, sep='')
        i += 1

                # uncomment to print data and coords, change coords = [x, y]

                # print("this is the estimated location")
                # print(coords)
                # print()
    return coords

guesses = triangulate(data)
print(guesses)
plt.scatter(guesses[0], guesses[1], alpha=0.5)
# plt.scatter(guesses[2], guesses[1], alpha=0.5)
plt.show()

# TODO: make a plot showing the data directly, and comparing it over time against the location guess

# print(data)

# ## if VIZ
# for i in range(0, n_feature):
#     x, y = fft(data[:, i+1])
#     print("All Data")
#     ind_y = np.argsort(y)[::-1]
#     print("Main periods: ", 1/x[ind_y[0]], 1/x[ind_y[1]], 1 /
#           x[ind_y[2]], 1/x[ind_y[3]], 1/x[ind_y[4]], 1/x[ind_y[5]])
#     # plt.scatter(x, y, label=i, alpha=.5, s=100*y)
#     plt.scatter(x, y, label=i, alpha=.5)
# plt.legend(loc='upper left')
# plt.show()
# ## endif

# period_poll = []
# t_window = 10
# n_window = math.floor(data[-1, 0]/t_window)
# n_winlen = math.floor(data.shape[0]/n_window)
# fft_freqs = np.fft.rfftfreq(n_winlen, d=sample_rate)
# fft_topk = 20
# fft_amps = np.ndarray((n_feature, n_window, len(fft_freqs))) # 2D (feature, window) array of (list of) amps
# maxima_store = []

# for i in range(0, n_feature):
#     print(i, " th feature stats", stats.describe(data[:, i+1]))
    
#     maxima_idx = np.array(argrelextrema(data[:, i+1], np.greater))[0]
#     threshold  = data[:, i+1].std()/2
#     maxima_idx_filtered = [m_i for m_i in maxima_idx if data[m_i, i+1] > threshold]
#     maxima_store.append(maxima_idx_filtered)
#     print('detected ', len(maxima_idx_filtered), ' peaks')
#     # TODO use detected main freqs to reduce duplicates
#     plt.xlim(0, 10)  # max freq should be at the 2nd column
#     plt.legend(loc='upper right')
#     plt.figure(str(i) + '-th data')
#     plt.plot(data[:, 0], data[:, i+1])
#     plt.scatter(data[maxima_idx_filtered,0], data[maxima_idx_filtered,i+1])

#     print('Feature {0}'.format(i))
#     plt.figure(str(i) + '-th spectrum')
#     for j in range(0, n_window):
#         x, y = fft(data[n_winlen*j:n_winlen*(j+1), i+1], topk=fft_topk)
#         fft_amps[i,j,:] = y.copy()
#         # plt.scatter(x+j*20, y, label=n_winlen*j+i, alpha=.5)
#         plt.scatter(1/x, y, label=n_winlen*j+i, alpha=.5, s=100*y/max(y))
#         print('Main freq:', x[np.argmax(y)], np.argmax(y))

#         main_signal_idx = np.argmax(y)
#         main_signal_period = 1/x[main_signal_idx]
#         main_signal_amp = y[main_signal_idx]

#         if main_signal_period >= swing_period[0] and main_signal_period <= swing_period[1]:
#             period_poll.append((main_signal_period, main_signal_amp))

#             # print('Main period:', 1/x[np.argmax(y)])
#         # plt.legend(loc='upper left')
#     # overall fft
    

# # TODO: handle empty array case
# period_pred = np.max([val for val, count in Counter([p for p,a in period_poll]).most_common(1)])
# print('Predicted main period', period_pred)
# print(period_poll)

# # Use period_pred to dedup maxima_store
# for p_idx_list in maxima_store:
#     pass
# # TODO: {outlier detection | swing detection} by levels: window, timestamp?





# plt.show()
# ## endif


# WARNING Scaler is in question...
# scaler = StandardScaler()
# data = scaler.fit_transform(data[:,1:])
# pca = PCA().fit(data[:,1:])
# print(pca.explained_variance_ratio_)

# df1 = pd.read_csv('gyroscope.csv', header=0)
# y1 = df1[acc_feature].values
# fft(y1)

# y2 = y1[2000:6000]
# fft(y2)

# #df.plot()
# #plt.show()


# TODO: CANCELLED adjacent frequencies combination
# This is not mathematically correct.
