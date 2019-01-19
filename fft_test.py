# acc file name
acc_path = 'gyroscope.csv'
# acc_feature = 'Absolute acceleration (m/s^2)'
acc_feature = 'Absolute (rad/s)'
acc_sample_rate = 0.005  # 5ms


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal




def fft(y):
    y -= np.mean(y)
    freq_map = np.fft.rfft(y)
    freq_map_abs = np.abs(freq_map)
    freq_frequencies = np.fft.rfftfreq(len(y), d=acc_sample_rate)

    #plt.figure()
    #plt.plot(freq_frequencies, freq_map_abs/freq_map_abs.max(), alpha=.2)
    #plt.show()

    # sort by abs, but take original spectrum (w/ imaginary part)
    freq_reserve_idx = np.argsort(freq_map_abs)[-10:][::-1]
    print("Window size: ", len(y)*1000*acc_sample_rate, "ms")
    print("Main freqs (Hz): ", freq_frequencies[freq_reserve_idx])
    print("Main amps  (rel to max): ",
          freq_map_abs[freq_reserve_idx]/freq_map_abs[freq_reserve_idx].max())
    print()
    # print(freq_reserve_idx)
    

    freq_mask = np.ones(len(freq_map), dtype=bool)
    freq_mask[freq_reserve_idx] = False

    #plt.scatter(freq_reserve_idx, freq_map_abs[freq_reserve_idx], color='red')

    freq_map_filtered = freq_map.copy()
    freq_map_filtered[freq_mask] = 0
    y_filtered = np.fft.irfft(freq_map_filtered)

    plt.scatter(freq_frequencies, np.abs(freq_map_filtered)/freq_map_abs.max(), alpha=.5)

    #plt.figure()
    #plt.plot(y_filtered)
    #plt.plot(y)
    return (freq_frequencies, freq_map_abs) # x: frequency, y: amplititude

df1 = pd.read_csv('gyroscope.csv', header=0)
y1 = df1[acc_feature].values
fft(y1)

y2 = y1[2000:6000]
fft(y2)

#df.plot()
#plt.show()



# fft(y[3357:3726]) # single swing
# fft(y[5000:6000])
# fft(y[5000:7000])
# fft(y[2000:6000])
plt.show()

plt.figure(2)
plt.plot(y1, alpha=.5)
plt.plot(y2, alpha=.5)
plt.show()



# TODO: adjacent frequencies combination
