# acc file name
acc_path = 'accelerometer.csv'
acc_feature = 'Absolute acceleration (m/s^2)'
acc_sample_rate = 0.005  # 5ms


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(acc_path, header=0)

df.plot()
plt.show()

y = df[acc_feature].values


def fft(y):
    y -= np.mean(y)
    freq_map = np.fft.rfft(y)
    freq_map_abs = np.abs(freq_map)
    freq_frequencies = np.fft.rfftfreq(len(freq_map), d=acc_sample_rate)

    plt.figure()
    plt.plot(freq_map_abs)

    # sort by abs, but take original spectrum (w/ imaginary part)
    freq_reserve_idx = np.argsort(freq_map_abs)[-10:][::-1]
    print("Window size: ", len(y)*1000*acc_sample_rate, "ms")
    print("Main freqs (Hz): ", freq_frequencies[freq_reserve_idx])
    print("Main amps  (rel to max): ", freq_map_abs[freq_reserve_idx]/freq_map_abs[freq_reserve_idx].max())
    print()
    # print(freq_reserve_idx)

    freq_mask = np.ones(len(freq_map), dtype=bool)
    freq_mask[freq_reserve_idx] = False

    plt.scatter(freq_reserve_idx, freq_map_abs[freq_reserve_idx], color='red')

    freq_map_filtered = freq_map.copy()
    freq_map_filtered[freq_mask] = 0
    y_filtered = np.fft.irfft(freq_map_filtered)

    plt.figure()
    plt.plot(y_filtered)
    plt.plot(y)


fft(y)
# single swing
fft(y[3357:3726])
fft(y[5000:6000])
fft(y[5000:7000])
fft(y[2000:6000])

#plt.show()


# TODO: adjacent frequencies combination