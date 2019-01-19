# acc file name
acc_path = 'accelerometer.csv'
acc_feature = 'Absolute acceleration (m/s^2)'
acc_sample_rate = 0.005 # 5ms


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(acc_path, header=0)

df.plot()
plt.show()

y = df[acc_feature].values
y -= np.mean(y)
freq_map = np.fft.rfft(y)
freq_map_abs = np.abs(freq_map)

plt.figure(1)
plt.plot(freq_map_abs)


# sort by abs, but take original spectrum (w/ imaginary part)
freq_reserve_idx = np.argsort(freq_map_abs)[-10:]
period_main = acc_sample_rate * freq_reserve_idx[-1]
print('Period main: {0}ms'.format(1000*period_main))
print(freq_reserve_idx)

freq_mask = np.ones(len(freq_map), dtype=bool)
freq_mask[freq_reserve_idx] = False

plt.scatter(freq_reserve_idx, freq_map_abs[freq_reserve_idx], color='red')

freq_map_filtered = freq_map.copy()
freq_map_filtered[freq_mask] = 0
y_filtered = np.fft.irfft(freq_map_filtered)

plt.figure(2)
plt.plot(y_filtered)
plt.plot(y)

plt.show()


