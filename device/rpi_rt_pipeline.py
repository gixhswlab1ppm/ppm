import numpy as np
from scipy import signal
import math
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import Counter
from scipy import stats
from scipy.signal import argrelextrema
import ujson
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

sample_rate = 0.005  # 5e-3s, 5ms
locations = {
7 : (0, 0),
8 : (10, 0),
9 : (5, 10),
}
visualize = True

if visualize == True:
    import matplotlib.pyplot as plt

def fft(y_temp, topk=.1):
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

def triangulate(data):
    """takes in time-series piezometer data nested array form:
    [[piezo1, piezo2, piezo3],...,[piezo1, piezo2, piezo3]]
    with the assumption that piezo1 is in the top left corner
    piezo2 is in the top right corner equidistant from the handle
    piezo3 is near the handle"""
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
def triangulate_centroid(readings, circles=[[-1,0], [1,0], [0, -math.sqrt(3)]]):
	"""
	Given a 1x3 array of readings, and 3x2 array of circles
	Returns the weighted centroid (1x2)
	"""
	return np.divide(np.dot(readings, circles), .0001 + np.sum(circles, axis=0))

def clustering_dedup(maximas): # maximas: timestamps
    # merge within window until no change
    # 2-level clustering (dedup then detect, respectively)
    idx_min = maximas.min()
    idx_max = maximas.max()
    import math
    seg_span = 1.7 # 1 sec per window

    segs = np.array(maximas).reshape(-1,1)
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
                segs = [*segs[:curr_seg_idx], [*curr_seg, *next_seg], *segs[curr_seg_idx+2:]]
                epoch_updated = True
            curr_seg_idx += 1
        print(len(segs))
        if epoch_updated == False:
            break
    print("!!!")
    return segs


def rt_process_file(mpu_obj):
    print("#######################################")
    swing_period = (1.5, 3)

    mpu = mpu_obj
    mpu = np.array(mpu)
    mpu_time_col = mpu[:, -1].copy() # [temp, IMU, piezo, time]
    mpu[:, -1] = mpu[:, 0].copy()
    mpu[:, 0] = mpu_time_col.copy()
    mpu_time_min = mpu[:,0].min()
    mpu_time_max = mpu[:,0].max()
    mpu[:, 0] -= mpu_time_min
    mpu[:, 0] = np.linspace(0, mpu_time_max - mpu_time_min, len(mpu))

    global sample_rate
    sample_rate = (mpu_time_max - mpu_time_min)/len(mpu)

    feature_cols = [1,2,3,4,5,6]
    n_feature = len(feature_cols)
    data = mpu[:, [0, *feature_cols]]
    for fc in feature_cols:
        data[:, fc] -= np.mean(data[:, fc])

    period_poll = []
    t_window = 10
    n_window = math.floor(data[-1, 0]/t_window)
    n_winlen = math.floor(data.shape[0]/n_window)
    fft_freqs = np.fft.rfftfreq(n_winlen, d=sample_rate)
    fft_topk = 20
    fft_amps = np.ndarray((n_feature, n_window, len(fft_freqs))) # 2D (feature, window) array of (list of) amps

    for i in range(0, n_feature):
        print("@@@@@@@@@@@@@@")
        print(i, "-th feature stats", stats.describe(data[:, i+1]))
        
        maxima_idx = np.array(argrelextrema(data[:, i+1], np.greater))[0]
        threshold  = data[:, i+1].std()*0.8
        maxima_idx_filtered = [m_i for m_i in maxima_idx if data[m_i, i+1] > threshold]
        ts_dedup = clustering_dedup(data[maxima_idx_filtered,0])
        print(i, "-th feature ", 'detected ', len(maxima_idx_filtered), ' peaks')

        if visualize == True:
            # plt.figure(str(i) + '-th data')
            plt.scatter(data[maxima_idx_filtered,0], [1-0.1*i]*len(maxima_idx_filtered), s=10*data[maxima_idx_filtered,i+1]/threshold)
            plt.scatter([np.mean(seg) for seg in ts_dedup], [1.5+0.05*i]*len(ts_dedup), s=[10*len(seg)for seg in ts_dedup], alpha=.5)
            if i==2 :
                plt.plot(data[:, 0], data[:, i+1])

        for j in range(0, n_window):
            x, y = fft(data[n_winlen*j:n_winlen*(j+1), i+1], topk=fft_topk)
            fft_amps[i,j,:] = y.copy()
            print('Swing period for window ', j, ' is ', 1/x[np.argmax(y)])

            main_signal_idx = np.argmax(y)
            main_signal_period = 1/x[main_signal_idx]
            main_signal_amp = y[main_signal_idx]

            if main_signal_period >= swing_period[0] and main_signal_period <= swing_period[1]:
                period_poll.append((main_signal_period, main_signal_amp))

    print('Swing period polls: ', period_poll)
    ## below is hit-pnt detection
    print("#######################################")
    # triangulate(mpu[:, [7,8,9]])
    try:
        period_pred = np.max([val for val, count in Counter([p for p,a in period_poll]).most_common(1)])
        print('Predicted swing period: ', period_pred)
    except:
        pass
    finally:
        pass



if __name__ == "__main__":
    rt_process_file(ujson.load(open(dir_path+'/1548709262.6291513.json','r')))
    if visualize == True:
        plt.show()


