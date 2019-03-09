import numpy as np
import math
from collections import Counter
from scipy import stats
from scipy.signal import argrelextrema
import ujson
import os
from scipy import stats
dir_path = os.path.dirname(os.path.realpath(__file__))

visualize = False
debug = False

hit_th = 10
hit_cluster_size = 1000
ts_col = 0
mpu_cols = [1, 2, 3, 4, 5, 6]
hit_cols = [7, 8, 9]

if visualize == True:
    import matplotlib.pyplot as plt

#  stateless algorithms


# center must be (0,0)
def triangulate_centroid(readings, circles=[[-1/2, -math.sqrt(3)/6], [0, 1./math.sqrt(3)], [1/2, -math.sqrt(3)/6]]):
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


def clustering_dedup(maximas, seg_span=1500):  # maximas: timestamps
        # merge within window until no change
        # 2-level clustering (dedup then detect, respectively)
        idx_min = maximas.min()
        idx_max = maximas.max()

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


def swing_count_svc(data, ts_col, feature_cols, pre_thresholds=None): # =[.3, .2, .34, 50, 30, 35]
    # data = shift_mean(data, ts_col, feature_cols)
    cluster_ts_per_feature = []
    curr_thresholds = np.ndarray((len(feature_cols)))
    for i in range(0, len(feature_cols)):
        maxima_idx = np.array(argrelextrema(data[:, i+1], np.greater))[0]
        curr_thresholds[i] = data[:, i+1].std()*0.6
        if pre_thresholds is None:
            threshold = curr_thresholds[i]
        else:
            threshold = pre_thresholds[i]

        maxima_idx_filtered = [
            m_i for m_i in maxima_idx if data[m_i, i+1] > threshold]
        if len(maxima_idx_filtered) > 0:
            cluster_ts = clustering_dedup(
                data[maxima_idx_filtered, 0], seg_span=1500)
            if visualize:
                plt.scatter(data[maxima_idx_filtered, 0],
                            data[maxima_idx_filtered, i+1], s=20, alpha=.5)
                plt.scatter([np.mean(tsd) for tsd in cluster_ts], pre_thresholds[i] * np.ones(len(cluster_ts)), s=40, alpha=.5)
                plt.plot(data[:, 0], data[:, i+1])
                plt.show()
            cluster_ts_per_feature.append(cluster_ts)
        else:
            cluster_ts_per_feature.append([])
    return cluster_ts_per_feature, curr_thresholds


def get_swing_count_from_ts_clusters_by_feature(ts_clusters_by_feature):
    return int(np.round(np.median([len(clusters) for clusters in ts_clusters_by_feature])))


def get_swing_centers_from_ts_cluster_by_feature(ts_clusters_by_feature):
    centers_by_feature = []
    for clusters in ts_clusters_by_feature:
        centers_by_feature.append([np.mean(cluster) for cluster in clusters])
    return centers_by_feature


def get_hit_dispersity_temporal(hit_event):
    reading_sum = np.sum(hit_event[:, 1:])
    t_center = np.sum(np.dot(hit_event[:, 0], hit_event[:, 1:]))/reading_sum
    dist_sum = np.sum(
        np.dot(np.abs(hit_event[:, 0]-t_center), hit_event[:, 1:]))
    return dist_sum/reading_sum


def get_hit_dispersity_spatial(hit_event):
    s_center = np.mean([triangulate_centroid(row)
                        for row in hit_event], axis=0)
    # if debug & visualize:
    #     centers = np.array([triangulate_centroid(row)
    #                     for row in hit_event])
    #     plt.scatter(centers[:, 0], centers[:, 1], alpha=.5)
    #     plt.plot([-1/2, 1/2, 0, -1/2], [-math.sqrt(3)/6, -math.sqrt(3)/6, math.sqrt(3)/3, -math.sqrt(3)/6])
    #     plt.show()
    return np.linalg.norm(s_center)


def hit_report_svc(data, ts_col, hit_cols):
    valued_indices_mask = np.zeros(len(data), dtype=bool)
    for hc in hit_cols:
        valued_indices_mask[np.argwhere(data[:, hc] > hit_th)] = True
    if np.sum(valued_indices_mask) > 0:
        hit_events = clustering_dedup(
            data[np.argwhere(valued_indices_mask == True), 0], seg_span=hit_cluster_size)
        n_event = len(hit_events)
        # features per "hit event"

        hit_dispersity_temporal = np.empty(
            n_event, dtype=float)  # bound by hit event length
        # hit_dispersity_spatial = np.empty(
        #     n_event, dtype=float)  # bound by hit triangle size
        hit_strength = np.empty(n_event, dtype=float)  # bound by hit reading

        for i, event_ts in enumerate(hit_events):
            indices = np.intersect1d(np.argwhere(
                data[:, ts_col] >= event_ts[0]), np.argwhere(data[:, ts_col] <= event_ts[-1]))
            hit_event_with_ts = data[:, [ts_col, *hit_cols]][indices]

            hit_dispersity_temporal[i] = np.min(
                (1., get_hit_dispersity_temporal(hit_event_with_ts)/(1 << 7)))
            # hit_dispersity_spatial[i] = get_hit_dispersity_spatial(
            #     hit_event_with_ts[:, 1:])
            # sum of max reading along each ts
            hit_strength[i] = np.min(
                (1., np.sum(np.max(hit_event_with_ts[:, 1:], axis=0))/(1 << 9)))
        return hit_dispersity_temporal, hit_strength, hit_events
    else:
        return [], [], []


def get_dataset(data, ts_col, Y_cols, X_cols):
    # X: mpu; Y: hit
    valued_indices_mask = np.zeros(len(data), dtype=bool)
    for hc in Y_cols:
        valued_indices_mask[np.argwhere(data[:, hc] > hit_th)] = True
    hit_events = clustering_dedup(
        data[np.argwhere(valued_indices_mask == True), 0], seg_span=hit_cluster_size)
    # take last element of each cluster as end of each swing window
    dataset = []
    for event_ts in hit_events:
        window = [event_ts[0], event_ts[-1]]

        X_indices = np.intersect1d(np.argwhere(
            data[:, ts_col] > window[0] - hit_cluster_size), np.argwhere(data[:, ts_col] <= window[1]))
        X = data[:, X_cols][X_indices].tolist()

        Y_indices = np.intersect1d(np.argwhere(
            data[:, ts_col] >= event_ts[0]), np.argwhere(data[:, ts_col] <= event_ts[-1]))
        hit_event_with_ts = data[:, [ts_col, *Y_cols]][Y_indices]

        hit_dispersity_temporal = np.min(
            (1., get_hit_dispersity_temporal(hit_event_with_ts)/(1 << 7)))
        hit_dispersity_spatial = get_hit_dispersity_spatial(
            hit_event_with_ts[:, 1:])
        # sum of max reading along each ts
        hit_strength = np.min(
            (1., np.sum(np.max(hit_event_with_ts[:, 1:], axis=0))/(1 << 9)))
        Y = [hit_dispersity_temporal]

        dataset.append((X, Y, window))
        # In temporal space, X extends earlier than Y:
        # X - 1 2 3 4 5
        # Y - / / / 2 3
        # win - / / / T T
    return dataset

# win_len=0 for unwindowed fft, unit: millisec
def fft_svc(data, ts_col, feature_cols, win_len=0):
    visualize = False
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


def train_rnn(X, Y, X_test, Y_test):
    n_labels = 1

    from sklearn.preprocessing import StandardScaler
    import tensorflow as tf

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.__enter__()

    X = tf.keras.preprocessing.sequence.pad_sequences(
        X, maxlen=16, dtype='float')
    X_test = tf.keras.preprocessing.sequence.pad_sequences(
        X_test, maxlen=16, dtype='float')
    Y = np.array(Y)
    Y_test = np.array(Y_test)

    # for i in range(6):
    #     col_2 = []
    #     for xi in X:
    #         col_2.append([xii[i] for xii in xi])
    #     for c in col_2:
    #         plt.figure(i)
    #         plt.plot(c)

    # plt.show()

    input_X = tf.keras.layers.Input(shape=(16, 6))
    # r_0 = tf.keras.layers.SimpleRNN(24)(input_X)
    r_0 = tf.keras.layers.Conv1D(24, 1)(input_X)
    fla = tf.keras.layers.Flatten()(r_0)
    dense_0 = tf.keras.layers.Dense(16, activation='relu', use_bias=False)(fla)
    bn_0 = tf.keras.layers.BatchNormalization()(dense_0)
    dropout_0 = tf.keras.layers.Dropout(.5)(bn_0)
    dense_1 = tf.keras.layers.Dense(8, activation='relu', use_bias=False)(dropout_0)
    bn_1 = tf.keras.layers.BatchNormalization()(dense_1)
    dropout_1 = tf.keras.layers.Dropout(.5)(bn_1)
    output_Y = tf.keras.layers.Dense(n_labels, activation='sigmoid', use_bias=False)(dropout_1)
    model = tf.keras.models.Model(inputs=input_X, outputs=output_Y)
    print(model.summary())

    # The following arch has better loss but is incompatible with tfLite
    # input_X = tf.keras.layers.Input(shape=(16, 6))
    # r_0 = tf.keras.layers.SimpleRNN(24)(input_X)
    # dense_0 = tf.keras.layers.Dense(16, activation='elu', use_bias=False)(r_0)
    # bn_0 = tf.keras.layers.BatchNormalization()(dense_0)
    # dropout_0 = tf.keras.layers.Dropout(.5)(bn_0)
    # dense_1 = tf.keras.layers.Dense(8, activation='elu', use_bias=False)(dropout_0)
    # bn_1 = tf.keras.layers.BatchNormalization()(dense_1)
    # dropout_1 = tf.keras.layers.Dropout(.5)(bn_1)
    # output_Y = tf.keras.layers.Dense(n_labels, activation='sigmoid', use_bias=False)(dropout_1)
    # model = tf.keras.models.Model(inputs=input_X, outputs=output_Y)

    model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mse'])

    # TODO !impt scale by each feature X = StandardScaler().fit_transform(X)
    history = model.fit(X, [Y], epochs=1024,
                        validation_data=(X_test, [Y_test]))
    
    tf.keras.models.save_model(model, 'magic_model')
    converter = tf.lite.TFLiteConverter.from_keras_model_file('magic_model')
    tflite_model = converter.convert()
    open('magic_model.tflite', 'wb').write(tflite_model)

    # converter = tf.lite.TFLiteConverter.from_session(sess, input_X, output_Y)
    # tflite_model = converter.convert()
    # open("converted_model.tflite", "wb").write(tflite_model)



    y_pred = model.predict(X)
    y_test_pred = model.predict(X_test)

    for i in range(n_labels):
        y_err = np.abs(np.array(Y[:, i])-y_pred[:, i].reshape(len(Y[:, i])))
        y_test_err = np.abs(
            np.array(Y_test[:, i])-y_test_pred[:, i].reshape(len(Y_test[:, i])))
        y_err_rate = np.divide(y_err, np.array(Y[:, i]))
        y_test_err_rate = np.divide(y_test_err, np.array(Y_test[:, i]))
        print('>>> ', i, '-th label')
        print('training set err mean: {0:.4f}, 25-quant: {1:.4f}, 50-quant: {2:.4f}, 75-quant: {3:.4f}'.format(np.mean(
            y_err), np.quantile(y_err, 0.25), np.quantile(y_err, 0.50), np.quantile(y_err, 0.75)))
        # print('training set err rate mean: {0:.4f}%, 25-quant: {1:.4f}%, 50-quant: {2:.4f}%, 75-quant: {3:.4f}%'.format(100*np.mean(
        #     y_err_rate), 100*np.quantile(y_err_rate, 0.25), 100*np.quantile(y_err_rate, 0.50), 100*np.quantile(y_err_rate, 0.75)))
        print('test set err mean: {0:.4f}, 25-quant: {1:.4f}, 50-quant: {2:.4f}, 75-quant: {3:.4f}'.format(np.mean(y_test_err),
                                                                                               np.quantile(y_test_err, 0.25), np.quantile(y_test_err, 0.50), np.quantile(y_test_err, 0.75)))
        # print('test set err rate mean: {0:.4f}%, 25-quant: {1:.4f}%, 50-quant: {2:.4f}%, 75-quant: {3:.4f}%'.format(100*np.mean(y_test_err_rate),
        #                                                                                             100*np.quantile(y_test_err_rate, 0.25), 100*np.quantile(y_test_err_rate, 0.50), 100*np.quantile(y_test_err_rate, 0.75)))
        plt.figure(0)
        plt.plot(np.sort(y_err_rate), label='d_y')
        plt.plot(np.sort(y_test_err_rate), label='d_y_test')
        plt.legend(loc='upper right')
        plt.ylim([0, 2])

        plt.figure(1)
        plt.plot(history.history['loss'], label='loss_y')
        plt.plot(history.history['val_loss'], label='loss_y_test')
        plt.legend(loc='upper right')

        plt.figure(2)
        args = np.argsort(Y[:,i])
        plt.plot(Y[:, i][args], label='y')
        plt.plot(y_pred[:, i][args], label='y_pred')
        plt.ylim([0, 1])
        plt.legend(loc='upper right')

        plt.figure(3)
        args = np.argsort(Y_test[:,i])
        plt.plot(Y_test[:, i][args], label='y_test')
        plt.plot(y_test_pred[:, i][args], label='y_test_pred')
        plt.ylim([0, 1])
        plt.legend(loc='upper right')
        plt.show()

    return model

def extract_raw_data(file_name):
    import os
    import json
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = dir_path[:-7] + '/data/' + file_name
    data = np.array(json.load(open(file_path, 'r')))
    data[:, mpu_cols[3:]] /= 512.
    if visualize:
        for h in hit_cols:
            plt.scatter(data[:, 0], np.ones(len(data)), alpha=.5, s=data[:, h], label=h)
        plt.legend(loc='upper right')
        plt.show()
        for h in hit_cols:
            plt.scatter(data[:, 0], data[:, h], alpha=.5, label=h)
        plt.legend(loc='upper right')
        plt.show()
        for m in mpu_cols:
            plt.plot(data[:, m])
            plt.show()
    return data

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # train_data_list = ['1551925483_0.json']
    train_data_list = ['1551918555_0.json', '1551926050_0.json', '1551926286_0.json']
    data_collection = [extract_raw_data(td) for td in train_data_list]
    train_data = np.vstack([get_dataset(d, ts_col, hit_cols, mpu_cols) for d in data_collection])
    test_data = get_dataset(extract_raw_data('1551925483_0.json'), ts_col, hit_cols, mpu_cols)


    training_set = list(train_data)
    X = [d[0] for d in training_set]
    y = [d[1] for d in training_set]

    test_set = list(test_data)
    X_test = [d[0] for d in test_set]
    y_test = [d[1] for d in test_set]

    model = train_rnn(X, y, X_test, y_test)

    for data in data_collection:
        # check if data points are ordered & have consistent segments
        plt.scatter(range(len(data)), data[:, ts_col], s=10, alpha=.5)
        plt.show()

        ts_clusters_by_feature = list(swing_count_svc(data, ts_col, mpu_cols))
        print(get_swing_count_from_ts_clusters_by_feature(ts_clusters_by_feature))

        hit_dispersity_temporal, hit_dispersity_spatial, hit_strength = hit_report_svc(data, ts_col, hit_cols)


        i = mpu_cols[1]
        plt.plot(data[:, ts_col], 130 + 10*(data[:, i+1] -
                                            data[:, i+1].mean())/data[:, i+1].std())
        for h in hit_cols:
            plt.scatter(data[:, ts_col], data[:, h], alpha=.5, label=h)
        for idx, clusters in enumerate(ts_clusters_by_feature):
            plt.scatter([np.mean(c) for c in clusters], (10*idx + 100) *
                        np.ones(len(clusters)), s=50, alpha=.2, label=mpu_cols[idx])
        plt.legend(loc='upper right')
        plt.show()

        # hit-point viz & correlation w/ swing counts
        for h in hit_cols:
            x = np.zeros(len(data), dtype=bool)
            x[np.argwhere(data[:, h] > 0)] = True
            plt.scatter(data[:, ts_col], x, alpha=.5, label=h, s=data[:, h])

        ts_centers_by_feature = get_swing_centers_from_ts_cluster_by_feature(
            ts_clusters_by_feature)
        for i, centers in enumerate(ts_centers_by_feature):
            plt.scatter(centers, 1+0.1*np.ones(len(centers)), alpha=.5, s=20+10*i)

        plt.legend(loc='upper right')
        plt.show()

        fft_result, swing_freq = list(fft_svc(data, ts_col, mpu_cols))
        print(swing_freq)

    pass
