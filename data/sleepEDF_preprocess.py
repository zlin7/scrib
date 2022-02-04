import sys
import mne
import numpy as np
import os
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import pandas as pd
from multiprocessing import Process
import pickle
from _settings import EDF_PATH


def train_val_test(root_folder, k, N, epoch_sec):
    all_index = np.unique([path[:6] for path in os.listdir(root_folder)])

    train_index = np.random.choice(all_index, int(len(all_index) * 0.8), replace=False)
    test_index = np.random.choice(list(set(all_index) - set(train_index)), int(len(all_index) * 0.1), replace=False)
    val_index = list(set(all_index) - set(train_index) - set(test_index))

    sample_package(root_folder, k, N, epoch_sec, 'train', train_index)
    sample_package(root_folder, k, N, epoch_sec, 'test', test_index)
    sample_package(root_folder, k, N, epoch_sec, 'val', val_index)


def sample_package(root_folder, k, N, epoch_sec, train_test_val, index):
    base_dirname = os.path.join(EDF_PATH, 'cassette_processed', train_test_val)
    for i, j in enumerate(index):
        if i % N == k:
            print('train', i, j, 'finished')

            # X load
            data = mne.io.read_raw_edf(
                root_folder + '/' + list(filter(lambda x: (x[:6] == j) and ('PSG' in x), os.listdir(root_folder)))[0])
            X = data.get_data()[:2, :]  # 'EEG Fpz-Cz','EEG Pz-Oz',
            ann = mne.read_annotations(root_folder + '/' + list(
                filter(lambda x: (x[:6] == j) and ('Hypnogram' in x), os.listdir(root_folder)))[0])
            labels = []
            for dur, des in zip(ann.duration, ann.description):
                for i in range(int(dur) // 30):
                    labels.append(des[-1])

            for slice_index in range(X.shape[1] // (100 * epoch_sec)):
                if labels[slice_index] == '?':
                    continue
                fname = 'cassette-' + j + '-' + str(slice_index) + '.pkl'
                path = os.path.join(base_dirname, fname)
                pickle.dump({'X': X[:, slice_index * 100 * epoch_sec: (slice_index + 1) * 100 * epoch_sec], \
                             'y': labels[slice_index]}, open(path, 'wb'))


if __name__ == '__main__':
    if not os.path.isdir(os.path.join(EDF_PATH, 'cassette_processed')):
        for _s in ['train', 'test', 'val']:
            os.makedirs(os.path.join(EDF_PATH, 'cassette_processed', _s))

    root_folder = os.path.join(EDF_PATH, 'sleep-edf-database-expanded-1.0.0', 'sleep-cassette')

    N, epoch_sec = 8, 30
    p_list = []
    for k in range(N):
        process = Process(target=train_val_test, args=(root_folder, k, N, epoch_sec))
        process.start()
        p_list.append(process)

    for i in p_list:
        i.join()

