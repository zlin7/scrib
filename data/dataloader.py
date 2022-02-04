import numpy as np
import os, glob
import bisect
import ipdb
import _settings as _settings
from _settings import ISRUC_NAME, EDF_NAME, ECG_NAME
import torch
from torch.utils.data import Dataset
import pandas as pd
import utils as utils
from importlib import reload
from sklearn.model_selection import train_test_split
import torchvision
import patoolib
import pyunpack
reload(_settings)

TRAIN = 0
VALID = 1
TEST = 2
SPLIT_NAMES = {TRAIN: 'train', VALID: 'val', TEST: 'test'}
SPLIT_ENUMS = {v:k for k,v in SPLIT_NAMES.items()}



def download_ISRUC(group=1):
    assert group == 1, "I only checked group 1 data"
    name = ISRUC_NAME
    data_dir = _settings.ISRUC_PATH
    if not os.path.isdir(data_dir): os.makedirs(data_dir)

        #https://sleeptight.isr.uc.pt/?page_id=48
    for patient_id in range(1, 100 + 1): #patient 1 to 100
        if os.path.isfile(os.path.join(data_dir, '%d/%d.edf'%(patient_id, patient_id))):
            continue

        rar_url = "http://dataset.isr.uc.pt/ISRUC_Sleep/subgroupI/%d.rar"%patient_id
        rar_dst = os.path.join(data_dir, "%d.rar"%patient_id)
        if not os.path.isfile(rar_dst): utils.download_file(rar_url, rar_dst)

        #unrar
        unzipped_dst = os.path.join(data_dir, "")
        #patoolib.extract_archive(rar_dst, outdir=unzipped_dst)
        pyunpack.Archive(rar_dst).extractall(unzipped_dst)
        #rename
        os.rename(os.path.join(data_dir, '%d/%d.rec'%(patient_id, patient_id)),
                  os.path.join(data_dir, '%d/%d.edf'%(patient_id, patient_id)))


        #Delete the rar
        os.remove(rar_dst)



class DatasetWrapper(Dataset):
    def __init__(self, mode=TRAIN):
        super(DatasetWrapper, self).__init__()
        self.mode = mode
        assert hasattr(self, 'DATASET'), "Please give this dataset a name"
        assert hasattr(self, 'LABEL_MAP'), "Please give a name to each class {NAME: class_id}"

    def is_train(self):
        return self.mode == TRAIN
    def is_test(self):
        return self.mode == TEST
    def is_valid(self):
        return self.mode == VALID

    #def get_class_frequencies(self):
    #    raise NotImplementedError()


class SLEEPEDFLoader(DatasetWrapper):
    DATASET = EDF_NAME
    DATA_PATH = os.path.join(_settings.EDF_PATH, 'cassette_processed')
    LABEL_MAP = {"W":0, "R":4, "N1":1, "N2":2, "N3":3}
    def __init__(self, mode=TRAIN):
        super(SLEEPEDFLoader, self).__init__(mode)
        self.dir = os.path.join(self.DATA_PATH, {TRAIN:'train', VALID:"val", TEST:'test'}[mode])
        self.list_IDs = os.listdir(self.dir)

        self.bandpass1 = (1, 5)
        self.bandpass2 = (30, 49)
        self.n_length = 100 * 30
        self.n_channels = 2
        self.n_classes = 5
        self.signal_freq = 100
        self.bound = 0.00025

    def __len__(self):
        return len(self.list_IDs)

    def idx2pid(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        return self.list_IDs[idx].split("-")[1]

    def __getitem__(self, index):
        import pickle
        path = os.path.join(self.dir,  self.list_IDs[index])
        sample = pickle.load(open(path, 'rb'))
        X, y = sample['X'], sample['y']

        # X = torch.FloatTensor(X)

        # original y.unique = [0, 1, 2, 3, 5]
        if y == 'W':
            y = 0
        elif y == 'R':
            y = 4
        elif y in ['1', '2', '3']:
            y = int(y)
        elif y == '4':
            y = 3
        else:
            y = 0

        return torch.FloatTensor(X), y, index


class ISRUCLoader(DatasetWrapper):

    DATASET = ISRUC_NAME
    DATA_PATH = _settings.ISRUC_PATH
    CHANNELS = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2']  # https://arxiv.org/pdf/1910.06100.pdf
    LABEL_MAP = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4} #R is called "5" in the oringinal data, but use 4 here
    CLASS_FREQ = {"W": 221, "N1": 126, "N2": 281, "N3": 214, "R":62} #Basing on patient_8_1
    EPOCH_LENGTH = 30 * 200  #30 seconds * 200 Hz

    @classmethod
    def split_data(cls, seed=_settings.RANDOM_SEED, split_ratio=[84, 6, 10],
                   toy_version=False): #train:test = 9:1 in SLEEPER
        if toy_version: return [1,24], [2], [3]
        patients = [int(x) for x in os.listdir(cls.DATA_PATH) if x != '8'] #patient 8's EDF file is missing channels
        n_patients = len(patients)
        assert n_patients == 100 - 1
        train_val, test = train_test_split(patients, test_size = split_ratio[2] / float(sum(split_ratio)),
                                           random_state=seed)
        train, val = train_test_split(train_val, test_size = split_ratio[1] / float(sum(split_ratio[:2])), random_state=seed)
        #print(train[:5])
        return sorted(train), sorted(val), sorted(test)
        #return [1, 24], [2], [3]

    @classmethod
    def clear_cache(cls):
        for pid in range(1, 100):
            caches = glob.glob(os.path.join(cls.DATA_PATH, f"{pid}/{pid}_*.pkl"))
            for cache_file in caches:
                if os.path.isfile(cache_file):
                    print("Removing {}".format(cache_file))
                    os.remove(cache_file)


    @classmethod
    def find_channels(cls, potential_channels):
        #channels = ['F3-A2', 'F4-A1', 'C3-A2', 'C4-A1', 'O1-A2', 'O2-A1'] #https://arxiv.org/pdf/1910.06100.pdf

        keep = {}
        for c in potential_channels:
            new_c = c.replace("-M2", "").replace("-A2", "").replace("-M1", "").replace("-A1", "")#https://www.ers-education.org/lrmedia/2016/pdf/298830.pdf
            if new_c in cls.CHANNELS:
                assert new_c not in keep
                keep[new_c] = c
        assert len(keep) == len(cls.CHANNELS), f"Something's wrong among columns={potential_channels}"
        return {v:k for k,v in keep.items()}

    @classmethod
    def load_data(cls, patient_ids=[1], clear_cache=False, save_mem=True):
        import mne
        import time; st = time.time()

        labels_1, labels_2 = {}, {}
        actual_data, actual_columns = {}, {}
        bad_pids = []
        for pid in patient_ids:
            cache_path = os.path.join(cls.DATA_PATH, f'{pid}/{pid}_Channels={"_".join(cls.CHANNELS)}.pkl')
            if clear_cache and os.path.isfile(cache_path): os.remove(cache_path)
            if not os.path.isfile(cache_path):
                EEG_raw_df = mne.io.read_raw_edf(os.path.join(cls.DATA_PATH, f'{pid}/{pid}.edf')).to_data_frame()
                try:
                    rename_dict = cls.find_channels(EEG_raw_df.columns)
                except Exception as err:
                    print(pid, err)
                    bad_pids.append(pid)
                    continue
                labels_1[pid] = pd.read_csv(os.path.join(cls.DATA_PATH, f"{pid}/{pid}_1.txt"), header=None)[0]
                labels_2[pid] = pd.read_csv(os.path.join(cls.DATA_PATH, f"{pid}/{pid}_2.txt"), header=None)[0]
                actual_data[pid] = EEG_raw_df.rename(columns=rename_dict).reindex(columns=cls.CHANNELS)
                actual_columns[pid] = {v: k for k, v in rename_dict.items()}

                # ipdb.set_trace()
                pd.to_pickle((labels_1[pid], labels_2[pid], actual_data[pid], actual_columns[pid]),
                             cache_path)
            else:
                labels_1[pid], labels_2[pid], actual_data[pid], actual_columns[pid] = pd.read_pickle(cache_path)
                assert len(actual_columns[pid]) == 6
            labels_1[pid][labels_1[pid] == 5] = 4
            labels_2[pid][labels_2[pid] == 5] = 4

            assert len(actual_data[pid]) % cls.EPOCH_LENGTH == 0
            n_epoch = int(len(actual_data[pid]) / cls.EPOCH_LENGTH)
            assert n_epoch == len(labels_1[pid])
            if n_epoch != len(labels_2[pid]):
                print(f"WARNING - Petient {pid}'s Label 2 is weird. Missing {n_epoch - len(labels_2[pid])} / {n_epoch}")

            if save_mem:
                epoch_cache_path = os.path.join(cls.DATA_PATH, f'{pid}/{pid}_epochs_Channels={"_".join(cls.CHANNELS)}')
                if not os.path.isdir(epoch_cache_path): os.makedirs(epoch_cache_path)
                actual_datapaths = []
                for curr_idx in range(n_epoch):
                    curr_idx_cache_path = os.path.join(epoch_cache_path, '%d.npy'%curr_idx)
                    actual_datapaths.append(curr_idx_cache_path)
                    if os.path.isfile(curr_idx_cache_path): continue
                    x = actual_data[pid].iloc[curr_idx * cls.EPOCH_LENGTH:(curr_idx+1) * cls.EPOCH_LENGTH].values.T
                    np.save(curr_idx_cache_path, x)
                actual_data[pid] = actual_datapaths

        print("Took %f seconds"%(time.time() - st))
        return labels_1, labels_2, actual_data, actual_columns

    def __init__(self, mode=TRAIN,
                 seq_len=1, overlap=False,
                 clear_cache=False,
                 to_tensor=True,
                   save_mem=True,
                 toy_version=False,
                 split_ratios=[75, 10,15]):
        super(ISRUCLoader, self).__init__(mode)
        self.save_mem = save_mem
        self.mode = mode
        self._seed = _settings.RANDOM_SEED

        self.seq_len = seq_len
        self.overlap = overlap

        self.patients = sorted(self.split_data(seed=self._seed, toy_version=toy_version, split_ratio=split_ratios)[self.mode])

        self.labels_1, self.labels_2, self.actual_data, self.actual_columns = self.load_data(self.patients, clear_cache, save_mem=save_mem)
        self.patients = sorted([pid for pid in self.labels_1.keys()]) #can have bad patient_ids

        self.npoints_by_patients = pd.Series(0, index=self.patients)

        for pid in self.patients:
            self.npoints_by_patients[pid] = len(self.labels_1[pid]) - self.seq_len + 1

        self.ndata = self.npoints_by_patients.sum()
        self.cumu_npoints_by_patients = self.npoints_by_patients.cumsum()

        self.to_tensor=to_tensor

    def get_class_frequencies(self):
        return pd.concat(self.labels_1.values(), ignore_index=True).value_counts()

    def __len__(self):
        return self.ndata

    def idx2pid(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        if idx >= self.ndata: raise IndexError("%d is out of range (%d elements)" % (idx, self.ndata))
        patient_id = self.cumu_npoints_by_patients.index[bisect.bisect(self.cumu_npoints_by_patients.values, idx)]
        return patient_id

    def get_raw_data(self, idx):
        import mne
        pid = self.idx2pid(idx)
        curr_idx = idx - (self.cumu_npoints_by_patients[pid] - self.npoints_by_patients[pid])
        EEG_raw = mne.io.read_raw_edf(os.path.join(self.DATA_PATH,  f'{pid}/{pid}.edf'), preload=True)
        return EEG_raw, curr_idx, pid, self.actual_columns[pid]

    def get_second_label(self, idx):
        patient_id = self.idx2pid(idx)
        curr_idx = idx - (self.cumu_npoints_by_patients[patient_id] - self.npoints_by_patients[patient_id])
        if self.mode == TEST:
            y = int(self.labels_2[patient_id][curr_idx])
        else:
            y = int(self.labels_2[patient_id][curr_idx])
        return y

    def __getitem__(self, idx):
        patient_id = self.idx2pid(idx)

        curr_idx = idx - (self.cumu_npoints_by_patients[patient_id] - self.npoints_by_patients[patient_id])
        st = curr_idx * self.EPOCH_LENGTH
        if self.save_mem:
            x = np.concatenate([np.load(self.actual_data[patient_id][_ci]) for _ci in range(curr_idx, curr_idx + self.seq_len)],
                               axis=1)
            #ipdb.set_trace()
        else:
            x = self.actual_data[patient_id].iloc[st:(st + self.EPOCH_LENGTH * self.seq_len)].values.T

        if self.mode == TEST:
            #y = None
            y = int(self.labels_1[patient_id][curr_idx])
        else:
            y = int(self.labels_1[patient_id][curr_idx])

        if self.to_tensor and y is not None: y = torch.tensor(y, dtype=torch.long)
        return torch.tensor(x, dtype=torch.float), y, idx

class ECGDataset(DatasetWrapper):
    DATASET = ECG_NAME
    DATA_PATH = os.path.join(_settings.ECG_PATH, "processed_data_full")
    #LABEL_MAP = {"Normal": 0, "AF": 1, "Other": 2, "Noisy": 3}
    LABEL_MAP = {"N": 0, "O": 1, "A": 2, "~": 3} #(5076, 2415, 758, 279)
    TOY_DATA_PATH = os.path.join(_settings.ECG_PATH, "processed_data_toy")

    @classmethod
    def float64_to_float32(cls, data_path=None, toy=True, oversample=False):
        import pickle as dill
        if data_path is None:
            data_path = os.path.join(_settings.ECG_PATH, 'processed_data_%s' % ('toy' if toy else 'full'))
            if oversample: data_path += '_oversampled'
        dst_path = data_path + "_float"
        if not os.path.isdir(dst_path): os.makedirs(dst_path)
        for dataset in ['train', 'val', 'test']:
            fname = 'mina_K_%s_beat.bin'%dataset
            with open(os.path.join(data_path, fname), 'rb') as fin:
                with open(os.path.join(dst_path, fname), 'wb') as fout:
                    np.save(fout, np.load(fin).astype(np.float32))
            fname = 'mina_X_%s.bin'%dataset
            with open(os.path.join(data_path, fname), 'rb') as fin:
                with open(os.path.join(dst_path, fname), 'wb') as fout:
                    np.save(fout, np.load(fin).astype(np.float32))
        fname = 'mina_knowledge.pkl'
        with open(os.path.join(data_path, fname), 'rb') as fin:
            with open(os.path.join(dst_path, fname), 'wb') as fout:
                knowledge = dill.load(fin)
                knowledge = {k:v.astype(np.float32) for k,v in knowledge.items()}
                dill.dump(knowledge, fout)
        import shutil
        for fname in ['challenge2017.pkl', 'mina_info.pkl']:
            shutil.copy2(os.path.join(data_path, fname), os.path.join(dst_path, fname))



    @classmethod
    def pre_process(cls, data_path=None, dst_path = None, toy=False, oversample=False):
        if data_path is None: data_path = os.path.join(_settings.ECG_PATH, 'challenge2017')
        if dst_path is None: dst_path = os.path.join(_settings.ECG_PATH, 'processed_data_%s' % ('toy' if toy else 'full'))
        if oversample: dst_path += "_oversampled"
        if not os.path.isdir(dst_path): os.makedirs(dst_path)
        import data.MINA_preprocess; reload(data.MINA_preprocess)
        data.MINA_preprocess.preprocess_physionet(data_path, 10 if toy else None, dst_path)
        data.MINA_preprocess.make_data_physionet(dst_path, oversample=oversample)
        data.MINA_preprocess.make_knowledge_physionet(dst_path)

    @classmethod
    def load_data_by_dataset(cls, dataset='train', data_path=DATA_PATH):
        assert dataset in {'train', 'val', 'test'}
        from collections import Counter
        import pickle as dill
        with open(os.path.join(data_path, 'mina_info.pkl'), 'rb') as fin:
            res = dill.load(fin)
            Y = res['Y_%s'%dataset]
            pids = res['pid_%s'%dataset]
            N = len(Y)
            assert N == len(pids)
        with open(os.path.join(data_path, 'mina_X_%s.bin'%dataset), 'rb') as fin:
            X = np.swapaxes(np.load(fin), 0, 1)
            assert 4 == X.shape[0] and X.shape[1] == N
        with open(os.path.join(data_path, 'mina_K_%s_beat.bin'%dataset), 'rb') as fin:
            K_beat = np.swapaxes(np.load(fin), 0, 1)
            assert K_beat.shape[0] == 4 and K_beat.shape[1] == N
        with open(os.path.join(data_path, 'mina_knowledge.pkl'), 'rb') as fin:
            res = dill.load(fin)
            K_rhythm = np.swapaxes(res['K_%s_rhythm'%dataset], 0, 1)
            K_freq = np.swapaxes(res['K_%s_freq' % dataset], 0, 1)
            assert K_rhythm.shape[0] == 4 and K_rhythm.shape[1] == N
            assert K_freq.shape[0] == 4 and K_freq.shape[1] == N

        print(Counter(Y))
        print(K_beat.shape, K_rhythm.shape, K_freq.shape)
        return X, Y, pids, K_beat, K_rhythm, K_freq #(nchannels=4, N, d)

    def __init__(self, mode=TRAIN, toy_version=False, float32=True, over_sample=False):
        super(ECGDataset, self).__init__(mode)
        data_path = self.TOY_DATA_PATH if toy_version else self.DATA_PATH
        if over_sample: data_path += '_oversampled'
        if float32: data_path += '_float'
        self.X, self.Y, self.pids, self.K_beat, self.K_rhythm, self.K_freq = self.load_data_by_dataset(SPLIT_NAMES[mode], data_path)

        self.Y = np.asarray([self.LABEL_MAP[y] for y in self.Y])
        self.N = len(self.Y)

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        return (self.X[:, i, :], self.K_beat[:, i, :], self.K_rhythm[:, i, :], self.K_freq[:, i, :]), self.Y[i], i

    def idx2pid(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        return f"{SPLIT_NAMES[self.mode]}{self.pids[idx]}"

    @classmethod
    def _collate_func(cls, batch):
        X = torch.tensor([[_x[0][0][c] for _x in batch] for c in range(4)], dtype=torch.float)
        K_beat = torch.tensor([[_x[0][1][c] for _x in batch] for c in range(4)], dtype=torch.float)
        K_rhythm = torch.tensor([[_x[0][2][c] for _x in batch] for c in range(4)], dtype=torch.float)
        K_freq = torch.tensor([[_x[0][3][c] for _x in batch] for c in range(4)], dtype=torch.float)
        Y = torch.tensor([_x[1] for _x in batch], dtype=torch.long)
        idx = torch.tensor([_x[2] for _x in batch], dtype=torch.long)
        return (X, K_beat, K_rhythm, K_freq), Y, idx

def get_default_dataloader(dataset, split):
    if dataset == _settings.ISRUC_NAME:
        _dataloader_kwargs = {'split_ratios': [75, 10, 15], 'toy_version': False, 'save_mem': True}
        return ISRUCLoader(mode=split, **_dataloader_kwargs)
    if dataset == _settings.EDF_NAME:
        return SLEEPEDFLoader(mode=split)
    if dataset == _settings.ECG_NAME:
        _dataloader_kwargs = {'toy_version': False, 'float32': True}
        return ECGDataset(mode=split, **_dataloader_kwargs)

if __name__ == '__main__':
    #ECGDataset.pre_process(toy=False)
    #ECGDataset.float64_to_float32()
    #o = ECGDataset(toy_version=True, float32=True)
    pass
    #o = ECGDataset(mode=TEST, toy_version=True)