import torch
import numpy as np
import torch.nn as nn
import os
from collections import Counter
import argparse
from tqdm import tqdm
from _settings import EDF_PATH
from _settings import ECG_NAME, EDF_NAME, ISRUC_NAME
import dl_models.models as models
import datetime
import data.dataloader as dld
from torch.utils.data import DataLoader
import utils
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
#import data.dataloader as dld
import ipdb

__CURR_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
_TRAINED_MODEL_DIR = os.path.join(__CURR_DIR_PATH, 'trained_models')
if not os.path.isdir(_TRAINED_MODEL_DIR): os.makedirs(_TRAINED_MODEL_DIR)

MODEL_DICT = {(ECG_NAME, False): (models.FreqNet_Dropout, {'n_class': 4}),
              (EDF_NAME, False): (models.CNNSleep, {'n_dim': 128, 'MCDropout': False}),
              (EDF_NAME, True): (models.CNNSleep, {'n_dim': 128, 'MCDropout': True}),
              (ISRUC_NAME, False): (models.CNNSleep, {'n_dim': 128, 'MCDropout': False, 'base_channels':6, 'avg_pool': 2}),
              (ISRUC_NAME, True): (models.CNNSleep, {'n_dim': 128, 'MCDropout': True, 'base_channels':6, 'avg_pool': 2})}

TRAIN_PARAMS = {(ECG_NAME, False): {'data': (dld.ECGDataset, {'mode': dld.TRAIN, "over_sample": True}),
                                    'criterion': (nn.CrossEntropyLoss, {}),
                                    'optimizer': (torch.optim.Adam, {'lr': 3e-3}),
                                    'batch': 128, 'epochs': 100,
                                    },
                (EDF_NAME, False): {'data': (dld.SLEEPEDFLoader, {'mode': dld.TRAIN}),
                                    'criterion': (nn.CrossEntropyLoss, {}),
                                    'optimizer': (torch.optim.Adam, {'lr': 2e-4}),
                                    'batch': 128, 'epochs': 40,
                                    },
                (EDF_NAME, True): {'data': (dld.SLEEPEDFLoader, {'mode': dld.TRAIN}),
                                    'criterion': (nn.CrossEntropyLoss, {}),
                                    'optimizer': (torch.optim.Adam, {'lr': 2e-4}),
                                    'batch': 128, 'epochs': 100,
                                    },
                (ISRUC_NAME, False): {'data': (dld.ISRUCLoader, {'mode': dld.TRAIN, 'save_mem': False}),
                                    'criterion': (nn.CrossEntropyLoss, {}),
                                    'optimizer': (torch.optim.Adam, {'lr': 4e-4}),
                                    'batch': 128, 'epochs': 100,
                                    },
                }
TRAIN_PARAMS[(ISRUC_NAME, True)] = TRAIN_PARAMS[(ISRUC_NAME, False)]

def data_to_device(data, device):
    if isinstance(data, tuple) or isinstance(data, list):
        return tuple([x.to(device) for x in data])
    return data.to(device)

def train(model, train_loader, criterion, optimizer, device,
          epochs=100,
          writer=None):
    model.train()
    model = model.to(device)
    step = 0
    for curr_epoch in range(epochs):
        curr_epoch += 1
        curr_preds, curr_labels = [], []
        for data, target, _ in tqdm(train_loader, desc='Epoch=%d'%curr_epoch, ncols=60):
            optimizer.zero_grad()
            output = model(data_to_device(data, device))
            loss = criterion(output, target.to(device))
            loss.backward()
            optimizer.step()

            step += 1
            curr_preds.extend(np.argmax(output.tolist(), axis=1))
            curr_labels.extend(target.tolist())
            writer.add_scalar('Loss/Train', loss.item(), step)
        acc = sum(np.asarray(curr_preds) == np.asarray(curr_labels)) / float(len(curr_labels))
        writer.add_scalar('Train/Acc', acc, step)

    results = {'state_dict': model.state_dict(),
               'step': step}
    return model, results

def train_main(dataset, dropout=False):
    utils.set_all_seeds()
    if dataset == ECG_NAME: assert not dropout
    train_id = dataset + ("_Dropout" if dropout else "")
    #train_id += "-%s"%(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))#TODO: remove this line
    save_path = os.path.join(_TRAINED_MODEL_DIR, train_id)
    if os.path.isdir(save_path): return
    os.makedirs(save_path)

    device = torch.device('cuda:{}'.format(0)) if torch.cuda.is_available() else torch.device('cpu')
    params = TRAIN_PARAMS[(dataset, dropout)]
    train_data = params['data'][0](**params['data'][1])
    collate_fn = train_data._collate_func if hasattr(train_data, '_collate_func') else None
    train_loader = DataLoader(dataset=train_data, batch_size=params['batch'], shuffle=True, num_workers=0, collate_fn=collate_fn)
    model = MODEL_DICT[(dataset, dropout)][0](**MODEL_DICT[(dataset, dropout)][1])
    criterion = params['criterion'][0](**params['criterion'][1])
    optimizer = params['optimizer'][0](model.parameters(), **params['optimizer'][1])
    #Tensorboard
    writer = SummaryWriter(log_dir=save_path)
    model, results = train(model, train_loader, criterion, optimizer, device, epochs=params['epochs'], writer=writer)

    torch.save(results, os.path.join(save_path, f'checkpoint_{results["step"]}.pth'))
    print(train_id)




if __name__ == '__main__':
    train_main(EDF_NAME, True)
    train_main(EDF_NAME, False)
    train_main(ISRUC_NAME, True)
    train_main(ISRUC_NAME, False)
    train_main(ECG_NAME)