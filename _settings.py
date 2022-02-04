import os
import sys
DATA_PATH = '/mnt/disks/data/Data'

__CUR_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.abspath(__CUR_FILE_PATH)))

#==============================Data Related

ISRUC_NAME = 'ISRUC_SLEEP1'
EDF_NAME = 'SLEEPEDF'
ECG_NAME = 'ECG_PhysioNet2017'
COVID_NAME = 'Xray'


ISRUC_PATH = os.path.join(DATA_PATH, ISRUC_NAME)
EDF_PATH = os.path.join(DATA_PATH, EDF_NAME)
ECG_PATH = os.path.join(DATA_PATH, ECG_NAME)
COVID_PATH = os.path.join(DATA_PATH, COVID_NAME)
WORKSPACE = os.path.join(__CUR_FILE_PATH, "Temp")




LOG_OUTPUT_DIR = os.path.join(WORKSPACE, 'logs')
RANDOM_SEED = 7

NCOLS = 80

import torch
import numpy as np
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)


#====================Paper Related
METHOD_NAME = "SCRIB"
METHOD_PLACEHOLDER = 'Method'