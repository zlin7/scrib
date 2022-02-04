import data.dataloader as dld

if __name__ == '__main__':
    #=================================ECG data
    #Download the data form https://physionet.org/content/challenge-2017/1.0.0/
    #Unzip and put it into _settings.ECG_PATH/challenge2017

    #Use the oversampled for training
    #dld.ECGDataset.pre_process(toy=False, oversample=True)
    #dld.ECGDataset.float64_to_float32(toy=False, oversample=True)

    #Use the non-oversampled for SCRIB
    #dld.ECGDataset.pre_process(toy=False, oversample=False)
    #dld.ECGDataset.float64_to_float32(toy=False, oversample=False)

    # Read the datasets:
    for _split in dld.SPLIT_ENUMS.values():
        dld.get_default_dataloader(dld.ECG_NAME, _split)

    #=================================ISRUC
    #First, download the data if necessary
    #Please change the ISRUC_PATH variable in _settings if you have already downloaded the data
    #dld.download_ISRUC()

    #Now, the data will be processed and cached the first time we initialize the Dataset.
    #for _split in dld.SPLIT_ENUMS.values():
    #    _ = dld.ISRUCLoader(_split, save_mem=True)

    #Read the datasets:
    for _split in dld.SPLIT_ENUMS.values():
        dld.get_default_dataloader(dld.ISRUC_NAME, _split)

    # =================================EDF
    #Download from https://physionet.org/content/sleep-edfx/1.0.0/
    # put it in EDF_PATH and unzip. The zipped version is about 8.1 GB
    #run sleepEDF_preprocess.py

    #Read the datasets:
    for _split in dld.SPLIT_ENUMS.values():
        dataset = dld.get_default_dataloader(dld.EDF_NAME, _split)

    pass
