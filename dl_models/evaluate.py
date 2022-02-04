from dl_models.train import *
from dl_models.train import __CURR_DIR_PATH, _TRAINED_MODEL_DIR


def load_model_weights(dataset, dropout=False):
    if dataset == ECG_NAME: dropout = False

    model_class, model_kwargs = MODEL_DICT[(dataset, dropout)]
    model = model_class(**model_kwargs)
    dir_path = os.path.join(_TRAINED_MODEL_DIR, dataset)
    if dropout: dir_path += '_Dropout'
    fname = [f_ for f_ in os.listdir(dir_path) if f_.startswith("checkpoint_")][0]
    state_dict = torch.load(os.path.join(dir_path, fname))['state_dict']
    if dataset != ECG_NAME: state_dict = {k: v for k,v in state_dict.items() if k.split(".")[0] not in {'fc', 'byol_mapping'}} #Get rid of unused parameters
    print(model.load_state_dict(state_dict))
    return model

def test_load_weights():
    mod = load_model_weights(EDF_NAME, True)
    mod = load_model_weights(EDF_NAME, False)
    mod = load_model_weights(ISRUC_NAME, True)
    mod = load_model_weights(ISRUC_NAME, False)
    mod = load_model_weights(ECG_NAME, False)

def eval_model(model, dataset, device=None, forward_kwargs={}, criterion=None):
    if device is None:
        device = torch.device('cuda:{}'.format(0)) if torch.cuda.is_available() else torch.device('cpu')
    model.eval()
    model = model.to(device)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=128, collate_fn=getattr(dataset, '_collate_func', None))

    all_val_pred = []
    all_val_gt = []
    all_val_loss = 0.0
    all_val_indices = []
    outputs = []
    with torch.no_grad():
        for data, target, indices in tqdm(dataloader, ncols=60, desc='Evaluating...'):
            data = data_to_device(data, device)
            target = target.to(device)
            output = model(data, **forward_kwargs)
            outputs.extend(output.tolist())
            all_val_pred.extend(np.argmax(output.tolist(), axis=1))
            all_val_gt.extend(target.tolist())
            all_val_indices.extend(indices.tolist())
            if criterion is not None: all_val_loss += criterion(output, target).cpu().numpy() * len(indices)
    return all_val_pred, all_val_gt, all_val_loss / len(all_val_gt), outputs, all_val_indices


def _eval_model_and_cache_df(dataset=ECG_NAME, dropout=False):
    if dataset == ECG_NAME: assert not dropout

    cache_dir = os.path.join(__CURR_DIR_PATH, 'cache')
    if not os.path.isdir(cache_dir): os.makedirs(cache_dir)
    cache_path = os.path.join(cache_dir, f'{dataset}{"_Dropout" if dropout else ""}.pkl')

    if not os.path.isfile(cache_path):
        model = load_model_weights(dataset, dropout)
        data = {_s: dld.get_default_dataloader(dataset, _s) for _s in dld.SPLIT_ENUMS.values()}
        K = len(data[dld.TRAIN].LABEL_MAP)
        def _eval_one_seed(seed, **kwargs):
            bdf = []
            utils.set_all_seeds(seed)
            for split, dataset_obj in data.items():
                tres = eval_model(model, dataset_obj, **kwargs)
                df = pd.DataFrame(utils.log_softmax_if_necessary(tres[3])).rename(
                    columns=lambda x: "P_%d" % x)  # outputs
                df['label'] = tres[1]
                df['idx'] = tres[4]
                df['dataset'] = dld.SPLIT_NAMES[split]
                df['subject'] = df['idx'].map(lambda idx: dataset_obj.idx2pid(idx))
                df['idx'] = df.apply(lambda r: f"{r['dataset']}{r['idx']}", axis=1)
                bdf.append(df)
            return pd.concat(bdf, ignore_index=True)
        df = _eval_one_seed(0)
        if dropout:
            sdfs = []
            for seed in range(20):
                print("Sampling with seed %d"%seed)
                sdf = _eval_one_seed(seed, forward_kwargs={'to_sample': True})
                sdf = utils.log_softmax_if_necessary(sdf.reindex(columns=["P_%d"%k for k in range(K)]))
                sdf['seed'] = seed
                sdfs.append(sdf.reset_index().rename(columns={'index': 'original_index'}))
            sdf = pd.concat(sdfs, ignore_index=True).set_index(['original_index', 'seed'])
            pd.to_pickle((df, sdf), cache_path)
        else:
            pd.to_pickle((df, None), cache_path)
    df, sdf = pd.read_pickle(cache_path)
    K = len([c for c in df.columns if c.startswith("P_")])
    columns = ["P_%d" % k for k in range(K)]
    df.loc[:, columns] = np.exp(df.loc[:, columns])
    if sdf is not None:
        pred = np.argmax(df.reindex(columns=['P_%d' % k for k in range(K)]).values, 1)
        pred_idx = np.expand_dims(pred, 1)
        pred_vars = np.exp(sdf.sort_index(axis=1)).var(level=0).sort_index()
        df['Var'] = np.take_along_axis(pred_vars.values, pred_idx, axis=1)
    return df

if __name__ == "__main__":
    test_load_weights()

    _eval_model_and_cache_df(EDF_NAME, True)
    _eval_model_and_cache_df(EDF_NAME, False)
    _eval_model_and_cache_df(ISRUC_NAME, True)
    _eval_model_and_cache_df(ISRUC_NAME, False)
    _eval_model_and_cache_df(ECG_NAME)