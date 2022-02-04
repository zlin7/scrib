import data.dataloader as dld
import pandas as pd, numpy as np
import os
from importlib import reload
import demos.demo_utils as demo_utils; reload(demo_utils)
import data.sim_data as sim_data; reload(sim_data)
from _settings import DATA_PATH, COVID_NAME, EDF_NAME, ECG_NAME, ISRUC_NAME, METHOD_NAME, METHOD_PLACEHOLDER
import utils
import dl_models.evaluate as evalm
import ipdb

__CUR_FILE_PATH = os.path.dirname(os.path.abspath(__file__))

SYNT_NAME = 'Synthetic'

CLASSES = {COVID_NAME: ['Covid19', 'pneumonia_virus', 'pneumonia_bacteria', 'healthy'],
           EDF_NAME: ['W', 'N1', 'N2', 'N3', 'R'],
           ISRUC_NAME: ['W', 'N1', 'N2', 'N3', 'R'],
           ECG_NAME: ['N', 'O', 'A', '~'],
           }

PERMUTE_MODES = {0: 'simple', 1: 'by_subject'}

def _get_K(dataset):
    if dataset in CLASSES: return len(CLASSES[dataset])
    assert dataset == SYNT_NAME
    return 5 #For Synthetic

#=================================================Dataset specific handling
def _permute_df(df, permute_seed=None, subject_col='subject'):
    if permute_seed is not None:
        np.random.seed(permute_seed)
        if subject_col is not None:
            all_subjects = sorted(df[df['dataset'].isin({'val', 'test'})][subject_col].unique())
            val_subjects = np.random.choice(all_subjects, len(all_subjects)//2, replace=False)
            test_subjects = set(all_subjects).difference(set(val_subjects))
            df.loc[df[subject_col].isin(val_subjects), 'dataset'] = 'val'
            df.loc[df[subject_col].isin(test_subjects), 'dataset'] = 'test'
        else:
            fidx = df[df['dataset'].isin({'test', 'val'})].index
            val_idx = np.random.choice(fidx, len(fidx)//2, replace=False)
            df.loc[val_idx, 'dataset'] = 'val'
            df.loc[fidx.difference(val_idx), 'dataset'] = 'test'
    return df

def _get_synthetic_data(seed=None):
    if seed is None: seed = 123456  # some random number otherwise this won't run
    datakwargs = {'signal': 3, 'method_id':3, 'high_signal':9, 'low_signal': 1, 'noise_level':3, 'nclass': 5, 'N': 10000}
    vprobs, vlabels, _ = sim_data._SimOutputData_cache(seed=seed, **datakwargs)
    tprobs, tlabels, _ = sim_data._SimOutputData_cache(seed=seed + 1, **datakwargs)
    N = datakwargs['N']
    probs, labels = np.concatenate([vprobs, tprobs], 0), np.concatenate([vlabels, tlabels])
    df = pd.DataFrame(np.log(probs), columns=['P_%d' % k for k in range(probs.shape[1])])
    df = df.reset_index().rename(columns={'index': 'idx'})
    df['dataset'] = 'test'
    df.loc[:N - 1, 'dataset'] = 'val'
    assert df['dataset'].value_counts()['val'] == N
    df['label'] = labels
    return df

def get_full_predictions(dataset=COVID_NAME, dropout=False,
                         permute_seed=None, permute_mode=0):
    if dataset not in {EDF_NAME, ISRUC_NAME}: dropout = False
    if dataset == SYNT_NAME:
        df = _get_synthetic_data(permute_seed)
        columns = ["P_%d" % k for k in range(5)]
        df.loc[:, columns] = np.exp(df.loc[:, columns])
        return df
    if dataset == COVID_NAME:
        fpath = os.path.join(os.path.join(__CUR_FILE_PATH, 'predictions', f"{dataset}{'_Dropout' if dropout else ''}.pkl"))
        assert os.path.isfile(fpath), "Please download or save the predictions of %s to %s"%(dataset,fpath)
        df = pd.read_pickle(fpath)
    else:
        df = evalm._eval_model_and_cache_df(dataset, dropout)
    df = _permute_df(df, permute_seed, subject_col='subject' if permute_mode == 1 else None)
    return df

def extract_data(df, K=None):
    if K is None:
        K = 0
        while "P_%d"%K in df.columns: K += 1
    cols = ['P_%d'%k for k in range(K)]
    return tuple([np.asarray(df[df['dataset'] == _d].loc[:, cols].values) for _d in ['train', 'val', 'test']])

def extract_label(df, K = None):
    if K is None:
        K = 0
        while "P_%d"%K in df.columns: K += 1
    return tuple([utils.to_onehot(np.asarray(df[df['dataset'] == _d]['label'].values), K) for _d in ['train', 'val', 'test']])

def extract_variance(df):
    if 'Var' not in df.columns: return None, None, None
    return tuple([np.asarray(df[df['dataset'] == _d]['Var'].values) for _d in ['train', 'val', 'test']])


#=======================================================Overall risk experiment
def _overall_exp_cache_by_method_r(dataset, permute_seed=None, permute_mode=0,
                                    r=0.01,
                                    method=METHOD_PLACEHOLDER, cache=1,
                                    **kwargs):
    df = get_full_predictions(dataset, True, permute_seed, permute_mode)
    K = _get_K(dataset)
    _, val_out, test_out = extract_data(df, K)
    _, val_lab, test_lab = extract_label(df)
    _, val_var, test_var = extract_variance(df) #For MC Dropout cases
    obj = demo_utils.OverallRiskDemo(val_out, val_lab, test_out, test_lab,
                                           val_var=val_var, test_var=test_var)
    return obj.run(method, r, **kwargs)

def run_overall_exp(dataset=COVID_NAME, permute_seed=None, permute_mode=0,
                    loss_func='overall la=0.01 lcs=10', rs=np.linspace(0.03, 0.1, 5),
                    all_methods = None,
                    cache=1,
                    **kwargs):
    if permute_seed is None: permute_mode = 0
    if all_methods is None:
        all_methods = ['SGR', METHOD_PLACEHOLDER]
        if dataset in {ISRUC_NAME, EDF_NAME}:
            all_methods += ["SGR_MCDropout"]

    args = (dataset, permute_seed, permute_mode)
    methods_kwargs = {'SGR': {},
                      "SGR_MCDropout": {},
                      METHOD_PLACEHOLDER: utils.merge_dict_inline(kwargs, {'loss_func': loss_func}),
                      "Global": utils.merge_dict_inline(kwargs, {'loss_func': loss_func}),
                      }
    from collections import defaultdict
    res, newres = defaultdict(dict), defaultdict(dict)
    for method, params in methods_kwargs.items():
        if method not in all_methods: continue
        params['method'] = method
        for r in rs:
            res[method][r] = _overall_exp_cache_by_method_r(*args, r=r, cache=cache, **params)
    return res


#=======================================================Class-specific risks
def _classSpecifc_singlemethod_cache(dataset, permute_seed=None, permute_mode=0,
                                     method=METHOD_PLACEHOLDER, cache=1,
                                     **kwargs):
    df = get_full_predictions(dataset, False, permute_seed, permute_mode)
    K = _get_K(dataset)
    _, val_out, test_out = extract_data(df, K)
    _, val_lab, test_lab = extract_label(df)
    _, val_var, test_var = extract_variance(df) #For MC Dropout cases
    obj = demo_utils.ClassSpecificRiskDemo(val_out, val_lab, test_out, test_lab,
                                           val_var=val_var, test_var=test_var)
    return obj.run(method, **kwargs)

def run_classSpecific_exp(dataset, permute_seed=None, permute_mode=0,
                          loss_func='classSpec la=0 lc=1 lcs=0.1', r=0.15, rks=None,
                          all_methods = ['SGR', METHOD_PLACEHOLDER, 'LABEL', 'Global'],
                          cache=1, **kwargs
                          ):
    args = (dataset, permute_seed, permute_mode)
    res, newres = {}, {}
    if rks is None: rks = tuple([r for _ in range(_get_K(dataset))])
    methods_kwargs = {'SGR': {'r': r},
               METHOD_PLACEHOLDER: utils.merge_dict_inline(kwargs, {'loss_func': loss_func, 'rks': rks, }),
               'LABEL': {'rks': rks},
               'Global': utils.merge_dict_inline(kwargs, {'loss_func': loss_func, 'rks': rks, }),
               }
    for method, params in methods_kwargs.items():
        if method not in all_methods: continue
        res[method] = _classSpecifc_singlemethod_cache(*args, method=method, cache=cache, **params)
    return res



#======================================================Meta routines
import matplotlib.pyplot as plt
def run_all_seeds_overall(run_kwargs, save_plot_path=None, plot_title=None, Nseed=20, plot_seed=1, test_only=True):
    if plot_seed != -1: plot_seed = min(plot_seed, Nseed-1)
    if plot_title is None: plot_title = run_kwargs['dataset']
    results = []
    for permute_seed in [i for i in range(Nseed)]:
        ores = run_overall_exp(permute_seed=permute_seed, **run_kwargs)
        if run_kwargs['dataset'] == ECG_NAME: ores = {k: v for k,v in ores.items() if 'Dropout' not in k}
        ores = {k: v for k, v in ores.items() if 'Entropy' not in k}
        res = demo_utils.organize_run_curve_results(ores, True)
        results.append(res)
        if permute_seed == plot_seed:
            demo_utils.plot_run_curve_results(res, title=plot_title, test_only=test_only)
            if save_plot_path is not None:
                plt.savefig(save_plot_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()
    results = demo_utils.summ_run_curve_results_bootstrapped(results)
    print(results[0]['test'])
    return results

def run_all_seeds_classSpec(run_kwargs, save_plot_path=None, plot_title=None, Nseed=20, plot_seed=-1):
    if plot_seed != -1: plot_seed = min(plot_seed, Nseed-1)
    K = _get_K(run_kwargs['dataset'])
    if 'rks' not in run_kwargs:
        same_risk = True
        run_kwargs['rks'] = tuple([run_kwargs['r'] for _ in range(K)])
        ideals = pd.Series(run_kwargs['r'], index=list(range(K)) + ['Overall'])
    else:
        same_risk=False
        ideals = pd.Series(list(run_kwargs['rks']) + [run_kwargs['r']], index=list(range(K))+['Overall'])
    if plot_title is None: plot_title = run_kwargs['dataset']
    results = []
    for permute_seed in range(Nseed):
        ores = run_classSpecific_exp(permute_seed=permute_seed, **run_kwargs)
        results.append(ores)
        res = demo_utils.organize_run_classSpecific_results(ores)
        if permute_seed == plot_seed:
            demo_utils.plot_run_classSpecific_results(res, ideals=ideals)
            if plot_title is not None: plt.title(plot_title)
            plt.show()
    flat_df = demo_utils.flatten_classSpecific_results(results)
    rets = {}
    for split in ['val', 'test']:
        ret = demo_utils.summ_run_classSpecific_results_bootstrapped(flat_df, bmk=ideals, keep_datasets=[split])[0]
        tdf = flat_df[(flat_df['class'] == 'Overall') & (flat_df['dataset'] == split)]
        for _m in ['% unsure', 'E[H]']:
            _mdf = tdf[tdf['measure'] == _m].groupby('method')['value'].mean()
            _sdf = tdf[tdf['measure'] == _m].groupby('method')['value'].std()
            ret[_m] = utils.merge_mean_std_tables(_mdf, _sdf)
        rets[split] = ret
    flat_df.loc[flat_df['method'] == METHOD_PLACEHOLDER, "method"] = METHOD_NAME
    flat_df.loc[flat_df['method'] == 'Global', "method"] = METHOD_NAME + "-"

    #=============Plotting
    demo_utils.plot_run_classSpec_box(flat_df, run_kwargs['r'])
    if plot_title is not None: plt.title(plot_title)
    if save_plot_path is not None and same_risk:
        plt.savefig(save_plot_path, dpi=600, bbox_inches='tight')
    else:
        plt.show()
    return flat_df, rets['test'], rets['val']
