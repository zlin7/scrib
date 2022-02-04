import numpy as np, pandas as pd
from importlib import reload
from demos.main import COVID_NAME, ISRUC_NAME, EDF_NAME, ECG_NAME, METHOD_PLACEHOLDER, METHOD_NAME
import demos.main; reload(demos.main)
import demos.demo_utils as demo_utils; reload(demo_utils)
import utils
import tqdm, ipdb, matplotlib.pyplot as plt


def _ambiguity_correlation(dataset, numHs=1000, permute_seed=0, **kwargs):
    K = demos.main._get_K(dataset)
    df = demos.main.get_full_predictions(dataset, False, permute_seed=permute_seed, permute_mode=0)
    _, val_out, test_out = demos.main.extract_data(df, K)
    N = len(val_out)
    np.random.seed(permute_seed+7)
    A_chance = []
    A_size = []
    for _ in range(numHs):
        ts = np.random.random_integers(0, N-1, K)
        s = sum([np.asarray(test_out[:, k] > val_out[r, k]) for k, r in enumerate(ts)])
        A_size.append(s.mean())
        A_chance.append(np.mean(s > 1))
    return pd.Series(A_chance), pd.Series(A_size)

def ambiguity_correlation(dataset, Nseeds=20, numHs=1000, **kwargs):
    seeds = list(range(Nseeds))
    corrs = pd.DataFrame(index=seeds)
    for seed in tqdm.tqdm(seeds, ncols=80):
        A_chance, A_size = _ambiguity_correlation(dataset, numHs, permute_seed=seed, **kwargs)
        corrs.loc[seed, 'RankCorr'] = A_chance.corr(A_size, method='spearman')
        corrs.loc[seed, 'Corr'] = A_chance.corr(A_size, method='pearson')
    return corrs


def demo_ambiguity_corr(datasets=[ISRUC_NAME]):
    dfs = []
    for dataset in datasets:
        corrs = ambiguity_correlation(dataset).describe().reindex(['mean', 'std'])
        corrs = corrs.unstack().reset_index().rename(columns={'level_0': 'corr', 'level_1': 'measure', 0: 'value'})
        corrs['dataset'] = dataset
        dfs.append(corrs)
    df = pd.concat(dfs)
    mdf = df[df['measure'] == 'mean'].pivot_table(index='dataset', columns='corr', values='value')
    sdf = df[df['measure'] == 'std'].pivot_table(index='dataset', columns='corr', values='value')
    df = utils.merge_mean_std_tables(mdf,sdf)
    return df, mdf, sdf



if __name__ == '__main__':
    #Correlation.
    import pprint
    df = demo_ambiguity_corr()
    pprint.pprint(df[0])
