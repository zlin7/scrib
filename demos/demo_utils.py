import scipy.stats
import matplotlib.pyplot as plt
import opt.SGR as risk_control
import opt.CoordDescent as cdt
import ipdb
import numpy as np
import utils
import opt.evaluation as ceval
import pandas as pd
from importlib import reload
from collections import defaultdict
reload(utils)
reload(ceval)

from _settings import METHOD_PLACEHOLDER, METHOD_NAME
_replace_method_name = lambda x: METHOD_NAME if x == METHOD_PLACEHOLDER else x

_SCORE_FUNCS_DICT={'max_prob': lambda x, y: np.max(x),
                   'softmax_response': lambda x, y: np.dot(x, y),
                   }

def SGR(out, labels, rstar, delta=0.8, score_func='max_prob'):
    assert len(labels.shape) == 2
    score_func = _SCORE_FUNCS_DICT[score_func]
    bound_cal = risk_control.risk_control()
    residuals = (np.argmax(out,1) != np.argmax(labels, 1))
    [theta, b_star] = bound_cal.bound(rstar, delta, kappa = np.max(out, 1), residuals=residuals, split=False)
    return theta, ceval.EvalRisks._make_single_pred_fs(theta, score_func)

def SGR_Variance(out, variance, labels, rstar, delta=0.8):
    assert len(labels.shape) == 2 and len(labels) == len(variance) == len(out)
    bound_cal = risk_control.risk_control()
    residuals = (np.argmax(out, 1) != np.argmax(labels, 1))
    [theta, b_star] = bound_cal.bound(rstar, delta, kappa=-variance, residuals=residuals, split=False)
    #ipdb.set_trace()
    def fs(x):
        K = len(x) - 1
        if -x[-1] > theta:
            pred = np.zeros(K)
            pred[np.argmax(x[:-1])] = 1
        else:
            pred = np.ones(K)
        return pred
    return theta, fs

def Method(out, labels, risks, loss_func='overall', B=10, batch=None, score_func='softmax_response', **kwargs):
    assert len(labels.shape) == 2
    score_func = _SCORE_FUNCS_DICT[score_func]
    kwargs.setdefault('max_step', None)
    best_ts, best_loss = cdt.CoordDescentCython.run(out, labels, risks, B=B, batch=batch, loss_func=loss_func,
                                                    **kwargs)
    fill_max = kwargs.get('fill_max', False)
    return best_ts, ceval.EvalRisks._make_set_pred_fs(best_ts, score_func, fill_empty='max' if fill_max else 'all')

def Method_global(out, labels, risks, loss_func='overall', seed=10, batch=None, score_func='softmax_response', **kwargs):
    assert len(labels.shape) == 2
    score_func = _SCORE_FUNCS_DICT[score_func]
    best_t, best_loss = cdt.CoordDescentGlobal.run(out, labels, risks, seed=seed, batch=batch, loss_func=loss_func,
                                                    **kwargs)
    best_ts = [best_t for _ in range(out.shape[1])]
    fill_max = kwargs.get('fill_max', False)
    return best_ts, ceval.EvalRisks._make_set_pred_fs(best_ts, score_func, fill_empty='max' if fill_max else 'all')

def Method_global_cov(out, labels, cov, seed=10, batch=None, score_func='softmax_response', **kwargs):
    assert len(labels.shape) == 2
    score_func = _SCORE_FUNCS_DICT[score_func]
    best_t, best_loss = cdt.CoordDescentGlobal.run_cov(out, labels, cov, seed=seed, batch=batch, **kwargs)
    best_ts = [best_t for _ in range(out.shape[1])]
    assert not kwargs.get('fill_max', False), "This method doesn't work with fill_max"
    return best_ts, ceval.EvalRisks._make_set_pred_fs(best_ts, score_func, fill_empty='all')

def LABEL(out, labels, alphas, score_func='softmax_response'):
    assert len(labels.shape) == 2
    score_func = _SCORE_FUNCS_DICT[score_func]
    K = out.shape[1]
    ts = []
    for k in range(K):
        truth = utils.one_hot_single(k, K)
        dists = np.asarray([score_func(out_i, truth) for ni,out_i in enumerate(out) if labels[ni,k] == 1])
        ts.append(np.percentile(dists, 100 * alphas[k]))
    return ts, ceval.EvalRisks._make_set_pred_fs(ts, score_func, fill_empty='max')

#Returned columns include
#      ['FN', 'FN&Sure', '#', '# Unsure', 'E[H]', '% unsure', '% Miss', 'risk', 'risk/Cov', 'E[H]|Unsure']
#Index includes classes and 'Overall'
def evaluate_df(Hfunc, out, labels):
    #out is NxK
    if len(labels.shape) == 1: labels = utils.to_onehot(labels, out.shape[1])
    return ceval.EvalRisks._eval_knary_thresholds(out, labels, fs=Hfunc, quiet=True)

##Overall Risk =====================================================================================

class OverallRiskDemo:
    def __init__(self, val_outs, val_labels, test_outs, test_labels,
                 val_var=None, test_var=None,
                 val_ent=None, test_ent=None):
        self.val_outs = val_outs
        self.val_labels = val_labels
        self.test_outs = test_outs
        self.test_labels = test_labels
        self.val_var, self.test_var = val_var, test_var
        if val_var is not None and len(val_var.shape) == 1: self.val_var = np.expand_dims(val_var, 1)
        if test_var is not None and len(test_var.shape) == 1: self.test_var = np.expand_dims(test_var, 1)

    def SGR(self, r, **kwargs):
        theta, Hfunc = SGR(self.val_outs, self.val_labels, r)
        val_df = evaluate_df(Hfunc, self.val_outs, self.val_labels)
        test_df = evaluate_df(Hfunc, self.test_outs, self.test_labels)
        return theta, val_df, test_df

    def SGR_Variance(self, r, **kwargs):
        assert self.val_var is not None and self.test_var is not None
        theta, Hfunc = SGR_Variance(self.val_outs, self.val_var[:, 0], self.val_labels, r)
        val_df = evaluate_df(Hfunc, np.concatenate([self.val_outs, self.val_var], 1), self.val_labels)
        test_df = evaluate_df(Hfunc, np.concatenate([self.test_outs, self.test_var], 1), self.test_labels)
        return theta, val_df, test_df

    def Method(self, r, loss_func ='overall', **kwargs):
        ts, Hfunc = Method(self.val_outs, self.val_labels, r, loss_func=loss_func, **kwargs)
        val_df = evaluate_df(Hfunc, self.val_outs, self.val_labels)
        test_df = evaluate_df(Hfunc, self.test_outs, self.test_labels)
        return ts, val_df, test_df

    def run(self, method, r, loss_func='overall', **kwargs):
        if method == 'SGR': return self.SGR(r)
        if method == 'SGR_MCDropout': return self.SGR_Variance(r)
        if method == METHOD_PLACEHOLDER: return self.Method(r, loss_func=loss_func, **kwargs)



def organize_run_curve_results(res, replace_riskone=True):
    new_res = {}
    for m, v in res.items():
        new_m = _replace_method_name(m)
        new_res[new_m] = {}
        for r, tres in v.items():
            new_res[new_m][r] = pd.Series({'r': tres[2].loc['Overall', 'risk/Cov'], 'A': tres[2].loc['Overall', '% unsure'],
                                       'val_r': tres[1].loc['Overall', 'risk/Cov'], 'val_A': tres[1].loc['Overall', '% unsure'],
                                       })
            if replace_riskone:
                if new_res[new_m][r]['r'] > 1.-1e-5 and new_res[new_m][r]['A'] >1.-1e-5: new_res[new_m][r]['r'] = 0.
                if new_res[new_m][r]['val_r'] >1.-1e-5 and new_res[new_m][r]['val_A'] >1.-1e-5: new_res[new_m][r]['r'] = 0.

    return {k:pd.DataFrame(v).T for k,v in new_res.items()}

def plot_run_curve_results(res, test_only=True, title='', add_val_legend=False):
    cplt = plt
    colors = {'SGR': 'black', 'SGR_MCDropout': 'blue', METHOD_NAME: "red"}

    #colors=['blue', 'red', 'black']
    for method, v in res.items():
        if method not in colors: continue
        cplt.plot(v['A'], 1-v['r'], label='%s%s' % (method, "" if test_only else ' test'), color=colors.get(method, 'gray'))
        if not test_only:
            #cplt.plot(v['val_A'], 1.-v['val_r'], label='%s valid' % method, color=color, linestyle='dashed')
            kwargs = {'label': '%s val' % (method)} if add_val_legend else {}
            cplt.plot(v['val_A'], 1 - v['val_r'], color=colors.get(method, 'gray'), linestyle='dashed', **kwargs)

    cplt.ylabel('Accuracy')
    cplt.xlabel('Chance-Ambiguity')
    cplt.title(title)
    cplt.legend()


def summ_run_curve_results_bootstrapped(results, **kwargs):
    #res[i] is a bootstrapped sample result generated by organize_run_curve_results
    AUCs = {'val':[], 'test':[]}
    R_diffs = {}
    for i, res_i in enumerate(results):
        for key in ['test', 'val']:
            pref = '' if key == 'test' else 'val_'
            AUC_input_i = {k:pd.Series(1-v[pref + 'r'].values, v[pref + 'A'].values) for k, v in res_i.items()}
            AUCs[key].append(ceval._compute_AUC_helper(AUC_input_i))
        for k,v in res_i.items():
            if k not in R_diffs: R_diffs[k] = {}
            R_diffs[k][i] = v['r'] - v.index
    R_diffs = {k: pd.DataFrame(v) for k,v in R_diffs.items()}
    mdf = pd.DataFrame(index=R_diffs.keys(), columns=['Risk(%)'])
    sdf = mdf.copy()
    for method, tdf in R_diffs.items():
        mses = np.sqrt(np.power(tdf, 2).mean(0))
        mdf.loc[method, 'Risk(%)'] = mses.mean()
        sdf.loc[method, 'Risk(%)'] = mses.std()
    Risk_Summaries = utils.merge_mean_std_tables(mdf * 100, sdf * 100)
    #AUCs = {k: pd.DataFrame(v) for k,v in AUCs.items()}
    newAUCs = {}
    for k, v in AUCs.items():
        v = pd.DataFrame(v)
        tdf = v.drop(['x_min', 'x_max'], axis=1)
        tdf = tdf.mean().sort_values(ascending=True)
        print(k, tdf.index[-1], '-', tdf.index[-2])
        v['max - 2nd'] = v[tdf.index[-1]] - v[tdf.index[-2]]
        newAUCs[k] = v
    return {k: v.describe() for k,v in newAUCs.items()}, pd.DataFrame(Risk_Summaries), R_diffs, newAUCs


## Class Specific Risk Comparison ================================================================
class ClassSpecificRiskDemo:
    def __init__(self, val_outs, val_labels, test_outs, test_labels,
                 val_var=None, test_var=None):
        self.val_outs = val_outs
        self.val_labels = val_labels
        self.test_outs = test_outs
        self.test_labels = test_labels
        self.val_var, self.test_var = val_var, test_var
        if val_var is not None and len(val_var.shape) == 1: self.val_var = np.expand_dims(val_var, 1)
        if test_var is not None and len(test_var.shape) == 1: self.test_var = np.expand_dims(test_var, 1)

    def SGR(self, r, **kwargs):
        return SGR(self.val_outs, self.val_labels, r)

    def SGR_Variance(self, r, **kwargs):
        assert self.val_var is not None and self.test_var is not None
        return SGR_Variance(self.val_outs, self.val_var[:, 0], self.val_labels, r)

    def Method(self, rs, loss_func ='classSpec', **kwargs):
        return Method(self.val_outs, self.val_labels, rs, loss_func=loss_func, **kwargs)

    def Method_global(self, rs, loss_func ='classSpec', **kwargs):
        kwargs = kwargs.copy()
        if 'B' in kwargs: kwargs['seed'] = kwargs.pop('B')
        for _key in ['max_step']: kwargs.pop(_key, None)
        return Method_global(self.val_outs, self.val_labels, rs, loss_func=loss_func, **kwargs)

    def LABEL(self, alphas, **kwargs):
        return LABEL(self.val_outs, self.val_labels, alphas)

    def run(self, method, r=None, rks=None, loss_func='classSpec', **kwargs):
        if method == 'SGR': res = self.SGR(r)
        if method == 'SGR_MCDropout': res = self.SGR_Variance(r)
        if method == 'LABEL': res = self.LABEL(rks)
        if method == METHOD_PLACEHOLDER: res = self.Method(rks, loss_func=loss_func, **kwargs)
        if method == 'Global': res = self.Method_global(rks, loss_func=loss_func, **kwargs)
        val_df = evaluate_df(res[1], self.val_outs, self.val_labels)
        test_df = evaluate_df(res[1], self.test_outs, self.test_labels)
        return res[0], val_df, test_df


def organize_run_classSpecific_results(res):
    measures = ['risk/Cov', '% unsure', '#']
    new_res = {"%s%s"%(pref,k):{} for k in measures for pref in ['val ', '']}
    for k in measures: new_res['val %s'%k] = {}
    for method,v in res.items():
        method_name = _replace_method_name(method)
        for k in measures:
            new_res[k][method_name] = v[2][k]
            new_res['val %s'%k][method_name] = v[1][k]
    column_orders = sorted(list(new_res['risk/Cov'].keys()), key=lambda x: 9999 if x in {METHOD_PLACEHOLDER, METHOD_NAME} else len(x))
    return {k:pd.DataFrame(v).reindex(columns=column_orders) for k,v in new_res.items()}

def plot_run_classSpecific_results(res, class_names=None, ideals=None, key='risk/Cov', **kwargs):
    ylim = kwargs.get('ylim', (0,1.05))
    df = res[key]
    if class_names is not None:
        df.index = class_names + [df.index[-1]]
    if ideals is not None:
        df["Target Risk"] = ideals
    #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.bar.html
    kwargs.setdefault('rot', 0)
    ax = df.plot.bar(**kwargs)
    ax.set_ylim(*ylim)
    ax.set_ylabel("risk")

def flatten_classSpecific_results(results):
    bdf = []
    for exp_i, res_i in enumerate(results):
        for method, (_t, val_df, test_df) in res_i.items():
            for dataset, df in zip(['val', 'test'], [val_df, test_df]):
                tdf = df.stack().reset_index().rename(columns={'level_0':"class", 'level_1':'measure', 0:"value"})
                tdf['dataset'] = dataset
                tdf['method'] = method
                tdf['exp'] = exp_i
                bdf.append(tdf)
    return pd.concat(bdf, ignore_index=True)

def summ_run_classSpecific_results_bootstrapped(flat_results, bmk=None, measure='risk/Cov',
                                                keep_datasets=['test', 'val'], **kwargs):
    nexps = flat_results['exp'].nunique()
    idx = pd.IndexSlice
    diff = flat_results.copy()
    diff = diff[diff['measure'] == measure].drop('measure', axis=1)
    if keep_datasets is not None: diff = diff[diff['dataset'].isin(set(keep_datasets))]
    if bmk is not None:
        diff['value'] = diff.apply(lambda r: r['value'] - bmk[r['class']], axis=1)
    else:
        tdf = diff.pivot_table(values='value', columns=['method'], index=['dataset', 'class', 'exp'])
        bmk_ = tdf['SGR']
        for method in tdf.columns: tdf[method] = tdf[method] - bmk_
        diff = tdf.stack().reset_index().rename(columns={0: 'value'})
    #Columns=['measure', 'dataset', 'class', 'exp', 'method', 'value']
    res_full = {}

    #=====Excess Risk
    key, _f = 'ExcessRisk', lambda s: s.clip(0)
    gb1 = diff.groupby(["method", "dataset", 'class'])["value"]
    ccol = utils.merge_mean_std_tables(gb1.apply(lambda s: _f(s).mean()), gb1.apply(lambda s: _f(s).std()))
    res_full[key] = ccol.unstack(level=2)
    gb2 = diff.groupby(["method", "dataset"])["value"]
    res_full[key]['avg'] = utils.merge_mean_std_tables(gb2.apply(lambda s: _f(s).mean()), gb2.apply(lambda s: _f(s).std()))
    res_full[key] = res_full[key].T

    return res_full, diff


def plot_run_classSpec_box(flat_df, target, methods=['SGR', 'LABEL', METHOD_NAME +'-', METHOD_NAME], measure='risk/Cov',
                           dataset='test', incl_overall=True,**kwargs):
    kwargs = kwargs.copy()
    df = flat_df.copy()
    df = df[df['dataset'] == dataset]
    K = df['class'].nunique() - 1
    df = df[df['measure'] == measure]
    if not incl_overall: df = df[df['class']!='Overall']
    df['target'] = target
    fig1 = plt.figure(figsize=kwargs.pop('figsize', None))
    cplt = plt.gca()
    data = [df[df['method'] == method]['value'].values for method in methods]
    cplt.violinplot(data, showextrema=False, showmedians=True)
    def set_axis_style(ax, labels):
        ax.get_xaxis().set_tick_params(direction='out')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_xlim(0.25, len(labels) + 0.75)
    set_axis_style(cplt, methods)
    left, right = cplt.get_xlim()
    cplt.hlines(target, xmin=left, xmax=right, color='r', linestyles='--')
    cplt.set_ylabel('$R_k(\mathbf{H})$ (all classes)')
    cplt.set_ylim(0, 1.05)
    return df
