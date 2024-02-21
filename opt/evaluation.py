import numpy as np

_ALLOW_DEBUG = False
import ipdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
import tqdm

import utils


class EvalRisks():
    def __init__(self):
        pass

    @classmethod
    def _eval_binary_thresholds(cls, preds, ys, t0, t1):
        ys_bool = np.asarray(ys, dtype=bool)
        maybe_1 = preds > t1
        maybe_0 = preds < t0
        not_sure = maybe_0 & maybe_1
        pred_1 = maybe_1 & (~ not_sure)
        pred_0 = maybe_0 & (~ not_sure)

        risk_0 = sum(pred_1 & (~ys_bool)) / sum(~ys_bool)
        risk_1 = sum(pred_0 & (ys_bool)) / sum(ys_bool)

        return risk_0, risk_1, sum(not_sure) / len(ys_bool)

    @classmethod
    def _eval_knary_thresholds(cls, output, ys=None, fs=None, K=None, quiet=False):
        ys_bool = np.asarray(ys, dtype=bool)
        K = K or ys.shape[1]
        if fs is None: fs = lambda x: [1 for _ in range(K)]
        iter_over = range(len(output)) if quiet else tqdm.tqdm(range(len(output)), ncols=80)
        maybe_preds = np.asarray([fs(output[j]) for j in iter_over]).astype(bool)
        unsure = np.sum(maybe_preds, axis=1) > 1

        results = []
        for i in range(K):
            results.append({"FN": sum((~maybe_preds[:, i]) & (ys_bool[:, i])),
                            "FN&Sure": sum((~maybe_preds[:, i]) & (~unsure) & (ys_bool[:, i])),
                            "#": sum(ys_bool[:, i]),
                            "# Unsure": sum(unsure & ys_bool[:, i]),
                            "E[H]": np.mean(np.sum(maybe_preds[ys_bool[:, i]], 1))
                            })
            continue
        results.append({"FN": sum((np.sum(ys_bool * maybe_preds, 1) < 1)),
                        "FN&Sure": sum(~unsure & (np.sum(ys_bool * maybe_preds, 1) < 1)),
                        "#": len(unsure),
                        "# Unsure": sum(unsure),
                        "E[H]":np.mean(np.sum(maybe_preds, 1)),
                        })
        results = pd.DataFrame(results, index=[i for i in range(K)] + ['Overall'])
        results['% unsure'] = results['# Unsure'] / results['#']
        results['% Miss'] = results['FN'] / results['#']
        results['risk'] = results['FN&Sure'] / results['#']
        results['risk/Cov'] = results['FN&Sure'] / (results['#'] - results['# Unsure'])
        results.loc[results['#'] == results['# Unsure'], 'risk/Cov'] = 1
        results['E[H]|Unsure'] = (results['E[H]'] * results['#'] - (results['#'] - results['# Unsure'])) / results['# Unsure']
        return results

    @classmethod
    def _eval_kary_thresholds_details(cls, output, ys, fs=None, K=None, quiet=False):
        K = K or ys.shape[1]
        ys = np.argmax(ys, 1)
        classes = [k for k in range(K)]
        if fs is None: fs = lambda x: [1 for _ in range(K)]
        iter_over = enumerate(output) if quiet else tqdm.tqdm(enumerate(output), ncols=80)
        ret = pd.DataFrame(0, index=classes, columns=classes + ['Unsure'])
        unsure_size_means = pd.Series(0, index=classes)
        for j, out in iter_over:
            pred = fs(out)
            y = ys[j]
            _cnt = sum(pred)
            if _cnt == 1:
                ret.loc[y, np.argmax(pred)] += 1
            if _cnt > 1:
                ret.loc[y, 'Unsure'] += 1
                unsure_size_means[y] += _cnt
        unsure_size_means /= ret.loc[:, 'Unsure']
        return ret, unsure_size_means

    @classmethod
    def _eval_kary_thresholds_F1(cls, res=None):
        #ret, unsure_size_means = cls._eval_kary_thresholds_details(output, ys, fs, K, quiet)
        ret, unsure_size_means = res
        conf_mat, unsure_cnt = ret.iloc[:, :-1], ret.iloc[:, -1]
        df = pd.DataFrame(index=ret.index)
        df['True Total #'] = ret.sum(1)
        df['True Sure #'] = conf_mat.sum(1)
        df['Pred #'] = conf_mat.sum(0)
        df['TP'] = np.diag(ret.iloc[:, :-1])
        df['Recall'] = df['TP'] / df['True Sure #']
        df['Precision'] = df['TP'] / df['Pred #']
        df['F1'] = 2 * df['Recall'] * df['Precision'] / (df['Precision'] + df['Recall'])

        return df#, ret, unsure_size_means

    @classmethod
    def _make_set_pred_fs(cls, ts, score_func=None, fill_empty='max'):
        K = len(ts)
        def fs(x):
            scores = [score_func(x, utils.one_hot_single(i,K)) for i in range(K)] if score_func is not None else x
            ret = np.asarray([scores[i] >= t for i,t in enumerate(ts)]).astype(bool)  # scores
            if np.sum(ret) == 0:
                # If thresholds are very high, it means we can get lower risk with no overlap, so we just argmax
                if fill_empty == 'max':
                    if score_func is None:
                        ret[np.argmax(x)] = 1
                    else:
                        ret[np.argmax(scores)] = 1
                if fill_empty == 'all':
                    ret = np.ones(K)
            return ret
        return fs

    @classmethod
    def _make_single_pred_fs(cls, t, score_func):
        def fs(x):
            K = len(x)
            pred = np.zeros(K)
            if score_func(x, None) >= t:
                pred[np.argmax(x)] = 1
            else:
                pred = np.ones(K)
            return pred
        return fs

def _compute_AUC_helper(results, xlim=None, plot=False, extend=True, extend_node=[(1.,1.), (0., 0.)]):
    #If extend, we will link the curve to the max/min x point of all methods to make a fair comparison
    #results[key]  = series
    res = {}
    if extend: assert xlim is None
    x2y = {}
    if xlim is None:
        for k, v in results.items():
            v = v.groupby(v.index).mean() #SGR + Dropout has some issues..
            x2y.update(v)
            results[k] = v.sort_index()
        res['x_min'], res['x_max'] = min(x2y.keys()), max(x2y.keys())
        x2y = {k:v for k,v in x2y.items() if k in {res['x_min'], res['x_max']}}
    else:
        res['x_min'], res['x_max'] = xlim
    colors=['blue', 'red', 'black']
    #ipdb.set_trace()
    for k,v in results.items():
        if not extend:
            v = v[(v.index > xlim[0]) & (v.index <= xlim[1])]
        if extend:
            for kk in x2y.keys(): v[kk] = x2y[kk]
            for kk, vv in extend_node: v[kk] = vv
            v = v.sort_index()
        if plot:
            plt.plot(v.index,v.values, label=k, color=colors.pop())
        if extend:
            res[k] = sklearn.metrics.auc(v.index,v.values)
        else:
            res[k] = sklearn.metrics.auc(v.index,v.values) / float(xlim[1] - xlim[0])
        if not isinstance(res[k], float): ipdb.set_trace()
    if plot: plt.show()
    return pd.Series(res)