import torch
import numpy as np
import pandas as pd
import tqdm
import scipy
import scipy.special
import scipy.stats
import scipy.optimize
import matplotlib.pyplot as plt
from importlib import reload
import utils
import ipdb



def mimic_synthetic_data_for_iid(preds, labels, noise_level=1., seed=0, Nsamples=1000):
    #For each class, sample from
    #mimic 2 things:
    # 1. the class counts
    # 2. class mean signal strength (softmax response)
    K = preds.shape[1]
    log_likelihoods = np.log(preds)
    np.random.seed(seed)
    new_outs = {}
    for k in set(labels):
        msk = labels == k
        mean = np.mean(log_likelihoods[msk], 0)
        new_outs[k] = mean + np.random.normal(size=(sum(msk), K)) * noise_level
        new_outs[k] = np.exp(new_outs[k])
        new_outs[k] /= np.sum(new_outs[k], 1, keepdims=True)
        ipdb.set_trace()
    pass

class SimOutputData:
    _PRESET_DATA_KWARGS = {1: {'signal': 3, 'method_id':3, 'high_signal':9, 'low_signal': 1, 'noise_level':4, 'nclass': 5, 'N': 10000},
                           2: {'signal': 3, 'method_id':3, 'high_signal':9, 'low_signal': 1, 'noise_level':3, 'nclass': 5, 'N': 10000}}
    def __init__(self):
        pass
    @classmethod
    def _make_1(cls, nclass, signal=3, noise_level=1, **kwargs):
        #Evenly distributed classes
        #Real probability - all uncertainty are data uncertainty
        scores = np.zeros(nclass)
        scores[np.random.randint(nclass)] = signal
        scores += np.random.random(nclass) * noise_level
        scores = scipy.special.softmax(scores)
        return scores, np.random.choice([k for k in range(nclass)], 1, p=scores)[0], None

    @classmethod
    def _make_2(cls, nclass, signal=1.5, high_signal=3, noise_level=1, **kwargs):
        #Class 0 has high_signal
        #Real probability - all uncertainty are data uncertainty
        scores = np.zeros(nclass)
        tint = np.random.randint(nclass)
        scores[tint] = (high_signal if tint == 0 else signal)
        scores += np.random.random(nclass) * noise_level
        scores = scipy.special.softmax(scores)
        return scores, np.random.choice([k for k in range(nclass)], 1, p=scores)[0], None

    @classmethod
    def _make_3(cls, nclass, signal=3, high_signal=4, low_signal=1, noise_level=1, **kwargs):
        #Class 0 has high_signal, 1 has low_signal
        #Real probability - all uncertainty are data uncertainty
        scores = np.zeros(nclass)
        tint = np.random.randint(nclass)
        scores[tint] = (high_signal if tint == 0 else (low_signal if tint == 1 else signal))
        scores += np.random.random(nclass) * noise_level
        scores = scipy.special.softmax(scores)
        return scores, np.random.choice([k for k in range(nclass)], 1, p=scores)[0], None

    @classmethod
    def _make_4(cls, nclass, signal=3, high_signal=4, low_signal=1, noise_level=1, **kwargs):
        raise NotImplementedError

    @classmethod
    def make_data(cls,  N, nclass, seed, method_id=1, signal=3, **kwargs):
        np.random.seed(seed)
        _make_func = getattr(cls, '_make_%d'%method_id)
        x, y, extra = [], [], []
        for i in range(N):
            xi, yi, extrai = _make_func(nclass, signal=signal, **kwargs)
            x.append(xi); y.append(yi); extra.append(extrai)
        x, y = np.stack(x,0), np.stack(y, 0)
        return x,y, extra

#@putils.persist_flex()
def _SimOutputData_cache(N, nclass, seed, method_id=1, signal=3, **kwargs):
    np.random.seed(seed)
    _make_func = getattr(SimOutputData, '_make_%d'%method_id)
    x, y, extra = [], [], []
    for i in range(N):
        xi, yi, extrai = _make_func(nclass, signal=signal, **kwargs)
        x.append(xi); y.append(yi); extra.append(extrai)
    x, y = np.stack(x,0), np.stack(y, 0)
    return x,y, extra


def _make_sim_noisy(n, nclass, seed, signal=0.1, p=None):
    np.random.seed(seed)
    labels = np.zeros((n, nclass))
    y = np.random.choice(np.arange(nclass), n, p=p)
    labels[np.arange(n), y] = 1

    def gen_one_pred(y):
        x = np.random.random(nclass)
        x[y] += signal * (3 if y <1 and nclass > 2 else 1)
        return scipy.special.softmax(x)

    preds = [gen_one_pred(y_i) for y_i in y]

    return np.asarray(preds), np.asarray(labels)

def _make_sim_real(n, nclass, seed, signal=3, **kwargs):
    probs, _ = _make_sim_noisy(n, nclass, seed=seed, signal=signal, **kwargs)
    np.random.seed(seed)
    classes = [i for i in range(nclass)]
    y = np.asarray([np.random.choice(classes, 1, p=probs[i])[0] for i in range(len(probs))])
    labels = np.zeros((n, nclass))
    labels[np.arange(n), y] = 1
    return probs, labels

def _make_sim_distorted(n, nclass, seed, signal=3, distort_class={1:0.5, 2:0.2}, **kwargs):
    probs, labels = _make_sim_real(n, nclass, seed, signal, **kwargs)
    for class_k, fac in distort_class.items():
        probs[:, class_k] *= fac
    probs /= np.sum(probs, 1, keepdims=True)
    #ipdb.set_trace()
    return probs, labels

def make_sim_data(n, nclass, seed, sim_type="", signal=3, **kwargs):
    if sim_type == 'noisy': return _make_sim_noisy(n,nclass,seed, signal, **kwargs)
    if sim_type == 'real': return _make_sim_real(n,nclass,seed, signal, **kwargs)
    if sim_type == 'distorted': return _make_sim_distorted(n,nclass,seed, signal, **kwargs)
    if isinstance(sim_type, int):
        probs, labels, _ = _SimOutputData_cache(n, nclass, seed, sim_type, signal, **kwargs)
        if len(labels.shape) == 1: labels = utils.to_onehot(labels, nclass)
        return probs, labels
    raise NotImplementedError()
