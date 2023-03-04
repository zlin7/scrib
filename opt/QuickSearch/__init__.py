_CYTHON_ENABLED = False
try:
    import pyximport; pyximport.install()
    import opt.QuickSearch.QuickSearch_cython as cdc
    _CYTHON_ENABLED = True
except:
    pass
assert _CYTHON_ENABLED, "This is a warning of potentially slow compute. You could uncomment this line and use the Python implementation instead of Cython."
import numpy as np

from .QuickSearch import (loss_class_specific_py, loss_overall_py,
                          naive_coord_descnet_class_specific_py,
                          naive_coord_descnet_overall_py, one_hot,
                          search_full_class_specific_py,
                          search_full_overall_py)


def loss_overall(idx2rnk, rnk2idx, labels, max_classes, ps, r=0.3, la=0.03, lc=10, lcs=0.01, fill_max=False):
    if not _CYTHON_ENABLED:
        preds = np.asarray(idx2rnk > ps, np.int)
        return loss_overall_py(preds, one_hot(labels, idx2rnk.shape[1]), max_classes, r, la, lc, lcs, fill_max)
    idx2rnk = np.asarray(idx2rnk, np.int)
    rnk2idx = np.asarray(rnk2idx, np.int)
    labels = np.asarray(labels, np.int)
    max_classes = np.asarray(max_classes, np.int)
    ps = np.asarray(ps, np.int)
    return cdc.loss_overall_q_(idx2rnk, rnk2idx, labels, max_classes, ps, r, la, lc, lcs, fill_max)


def loss_class_specific(idx2rnk, rnk2idx, labels, max_classes, ps, rks, class_weights=None,
                             la=0.03, lc=10, lcs=0.01, fill_max=False):
    if not _CYTHON_ENABLED:
        preds = np.asarray(idx2rnk > ps, np.int)
        return loss_class_specific_py(preds, one_hot(labels, idx2rnk.shape[1]), max_classes, rks,
                                      class_weights, la, lc, lcs, fill_max)
    idx2rnk = np.asarray(idx2rnk, np.int)
    rnk2idx = np.asarray(rnk2idx, np.int)
    labels = np.asarray(labels, np.int)
    max_classes = np.asarray(max_classes, np.int)
    ps = np.asarray(ps, np.int)
    if class_weights is not None: class_weights = np.asarray(class_weights, np.float)
    if rks is not None: rks = np.asarray(rks, np.float)
    return cdc.loss_class_specific_q_(idx2rnk, rnk2idx, labels, max_classes, ps, rks, class_weights, la, lc, lcs, fill_max)

def search_full_overall(idx2rnk, rnk2idx, labels, max_classes, ps, k, r, la=0.03, lc=10, lcs=0.01, fill_max=False):
    if not _CYTHON_ENABLED:
        return search_full_overall_py(idx2rnk, max_classes, rnk2idx, labels, r, ps, k, la, lc, lcs, fill_max=fill_max)
    idx2rnk = np.asarray(idx2rnk, np.int)
    rnk2idx = np.asarray(rnk2idx, np.int)
    labels = np.asarray(labels, np.int)
    max_classes = np.asarray(max_classes, np.int)
    ps = np.asarray(ps, np.int)
    return cdc.search_full_overall(idx2rnk, rnk2idx, labels, max_classes, ps, k, r,
                                   la, lc, lcs, fill_max)

def search_full_class_specific(idx2rnk, rnk2idx, labels, max_classes, ps, k, rks, la=0.03, lc=10, lcs=0.01, fill_max=False):
    if not _CYTHON_ENABLED:
        return search_full_class_specific_py(idx2rnk, max_classes, rnk2idx, labels, rks,
                                             None, ps, k, la, lc, lcs, fill_max)
    idx2rnk = np.asarray(idx2rnk, np.int)
    rnk2idx = np.asarray(rnk2idx, np.int)
    labels = np.asarray(labels, np.int)
    max_classes = np.asarray(max_classes, np.int)
    ps = np.asarray(ps, np.int)
    if rks is not None: rks = np.asarray(rks, np.float)
    return cdc.search_full_class_specific(idx2rnk, rnk2idx, labels, max_classes, ps, k, rks,
                                          la, lc, lcs, fill_max)

def main_coord_descent_overall(idx2rnk, rnk2idx, labels, max_classes, init_ps, r,
                                  max_step=None, la=0.03, lc=10, lcs=0.01,
                                  fill_max=False):
    if not _CYTHON_ENABLED:
        assert max_step is None
        return naive_coord_descnet_overall_py(idx2rnk, max_classes, rnk2idx, labels, init_ps, r,
                                              la, lc, lcs, fill_max)
    idx2rnk = np.asarray(idx2rnk, np.int)
    rnk2idx = np.asarray(rnk2idx, np.int)
    labels = np.asarray(labels, np.int)
    max_classes = np.asarray(max_classes, np.int)
    init_ps = np.asarray(init_ps, np.int)
    return cdc.main_coord_descent_overall_(idx2rnk, rnk2idx, labels, max_classes, init_ps, r,
                                           max_step, la, lc, lcs, fill_max)


def main_coord_descent_class_specific(idx2rnk, rnk2idx, labels, max_classes, init_ps, rks,
                                      class_weights=None, max_step=None, la=0.03, lc=10, lcs=0.01,
                                      fill_max=False):
    if not _CYTHON_ENABLED:
        assert max_step is None
        return naive_coord_descnet_class_specific_py(idx2rnk, max_classes, rnk2idx, labels, init_ps, rks,
                                                     class_weights, la, lc, lcs, fill_max)
    idx2rnk = np.asarray(idx2rnk, np.int)
    rnk2idx = np.asarray(rnk2idx, np.int)
    labels = np.asarray(labels, np.int)
    max_classes = np.asarray(max_classes, np.int)
    init_ps = np.asarray(init_ps, np.int)
    if class_weights is not None: class_weights = np.asarray(class_weights, np.float)
    if rks is not None: rks = np.asarray(rks, np.float)
    return cdc.main_coord_descent_class_specific_(idx2rnk, rnk2idx, labels, max_classes, init_ps, rks,
                                                  class_weights, max_step, la, lc, lcs, fill_max)


def main_coord_descent_class_specific_globalt(rnk2ik, labels, max_classes, rks, ascending=False, class_weights=None, la=0.04, lc=1, lcs=0.1, fill_max=False):
    assert _CYTHON_ENABLED, "This function currently only has cython version"
    rnk2ik = np.asarray(rnk2ik, np.int)
    labels = np.asarray(labels, np.int)
    max_classes = np.asarray(max_classes, np.int)
    if rks is not None: rks = np.asarray(rks, np.float)
    return cdc.main_coord_descent_class_specific_globalt_(rnk2ik, labels, max_classes, rks, ascending,
                                                          class_weights, la, lc, lcs, fill_max)