try:
    import QuickSearch as quicksearch
except Exception as err:
    print(err)
    import QuickSearch.QuickSearch as quicksearch
from importlib import reload
reload(quicksearch)

import pandas as pd
import numpy as np
import timeit
import ipdb

def sort_outputs(out):
    scores = np.sort(out, axis=0)
    scores_idx = np.argsort(out, axis=0)
    idx2rnk = np.asarray(pd.DataFrame(out).rank(ascending=True), np.int) - 1
    return scores, scores_idx, idx2rnk

one_hot = quicksearch.one_hot

def get_data():
    np.random.seed(7)
    out = np.random.uniform(0, 1, (1000, 3))
    labels = np.random.randint(0, 3, 1000)
    #ps = np.random.randint(0, 1000, 3)
    ps = np.asarray([556,536, 529])
    scores, scores_idx, idx2rnk = sort_outputs(out)
    ts = np.asarray([scores[ps[i], i] for i in range(3)])
    return ts, out, labels, ps, scores, scores_idx, idx2rnk


def test(f_py, f_c, repeat, desc):
    t_py, t_c = timeit.timeit(f_py, number=repeat), timeit.timeit(f_c, number=repeat)
    print(f"{desc} Time: {t_py:.5f} vs {t_c:.5f}, mult={t_py/t_c:.2f}")
    print(f_py(), f_c())
    print()


if __name__ == '__main__':
    ts, out, labels, ps, scores, scores_idx, idx2rnk = get_data()
    max_classes = np.asarray(np.argmax(out, axis=1), np.int)
    scores_idx = np.asarray(scores_idx,dtype=np.int)
    labels_py = one_hot(labels, 3)
    rks = np.asarray([0.3, 0.3, 0.3], dtype=np.float)

    fill_max = False

    pred = np.asarray(out > ts, dtype=np.int)
    loss_kwargs = {'la': 0.03, 'lc': 10, 'lcs': 0.01, 'fill_max': fill_max}


    test(lambda: quicksearch.loss_class_specific_py(pred, labels_py, max_classes, rks, **loss_kwargs),
         lambda: quicksearch.loss_class_specific(idx2rnk, scores_idx, labels, max_classes, ps, rks, **loss_kwargs),
         repeat=0, desc='Loss ClassSpec')

    test(lambda: quicksearch.loss_overall_py(pred, labels_py, max_classes, 0.3, **loss_kwargs),
         lambda: quicksearch.loss_overall(idx2rnk, scores_idx, labels, max_classes, ps, 0.3, **loss_kwargs),
         repeat=0, desc='Loss Overall')


    test(lambda: quicksearch.search_full_class_specific_py(out, scores, scores_idx, labels, alphas=rks, class_weights=None, ps=ps, d=0, **loss_kwargs),
         lambda: quicksearch.search_full_class_specific(idx2rnk, scores_idx, labels, max_classes, ps, 0, rks, **loss_kwargs),
         repeat=0, desc='Search')

    test(lambda: quicksearch.naive_coord_descnet_class_specific_py(out, scores, scores_idx, labels, ps, rks, class_weights=False, **loss_kwargs),
         lambda: quicksearch.main_coord_descent_class_specific(idx2rnk, scores_idx, labels, max_classes, ps, rks, class_weights=None, **loss_kwargs),
         repeat=0, desc='Full Run')


    test(lambda: quicksearch.search_full_overall_py(out, scores, scores_idx, labels, alpha=0.3, ps=ps, d=0, **loss_kwargs),
         lambda: quicksearch.search_full_overall(idx2rnk, scores_idx, labels, max_classes, ps, 0, 0.3, **loss_kwargs),
         repeat=0, desc='Search Overall')
    test(lambda: quicksearch.naive_coord_descnet_overall_py(out, scores, scores_idx, labels, ps, 0.3, **loss_kwargs),
         lambda: quicksearch.main_coord_descent_overall(idx2rnk, scores_idx, labels, max_classes, ps, 0.3, **loss_kwargs),
         repeat=1, desc='Full Run Overall')
