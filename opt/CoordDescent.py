import time

import numpy as np
import pandas as pd
import tqdm

_ALLOW_RUN = True
import opt.QuickSearch as qs

OVERALL_LOSSFUNC = 'overall'
CLASSPECIFIC_LOSSFUNC = 'classSpec'

def parse_loss_func(loss_func:str):
    loss_kwargs = {}
    loss_name = loss_func.split()[0]
    if len(loss_func.split()) > 1:
        for s in loss_func.split()[1:]:
            loss_kwargs[s.split('=')[0]] = float(s.split('=')[1])
    return loss_name, loss_kwargs

class CoordDescentGlobal():
    def __init__(self, model_output, labels, rks,
                 loss_func='overall',
                 class_weights=False,
                 debug=False,
                 fill_max=False):
        self.N, self.K = model_output.shape

        self.loss_name, self.loss_kwargs = parse_loss_func(loss_func)
        if self.loss_name == OVERALL_LOSSFUNC:
            assert isinstance(rks, float)
        elif rks is not None:
            rks = np.asarray(rks)

        self.max_classes = np.argmax(model_output, 1)

        if len(labels.shape) == 1: labels = qs.one_hot(labels, self.K)
        self.loss_kwargs.update({"fill_max": fill_max})
        if self.loss_name == CLASSPECIFIC_LOSSFUNC:
            def _loss_eval(preds):
                return qs.loss_class_specific_py(preds, self.labels, self.max_classes, self.rks, self.class_weights, **self.loss_kwargs)
        elif self.loss_name == OVERALL_LOSSFUNC:
            def _loss_eval(preds):
                return qs.loss_overall_py(preds, self.labels, self.max_classes, self.rks, **self.loss_kwargs)
        else:
            raise NotImplementedError()
        self.loss_eval = _loss_eval
        self.model_output = model_output
        sorted_ps = pd.DataFrame(model_output).stack().reset_index().rename(columns={'level_0': 'i', 'level_1': 'k', 0: 'p'})
        self.sorted_ps = sorted_ps.sort_values('p', ascending=True).reset_index().drop('index', axis=1)
        self.sorted_ps = self.sorted_ps.sort_values('p', ascending=False)
        self.labels = labels
        self.rks = rks
        if isinstance(class_weights, bool):
            if class_weights:
                class_weights = np.asarray(np.unique(labels, return_counts=True)[1], dtype=np.float) * self.K / float(self.N)
            else:
                class_weights = None
        elif class_weights is not None:
            class_weights = np.asarray(class_weights, np.float)
        self.class_weights = class_weights
        self.fill_max = fill_max
        self.debug = debug

    def search_cov(self, cov):
        def _get_cov(idx):
            t = self.sorted_ps.loc[idx, 'p']
            preds = self.model_output > t
            curr_cov = sum(np.sum(preds, 1) == 1) / float(self.N)
            return curr_cov
        st = time.time()
        lbidx, ubidx = self.sorted_ps.index[-1], self.sorted_ps.index[0]
        while lbidx < ubidx:
            if lbidx == ubidx - 1:
                if abs(_get_cov(lbidx) - cov) < abs(_get_cov(ubidx) - cov):
                    mid = lbidx
                    break
                else:
                    mid = ubidx
                    break
            mid = (lbidx + ubidx) // 2
            curr_cov = _get_cov(mid)
            if curr_cov > cov: #need lower threshold
                ubidx = mid
            else:
                lbidx = mid
        t = self.sorted_ps.loc[mid, 'p']
        return t, (curr_cov - cov) ** 2.

    def search_slow(self):
        preds = np.zeros(self.model_output.shape)
        best_loss, best_global_rnk = np.inf, None
        sorted_idxs = self.sorted_ps.drop('p', axis=1)
        for rnk, row in tqdm.tqdm(sorted_idxs.iterrows(), ncols=80, desc='Chck t'):
            preds[row['i'], row['k']] = 1
            curr_loss = self.loss_eval(preds)
            if curr_loss < best_loss:
                best_loss, best_global_rnk = curr_loss, rnk - 1
        if best_global_rnk >= 0:
            t = self.sorted_ps.loc[best_global_rnk, 'p']
        else:
            t = self.sorted_ps.loc[0, 'p'] - 1e-5
        return t, best_loss

    def search_fast(self):
        sorted_idxs = self.sorted_ps.drop('p', axis=1)
        best_global_rnk, best_loss = qs.main_coord_descent_class_specific_globalt(sorted_idxs.values, np.argmax(self.labels, 1),
                                                                                  self.max_classes, self.rks,
                                                                                  ascending=False, class_weights=self.class_weights,
                                                                                  **self.loss_kwargs)
        best_global_rnk = len(sorted_idxs) - 1 - best_global_rnk
        if best_global_rnk >= 0:
            t = self.sorted_ps.loc[best_global_rnk, 'p']
        else:
            t = self.sorted_ps.loc[0, 'p'] - 1e-5
        return t, best_loss

    @classmethod
    def run(cls, preds, ys, rks, batch=None, seed=0, **kwargs):
        np.random.seed(seed)
        if batch is None:
            searcher = CoordDescentGlobal(preds, ys, rks, **kwargs)
        else:
            subsample = np.random.choice(len(preds), batch, replace=False)
            searcher = CoordDescentGlobal(preds[subsample], ys[subsample], rks, **kwargs)
        return searcher.search_fast()


    @classmethod
    def run_cov(cls, preds, ys, cov, batch=None, seed=0, **kwargs):
        np.random.seed(seed)
        if batch is None:
            searcher = CoordDescentGlobal(preds, ys, -1., **kwargs)
        else:
            subsample = np.random.choice(len(preds), batch, replace=False)
            searcher = CoordDescentGlobal(preds[subsample], ys[subsample], -1., **kwargs)
        return searcher.search_cov(cov)

class CoordDescentCython():
    def __init__(self, model_output, labels, rks,
                 loss_func='overall',
                 class_weights=False,
                 debug=False, restart_n=1000, restart_range=0.1,
                 init_range=None,
                 max_step=None,
                 fill_max=False):
        self.N, self.K = model_output.shape

        self.loss_name, self.loss_kwargs = parse_loss_func(loss_func)
        if self.loss_name == OVERALL_LOSSFUNC:
            assert isinstance(rks, float)
        elif rks is not None:
            rks = np.asarray(rks)

        self.max_classes = np.argmax(model_output, 1)

        self.loss_kwargs.update({"fill_max": fill_max})
        if self.loss_name == CLASSPECIFIC_LOSSFUNC:
            def _search_func(ps):
                return qs.main_coord_descent_class_specific(self.idx2rnk, self.rnk2idx,
                                                            self.labels, self.max_classes,
                                                            ps, self.rks, self.class_weights,
                                                            max_step=max_step,
                                                            **self.loss_kwargs)
            self.search_func = _search_func
            self.loss_eval = lambda ps: qs.loss_class_specific(self.idx2rnk, self.rnk2idx, self.labels, self.max_classes,
                                                               ps, self.rks, self.class_weights,
                                                               **self.loss_kwargs)

        elif self.loss_name == OVERALL_LOSSFUNC:
            self.search_func = lambda ps: qs.main_coord_descent_overall(self.idx2rnk, self.rnk2idx,
                                                                        self.labels, self.max_classes,
                                                                        ps, self.rks, max_step=max_step,
                                                                        **self.loss_kwargs)
            self.loss_eval = lambda ps: qs.loss_overall(self.idx2rnk, self.rnk2idx, self.labels, self.max_classes,
                                                        ps, self.rks, **self.loss_kwargs)
        else:
            raise NotImplementedError()

        self.model_output = model_output
        self.rnk2idx = np.asarray(np.argsort(model_output, axis=0), np.int)
        self.idx2rnk = np.asarray(pd.DataFrame(model_output).rank(ascending=True), np.int)
        if np.min(self.idx2rnk) == 1: self.idx2rnk -= 1

        self.labels = np.asarray(labels, np.int)
        self.rks = rks
        if isinstance(class_weights, bool):
            if class_weights:
                class_weights = np.asarray(np.unique(labels, return_counts=True)[1], dtype=np.float) * self.K / float(self.N)
            else:
                class_weights = None
        elif class_weights is not None:
            class_weights = np.asarray(class_weights, np.float)
        self.class_weights = class_weights
        self.fill_max = fill_max
        self.restart_n = restart_n
        self.restart_range = restart_range
        self.init_range = init_range or (int(np.ceil(self.N / 2)), self.N - 1)
        self.debug = debug

        assert max_step is None
        self.max_step = max_step

    def _p2t(self, p):
        return [self.model_output[self.rnk2idx[p[k], k], k] for k in range(self.K)]

    def sample_new_loc(self, old_p, restart_range=0.1):
        diff = np.random.uniform(-restart_range, restart_range, self.K)
        new_p = old_p.copy()
        for k in range(self.K):
            new_p[k] = max(min(int(new_p[k] + diff[k] * self.N), self.N-1), 0)
        return new_p

    def search(self, seed=7, get_complexity=False):
        np.random.seed(seed)
        total_searches = 0
        best_ps = np.random.randint(*self.init_range, self.K)

        st = time.time()
        best_loss, best_ps, n_searches = self.search_func(best_ps)
        total_searches += n_searches
        ed1 = time.time()
        if self.restart_n > 0:
            keep_going = True
            while keep_going:
                keep_going = False
                curr_restart_best_loss, curr_restart_best_ps = np.inf, None

                for _ in range(self.restart_n):
                    new_ps_ = self.sample_new_loc(best_ps, self.restart_range)
                    loss_ = self.loss_eval(new_ps_)
                    if loss_ < best_loss:
                        if self.debug: print("Neighborhood has a better loc with loss={} < {} ".format(loss_, best_loss))
                        best_loss, best_ps, n_searches = self.search_func(new_ps_)
                        total_searches += n_searches
                        keep_going = True
                        break
                    elif loss_ < curr_restart_best_loss:
                        curr_restart_best_loss, curr_restart_best_ps = loss_, new_ps_
                if not keep_going:
                    if self.debug: print(f"Tried {curr_restart_best_ps} vs {best_ps}, loss:{curr_restart_best_loss} > {best_loss}")
        ed2 = time.time()
        if self.debug: print(f"{ed1-st:.3f} + {ed2-ed1:.3f} seconds")
        if get_complexity: return self._p2t(best_ps), best_loss, total_searches
        return self._p2t(best_ps), best_loss

    @classmethod
    def run(cls, preds, ys, rks, B=10, batch=None, **kwargs):
        if len(ys.shape) == 2: ys = np.argmax(ys, 1)
        best_loss, best_ts = np.inf, None

        if batch is None:
            searcher = CoordDescentCython(preds, ys, rks, **kwargs)
        for seed in range(B):
            np.random.seed(seed)
            if batch is not None:
                subsample = np.random.choice(len(preds), batch, replace=False)
                searcher = CoordDescentCython(preds[subsample], ys[subsample], rks, **kwargs)
            ts, _l = searcher.search(seed+1)
            print(f"{seed}: loss={_l}")
            if _l < best_loss:
                best_loss,best_ts = _l, ts
        return best_ts, best_loss

if __name__ == '__main__' and _ALLOW_RUN:

    import data.sim_data as sim_data
    import opt.evaluation as evaluation

    N, K = 10000, 5
    sim_type='real'
    valid_output, valid_y = sim_data.make_sim_data(N, K, 123, sim_type=sim_type, signal=1.5)
    test_output, test_y = sim_data.make_sim_data(N, K, 778, sim_type=sim_type, signal=1.5)
    r=0.3
    rks=[0.3, 0.3, 0.3, 0.3, 0.3]

    score_func = lambda pred, y: np.dot(pred, y)
    best_ts, best_loss = CoordDescentCython.run(valid_output, valid_y, rks, B=5, batch=None, loss_func='classSpec')
    best_ts, best_loss = CoordDescentCython.run(valid_output, valid_y, 0.3, B=5, batch=None, loss_func='overall')#can be "classSpec" as well, in which case alphas should be list-like
    best_t, best_loss = CoordDescentGlobal.run_cov(valid_output, valid_y, 0.5, batch=None, loss_func='classSpec', fill_max=True)
    best_t, best_loss = CoordDescentGlobal.run(valid_output, valid_y, rks, batch=None, loss_func='classSpec', fill_max=False)

    print(best_t, best_loss)
