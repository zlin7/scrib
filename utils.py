import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time, sys
import ipdb
import json
import requests
import datetime
import re
import scipy.special
import _settings

LOG_FOLDER = _settings.LOG_OUTPUT_DIR
TODAY_STR = datetime.datetime.today().strftime('%Y%m%d')
#=============Plotting

def one_hot_single(y, nclass=10):
    onehot = np.zeros(nclass)
    onehot[y] = 1
    return onehot

def to_onehot(labels, K):
    new_labels = np.zeros((len(labels), K))
    new_labels[np.arange(len(labels)), labels] = 1
    return new_labels

def project_to_triangle(p, scale=2):
    return np.asarray([p[1] + 2 * p[2], np.sqrt(3) * p[1]]) * scale/2.
def plot_scatter_prob(preds, ys, k=3):

    assert k == 3
    ys_bool = np.asarray(ys, dtype=bool)
    for i in range(k):
        ps = preds[ys_bool[:, i], :]
        ps_2d = [project_to_triangle(_p) for _p in ps]
        col = ['red', 'blue', 'green'][i]
        plt.scatter([x for x,y in ps_2d],[y for x,y in ps_2d], color=col, label='class_%d'%i, alpha=0.2, marker='^')

        #plt.scatter(transform(one_hot_single(i, 3))[0], transform(one_hot_single(i, 3))[1], color=col, label='true_%d'%i)
    plt.legend()
    return

def log_softmax(x):
    return np.log(scipy.special.softmax(x, 1))

def log_softmax_if_necessary(output):
    if abs(np.mean(np.sum(np.exp(output), 1)) - 1) > 1e-3 and abs(np.mean(np.sum(output, 1))) > 1e-3:
        output = log_softmax(output)
        assert abs(np.mean(np.sum(np.exp(output), 1)) - 1) <= 1e-3
    return output

def conv_size_calc(L_in, kernel_size, dilation=1, padding=1, stride=1):
    #ipdb.set_trace()
    L_out = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
    #ipdb.set_trace()
    return int(np.floor(L_out / stride + 1))

def update_L_out(l, L_in):
    import torch
    if isinstance(l, torch.nn.Conv1d):
        return conv_size_calc(L_in, l.kernel_size[0], l.dilation[0], l.padding[0], l.stride[0])
    if isinstance(l, torch.nn.MaxPool1d):
        return conv_size_calc(L_in, l.kernel_size, l.dilation, l.padding, l.stride)
    raise NotImplementedError()

def update_L_out_layers(layers, L_in):
    L_out = L_in
    for l in layers:
        try:
            L_out = update_L_out(l, L_out)
        except NotImplementedError:
             pass
    return L_out


def plot_F1(out, y):
    import sklearn.metrics
    import tqdm
    thresholds = sorted(out)
    to_plot = [[], []]
    for i in tqdm.tqdm(range(0, len(thresholds), 100), ncols=80):
        t = thresholds[i]
        to_plot[0].append(t)
        to_plot[1].append(sklearn.metrics.f1_score(y, out > t))
    plt.plot(to_plot[0], to_plot[1])

#========================
import  logging
def get_logger(name=None, log_path=None, level = logging.INFO):
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if len(logger.handlers) > 0 and log_path is None: return logger
    if log_path is not None:
        if not log_path.endswith(".log"):
            log_path = os.path.join(log_path, "%s.log"%TODAY_STR)
        log_dir = os.path.dirname(log_path)
        if log_dir == '':
            log_dir = LOG_FOLDER
            log_path = os.path.join(log_dir, log_path)
        if not os.path.isdir(log_dir): os.makedirs(log_dir)
    else:
        if name is None:
            log_path = os.path.join(LOG_FOLDER, "%s.log"%TODAY_STR)
        else:
            log_path = os.path.join(LOG_FOLDER, name, "%s.log"%TODAY_STR)

    if log_path is not None:
        for handler in [] if len(logger.handlers) == 0 else logger.handlers:
            if os.path.normpath(handler.baseFilename) == log_path:
                break
        else:
            logger.handlers = [] #TODO: This does not make sense in the current case. Maybe change the above to assuming there's only one handler
            fileHandler = logging.FileHandler(log_path, mode='a')
            fileHandler.setFormatter(logFormatter)
            logger.addHandler(fileHandler)
        #logger.warning("\n\nStart of a new log")
    return logger


#=========================
class ProgressBar:
    #If iterable is an intertor, pass n! if not, can ignore
    def __init__(self, iterable, taskname=None, barLength=40, stride = 1, n = None):
        self.l = iterable
        if n is None:
            try:
                self.n = len(self.l)
            except TypeError:
                self.l = list(self.l)
                self.n = len(self.l)
        else:
            self.n = n
        if not hasattr(self.l, '__next__'):
            self.l = iter(self.l)
        self.cur = 0
        self.starttime = time.time()
        self.barLength = barLength
        self.taskname = taskname
        self.last_print_time = time.time()
        self.stride = stride

    def __iter__(self):
        return self
    def _update_progress(self):
        status = "Done...\r\n" if self.cur == self.n else "\r"
        progress = float(self.cur) / self.n
        curr_time = time.time()

        block = int(round(self.barLength * progress))
        text = "{}Percent: [{}] {:.2%} Used Time:{:.2f} seconds {}".format("" if self.taskname is None else "Working on {}. ".format(self.taskname),
                                                                      "#" * block + "-"*(self.barLength - block),
                                                                      progress, curr_time - self.starttime, status)
        sys.stdout.write(text)
        sys.stdout.flush()

    def __next__(self):
        if self.cur % self.stride == 0:
            self._update_progress()
        #if self.cur >= self.n:
        #    raise StopIteration
        #else:
        #    self.cur += 1
        #    return self.l[self.cur - 1]
        item = next(self.l)
        self.cur += 1
        return item



def download_file(url, dst, overwrite=False):
    # NOTE the stream=True parameter below
    if not overwrite: assert not os.path.isfile(dst), "{} already exists.".format(dst)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dst, 'wb') as f:
            #for chunk in r.iter_content(chunk_size=8192):
            for chunk in ProgressBar(r.iter_content(chunk_size=8192), n=int(r.headers.get('content-length', 0)) / 8192):
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                #if chunk:
                f.write(chunk)
    return dst


#========================================================
def pivot_with_mean_std_tstat(flat_df, columns, index, values, axis=0, **kwargs):
    #compare across axis
    mdf = flat_df.pivot_table(columns=columns, index=index, values=values, aggfunc='mean')
    sdf = flat_df.pivot_table(columns=columns, index=index, values=values, aggfunc='std')
    df = merge_mean_std_tables(mdf, sdf, **kwargs)
    cdf = flat_df.pivot_table(columns=columns, index=index, values=values, aggfunc='size')
    if axis == 1: df, cdf, mdf, sdf, = df.T, cdf.T, mdf.T, sdf.T
    for c in df.columns:
        for ascend in [True, False]:
            ser = mdf[c].sort_values(inplace=False, ascending=ascend)
            i1, i2 = ser.index[0], ser.index[1]
            #ipdb.set_trace()
            t_stat, pval = ttest_from_stats(mdf.loc[i1, c], sdf.loc[i1, c], cdf.loc[i1, c],
                                            mdf.loc[i2, c], sdf.loc[i2, c], cdf.loc[i2, c])
            df.loc[i1, c] += "[{:.2e}->{}]".format(pval, i2)
    if axis == 1: df, cdf, mdf, sdf, = df.T, cdf.T, mdf.T, sdf.T
    return df

def merge_mean_std_tables(mean_df, std_df, prec1=4, prec2=4):
    format_ = "{:.%df}({:.%df})"%(prec1, prec2)
    if isinstance(mean_df, pd.DataFrame):
        ndf=  pd.DataFrame(index=mean_df.index, columns=mean_df.columns)
        for c in mean_df.columns:
            ndf[c] = merge_mean_std_tables(mean_df[c], std_df[c], prec1, prec2)
        return ndf
    nser = pd.Series("", index=mean_df.index)
    for i in nser.index:
        nser[i] = format_.format(mean_df[i], std_df[i])
    return nser

def ttest(a, b):
    import scipy.stats
    return scipy.stats.ttest_ind(a, b, equal_var=False) #can be greater or less

def ttest_from_stats(m1, s1, n1, m2, s2, n2, alternative='two-sided', **kwargs):
    import scipy.stats
    kwargs.setdefault('equal_var', False)
    assert alternative == 'two-sided', "ttest_from_stats: this scipy version only supports two-sided"
    return scipy.stats.ttest_ind_from_stats(m1, s1, n1, m2, s2, n2, **kwargs)

def merge_dict_inline(d1,d2):
    d1 = d1.copy()
    d1.update(d2)
    return d1



def iterate_over_func_params(func, fixed_kwargs, iterate_kwargs):
    import itertools
    keys = list(iterate_kwargs.keys())
    vals = list(iterate_kwargs.values())
    for args in itertools.product(*vals):
        kwargs = {k: v for k, v in fixed_kwargs.items()}
        curr_kwargs = {k: v for k,v in zip(keys, args)}
        print(curr_kwargs)
        kwargs.update(curr_kwargs)
        _ = func(**kwargs)

class TaskPartitioner():
    def __init__(self):
        self.task_list = None

    def add_task(self, func, *args, **kwargs):
        if self.task_list is None:
            self.task_list = []
        else:
            assert isinstance(self.task_list, list), "Trying to add a task without key to a keyed TaskPartitioner"
        self.task_list.append((func, args, kwargs))


    def add_task_with_key(self, key, func, *args, **kwargs):
        if self.task_list is None:
            self.task_list = dict()
        else:
            assert isinstance(self.task_list, dict), "Trying to add a keyed task without key to a non-eyed TaskPartitioner"
        self.task_list[key] = (func, args, kwargs)

    def __len__(self):
        return len(self.task_list)

    def run(self, ith, shuffle=True, npartition=3, suppress_exception=False, cache_only=False, debug=False):
        import tqdm
        n = len(self.task_list)
        keyed = isinstance(self.task_list, dict)
        if ith is None:
            ith, npartition = 0, 1
        if shuffle:
            np.random.seed(npartition)  # being lazy
            perm = np.random.permutation(len(self.task_list))
        else:
            perm= np.arange(n)
        if keyed:
            task_ids = [key for i, key in enumerate(self.task_list.keys()) if perm[i] % npartition == ith]
        else:
            task_ids = [perm[i] for i in range(n) if i % npartition == ith]
        res = {}
        for task_id in tqdm.tqdm(task_ids, ncols=int(_settings.NCOLS / 2 * 1.5)):
            func, arg, kwargs = self.task_list[task_id]
            if debug:
                print(func, arg, kwargs)
            try:
                res[task_id] = func(*arg, **kwargs)
                if cache_only: res[task_id] = True
            except Exception as err:
                if suppress_exception:
                    print(err)
                else:
                    raise err
        return res


def iterate_over_func_params_scheduler(func, fixed_kwargs, iterate_kwargs, task_runner=None, run=False):
    import itertools
    fixed_kwargs = fixed_kwargs.copy()
    iterate_kwargs = iterate_kwargs.copy()
    keys = list(iterate_kwargs.keys())
    vals = list(iterate_kwargs.values())
    if task_runner is None:
        task_runner= TaskPartitioner()
    for args in itertools.product(*vals):
        kwargs = {k: v for k, v in fixed_kwargs.items()}
        curr_kwargs = {k: v for k,v in zip(keys, args)}
        kwargs.update(curr_kwargs)
        task_runner.add_task(func, **kwargs)
    if run:
        task_runner.run(0, npartition=1)
    return task_runner

def set_all_seeds(random_seed=_settings.RANDOM_SEED):
    import torch
    import numpy as np
    # torch.set_deterministic(True)#This is only available in 1.7
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    print("Setting seeds to %d" % random_seed)
    #os.environ["DEBUSSY"] = str(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
