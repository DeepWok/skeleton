import os
import csv
import math
import shutil
import torch

import numpy as np

from ..utils import topk
from .logger import log


class AccuracyCounter:
    def __init__(self, num, k=(1, ), single_graph = True):
        super().__init__()
        self.num = num
        self.k = k
        self.correct = [0] * len(k)
        self.size = 0
        self.single_graph = single_graph

    def add(self, output, target=None, graph_mask=None):
        self.size += target.size(0)
        if self.single_graph:
            for i, a in enumerate(topk(output, target, self.k, True, graph_mask)):
                self.correct[i] += a
        else:
            self.correct[0] = (torch.round(output) == target).float()

    def accuracies(self):
        for i in range(len(self.k)):
            yield self.correct[i] / self.size

    def errors(self):
        for a in self.accuracies():
            yield 1 - a

    def progress(self):
        return self.size / self.num


class MovingAverage:
    def __init__(self, num):
        super().__init__()
        self.num = num
        self.items = []

    def add(self, value):
        self.items.append(float(value))
        if len(self.items) > self.num:
            self.items = self.items[-self.num:]

    def latest(self):
        return self.items[-1]

    def mean(self):
        return np.mean(self.items)

    def std(self):
        return np.std(self.items)

    def flush(self):
        self.items = []

    def __format__(self, mode):
        text = f'{self.mean():.5f}'
        if 's' not in mode:
            return text
        return text + f'±{self.std():.2%}'

    def __float__(self):
        return self.mean()


class History:
    def __init__(self, prefix, load_name, save_name, suffix, header):
        super().__init__()
        os.makedirs('history', exist_ok=True)
        self.header = header
        if load_name:
            self.load_path = f'history/{prefix}.{load_name}.{suffix}.csv'
        else:
            self.load_path = None
        self.save_path = f'history/{prefix}.{save_name}.{suffix}.csv'
        if self.load_path and self.save_path != self.load_path:
            try:
                shutil.copyfile(self.load_path, self.save_path)
            except FileNotFoundError:
                pass
        exists = os.path.exists(self.save_path)
        self.file = open(self.save_path, 'a')
        if exists:
            log.debug(
                f'Continued history in {self.save_path!r}, '
                f'inherited from {self.load_path!r}.')
        else:
            log.debug(f'Created history in {self.save_path!r}.')
            self.file.write(', '.join(header) + '\n')

    def record(self, info):
        keys = info.keys()
        self.file.write(','.join(str(info[k]) for k in keys) + '\n')

    def flush(self):
        self.file.flush()


def csvread(file_name, xname, ynames, bins=False):
    from scipy.stats import binned_statistic
    csv_name = f'history/{file_name}.csv'
    x, ys = [], {}
    with open(csv_name, 'r') as f:
        for row in csv.DictReader(f):
            row = {k.strip(): float(v) for k, v in row.items()}
            x.append(row[xname])
            for yname in ynames:
                ys.setdefault(yname, []).append(row[yname])
    # clean up
    ivm = {}
    maxx = 0
    for i, v in enumerate(x):
        if i % 100 == 0:
            log.debug(f'Processing {i}/{len(x)}...', update=True)
        maxx = max(v, maxx)
        if ivm and v < maxx:
            for j, w in tuple(ivm.items()):
                if w >= v:
                    del ivm[j]
        ivm[i] = v
    nx, nys = [], {}
    for i in sorted(ivm):
        nx.append(x[i])
        for yname, y in ys.items():
            nys.setdefault(yname, []).append(y[i])
    if bins:
        bnys = {}
        bin_max = math.ceil(max(nx))
        for yname, ny in nys.items():
            bnys[yname], *_ = binned_statistic(
                nx, ny, bins=bin_max, range=[0, bin_max])
        nx = list(range(1, bin_max + 1))
        nys = bnys
    return nx, [nys[yn] for yn in ynames]


def plot(name, x, ys, percent=False):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib import ticker
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(1, 1, 1)
    for yname, y in ys.items():
        # a hacky way to ensure x and y have the same length
        length = min(len(x), len(y))
        y = 100 * np.array(y) if percent else y
        ax.plot(x[:length], y[:length], linewidth=0.5, label=yname)
    ax.legend()
    if percent:
        ax.yaxis.set_major_formatter(ticker.PercentFormatter())
        ax.set_ylim(0, 100)
    else:
        ax.set_ylim(0, max(max(y) for y in ys.values()))
    ax.grid()
    fig.tight_layout()
    fig.savefig(f'history/{name}.pdf')
