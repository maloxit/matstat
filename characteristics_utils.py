import math
import matplotlib.pyplot as pyplt
import numpy as np


def get_table_func(sample):
    seq = sorted(sample)
    n = len(seq)
    acc = 0.
    table = {'x': seq, 'y': []}
    for value in seq:
        acc += 1 / n
        table['y'].append(acc)
    return table


def data_mean(data: list):
    n = len(data)
    acc = 0.
    for value in data:
        acc += value / n
    return acc


def data_sqr_mean(data: list):
    n = len(data)
    acc = 0.
    for value in data:
        acc += (value ** 2) / n
    return acc


def data_dispersion(data: list):
    return data_sqr_mean(data) - data_mean(data) ** 2


def seq_median(seq: list):
    n = len(seq)
    if n % 2 == 1:
        return seq[n // 2]
    else:
        return (seq[n // 2 - 1] + seq[n // 2]) / 2


def sample_z_r(sample: list):
    return (min(sample) + max(sample)) / 2


def seq_quartile(seq, p):
    n_p = p * len(seq)
    if p.is_integer():
        return seq[math.floor(n_p) - 1]
    else:
        return seq[math.floor(n_p)]


def seq_z_q(seq):
    return (seq_quartile(seq, 1. / 4) + seq_quartile(seq, 3. / 4)) / 2


def seq_z_tr(seq):
    size = len(seq)
    r = int(size / 4)
    sum = 0
    for i in range(r, size - r):
        sum += seq[i]
    return sum / (size - 2 * r)


def outliers_bounds(data):
    q_1, q_3 = np.quantile(data, [0.25, 0.75])
    return q_1 - 3 / 2 * (q_3 - q_1), q_3 + 3 / 2 * (q_3 - q_1)


def number_of_outliers(data):
    x1, x2 = outliers_bounds(data)
    filtered = [x for x in data if x > x2 or x < x1]
    return len(filtered)
