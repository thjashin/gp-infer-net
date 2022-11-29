#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
import os
import csv
import gc
from collections import namedtuple
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
import gzip
from six.moves import cPickle as pickle
import tarfile

import six
import numpy as np
import pandas


root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ToyData1D(object):
    def __init__(self, train_x, train_y, test_x=None, x_min=None, x_max=None,
                 n_test=None, normalize=False, dtype=np.float64):
        self.train_x = np.array(train_x, dtype=dtype)[:, None]
        self.train_y = np.array(train_y, dtype=dtype)[:, None]
        self.n_train = self.train_x.shape[0]
        if test_x is not None:
            self.test_x = np.array(test_x, dtype=dtype)[:, None]
            self.x_min = np.min(test_x)
            self.x_max = np.max(test_x)
            self.n_test = self.test_x.shape[0]
        else:
            self.x_min = x_min
            self.x_max = x_max
            self.n_test = n_test
            self.test_x = np.linspace(
                x_min, x_max, num=n_test, dtype=dtype)[:, None]
        if normalize:
            self.normalize()

    def normalize(self):
        self.mean_x = np.mean(self.train_x, axis=0, keepdims=True)
        self.std_x = np.std(self.train_x, axis=0, keepdims=True) + 1e-6
        self.mean_y = np.mean(self.train_y, axis=0, keepdims=True)
        self.std_y = np.std(self.train_y, axis=0, keepdims=True) + 1e-6

        for x in [self.train_x, self.test_x]:
            x -= self.mean_x
            x /= self.std_x

        for x in [self.x_min, self.x_max]:
            x -= self.mean_x.squeeze()
            x /= self.std_x.squeeze()

        self.train_y -= self.mean_y
        self.train_y /= self.std_y

    def unnormalize_x(self, x):
        return x * self.std_x + self.mean_x

    def unnormalize_y(self, y):
        return y * self.std_y + self.mean_y


def load_snelson_data(n=100, dtype=np.float64):
    def _load_snelson(filename):
        with open(os.path.join(root, "data", "snelson", filename), "r") as f:
            return np.array([float(i) for i in f.read().strip().split("\n")],
                            dtype=dtype)

    train_x = _load_snelson("train_inputs")
    train_y = _load_snelson("train_outputs")
    test_x = _load_snelson("test_inputs")
    perm = np.random.permutation(train_x.shape[0])
    train_x = train_x[perm][:n]
    train_y = train_y[perm][:n]
    return ToyData1D(train_x, train_y, test_x=test_x)


class UCIRegression(object):
    def __init__(self, name, N, D, data_path):
        self.data_path = data_path
        if not os.path.isdir(data_path):
            os.makedirs(data_path)
        self.name, self.N, self.D = name, N, D

    def csv_file_path(self, name):
        return os.path.join(self.data_path, '{}.csv'.format(name))

    def read_data(self):
        data = pandas.read_csv(self.csv_file_path(self.name),
                               header=None, delimiter=',').values
        self.x = data[:, :-1]
        self.y = data[:, -1, None]
        del data
        gc.collect()

    def download_data(self):
        raise NotImplementedError()

    def load(self, seed=0, split=0, prop=0.9):
        path = self.csv_file_path(self.name)
        if not os.path.isfile(path):
            self.download_data()

        self.read_data()
        self.split(seed, split, prop)
        self.normalize()

    def split(self, seed, split, prop):
        ind = np.arange(self.N)

        rng = np.random.RandomState(seed + split)
        rng.shuffle(ind)

        n = int(self.N * prop)

        self.train_x = self.x[ind[:n], :]
        self.test_x = self.x[ind[n:], :]

        self.train_y = self.y[ind[:n], :]
        self.test_y = self.y[ind[n:], :]

    def normalize(self):
        self.mean_x = np.mean(self.train_x, axis=0, keepdims=True)
        self.std_x = np.std(self.train_x, axis=0, keepdims=True) + 1e-6
        self.mean_y = np.mean(self.train_y, axis=0, keepdims=True)
        self.std_y = np.std(self.train_y, axis=0, keepdims=True) + 1e-6

        for x in [self.x, self.train_x, self.test_x]:
            x -= self.mean_x
            x /= self.std_x

        for y in [self.y, self.train_y, self.test_y]:
            y -= self.mean_y
            y /= self.std_y


uci_base = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'


class Boston(UCIRegression):
    def __init__(self, data_path):
        super(Boston, self).__init__("boston", 506, 12, data_path)

    def download_data(self):
        url = '{}{}'.format(uci_base, 'housing/housing.data')

        data = pandas.read_fwf(url, header=None).values
        with open(self.csv_file_path(self.name), 'w') as f:
            csv.writer(f).writerows(data)


class Concrete(UCIRegression):
    def __init__(self, data_path):
        super(Concrete, self).__init__("concrete", 1030, 8, data_path)

    def download_data(self):
        url = '{}{}'.format(uci_base, 'concrete/compressive/Concrete_Data.xls')

        data = pandas.read_excel(url).values
        with open(self.csv_file_path(self.name), 'w') as f:
            csv.writer(f).writerows(data)


class Energy(UCIRegression):
    def __init__(self, data_path):
        super(Energy, self).__init__("energy", 768, 8, data_path)

    def download_data(self):
        url = '{}{}'.format(uci_base, '00242/ENB2012_data.xlsx')

        data = pandas.read_excel(url).values
        data = data[:, :-1]

        with open(self.csv_file_path(self.name), 'w') as f:
            csv.writer(f).writerows(data)


class Kin8nm(UCIRegression):
    def __init__(self, data_path):
        super(Kin8nm, self).__init__("kin8nm", 8192, 8, data_path)

    def download_data(self):
        url = 'http://mldata.org/repository/data/download/csv/uci-20070111-kin8nm'

        data = pandas.read_csv(url, header=None).values

        with open(self.csv_file_path(self.name), 'w') as f:
            csv.writer(f).writerows(data)


class Power(UCIRegression):
    def __init__(self, data_path):
        super(Power, self).__init__("power", 9568, 4, data_path)

    def download_data(self):
        url = '{}{}'.format(uci_base, '00294/CCPP.zip')
        with urlopen(url) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall('/tmp/')

        data = pandas.read_excel('/tmp/CCPP//Folds5x2_pp.xlsx').values

        with open(self.csv_file_path(self.name), 'w') as f:
            csv.writer(f).writerows(data)


class Protein(UCIRegression):
    def __init__(self, data_path):
        super(Protein, self).__init__("protein", 45730, 9, data_path)

    def download_data(self):
        url = '{}{}'.format(uci_base, '00265/CASP.csv')

        data = pandas.read_csv(url).values
        data = np.concatenate([data[:, 1:], data[:, 0, None]], 1)

        with open(self.csv_file_path(self.name), 'w') as f:
            csv.writer(f).writerows(data)


class WineRed(UCIRegression):
    def __init__(self, data_path):
        super(WineRed, self).__init__("wine_red", 1599, 11, data_path)

    def download_data(self):
        url = '{}{}'.format(uci_base, 'wine-quality/winequality-red.csv')

        data = pandas.read_csv(url, delimiter=';').values

        with open(self.csv_file_path(self.name), 'w') as f:
            csv.writer(f).writerows(data)


uci_reg_path = os.path.join(root, "data", "uci_reg")
uci_reg_classes = [Boston, Concrete, Energy, Kin8nm, Power, Protein, WineRed]
uci_reg_list = [i(uci_reg_path) for i in uci_reg_classes]
uci_mapping = dict((i.name, i) for i in uci_reg_list)


def load_regression(name):
    if name in uci_mapping:
        return uci_mapping[name]
    elif name == "airline":
        return load_airline()


def load_airline():
    def _preprocess(data_path):
        # Import the data
        data = pandas.read_pickle(
            os.path.join(root, "data", "airline", 'airline.pickle'))

        # Convert time of day from hhmm to minutes since midnight
        data.ArrTime = 60 * np.floor(data.ArrTime / 100) + np.mod(
            data.ArrTime, 100)
        data.DepTime = 60 * np.floor(data.DepTime / 100) + np.mod(
            data.DepTime, 100)

        # Pick out the data
        Y = data['ArrDelay'].values
        names = ['Month', 'DayofMonth', 'DayOfWeek', 'plane_age', 'AirTime',
                 'Distance', 'ArrTime', 'DepTime']
        X = data[names].values

        print("X:", X.shape)
        print("Y:", Y.shape)

        train_x = X[:70000]
        train_y = Y[:70000].reshape(-1, 1)
        test_x = X[70000:80000]
        test_y = Y[70000:80000].reshape(-1, 1)

        mean_x = np.mean(train_x, axis=0, keepdims=True)
        std_x = np.std(train_x, axis=0, keepdims=True) + 1e-6
        mean_y = np.mean(train_y, axis=0, keepdims=True)
        std_y = np.std(train_y, axis=0, keepdims=True) + 1e-6

        def _normalize_x(tmp):
            return (tmp - mean_x) / std_x

        def _normalize_y(tmp):
            return (tmp - mean_y) / std_y

        np.savez_compressed(data_path,
                            train_x=_normalize_x(train_x),
                            train_y=_normalize_y(train_y),
                            test_x=_normalize_x(test_x),
                            test_y=_normalize_y(test_y),
                            mean_x=mean_x,
                            mean_y=mean_y,
                            std_x=std_x,
                            std_y=std_y)

    data_path = os.path.join(root, "data", "airline", "airline.npz")
    if not os.path.exists(data_path):
        _preprocess(data_path)

    data = np.load(data_path)
    _Tuple = namedtuple("_Tuple", data.keys())
    ret = _Tuple(**data)
    return ret


def to_one_hot(x, depth):
    """
    Get one-hot representation of a 1-D numpy array of integers.
    :param x: 1-D Numpy array of type int.
    :param depth: A int.
    :return: 2-D Numpy array of type int.
    """
    ret = np.zeros((x.shape[0], depth))
    ret[np.arange(x.shape[0]), x] = 1
    return ret


def load_mnist_realval(one_hot=True, dequantify=False, dtype=np.float32):
    """
    Loads the real valued MNIST dataset.
    :param path: Path to the dataset file.
    :param one_hot: Whether to use one-hot representation for the labels.
    :param dequantify:  Whether to add uniform noise to dequantify the data
        following (Uria, 2013).
    :return: The MNIST dataset.
    """
    path = os.path.join(root, "data", "mnist", "mnist.pkl.gz")
    f = gzip.open(path, 'rb')
    if six.PY2:
        train_set, valid_set, test_set = pickle.load(f)
    else:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    f.close()
    x_train, t_train = train_set[0], train_set[1]
    x_valid, t_valid = valid_set[0], valid_set[1]
    x_test, t_test = test_set[0], test_set[1]
    if dequantify:
        x_train += np.random.uniform(0, 1. / 256,
                                     size=x_train.shape)
        x_valid += np.random.uniform(0, 1. / 256,
                                     size=x_valid.shape)
        x_test += np.random.uniform(0, 1. / 256,
                                    size=x_test.shape)
    n_y = t_train.max() + 1
    t_transform = (lambda x: to_one_hot(x, n_y)) if one_hot else (lambda x: x)
    return x_train.astype(dtype), t_transform(t_train), x_valid.astype(dtype), \
        t_transform(t_valid), x_test.astype(dtype), t_transform(t_test)


def load_cifar10(normalize=True, dequantify=False, one_hot=True,
                 dtype=np.float32):
    """
    Loads the cifar10 dataset.
    :param path: Path to the dataset file.
    :param normalize: Whether to normalize the x data to the range [0, 1].
    :param dequantify: Whether to add uniform noise to dequantify the data
        following (Uria, 2013).
    :param one_hot: Whether to use one-hot representation for the labels.
    :return: The cifar10 dataset.
    """
    data_dir = os.path.join(root, "data", "cifar10")
    path = os.path.join(data_dir, "cifar-10-python.tar.gz")
    batch_dir = os.path.join(data_dir, "cifar-10-batches-py")
    if not os.path.isfile(os.path.join(batch_dir, 'data_batch_5')):
        with tarfile.open(path) as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, data_dir)

    train_x, train_y = [], []
    for i in range(1, 6):
        batch_file = os.path.join(batch_dir, 'data_batch_' + str(i))
        with open(batch_file, 'rb') as f:
            if six.PY2:
                data = pickle.load(f)
            else:
                data = pickle.load(f, encoding='latin1')
            train_x.append(data['data'])
            train_y.append(data['labels'])
    train_x = np.vstack(train_x)
    train_y = np.hstack(train_y)

    test_batch_file = os.path.join(batch_dir, 'test_batch')
    with open(test_batch_file, 'rb') as f:
        if six.PY2:
            data = pickle.load(f)
        else:
            data = pickle.load(f, encoding='latin1')
        test_x = data['data']
        test_y = np.asarray(data['labels'])

    train_x = train_x.astype(dtype)
    test_x = test_x.astype(dtype)
    if dequantify:
        train_x += np.random.uniform(0, 1,
                                     size=train_x.shape).astype(dtype)
        test_x += np.random.uniform(0, 1, size=test_x.shape).astype(dtype)
    if normalize:
        train_x = train_x / 256
        test_x = test_x / 256

    t_transform = (lambda x: to_one_hot(x, 10)) if one_hot else (lambda x: x)
    return train_x, t_transform(train_y), test_x, t_transform(test_y)


if __name__ == "__main__":
    data = load_airline()
    print("train_x:", data.train_x.shape)
    print(data.train_x[0])
    print("train_y:", data.train_y.shape)
