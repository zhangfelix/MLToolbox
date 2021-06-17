# coding=utf-8
'''
data preprocess
'''
from math import ceil, floor
from sklearn.datasets import load_iris
import numpy as np
def date_loader(data_set_name):
    '''
    Load data from internal database.
    '''
    if data_set_name == 'knn':
        return load_iris(return_X_y=True)
    return np.array([]), np.array([])

def train_test_split(X,y,train_size=None,test_size=None,shuffle=True):
    '''
    Split whole dataset into train set and test set.
    '''
    # Validate parameters
    # Validate X and y
    len_x = len(X)
    if len_x != len(y):
        raise ValueError("Features X and labels y should have same length.")
    # validate train_size and test_size
    if train_size is not None and test_size is not None:
        if train_size <= 0 or test_size <= 0:
            raise ValueError("Train and test size should > 0")
        if train_size + test_size > 1:
            raise ValueError("The sum of train and test size should < 1.")

    if train_size is None and test_size is not None:
        if test_size <= 0:
            raise ValueError("Test size should > 0")
        if test_size >= 1:
            raise ValueError("Test size should < 1")

    if train_size is not None and test_size is None:
        if train_size <= 0:
            raise ValueError("Train size should > 0")
        if train_size >= 1:
            raise ValueError("Train size should < 1")

    if train_size is None and test_size is None:
        raise ValueError("Specify at least one of train_size and test_size.")


    if train_size is None:
        train_size = 1 - test_size
    elif test_size is None:
        test_size = 1 - train_size

    n_train = floor(len_x * train_size)
    n_test = ceil(len_x * test_size)
    if shuffle is True:
        rng = np.random.default_rng()
        idxes = rng.permutation(len_x)
    else:
        idxes = range(len_x)
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for idx in idxes[:n_train]:
        X_train = X_train.append(X[idx])
        y_train = y_train.append(y[idx])
    for idx in idxes[n_train:n_train+n_test]:
        X_test = X_test.append(X[idx])
        y_test = y_test.append(y[idx])

    # return X_train, X_test, y_train, y_test
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
