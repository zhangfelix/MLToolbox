# coding=utf-8
'''
data preprocess
'''
from sklearn.datasets import load_iris
import numpy as np
def date_loader(data_set_name):
    '''
    load data from internal database.
    '''
    if data_set_name == 'knn':
        return load_iris(return_X_y=True)
    return np.array([]), np.array([])

def train_test_split(X,y,train_size=None,test_size=None,shuffle=True):
    '''
    split whole dataset into train set and test set.
    '''
