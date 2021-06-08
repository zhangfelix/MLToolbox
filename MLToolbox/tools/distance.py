# coding=utf-8
import numpy as np

def euler_distance(a=np.array([]), b=np.array([])):
    '''
    Calculate Euclidean distance.
    '''
    return np.sqrt(np.sum(np.square(a - b)))
