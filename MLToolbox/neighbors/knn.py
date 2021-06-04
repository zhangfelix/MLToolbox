'''
knn class.
'''
# coding=utf-8
import numpy as np
from MLToolbox.tools import data as tdata, distance as dis, sort

class KNN:
    '''
    knn class
    '''
    def _normal_vote(self):
        keys = {}
        most = 0
        key = None
        for i in self.topk['labels']:
            keys[i] = 1 if i not in keys else keys[i] + 1

        for k in keys:
            if k is not None and most < keys[k]:
                most = keys[k]
                key = k
        return key

    def _weight_vote(self):
        keys = {}
        most = 0
        key = None
        for idx, i in enumerate(self.topk['labels']):
            keys[i] = 1/self.topk['distances'][idx] if i not in keys else keys[i] + 1/self.topk['distances'][idx]

        for k in keys:
            if k is not None and most < keys[k]:
                most = keys[k]
                key = k
        return key

    vote_methods = {'normal':_normal_vote,
                    'weight':_weight_vote}
    def __init__(self, vote_method='normal', filename=None, data=(np.ndarray([]),np.ndarray([])), \
                 k=1, distance=dis.euler_distance):
        '''
        class init.
        '''
        if vote_method not in self.vote_methods:
            self.vote_method = 'normal'
        else:
            self.vote_method = vote_method
        if filename is not None:
            self.data, self.label  = tdata.date_loader(filename)
        else:
            self.data, self.label = data

        self.k = k
        self.topk = {'points':[None] * self.k,
                     'distances':[None] * self.k,
                     'labels':[None] * self.k}
        self.distance = distance

    def calculate(self, new_points):
        '''
        calculate which label new points belong to.
        '''
        new_labels = [None] * len(new_points)
        for idx, new_point in enumerate(new_points):
            self.k_nestest(new_point)
            new_labels[idx] = self.vote()
        return new_labels

    def k_nestest(self, new_point):
        '''
        find the k nestest points to the new point.
        '''
        # calculate distance with different methods.
        for i,j in zip(self.data, self.label):
            self._top_k(new_point, i, j)

    def vote(self):
        '''
        vote which label the point belong to.
        '''
        # TODO(zhangfelixx@hotmail.com): vote which class the new point belong to.
        return self.vote_methods[self.vote_method]()

    def _top_k(self, new_point, i, j):
        '''
        get top k nestest points.
        '''
        # get top k nestest points, and set them in the topk.
        new_dis = self.distance(new_point, i)
        pos = sort.add_to_topk(self.topk['distances'], new_dis)
        if pos is not None:
            self.topk['points'] = self.topk['points'][:pos] + \
                [i] + self.topk['points'][pos:-1]
            self.topk['distances'] = self.topk['distances'][:pos] + \
                [new_dis] + self.topk['distances'][pos:-1]
            self.topk['labels'] = self.topk['labels'][:pos] + \
                [j] + self.topk['labels'][pos:-1]
