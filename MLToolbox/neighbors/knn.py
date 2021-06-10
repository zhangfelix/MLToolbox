# coding=utf-8
'''
knn class.
'''
import numpy as np
from MLToolbox.tools import data as tdata, distance as dis, sort

class KNN:
    '''
    knn class
    '''
    def _normal_vote(self, idx):
        keys = {}
        most = 0
        key = None
        for i in self.topk['labels'][idx]:
            keys[i] = 1 if i not in keys else keys[i] + 1

        for k in keys:
            if k is not None and most < keys[k]:
                most = keys[k]
                key = k
        return key

    def _weight_vote(self, idx):
        keys = {}
        most = 0
        key = None
        for i, k in enumerate(self.topk['labels'][idx]):
            keys[k] = 1/self.topk['distances'][idx][i] if k not in keys else keys[k] + 1/self.topk['distances'][idx][i]

        for k in keys:
            if k is not None and most < keys[k]:
                most = keys[k]
                key = k
        return key

    vote_methods = {'normal':_normal_vote,
                    'weight':_weight_vote}
    def __init__(self, vote_method='normal', data=None, \
                 k=1, distance=dis.euler_distance):
        '''
        class init.
        '''
        if vote_method not in self.vote_methods:
            self.vote_method = 'normal'
        else:
            self.vote_method = vote_method
        if data is not None:
            self.data, self.label  = data[0], data[1]
        else:
            self.data, self.label = tdata.date_loader('knn')

        self.k = k
        self.topk = {}
        self.distance = distance

    def calculate(self, new_points):
        '''
        calculate which label new points belong to.
        '''
        new_labels = np.array([None] * len(new_points))
        self.topk['points'] = [[None]*self.k]*len(new_points)
        self.topk['distances'] = [[None]*self.k]*len(new_points)
        self.topk['labels'] = [[None]*self.k]*len(new_points)
        self.topk['indices'] = [[None]*self.k]*len(new_points)

        for idx, new_point in enumerate(new_points):
            self.k_nestest(idx, new_point)
            new_labels[idx] = self.vote(idx)

        self.topk['points'] = np.array(self.topk['points'])
        self.topk['distances'] = np.array(self.topk['distances'])
        self.topk['labels'] = np.array(self.topk['labels'])
        self.topk['indices'] = np.array(self.topk['indices'])
        return new_labels

    def k_nestest(self, idx, new_point):
        '''
        find the k nestest points to the new point.
        '''
        # calculate distance with different methods.
        for ind,(i,j) in enumerate(zip(self.data, self.label)):
            self._top_k(idx, new_point, ind, i, j)

    def vote(self, idx):
        '''
        vote which label the point belong to.
        '''
        # TODO(zhangfelixx@hotmail.com): vote which class the new point belong to.
        return self.vote_methods[self.vote_method](self, idx)

    def _top_k(self, idx, new_point, ind, i, j):
        '''
        get top k nestest points.
        '''
        # get top k nestest points, and set them in the topk.
        new_dis = self.distance(new_point, i)
        pos = sort.add_to_topk_binary(self.topk['distances'][idx], new_dis)
        if pos is not None:
            self.topk['points'][idx] = self.topk['points'][idx][:pos]+[i]+self.topk['points'][idx][pos:-1]
            self.topk['distances'][idx] = self.topk['distances'][idx][:pos]+[new_dis]+self.topk['distances'][idx][pos:-1]
            self.topk['labels'][idx] = self.topk['labels'][idx][:pos]+[j]+self.topk['labels'][idx][pos:-1]
            self.topk['indices'][idx] = self.topk['indices'][idx][:pos]+[ind]+self.topk['indices'][idx][pos:-1]
