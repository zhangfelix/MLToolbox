# coding=utf-8
'''
KMeans class
'''
import numpy as np

class KMeans:
    '''
    KMeans class
    '''
    def __init__(self, n_clusters=8, init = 'ramdon', max_iter = 300, tolerance = 0.0001):
        self.n_clusters = n_clusters
        self.init = init
        self.centroids = self.initialize_centroids()
        self.max_iter = max_iter
        self.tolerance = tolerance
    def fit(self, data):
        '''
        Compute kmeans.
        '''
        self.initialize_centroids(data)

    def predict(self, new_points):
        '''
        Predict the clusters which new_points belong to.
        '''

    def initialize_centroids(self, data):
        '''Initialize centroids.'''
        if self.init == 'ramdon':
            return self._ramdon_initialize_centroids()
        raise ValueError("init parameter may take values from ['ramdon]")

    def _ramdon_initialize_centroids(self, data):
        '''Initialize centroids at ramdon from data.'''
        rng = np.random.default_rng()
        return rng.choice(data, self.n_clusters, replace=False)
