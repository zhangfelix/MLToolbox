# coding=utf-8
"""
KMeans class
"""
import numpy as np
from MLToolbox.tools import distance as dis

class KMeans: # pylint: disable=too-many-instance-attributes
    """
    KMeans class
    """
    def __init__(self, n_clusters=8, init = 'ramdon', seed = None, \
                 max_iter = 300, tolerance = 0.0001): # pylint: disable=too-many-arguments
        self.n_clusters = n_clusters
        self.clusters = [[] for _ in range(self.n_clusters)]
        self.init = init
        self.centroids = None
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.isfitted = False
        self.data = []
        self.seed = seed
    def fit(self, data):
        """
        Compute kmeans.
        """
        self.data = data
        self.centroids = self.initialize_centroids()
        self.centroids_iterator()

    def centroids_iterator(self):
        """Calculate centroids iteratively"""
        iter_counter = 0
        while iter_counter < self.max_iter:
            self.clusters = [[] for _ in range(self.n_clusters)]
            cluster_counter = np.zeros((self.n_clusters, 1))
            new_centroids = np.zeros((self.n_clusters, self.data.shape[1]))
            for idx, (_, label) in enumerate(self._transform(self.data)):
                self.clusters[label].append(idx)
                cluster_counter[label] += 1
                new_centroids[label] = new_centroids[label] + self.data[idx]
            new_centroids = new_centroids/cluster_counter
            tolerances = np.array([dis.euler_distance(x,y) for x,y in \
                                   zip(new_centroids,self.centroids)])
            self.centroids = new_centroids
            if all(tolerances < self.tolerance):
                break

    def predict(self, new_points):
        """
        Predict the clusters which new_points belong to.
        """
        return np.array([x for _, x in self._transform(new_points)])

    def initialize_centroids(self):
        """Initialize centroids."""
        if self.init == 'ramdon':
            return self._ramdon_initialize_centroids()
        raise ValueError("init parameter may take values from ['ramdon]")

    def transform(self, X): # pylint: disable=invalid-name
        """Transform X to the distances matrix."""
        transformed_matrix = []
        for transformed_matrix_row, _ in self._transform(X):
            transformed_matrix.append(transformed_matrix_row)
        return np.array(transformed_matrix)
        # transformed_matrix, _ = self._transform(X)
        # return transformed_matrix

    def _transform(self, X): # pylint: disable=invalid-name
        """Compute the distances from each sample to centroids."""
        # transformed_matrix = np.zeros((len(X), self.n_clusters), dtype=float)
        # transformed_matrix = []
        # labels = []
        for x in X: # pylint: disable=invalid-name
            min_dis = float("inf")
            label = None
            transformed_matrix_row = []
            for idx, centroid in enumerate(self.centroids):
                temp_dis = dis.euler_distance(x, centroid)
                transformed_matrix_row.append(temp_dis)
                if min_dis > temp_dis:
                    min_dis = temp_dis
                    label = idx
            yield transformed_matrix_row, label
            # transformed_matrix.append(transformed_matrix_row)
            # labels.append(label)
        # return np.array(transformed_matrix), np.array(labels)


    def _ramdon_initialize_centroids(self):
        """Initialize centroids at ramdon from self.data."""
        rng = np.random.default_rng(self.seed)
        return rng.choice(self.data, self.n_clusters, replace=False)
