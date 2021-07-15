'''test kmeans
'''

from MLToolbox.cluster.kmeans import KMeans
from MLToolbox.tools import data
from sklearn.cluster import KMeans as KMeans_skl
import numpy as np

class TestKMeans:
    '''
    test MLToolbox.cluster.kmeans
    '''
    def test_kmeans_fit(self):
        '''
        test MLToolbox.cluster.kmeans.KMeans.fit
        '''
        ## X,y = data.date_loader('knn')
        ## kmeans_mlt = KMeans(n_clusters=4)
        ## kmeans_skl = KMeans_skl(n_clusters=4)
        ## kmeans_mlt.fit(X)
        ## kmeans_skl.fit(X)
        # test_points =
        # TODO: complete test kmeans_fit
        data = np.array([[1, 2], [1, 4], [1, 0],[10, 2], [10, 4], [10, 0]])
        kmeans_mlt = KMeans(2)
        kmeans_skl = KMeans_skl(n_clusters=2)
        kmeans_mlt.fit(data)
        kmeans_skl.fit(data)
        print('kmeans_mlt.centroids:',kmeans_mlt.centroids)
        print('kmeans_mlt.cluster:',kmeans_mlt.clusters)
        print('kmeans_skl.cluster_centers_:',kmeans_skl.cluster_centers_)
        print('kmeans_skl.labels_:',kmeans_skl.labels_)
        assert False
