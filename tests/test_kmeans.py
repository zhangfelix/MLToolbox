'''test kmeans
'''

from MLToolbox.cluster.kmeans import KMeans
from MLToolbox.tools import data
from sklearn.cluster import KMeans as KMeans_skl

class TestKMeans:
    '''
    test MLToolbox.cluster.kmeans
    '''
    def test_kmeans_fit(self):
        '''
        test MLToolbox.cluster.kmeans.KMeans.fit
        '''
        X,y = data.date_loader('knn')
        kmeans_mlt = KMeans(n_clusters=4)
        kmeans_skl = KMeans_skl(n_clusters=4)
        kmeans_mlt.fit(X)
        kmeans_skl.fit(X)
        # test_points =
        # TODO: complete test kmeans_fit
