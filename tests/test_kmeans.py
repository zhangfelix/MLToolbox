'''test kmeans
'''

from MLToolbox.cluster.kmeans import KMeans
# from MLToolbox.tools import data
from sklearn.cluster import KMeans as KMeans_skl
import numpy as np

class TestKMeans:
    '''
    test MLToolbox.cluster.kmeans
    '''
    def test_kmeans(self):
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
        seed = 1
        kmeans_mlt = KMeans(2,seed=seed)
        kmeans_skl = KMeans_skl(n_clusters=2,random_state=seed)
        kmeans_mlt.fit(data)
        kmeans_skl.fit(data)
        print('kmeans_mlt.centroids:',kmeans_mlt.centroids)
        print('kmeans_mlt.cluster:',kmeans_mlt.clusters)
        print('kmeans_skl.cluster_centers_:',kmeans_skl.cluster_centers_)
        print('kmeans_skl.labels_:',kmeans_skl.labels_)
        mlt_centroids_set = {tuple(x) for x in kmeans_mlt.centroids}
        skl_centroids_set = {tuple(x) for x in kmeans_skl.cluster_centers_}
        assert mlt_centroids_set == skl_centroids_set
        kmeans_skl_clusters = [[] for _ in range(kmeans_skl.n_clusters)]
        for idx, i in enumerate(kmeans_skl.labels_):
            kmeans_skl_clusters[i].append(idx)
            # print("i:",i)
            # print("kmeans_skl_clusters[",i,"]:",kmeans_skl.labels_[i])
            # print("kmeans_skl_clusters:", kmeans_skl_clusters)
        mlt_clusters_set = {tuple(x) for x in kmeans_mlt.clusters}
        skl_clusters_set = {tuple(x) for x in kmeans_skl_clusters}
        print("skl_clusters_set:",skl_clusters_set)
        assert mlt_clusters_set == skl_clusters_set
        X = np.array([[2,6],[7,8],[5,3]])
        kmeans_skl_transform = kmeans_skl.transform(X)
        print("kmeans_skl_transform:",kmeans_skl_transform)
        kmeans_mlt_transform = kmeans_mlt.transform(X)
        print("kmeans_mlt_transform:",kmeans_mlt_transform)
        assert (kmeans_skl_transform == kmeans_mlt_transform).all()
        rng = np.random.default_rng()
        new_points = rng.integers(low=0, high=10, size=20).reshape(10,2)
        new_labels_mlt = kmeans_mlt.predict(new_points)
        new_labels_skl = kmeans_skl.predict(new_points)
        # print(new_labels_mlt)
        # print(new_labels_skl)
        assert (new_labels_mlt == new_labels_skl).all()
