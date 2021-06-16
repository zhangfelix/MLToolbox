# coding=utf-8
'''
test knn
'''
import numpy as np
from MLToolbox.tools import data
from MLToolbox.neighbors.knn import KNN
from sklearn.neighbors import KNeighborsClassifier
class TestKNN:
    '''
    test MLToolbox.neighbors.knn
    '''
    times = 0
    def test_knn_calculate(self):
        '''
        test MLToolbox.neighbors.knn.KNN
        '''
        X,y = data.date_loader('knn')
        rng = np.random.default_rng()
        k = 3
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh_fit = neigh.fit(X,y)
        knn_mlt = KNN(data=(X,y),k=k)
        neigh_1 = KNeighborsClassifier(n_neighbors=k,weights='distance')
        neigh_fit_1 = neigh_1.fit(X,y)
        knn_mlt_1 = KNN(vote_method='weight',data=(X,y),k=k)
        for _ in range(100):
            print("times:",self.times)
            rng = np.random.default_rng()
            temp1 = 5 * rng.random((5,1)) + 4
            temp2 = 3 * rng.random((5,1)) + 2
            temp3 = 6 * rng.random((5,1)) + 1
            temp4 = 3 * rng.random((5,1)) + 0
            new_points = np.hstack((temp1,temp2,temp3,temp4))
            result_skl = neigh_fit.predict(new_points)
            result_mlt = knn_mlt.calculate(new_points)
            #dis_skl, ind_skl = neigh.kneighbors(new_points)
            #dis_mlt = knn_mlt.topk['distances']
            #ind_mlt = knn_mlt.topk['indices']
            assert (result_skl == result_mlt).all()
            #assert (dis_skl == dis_mlt).all()
            #assert (ind_skl == ind_mlt).all()
            result_skl_1 =neigh_fit_1.predict(new_points)
            result_mlt_1 = knn_mlt_1.calculate(new_points)
            #dis_skl_1, ind_skl_1 = neigh_1.kneighbors(new_points)
            #dis_mlt_1 = knn_mlt_1.topk['distances']
            #ind_mlt_1 = knn_mlt_1.topk['indices']
            assert (result_skl_1 == result_mlt_1).all()
            #assert (dis_skl_1 == dis_mlt_1).all()
            #assert (ind_skl_1 == ind_mlt_1).all()
