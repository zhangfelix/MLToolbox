'''
tools tests
'''
import math
import pytest
from MLToolbox.tools import sort, distance, data
import numpy as np

class TestTools:
    '''
    test MLToolbox.tools
    '''
    def test_add_to_topk(self):
        '''
        test sort.add_to_topk
        '''
        lst = [1,2,3,4,5,5,5,6,7,8]
        new_num = 5
        assert sort.add_to_topk(lst,new_num) >= 4
        assert sort.add_to_topk(lst,new_num) <= 7

    def test_add_to_topk_binary(self):
        '''
        test sort.add_to_topk_binary
        '''
        lst_1 = [1,2,3,4,5,5,5,6,7,8]
        new_num_1 = 5
        assert sort.add_to_topk_binary(lst_1,new_num_1) == 7
        assert sort.add_to_topk_binary(lst_1[::-1],new_num_1,True) == 6
        lst_2 = [1,2,3,4,5,5,5]
        new_num_2 = 5
        # left = 0, right = 6, pivot = 3
        # lst[piovt]=4 <5, left=pivot=3, piovt = 4
        # lst[pivot]=5 <=5, left=pivot=4, pivot = 5
        # lst[pivot]=5 <=5, left=pivot=5,
        assert sort.add_to_topk_binary(lst_2,new_num_2) is None
        assert sort.add_to_topk_binary(lst_2[::-1],new_num_2,True) == 3
        lst_3 = [5,5,5,6,7,8,9,10,11,12]
        new_num_3 = 5
        assert sort.add_to_topk_binary(lst_3,new_num_3) == 3
        assert sort.add_to_topk_binary(lst_3[::-1],new_num_3,True) is None
        lst_4 = [1,2,3,None,None]
        new_num_4 = 5
        assert sort.add_to_topk_binary(lst_4,new_num_4) == 3
        lst_4 = [3,2,1,None,None]
        assert sort.add_to_topk_binary(lst_4,new_num_4,True) == 0
        lst_5 = [1,2,3,4,6,7,8,None,None]
        new_num_5 = 5
        assert sort.add_to_topk_binary(lst_5,new_num_5) == 4
        lst_5 = [8,7,6,4,3,2,1,None,None]
        assert sort.add_to_topk_binary(lst_5,new_num_5,True) == 3
        lst_6 = [None,None,None,None,None,None]
        new_num_6 = 5
        assert sort.add_to_topk_binary(lst_6,new_num_6) == 0
        assert sort.add_to_topk_binary(lst_6,new_num_6,True) == 0

    def test_euler_distance(self):
        '''
        test distance.euler_distance
        '''
        a = np.array([1,2,3,4,5])
        b = np.array([2,3,4,5,6])
        assert distance.euler_distance(a,b) == math.sqrt(5)

    def test_date_loader(self):
        '''
        test data.data_loader
        '''
        x,y = data.date_loader('knn')
        assert isinstance(x,np.ndarray)
        assert isinstance(y,np.ndarray)

    def test_train_test_split(self):
        '''
        Test data.train_test_split.
        '''
        # X,y have different size
        X=[[1,1,1],[2,2,2],[3,3,3]]
        y=[1,2]
        with pytest.raises(ValueError, match=r"Features X and labels y should have same length."):
            data.train_test_split(X,y,train_size=0.8,test_size=0.2)

        # tran_size or test_size < 0
        X=[[1,1,1],[2,2,2],[3,3,3]]
        y=[1,2,3]
        # train_size and test_size are not none
        with pytest.raises(ValueError, match=r"Train and test size should > 0"):
            data.train_test_split(X,y,0,0)
        with pytest.raises(ValueError, match=r"Train and test size should > 0"):
            data.train_test_split(X,y,-0.2,0)
        with pytest.raises(ValueError, match=r"Train and test size should > 0"):
            data.train_test_split(X,y,0,-0.2)
        with pytest.raises(ValueError, match=r"Train and test size should > 0"):
            data.train_test_split(X,y,-0.2,-0.2)
        with pytest.raises(ValueError, match=r"Train and test size should > 0"):
            data.train_test_split(X,y,0.2,-0.2)
        with pytest.raises(ValueError, match=r"Train and test size should > 0"):
            data.train_test_split(X,y,-0.2,0.2)

        # test_size is none.
        with pytest.raises(ValueError, match=r"Train size should > 0"):
            data.train_test_split(X,y,-0.2,None)
        with pytest.raises(ValueError, match=r"Train size should > 0"):
            data.train_test_split(X,y,0,None)
        with pytest.raises(ValueError, match=r"Train size should < 1"):
            data.train_test_split(X,y,1,None)
        with pytest.raises(ValueError, match=r"Train size should < 1"):
            data.train_test_split(X,y,1.2,None)

        # train_size is none
        with pytest.raises(ValueError, match=r"Test size should > 0"):
            data.train_test_split(X,y,None,-0.2)
        with pytest.raises(ValueError, match=r"Test size should > 0"):
            data.train_test_split(X,y,None,0)
        with pytest.raises(ValueError, match=r"Test size should < 1"):
            data.train_test_split(X,y,None,1)
        with pytest.raises(ValueError, match=r"Test size should < 1"):
            data.train_test_split(X,y,None,1.2)
