'''
tools tests
'''
from MLToolbox.tools import sort

class TestTools:
    '''
    test tools
    '''
    def test_add_to_topk(self):
        '''
        test sort
        '''
        lst = [1,2,3,4,5,5,5,6,7,8]
        new_num = 5
        assert sort.add_to_topk(lst,new_num) >= 4
        assert sort.add_to_topk(lst,new_num) <= 7

    def test_add_to_topk_binary(self):
        '''
        test add_to_topk_binary
        '''
        lst_1 = [1,2,3,4,5,5,5,6,7,8]
        new_num_1 = 5
        assert sort.add_to_topk_binary(lst_1,new_num_1) == 7
        lst_2 = [1,2,3,4,5,5,5]
        new_num_2 = 5
        # left = 0, right = 6, pivot = 3
        # lst[piovt]=4 <5, left=pivot=3, piovt = 4
        # lst[pivot]=5 <=5, left=pivot=4, pivot = 5
        # lst[pivot]=5 <=5, left=pivot=5,
        assert sort.add_to_topk_binary(lst_2,new_num_2) == 6
        lst_3 = [5,5,5,6,7,8,9,10,11,12]
        new_num_3 = 5
        assert sort.add_to_topk_binary(lst_3,new_num_3) == 3
        lst_4 = [1,2,3,None,None]
        new_num_4 = 5
        assert sort.add_to_topk_binary(lst_4,new_num_4) == 3
        lst_5 = [1,2,3,4,6,7,8,None,None]
        new_num_5 = 5
        assert sort.add_to_topk_binary(lst_5,new_num_5) == 4
