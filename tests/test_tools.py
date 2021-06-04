from MLToolbox.tools import sort

class TestTools:
    def test_sort(self):
        lst = [1,2,3,4,5,5,5,6,7,8]
        new_num = 5
        assert sort.add_to_topk(lst,new_num) >= 4
        assert sort.add_to_topk(lst,new_num) <= 6
