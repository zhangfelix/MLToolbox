# import sys
# import os

# if not __package__:
#     path = os.path.join(os.path.dirname(__file__), os.pardir)
#     sys.path.insert(0, path)

from MLToolbox.neighbors.knn import KNN
from MLToolbox.tools import sort
def main():
    print('You find the entry_point!')

if __name__ == "__main__":
    main()
    # a = KNN()
    lst_2 = [1,2,3,4,5,5,5]
    new_num_2 = 5
    # left = 0, right = 6, pivot = 3
    # lst[piovt]=4 <5, left=pivot=3, piovt = 4
    # lst[pivot]=5 <=5, left=pivot=4, pivot = 5
    # lst[pivot]=5 <=5, left=pivot=5,
    print(sort.add_to_topk_binary(lst_2,new_num_2))
    # assert sort.add_to_topk_binary(lst_2,new_num_2) == 4
